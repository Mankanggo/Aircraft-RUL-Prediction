"""
The health index (HI) is a SUPERVISED feature: a ridge regression fitted on the
training-fold engines' labels (min(RUL, 125) / 125). These tests pin down exactly
where labels can and cannot enter:

  * labels enter ONLY through `training_rul`, ONLY inside `fit`, and ONLY for the
    engines passed to `fit` (the training fold);
  * training rows get cross-fitted HI values: an engine's HI never comes from a ridge
    fitted on its own labels;
  * `transform` (used for validation samples and the official test) never reads any
    label: validation/test feature values are invariant to their own labels;
  * any RUL column supplied in the input is ignored;
  * changing the TRAINING labels does change the HI - by design (it is supervised).
"""

import numpy as np
import pandas as pd
import pytest

import src.components.data_transformation as dt
from src.components.data_transformation import FEATURE_GROUPS, FeatureConfig, FoldPreprocessor
from src.components.validation import iter_folds, last_observed_rows, make_cv_plan, make_truncated_samples

HI_CONFIG = FeatureConfig(groups=("current", "mean"), mean_windows=(10,), health_index=True)
FULL = FeatureConfig(groups=FEATURE_GROUPS, mean_windows=(5, 10, 20, 30), std_windows=(10, 30), slope_windows=(10, 20, 30),
                     minmax_windows=(10, 30), derived=True, health_index=True)
HI_COLS = ["hi", "hi_slope10", "hi_slope30", "hi_expmax", "hi_expmin"]


@pytest.fixture(scope="module")
def fold(cmapss):
    train = cmapss["FD002"]["train"]
    folds, cuts = make_cv_plan(train)
    _, tr, va = next(iter_folds(train, folds))
    return tr, va, cuts[cuts.fold == 0]


def _val_features(pre, va, cuts):
    samples = make_truncated_samples(va, cuts)
    return last_observed_rows(pre.transform(samples, group_cols=("sample_id",))).set_index("sample_id")


def test_rul_column_in_input_is_ignored(fold):
    tr, va, cuts = fold
    rng = np.random.default_rng(0)
    a = FoldPreprocessor(config=FULL).fit(tr)
    b = FoldPreprocessor(config=FULL).fit(tr.assign(RUL=rng.integers(0, 500, len(tr))))
    np.testing.assert_array_equal(a.hi_coef_, b.hi_coef_)
    fa = _val_features(a, va.assign(RUL=0), cuts)
    fb = _val_features(b, va.assign(RUL=rng.integers(0, 500, len(va))), cuts)
    pd.testing.assert_frame_equal(fa[a.feature_names_], fb[b.feature_names_])


def test_transform_never_reads_labels(fold, monkeypatch):
    """Validation/test features are computed with zero label access."""
    tr, va, cuts = fold
    pre = FoldPreprocessor(config=FULL).fit(tr)

    def forbidden(*_a, **_k):
        raise AssertionError("label access during transform")

    monkeypatch.setattr(dt, "training_rul", forbidden)
    feats = _val_features(pre, va, cuts)  # would raise if transform touched labels
    assert set(HI_COLS) <= set(feats.columns)


def test_validation_features_invariant_to_validation_labels(fold):
    """
    Change every validation engine's label (its lifespan) without touching the
    cycles up to each cut: append 40 fake post-failure cycles, or delete all cycles
    after the cut. True RUL at the cut changes; the features at the cut must not.
    """
    tr, va, cuts = fold
    pre = FoldPreprocessor(config=FULL).fit(tr)
    base = _val_features(pre, va, cuts)

    longer = pd.concat([va, va.loc[va.groupby("unit_id").cycle.idxmax()].loc[lambda d: d.index.repeat(40)]
                        .assign(cycle=lambda d: d.cycle + d.groupby("unit_id").cumcount() + 1)], ignore_index=True)
    cut_of = cuts.groupby("unit_id").cut_cycle.max()
    shorter = va[va.cycle <= va.unit_id.map(cut_of)]
    for altered in (longer, shorter):
        new_life = altered.groupby("unit_id").cycle.max()
        assert not new_life.equals(va.groupby("unit_id").cycle.max()), "labels must actually change"
        feats = _val_features(pre, altered, cuts)
        pd.testing.assert_frame_equal(feats[pre.feature_names_], base[pre.feature_names_])


def test_label_access_is_training_fold_only(fold, monkeypatch):
    tr, va, cuts = fold
    calls = []
    real = dt.training_rul

    def spy(df):
        calls.append(set(df.unit_id))
        return real(df)

    monkeypatch.setattr(dt, "training_rul", spy)
    pre = FoldPreprocessor(config=FULL).fit(tr)
    _val_features(pre, va, cuts)
    assert len(calls) == 1, "labels read exactly once, in fit"
    assert calls[0] == set(tr.unit_id) and calls[0].isdisjoint(set(va.unit_id))


def _unscaled_hi(pre, feats):
    return feats[HI_COLS].astype(float) * pre.std_[HI_COLS] + pre.mean_[HI_COLS]


def test_training_rows_hi_does_not_use_their_own_labels(fold, monkeypatch):
    """
    Cross-fitting: a training engine's HI values come from a ridge fitted on OTHER training
    engines, so corrupting that engine's labels leaves its own HI unchanged (it only
    changes the HI of engines whose ridge used it).
    """
    tr, _, _ = fold
    victim = sorted(tr.unit_id.unique())[3]
    a_pre = FoldPreprocessor(config=HI_CONFIG)
    a = a_pre.fit_transform(tr)
    real = dt.training_rul

    def corrupted(df):
        y = real(df).astype(float)
        m = (df["unit_id"] == victim).to_numpy()
        y[m] = np.random.default_rng(1).uniform(0, 400, m.sum())  # nonsense labels for one engine
        return y

    monkeypatch.setattr(dt, "training_rul", corrupted)
    b_pre = FoldPreprocessor(config=HI_CONFIG)
    b = b_pre.fit_transform(tr)
    is_victim = tr["unit_id"] == victim
    hi_a, hi_b = _unscaled_hi(a_pre, a), _unscaled_hi(b_pre, b)
    np.testing.assert_allclose(hi_a[is_victim].to_numpy(), hi_b[is_victim].to_numpy(), rtol=1e-4, atol=1e-4)
    assert not np.allclose(hi_a[~is_victim]["hi"], hi_b[~is_victim]["hi"], atol=1e-4), "corruption must have an effect"
    assert not np.allclose(a_pre.hi_coef_, b_pre.hi_coef_)  # the all-engine ridge (for new data) does use it


def test_hi_does_depend_on_training_labels(fold, monkeypatch):
    """Documents the design: the HI is supervised, so permuting TRAINING labels changes it."""
    tr, va, cuts = fold
    a = FoldPreprocessor(config=HI_CONFIG).fit(tr)
    real = dt.training_rul
    monkeypatch.setattr(dt, "training_rul", lambda df: np.random.default_rng(0).permutation(real(df)))
    b = FoldPreprocessor(config=HI_CONFIG).fit(tr)
    assert not np.allclose(a.hi_coef_, b.hi_coef_)
    fa, fb = _val_features(a, va, cuts), _val_features(b, va, cuts)
    assert not np.allclose(fa["hi"], fb["hi"])
    # the non-HI features are unaffected by training labels
    other = [c for c in a.feature_names_ if not c.startswith("hi")]
    pd.testing.assert_frame_equal(fa[other], fb[other])
