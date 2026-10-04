"""
Correctness and causality of every feature group (incl. derived sensors and the
fitted health index), checked against brute-force recomputation.
"""

import numpy as np
import pandas as pd
import pytest

from src.components.data_transformation import (
    FEATURE_GROUPS,
    FeatureConfig,
    FoldPreprocessor,
    add_causal_features,
    add_derived_sensors,
)
from src.components.validation import build_fold_data, iter_folds, make_cv_plan
from src.data_schema import SENSOR_COLUMNS

FULL = FeatureConfig(groups=FEATURE_GROUPS, mean_windows=(5, 10, 20, 30), std_windows=(10, 30), slope_windows=(10, 20, 30),
                     minmax_windows=(10, 30), derived=True, health_index=True, include_cycle=True)


def _toy(n_units=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, 40 + 10 * u):
            rows.append({"unit_id": u, "cycle": c, "a": rng.normal() + 0.05 * c, "b": rng.normal()})
    return pd.DataFrame(rows)


def test_feature_values_match_brute_force():
    df = _toy()
    cfg = FeatureConfig(groups=FEATURE_GROUPS, mean_windows=(5, 10), std_windows=(10,), slope_windows=(10,),
                        minmax_windows=(10,), delta_lags=(7,), momentum_long=20, smooth_window=5, baseline_cycles=8)
    f = add_causal_features(df, ["a", "b"], config=cfg)
    for u, g in df.groupby("unit_id"):
        x = g["a"].to_numpy()
        fu = f[f.unit_id == u].sort_values("cycle")
        for i in [0, 3, 6, 9, 15, len(x) - 1]:
            t = i + 1
            m5 = lambda j: x[max(0, j - 4): j + 1].mean()  # trailing mean of 5 ending at index j
            w10 = x[max(0, i - 9): i + 1]
            row = fu.iloc[i]
            assert np.isclose(row["a"], x[i])
            assert np.isclose(row["a_mean10"], w10.mean())
            assert np.isclose(row["a_std10"], w10.std(ddof=1) if len(w10) > 1 else 0.0)
            assert np.isclose(row["a_min10"], w10.min()) and np.isclose(row["a_max10"], w10.max())
            slope = np.polyfit(np.arange(len(w10)), w10, 1)[0] if len(w10) > 1 else 0.0
            assert np.isclose(row["a_slope10"], slope)
            assert np.isclose(row["a_diff7"], m5(i) - m5(max(i - 7, 0)))
            assert np.isclose(row["a_momentum"], m5(i) - x[max(0, i - 19): i + 1].mean())
            assert np.isclose(row["a_delta_base"], m5(i) - x[: min(t, 8)].mean())
            assert np.isclose(row["a_expmax"], max(m5(j) for j in range(i + 1)))
            assert np.isclose(row["a_expmin"], min(m5(j) for j in range(i + 1)))


def test_all_groups_prefix_invariant_float64():
    df = _toy(n_units=4, seed=1)
    cfg = FeatureConfig(groups=FEATURE_GROUPS, mean_windows=(5, 10, 20, 30), std_windows=(10, 30), slope_windows=(10, 20, 30))
    full = add_causal_features(df, ["a", "b"], config=cfg)
    cols = [c for c in full.columns if c not in ("unit_id", "cycle")]
    for u in df.unit_id.unique():
        for cut in (1, 2, 9, 25):
            pre = add_causal_features(df[(df.unit_id == u) & (df.cycle <= cut)], ["a", "b"], config=cfg)
            ref = full[(full.unit_id == u) & (full.cycle <= cut)]
            np.testing.assert_allclose(pre[cols].to_numpy(), ref[cols].to_numpy(), rtol=1e-12, atol=1e-12)


def test_derived_sensors_are_row_wise(cmapss):
    df = cmapss["FD001"]["train"].head(300)
    d = add_derived_sensors(df)
    assert np.allclose(d["Wf_est"], df["phi"] * df["Ps30"])
    assert np.allclose(d["dT_HPC"], df["T30"] - df["T24"])
    assert np.allclose(d["Nc_over_Nf"], df["Nc"] / df["Nf"])
    assert np.allclose(d["T50_over_T30"], df["T50"] / df["T30"])
    # row-wise: shuffling rows permutes outputs identically
    shuffled = add_derived_sensors(df.sample(frac=1, random_state=0)).loc[df.index]
    pd.testing.assert_frame_equal(shuffled, d)


@pytest.mark.parametrize("subset", ["FD001", "FD004"])
@pytest.mark.parametrize("normalize", [True, False])
def test_full_pipeline_prefix_invariant(cmapss, subset, normalize):
    train = cmapss[subset]["train"]
    cfg = FULL.with_(normalize=normalize)
    pre = FoldPreprocessor(config=cfg).fit(train)
    units = np.random.default_rng(0).choice(train.unit_id.unique(), size=6, replace=False)
    df = train[train.unit_id.isin(units)]
    full = pre.transform(df)
    for u in units:
        life = df.loc[df.unit_id == u, "cycle"].max()
        for cut in (1, 7, life // 2, life - 6):
            part = pre.transform(df[(df.unit_id == u) & (df.cycle <= cut)]).sort_values("cycle")
            ref = full[(full.unit_id == u) & (full.cycle <= cut)].sort_values("cycle")
            np.testing.assert_allclose(part[pre.feature_names_].to_numpy(), ref[pre.feature_names_].to_numpy(),
                                       rtol=1e-5, atol=1e-5)


def test_future_tampering_with_full_config(cmapss):
    train = cmapss["FD003"]["train"]
    pre = FoldPreprocessor(config=FULL).fit(train)
    df = train[train.unit_id.isin([1, 2, 3])].copy()
    df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].astype(float)
    base = pre.transform(df)
    future = df.cycle > 60
    df.loc[future, SENSOR_COLUMNS] *= np.random.default_rng(1).uniform(0.7, 1.3, size=(future.sum(), len(SENSOR_COLUMNS)))
    after = pre.transform(df)
    np.testing.assert_allclose(base.loc[~future, pre.feature_names_].to_numpy(), after.loc[~future, pre.feature_names_].to_numpy(),
                               rtol=1e-6, atol=1e-6)


def test_health_index_fitted_on_training_engines_only(cmapss):
    train = cmapss["FD002"]["train"]
    folds, cuts = make_cv_plan(train)
    _, tr, va = next(iter_folds(train, folds))
    a = FoldPreprocessor(config=FULL).fit(tr)
    weird = pd.concat([tr, va.assign(**{c: va[c] * 5.0 for c in SENSOR_COLUMNS})])
    # fitting on training engines gives the same HI whatever the validation engines contain...
    b = FoldPreprocessor(config=FULL).fit(weird[weird.unit_id.isin(tr.unit_id)])
    np.testing.assert_allclose(a.hi_coef_, b.hi_coef_)
    # ...and transform never refits it
    coef = a.hi_coef_.copy()
    a.transform(va.assign(**{c: va[c] * 5.0 for c in SENSOR_COLUMNS}))
    np.testing.assert_array_equal(a.hi_coef_, coef)
    assert {"health_index", "cycle"} <= set(a.feature_meta_.group)
    assert a.feature_meta_.feature.is_unique and list(a.feature_meta_.feature) == a.feature_names_


def test_build_fold_data_full_config_is_engine_disjoint(cmapss):
    train = cmapss["FD001"]["train"]
    train = train[train.unit_id <= 30]
    folds, cuts = make_cv_plan(train, n_splits=3, n_cuts=2)
    fds = build_fold_data(train, folds, cuts, lambda: FoldPreprocessor(config=FULL))
    fold_of = folds.set_index("unit_id")["fold"]
    for fd in fds:
        assert fd.preprocessor.fit_units_.isdisjoint(set(fd.val_meta.unit_id))
        assert (fd.val_meta.unit_id.map(fold_of) == fd.fold).all()
        assert fd.X_train.shape[1] == fd.X_val.shape[1] == len(fd.feature_names)
        assert np.isfinite(fd.X_train).all() and np.isfinite(fd.X_val).all()
        assert (fd.val_meta.y_true == fd.val_meta.lifespan - fd.val_meta.cut_cycle).all()
