"""Engine-level fold construction and test-like truncation."""

import numpy as np
import pandas as pd
import pytest

from src.components.validation import (
    assert_engine_disjoint,
    engine_lifespans,
    iter_folds,
    make_cv_plan,
    make_engine_folds,
    make_truncated_samples,
    sample_truncation_cuts,
)
from src.data_schema import SUBSETS


@pytest.mark.parametrize("subset", SUBSETS)
def test_folds_partition_engines(cmapss, subset):
    train = cmapss[subset]["train"]
    folds = make_engine_folds(train, n_splits=5, seed=42)
    assert folds["unit_id"].is_unique
    assert set(folds["unit_id"]) == set(train["unit_id"])
    sizes = folds["fold"].value_counts()
    assert len(sizes) == 5 and sizes.max() - sizes.min() <= 1

    n_rows = 0
    for _, tr, va in iter_folds(train, folds):
        assert set(tr["unit_id"]).isdisjoint(set(va["unit_id"]))
        assert len(tr) + len(va) == len(train)
        # whole engines: every row of a validation engine is in the validation part
        assert (va.groupby("unit_id").size() == train[train["unit_id"].isin(va["unit_id"])].groupby("unit_id").size()).all()
        n_rows += len(va)
    assert n_rows == len(train), "every row is validated exactly once"


@pytest.mark.parametrize("subset", SUBSETS)
def test_folds_are_lifespan_balanced(cmapss, subset):
    train = cmapss[subset]["train"]
    folds = make_engine_folds(train, n_splits=5, seed=42)
    overall = folds["lifespan"].median()
    medians = folds.groupby("fold")["lifespan"].median()
    assert (abs(medians - overall) / overall < 0.10).all()


def test_folds_are_deterministic(cmapss):
    train = cmapss["FD002"]["train"]
    a = make_engine_folds(train, seed=7)
    pd.testing.assert_frame_equal(a, make_engine_folds(train, seed=7))
    assert not a["fold"].equals(make_engine_folds(train, seed=8)["fold"])


def test_assert_engine_disjoint_detects_overlap():
    a = pd.DataFrame({"unit_id": [1, 1, 2], "cycle": [1, 2, 1]})
    b = pd.DataFrame({"unit_id": [2, 3], "cycle": [5, 1]})
    with pytest.raises(AssertionError):
        assert_engine_disjoint(a, b)
    assert_engine_disjoint(a, b[b.unit_id == 3])


@pytest.mark.parametrize("strategy", ["test_lengths", "uniform"])
def test_truncation_cuts_are_valid(cmapss, strategy):
    train, test = cmapss["FD004"]["train"], cmapss["FD004"]["test"]
    ref = test.groupby("unit_id")["cycle"].max().to_numpy()
    cuts = sample_truncation_cuts(train, n_cuts=10, strategy=strategy, reference_lengths=ref, min_rul=6, max_rul=150, seed=1)
    life = engine_lifespans(train)
    assert (cuts["lifespan"] == cuts["unit_id"].map(life)).all()
    assert cuts["rul_at_cut"].between(6, 150).all() and (cuts["cut_cycle"] >= 1).all()
    assert (cuts["cut_cycle"] < cuts["lifespan"]).all()
    assert not cuts.duplicated(["unit_id", "cut_cycle"]).any()
    assert cuts.groupby("unit_id").size().max() <= 10
    assert cuts["sample_id"].is_unique
    if strategy == "test_lengths":
        assert cuts["cut_cycle"].isin(set(ref)).all(), "observed lengths must come from the test length set"
    pd.testing.assert_frame_equal(
        cuts, sample_truncation_cuts(train, 10, strategy, ref, 6, 150, seed=1)
    )


def test_max_rul_bounds_the_target(cmapss):
    train = cmapss["FD003"]["train"]  # longest-lived engines (up to 525 cycles)
    default = sample_truncation_cuts(train, n_cuts=10, seed=0)
    assert default["rul_at_cut"].between(6, 150).all()
    unbounded = sample_truncation_cuts(train, n_cuts=10, max_rul=None, seed=0)
    assert unbounded["rul_at_cut"].max() > 150
    # an engine shorter than max_rul + 1 is cut anywhere from cycle 1
    short = train[train["unit_id"] == train.groupby("unit_id")["cycle"].max().idxmin()]
    c = sample_truncation_cuts(short, n_cuts=200, max_rul=1000, seed=0)
    assert c["cut_cycle"].min() >= 1 and c["rul_at_cut"].max() == short["cycle"].max() - c["cut_cycle"].min()


def test_truncated_samples_are_exact_prefixes(cmapss):
    train = cmapss["FD001"]["train"]
    cuts = sample_truncation_cuts(train, n_cuts=4, strategy="uniform", seed=3)
    samples = make_truncated_samples(train, cuts)
    for _, c in cuts.sample(20, random_state=0).iterrows():
        s = samples[samples["sample_id"] == c["sample_id"]].drop(columns="sample_id").reset_index(drop=True)
        prefix = train[(train["unit_id"] == c["unit_id"]) & (train["cycle"] <= c["cut_cycle"])]
        prefix = prefix.drop(columns="RUL").reset_index(drop=True)
        pd.testing.assert_frame_equal(s[prefix.columns], prefix)


@pytest.mark.parametrize("subset", SUBSETS)
def test_cv_plan_cuts_belong_to_their_fold(cmapss, subset):
    train, test = cmapss[subset]["train"], cmapss[subset]["test"]
    ref = test.groupby("unit_id")["cycle"].max().to_numpy()
    folds, cuts = make_cv_plan(train, ref)
    fold_of = folds.set_index("unit_id")["fold"]
    assert (cuts["unit_id"].map(fold_of) == cuts["fold"]).all()
    assert cuts["sample_id"].is_unique
    # nearly every engine is validated (engines too short for any feasible cut are skipped)
    assert cuts["unit_id"].nunique() >= 0.98 * train["unit_id"].nunique()
