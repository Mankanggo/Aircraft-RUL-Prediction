"""Correctness of the RUL target for training, test and validation samples."""

import numpy as np
import pandas as pd
import pytest

from src.components.validation import make_truncated_samples, sample_truncation_cuts
from src.data_schema import SUBSETS
from src.utils import add_train_rul, cap_rul


@pytest.mark.parametrize("subset", SUBSETS)
def test_train_rul_counts_down_to_zero_at_failure(cmapss, subset):
    df = cmapss[subset]["train"]
    life = df.groupby("unit_id")["cycle"].transform("max")
    assert (df["RUL"] == life - df["cycle"]).all()
    last = df.loc[df.groupby("unit_id")["cycle"].idxmax()]
    assert (last["RUL"] == 0).all(), "failure cycle must have RUL 0"
    first = df.loc[df.groupby("unit_id")["cycle"].idxmin()]
    assert (first["RUL"] == first["unit_id"].map(df.groupby("unit_id")["cycle"].max()) - 1).all()
    steps = df.sort_values(["unit_id", "cycle"]).groupby("unit_id")["RUL"].diff().dropna()
    assert (steps == -1).all(), "RUL must decrease by exactly 1 per cycle"
    assert df["RUL"].min() == 0 and pd.api.types.is_integer_dtype(df["RUL"])


@pytest.mark.parametrize("subset", SUBSETS)
def test_test_rul_matches_rul_file_at_last_cycle(cmapss, subset):
    df, rul = cmapss[subset]["test"], cmapss[subset]["rul"]
    assert len(rul) == df["unit_id"].nunique()
    assert list(rul["unit_id"]) == list(range(1, len(rul) + 1))
    last = df.loc[df.groupby("unit_id")["cycle"].idxmax()].set_index("unit_id")["RUL"]
    assert (last.sort_index().to_numpy() == rul.set_index("unit_id")["RUL"].sort_index().to_numpy()).all()
    steps = df.sort_values(["unit_id", "cycle"]).groupby("unit_id")["RUL"].diff().dropna()
    assert (steps == -1).all()
    assert (df["RUL"] >= rul["RUL"].min()).all()


def test_cap_rul_semantics():
    rul = pd.Series([300, 200, 126, 125, 124, 10, 0])
    assert cap_rul(rul, None) is rul
    capped = cap_rul(rul, 125)
    assert capped.tolist() == [125, 125, 125, 125, 124, 10, 0]
    assert cap_rul(np.array([130, 5]), 100).tolist() == [100, 5]
    assert cap_rul(150, 125) == 125
    for bad in (0, -5):
        with pytest.raises(ValueError):
            cap_rul(rul, bad)


@pytest.mark.parametrize("subset", SUBSETS)
@pytest.mark.parametrize("cap", [90, 125, 150])
def test_capped_train_target(cmapss, subset, cap):
    raw = cmapss[subset]["train"].drop(columns="RUL")
    capped = add_train_rul(raw, cap=cap)
    uncapped = add_train_rul(raw)
    assert (capped["RUL"] == np.minimum(uncapped["RUL"], cap)).all()
    assert capped["RUL"].max() <= cap
    below = uncapped["RUL"] < cap
    assert (capped.loc[below, "RUL"] == uncapped.loc[below, "RUL"]).all(), "linear part must be untouched"
    steps = capped.sort_values(["unit_id", "cycle"]).groupby("unit_id")["RUL"].diff().dropna()
    assert steps.isin([0, -1]).all(), "capped target is flat, then decreases by 1 per cycle"


def test_add_train_rul_does_not_mutate_input(cmapss):
    raw = cmapss["FD001"]["train"].drop(columns="RUL")
    before = raw.copy()
    add_train_rul(raw, cap=125)
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize("subset", SUBSETS)
def test_validation_target_equals_train_target_at_cut(cmapss, subset):
    """A truncated engine's target must be the same number add_train_rul gives at that cycle."""
    train = cmapss[subset]["train"]
    cuts = sample_truncation_cuts(train, n_cuts=3, strategy="uniform", seed=0)
    at_cut = cuts.merge(train[["unit_id", "cycle", "RUL"]], left_on=["unit_id", "cut_cycle"], right_on=["unit_id", "cycle"])
    assert len(at_cut) == len(cuts)
    assert (at_cut["rul_at_cut"] == at_cut["RUL"]).all()
    assert (cuts["rul_at_cut"] == cuts["lifespan"] - cuts["cut_cycle"]).all()

    samples = make_truncated_samples(train, cuts)
    assert "RUL" not in samples.columns, "truncated samples must not carry the target"
    last = samples.groupby("sample_id")["cycle"].agg(["max", "size"])
    expected = cuts.set_index("sample_id")["cut_cycle"]
    assert (last["max"] == expected).all() and (last["size"] == expected).all()
