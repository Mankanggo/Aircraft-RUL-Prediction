import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_schema import SUBSETS  # noqa: E402
from src.utils import add_test_rul, add_train_rul, read_cmapss_file, read_rul_file  # noqa: E402

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
HAS_DATA = all(os.path.exists(os.path.join(RAW_DIR, f"train_{s}.txt")) for s in SUBSETS)
requires_data = pytest.mark.skipif(not HAS_DATA, reason="raw C-MAPSS files not found in data/raw")


@pytest.fixture(scope="session")
def cmapss():
    """All subsets, loaded once: {subset: {"train", "test", "rul"}} (train/test with RUL)."""
    if not HAS_DATA:
        pytest.skip("raw C-MAPSS files not found in data/raw")
    out = {}
    for s in SUBSETS:
        rul = read_rul_file(s, RAW_DIR)
        out[s] = {
            "train": add_train_rul(read_cmapss_file(s, "train", RAW_DIR)),
            "test": add_test_rul(read_cmapss_file(s, "test", RAW_DIR), rul),
            "rul": rul,
        }
    return out
