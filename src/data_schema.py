"""
Verified schema for the NASA C-MAPSS turbofan run-to-failure data.

Source of truth for sensor names/descriptions/units:
    Saxena, Goebel, Simon & Eklund (2008), "Damage Propagation Modeling for
    Aircraft Engine Run-to-Failure Simulation", PHM08 - Table 2,
    "Parameters available to participants as sensor data" (docs/).

Table 2 lists exactly 21 sensor parameters and docs/readme.txt states that
columns 6-26 are "sensor measurement 1..21". The mapping below is positional
(sensor k = k-th row of Table 2). The paper does not state the column order
explicitly, so notebooks/01_eda.ipynb checks it against physical magnitudes
(e.g. sensor 1 = 518.67 degR = ISA sea-level T2, sensor 5 = 14.62 psia = P2).

The three operational settings are not mapped to named quantities in the
paper or readme (the paper only says the operating conditions are
combinations of altitude, TRA and Mach number), so they keep generic names.
"""

from dataclasses import dataclass
from typing import Dict, List

SUBSETS: List[str] = ["FD001", "FD002", "FD003", "FD004"]

# Metadata from docs/readme.txt (number of trajectories, conditions, fault modes)
SUBSET_INFO: Dict[str, Dict[str, object]] = {
    "FD001": {"conditions": 1, "fault_modes": "HPC", "train_units": 100, "test_units": 100},
    "FD002": {"conditions": 6, "fault_modes": "HPC", "train_units": 260, "test_units": 259},
    "FD003": {"conditions": 1, "fault_modes": "HPC + Fan", "train_units": 100, "test_units": 100},
    "FD004": {"conditions": 6, "fault_modes": "HPC + Fan", "train_units": 248, "test_units": 249},
}

ID_COLUMNS: List[str] = ["unit_id", "cycle"]
SETTING_COLUMNS: List[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]


@dataclass(frozen=True)
class Sensor:
    index: int  # 1-based position among the 21 sensor columns
    symbol: str  # Table 2 symbol, used as the column name
    description: str
    units: str  # "--" = dimensionless, as printed in Table 2

    @property
    def raw_name(self) -> str:
        return f"sensor_{self.index}"


# Table 2, in order. Units: degR = degrees Rankine, pps/psi = (lbm/s)/psi.
SENSORS: List[Sensor] = [
    Sensor(1, "T2", "Total temperature at fan inlet", "°R"),
    Sensor(2, "T24", "Total temperature at LPC outlet", "°R"),
    Sensor(3, "T30", "Total temperature at HPC outlet", "°R"),
    Sensor(4, "T50", "Total temperature at LPT outlet", "°R"),
    Sensor(5, "P2", "Pressure at fan inlet", "psia"),
    Sensor(6, "P15", "Total pressure in bypass-duct", "psia"),
    Sensor(7, "P30", "Total pressure at HPC outlet", "psia"),
    Sensor(8, "Nf", "Physical fan speed", "rpm"),
    Sensor(9, "Nc", "Physical core speed", "rpm"),
    Sensor(10, "epr", "Engine pressure ratio (P50/P2)", "--"),
    Sensor(11, "Ps30", "Static pressure at HPC outlet", "psia"),
    Sensor(12, "phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    Sensor(13, "NRf", "Corrected fan speed", "rpm"),
    Sensor(14, "NRc", "Corrected core speed", "rpm"),
    Sensor(15, "BPR", "Bypass Ratio", "--"),
    Sensor(16, "farB", "Burner fuel-air ratio", "--"),
    Sensor(17, "htBleed", "Bleed Enthalpy", "--"),
    Sensor(18, "Nf_dmd", "Demanded fan speed", "rpm"),
    Sensor(19, "PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    Sensor(20, "W31", "HPT coolant bleed", "lbm/s"),
    Sensor(21, "W32", "LPT coolant bleed", "lbm/s"),
]

# The six operating-condition centres observed in FD002/FD004 (FD001/FD003 run
# only at the first one). The settings carry tiny noise (< 0.01) around these
# centres, so rounding to (0, 2, 0) decimals recovers them exactly (see EDA).
SETTING_ROUNDING: Dict[str, int] = {"op_setting_1": 0, "op_setting_2": 2, "op_setting_3": 0}
OPERATING_CONDITIONS: List[tuple] = [
    (0.0, 0.0, 100.0),
    (10.0, 0.25, 100.0),
    (20.0, 0.7, 100.0),
    (25.0, 0.62, 60.0),
    (35.0, 0.84, 100.0),
    (42.0, 0.84, 100.0),
]

SENSOR_COLUMNS: List[str] = [s.symbol for s in SENSORS]
FEATURE_COLUMNS: List[str] = SETTING_COLUMNS + SENSOR_COLUMNS

# EDA (notebooks/01_eda.ipynb, Section 16): constant in FD001/FD003 and fully
# determined by the operating condition in FD002/FD004 -> no health information.
NON_HEALTH_SENSORS: List[str] = ["T2", "P2", "Nf_dmd", "PCNfR_dmd"]
# Default candidate health sensors. Further selection is a hypothesis to validate.
CANDIDATE_SENSORS: List[str] = [c for c in SENSOR_COLUMNS if c not in NON_HEALTH_SENSORS]
RAW_COLUMNS: List[str] = ID_COLUMNS + FEATURE_COLUMNS  # 26 columns, file order

SENSOR_BY_SYMBOL: Dict[str, Sensor] = {s.symbol: s for s in SENSORS}
RAW_TO_SYMBOL: Dict[str, str] = {s.raw_name: s.symbol for s in SENSORS}

RUL_COLUMN = "RUL"


def sensor_label(symbol: str) -> str:
    """Human-readable axis label, e.g. 'T24 - Total temperature at LPC outlet (°R)'."""
    s = SENSOR_BY_SYMBOL[symbol]
    unit = "" if s.units == "--" else f" ({s.units})"
    return f"{s.symbol} - {s.description}{unit}"
