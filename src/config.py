from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_DATA_FILE = RAW_DATA_DIR / "household_power_consumption.txt"
PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR / "hourly_energy_consumption.csv"
)


DATE_COLUMN = "Date"
TIME_COLUMN = "Time"

RAW_FEATURE_COLUMNS = [
    "Global_active_power",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

MODEL_FEATURE_COLUMNS = [
    "global_energy_kwh",
    "kitchen_energy_kwh",
    "laundry_energy_kwh",
    "water_heater_ac_energy_kwh",
]

DATETIME_COLUMN = "datetime"

RESAMPLE_FREQUENCY = "h"

MIN_VALID_MINUTES_PER_HOUR = 60

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

RANDOM_SEED = 42

N_STATES_CANDIDATES = [2, 5, 8, 9, 10, 11, 12]
N_INITIALIZATIONS = 5
MAX_ITERATIONS = 200
CONVERGENCE_TOLERANCE = 1e-3
COVARIANCE_TYPE = "diag"
MIN_COVARIANCE = 1e-3