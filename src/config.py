from pathlib import Path


# ---------------------------------------------------------
# Project folders
# ---------------------------------------------------------

# __file__ is the path of this file: src/config.py   Path(__file__) lo trasforma in un oggetto Path.
# .resolve() ottiene il percorso assoluto completo. parent is the src folder parent.parent is the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------
# Data files
# ---------------------------------------------------------

RAW_DATA_FILE = RAW_DATA_DIR / "household_power_consumption.txt"
#Il dataset originale
PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR / "hourly_energy_consumption.csv"
)
#risultato della nostra pulizia e aggregazione.


# ---------------------------------------------------------
# Original dataset columns
# ---------------------------------------------------------

DATE_COLUMN = "Date"
TIME_COLUMN = "Time"

RAW_FEATURE_COLUMNS = [
    #colonne che leggeremo dal dataset.
    "Global_active_power",
    #potenza attiva globale misurata per l’intera abitazione.
    "Sub_metering_1",
    #consumo relativo alla cucina.
    "Sub_metering_2",
    #consumo relativo alla lavanderia.
    "Sub_metering_3",
    #consumo associato allo scaldabagno elettrico e al sistema di climatizzazione.
]


# ---------------------------------------------------------
# Processed feature names
# ---------------------------------------------------------

MODEL_FEATURE_COLUMNS = [
    #Questi non sono i nomi originali del dataset. Saranno creati in data.py.
    "global_energy_kwh",
    "kitchen_energy_kwh",
    "laundry_energy_kwh",
    "water_heater_ac_energy_kwh",
]


# ---------------------------------------------------------
# Temporal preprocessing
# ---------------------------------------------------------

DATETIME_COLUMN = "datetime"

# Pandas uses "h" to indicate hourly aggregation.
RESAMPLE_FREQUENCY = "h"
#Il dataset originale contiene una misurazione al minuto. Noi lavoreremo con dati orari. un regime orario è più significativo di un regime che cambia ogni minuto.

MIN_VALID_MINUTES_PER_HOUR = 60
#Consideriamo valida un’ora solo se possediamo tutte e 60 le osservazioni al minuto.

# ---------------------------------------------------------
# Temporal dataset split
# ---------------------------------------------------------

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


# ---------------------------------------------------------
# Model settings
# ---------------------------------------------------------

RANDOM_SEED = 42

N_STATES_CANDIDATES = [2, 3, 4, 5]
#Non decidiamo immediatamente quanti regimi siano presenti. Addestreremo modelli con: K=2,K=3,K=4,K=5. K è il numero di stati latenti.

N_INITIALIZATIONS = 5
#Expectation Maximization può convergere verso differenti massimi locali. Alla fine conserveremo il modello con la likelihood di training più alta.
MAX_ITERATIONS = 100
#limite al numero di iterazioni di EM.
CONVERGENCE_TOLERANCE = 1e-3
COVARIANCE_TYPE = "diag"
MIN_COVARIANCE = 1e-3