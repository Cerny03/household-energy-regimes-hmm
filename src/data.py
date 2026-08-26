import pandas as pd

from src.config import (
    #src.config serve a dire il percorso del file 
    RAW_DATA_FILE,
    DATE_COLUMN,
    TIME_COLUMN,
    DATETIME_COLUMN,
    RAW_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    RESAMPLE_FREQUENCY,
    MIN_VALID_MINUTES_PER_HOUR,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    TEST_FRACTION,
)

def load_raw_data():
    """
    Load the original household energy consumption dataset.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing date, time and the selected energy features.
    """

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{RAW_DATA_FILE}\n"
            "Place household_power_consumption.txt inside data/raw."
        )

    columns_to_load = [
        DATE_COLUMN,
        TIME_COLUMN,
        *RAW_FEATURE_COLUMNS,
        #* serve a spacchettare gli elementi di una lista dentro un'altra lista.
    ]

    data = pd.read_csv(
        RAW_DATA_FILE,
        sep=";",
        usecols=columns_to_load,
        na_values="?",
        #quando trovi il simbolo "?" nel CSV, trattalo come un valore mancante.
        low_memory=False,
        #analizzare i dati in modo più globale prima di determinare i tipi delle colonne.
    )

    return data

def create_datetime(data):
    """
    Combine the Date and Time columns into a single datetime column.
    """

    data = data.copy()
    #crea una copia del DataFrame ricevuto.

    data[DATETIME_COLUMN] = pd.to_datetime(
        #dice a pandas Interpreta questa stringa come una data e un'ora.
        data[DATE_COLUMN] + " " + data[TIME_COLUMN],
        format="%d/%m/%Y %H:%M:%S",
        #pandas interpreta correttamente: 16/12/200617:24:00.
    )

    data = data.sort_values(DATETIME_COLUMN)
    #ordina tutte le righe secondo il timestamp. Concettualmente vogliamo: t1<t2<...<tn : Per un HMM questa proprietà è fondamentale.
    data = data.reset_index(drop=True)
    #drop=trueNon conservare il vecchio indice come nuova colonna.

    return data

def print_time_summary(data):
    """
    Print basic information about the temporal structure of the dataset.
    """

    print("\nTime information:")

    print("First timestamp:")
    print(data[DATETIME_COLUMN].min())

    print("\nLast timestamp:")
    print(data[DATETIME_COLUMN].max())

    print("\nDatetime type:")
    print(data[DATETIME_COLUMN].dtype)

    print("\nDuplicated timestamps:")
    print(data[DATETIME_COLUMN].duplicated().sum())
    #data[DATETIME_COLUMN].duplicated() produce una serie di True e False. Contiamo i true e speriamo che il numero di timestamp duplicati sia 0

    print("\nChronologically ordered:") 
    print(data[DATETIME_COLUMN].is_monotonic_increasing)


def analyze_missing_values(data):
    """
    Analyze how missing values are distributed across the energy features.
    """

    missing = data[RAW_FEATURE_COLUMNS].isna()
    #selezioniamo soltanto le quattro variabili energetiche. missing contiene soltanto l'informazione "questo valore è mancante?".

    rows_with_any_missing = missing.any(axis=1)
    rows_with_all_missing = missing.all(axis=1)
    #controlla se tutte e quattro le feature sono mancanti nella stessa riga.

    number_any_missing = rows_with_any_missing.sum()
    #conta quante righe hanno almeno un valore mancante.
    number_all_missing = rows_with_all_missing.sum()
    #quanti minuti non contengono proprio nessuna delle nostre misurazioni energetiche.

    number_partial_missing = (
        rows_with_any_missing & ~rows_with_all_missing
    ).sum()
    #righe che hanno almeno un valore mancante, ma non tutti i valori mancanti.

    missing_percentage = (
        number_any_missing / len(data) * 100
    )

    print("\nMissing-value structure:")

    print("Rows with at least one missing feature:")
    print(number_any_missing)

    print("\nRows with all energy features missing:")
    print(number_all_missing)

    print("\nRows with only some energy features missing:")
    print(number_partial_missing)

    print("\nPercentage of rows with missing measurements:")
    print(f"{missing_percentage:.2f}%")
    #:.2f significa: visualizza il numero con due cifre dopo la virgola.


def analyze_missing_blocks(data):
    """
    Analyze consecutive blocks of missing measurements.
    """

    missing_rows = data[RAW_FEATURE_COLUMNS].isna().all(axis=1)
    #trova le righe in cui tutte le feature energetiche sono mancanti.

    missing_times = data.loc[
        #seleziona determinate righe e determinate colonne.
        missing_rows,
        #seleziona solo i minuti completamente mancanti.
        [DATETIME_COLUMN],
    ].copy()

    time_difference = missing_times[DATETIME_COLUMN].diff()
    #calcola la differenza temporale tra un timestamp mancante e quello mancante precedente.

    new_block = time_difference != pd.Timedelta(minutes=1)
    #la distanza dal missing precedente è diversa da 1 minuto?

    missing_times["block"] = new_block.cumsum()
    #numeri diventano gli identificatori dei blocchi.

    block_lengths = missing_times.groupby("block").size()
    #raggruppiamo le righe con lo stesso numero di blocco e contiamo quante ce ne sono.

    print("\nMissing blocks:")

    print("Number of missing blocks:")
    print(len(block_lengths))

    print("\nLongest missing block:")
    print(block_lengths.max(), "minutes")

    print("\nMedian missing block length:")
    print(block_lengths.median(), "minutes")

    print("\nBlocks of 5 minutes or less:")
    print((block_lengths <= 5).sum())

    print("\nBlocks longer than 60 minutes:")
    print((block_lengths > 60).sum())

def print_data_summary(data):
    """
    Print basic information about the loaded dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset returned by load_raw_data().
    """

    print("Dataset shape:")
    print(data.shape)

    print("\nFirst five rows:")
    print(data.head())

    print("\nColumn data types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isna().sum())
    #conta quanti valori mancanti ci sono in ogni colonna. somma i True colonna per colonna.


def aggregate_hourly_data(data):
    """
    Aggregate minute-level measurements into hourly energy consumption.
    Only complete hours are considered valid.
    """

    data = data.copy()

    data = data.set_index(DATETIME_COLUMN)
    #fai diventare il timestamp indice del DataFrame. resample() lavora sull'indice temporale.

    hourly_sum = (
        #hourly_sum è semplicemente una nuova tabella pandas che contiene, per ogni ora, la somma delle feature energetiche originali minuto per minuto.
        data[RAW_FEATURE_COLUMNS]
        .resample(RESAMPLE_FREQUENCY)
        #raggruppa i dati in finestre di un'ora.
        .sum(min_count=MIN_VALID_MINUTES_PER_HOUR)
        #somma i valori. se: MIN_VALID_MINUTES_PER_HOUR = 60 vuoi tutti i 60 minuti, basta che non ci sia un valore per non coniderare quell'ora
    )

    valid_minutes = (
        data["Global_active_power"]
        .resample(RESAMPLE_FREQUENCY)
        .count()
        #.count() conta i valori non mancanti. quanti valori validi di Global_active_power ci sono in ogni ora
    )

    hourly_data = pd.DataFrame(index=hourly_sum.index)
    #crea un nuovo DataFrame che abbia una riga per ogni ora presente in hourly_sum.

    hourly_data["global_energy_kwh"] = (
        #rappresenta l'energia totale consumata dalla casa durante quell'ora, in kWh.
        hourly_sum["Global_active_power"] / 60
        #Global_active_power in hourly sum contiene la somma per ogni minuto di kwh, mi serve la media
    )

    hourly_data["kitchen_energy_kwh"] = (
        #Nel dataset, Sub_metering_1, Sub_metering_2 e Sub_metering_3 sono già misure di energia, espresse in Wh. Dividiamo per 1000 per ottenere kWh.
        hourly_sum["Sub_metering_1"] / 1000
    )

    hourly_data["laundry_energy_kwh"] = (
        hourly_sum["Sub_metering_2"] / 1000
    )

    hourly_data["water_heater_ac_energy_kwh"] = (
        hourly_sum["Sub_metering_3"] / 1000
    )

    hourly_data["valid_minutes"] = valid_minutes
    #salvi anche quanti minuti validi aveva ogni ora, aggiungendo una nuova colonna

    hourly_data = hourly_data.reset_index()
    #il timestamp smette di essere indice e torna a essere una normale colonna.

    return hourly_data

def add_sequence_ids(data):
    """
    Assign an identifier to each continuous sequence of complete hours.
    Incomplete hours do not belong to any sequence.
    """

    data = data.copy()

    valid_hours = data[MODEL_FEATURE_COLUMNS].notna().all(axis=1)

    previous_hour_valid = valid_hours.shift(
        1,
        fill_value=False,
        #Per la prima riga non esiste un'ora precedente nel DataFrame. Per questo diciamo: fill_value=False
    )
    #shift(1) sposta tutti i valori di una posizione verso il basso.

    new_sequence = valid_hours & ~previous_hour_valid
    #identifichiamo le nuove sequenze di ore, in base a quando un'ora è valida e la precedente no.

    sequence_ids = new_sequence.cumsum()
    #cosi abbiamo quali ore appartengono a quale sequenza. Ogni volta che c'è un nuovo blocco di ore valide, il numero della sequenza aumenta di 1.

    data["sequence_id"] = sequence_ids
    data.loc[~valid_hours, "sequence_id"] = pd.NA
    #non vogliamo che le ore incomplete siano in una sequenza, quindi assegniamo loro un valore mancante.

    data["sequence_id"] = data["sequence_id"].astype("Int64")
    return data


def split_hourly_data(data):
    """
    Split the hourly dataset chronologically into
    training, validation and test sets.
    """

    total_fraction = (
        TRAIN_FRACTION
        + VALIDATION_FRACTION
        + TEST_FRACTION
    )

    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError(
            "Train, validation and test fractions must sum to 1."
        )

    number_of_hours = len(data)
    #T=34589. Quindi lo split avviene sull'intera griglia oraria, comprese le ore incomplete. Perché vogliamo dividere il periodo temporale, non semplicemente dividere le osservazioni valide.

    train_end = int(
        number_of_hours * TRAIN_FRACTION
    )

    validation_end = int(
        number_of_hours
        * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    train_data = data.iloc[:train_end].copy()
    #iloc seleziona le righe in base alla loro posizione numerica.

    validation_data = data.iloc[
        train_end:validation_end
    ].copy()

    test_data = data.iloc[
        validation_end:
    ].copy()

    train_data = train_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    train_data = add_sequence_ids(train_data)
    #train, validation e test devono essere considerate tre parti indipendenti.
    validation_data = add_sequence_ids(validation_data)
    test_data = add_sequence_ids(test_data)

    return train_data, validation_data, test_data


def compute_standardization_parameters(train_data):
    """
    Compute feature means and standard deviations using
    only complete observations from the training set.
    """

    valid_train_data = (
        train_data[MODEL_FEATURE_COLUMNS]
        .dropna()
    )

    feature_means = valid_train_data.mean()

    feature_stds = valid_train_data.std(ddof=0)
    #ddof=0 perché qui non stiamo cercando una stima statistica corretta della deviazione standard. Delta Degrees of Freedom.
    if (feature_stds == 0).any():
        raise ValueError(
            "At least one feature has zero standard deviation."
        )

    return feature_means, feature_stds

def standardize_data(
    data,
    feature_means,
    feature_stds,
):
    """
    Standardize model features using training-set parameters.
    """

    standardized_data = data.copy()

    standardized_data[MODEL_FEATURE_COLUMNS] = (
        standardized_data[MODEL_FEATURE_COLUMNS]
        - feature_means
    ) / feature_stds

    return standardized_data
#i buchi temporali rimangono esattamente dove erano.

def standardize_splits(
    train_data,
    validation_data,
    test_data,
):
    """
    Standardize train, validation and test sets using
    parameters computed only from the training set.
    """

    feature_means, feature_stds = (
        compute_standardization_parameters(train_data)
    )

    train_standardized = standardize_data(
        train_data,
        feature_means,
        feature_stds,
    )

    validation_standardized = standardize_data(
        validation_data,
        feature_means,
        feature_stds,
    )

    test_standardized = standardize_data(
        test_data,
        feature_means,
        feature_stds,
    )

    return (
        train_standardized,
        validation_standardized,
        test_standardized,
        feature_means,
        feature_stds,
    )

def print_standardization_summary(
    train_standardized,
    validation_standardized,
    test_standardized,
    feature_means,
    feature_stds,
):
    """
    Print preprocessing parameters and standardized-data statistics.
    """

    print("\nTraining feature means before standardization:")
    print(feature_means)

    print("\nTraining feature standard deviations before standardization:")
    print(feature_stds)

    datasets = [
        ("Training", train_standardized),
        ("Validation", validation_standardized),
        ("Test", test_standardized),
    ]

    for name, current_data in datasets:

        valid_data = (
            current_data[MODEL_FEATURE_COLUMNS]
            .dropna()
        )

        print(f"\n{name} standardized feature means:")
        print(valid_data.mean())

        print(f"\n{name} standardized feature standard deviations:")
        print(valid_data.std(ddof=0))


def print_split_summary(
    train_data,
    validation_data,
    test_data,
):
    """
    Print information about the temporal dataset split.
    """

    datasets = [
        ("Training", train_data),
        ("Validation", validation_data),
        ("Test", test_data),
    ]

    print("\nTemporal split:")

    for name, current_data in datasets:

        valid_hours = (
            current_data[MODEL_FEATURE_COLUMNS]
            .notna()
            .all(axis=1)
        )

        number_of_sequences = (
            current_data.loc[
                valid_hours,
                "sequence_id",
                #seleziona solamente le righe corrispondenti a ore valide e, di queste righe, prendi solamente la colonna "sequence_id".
            ]
            .nunique()
            #nunique significa number of unique values, cioè: conta quanti valori diversi ci sono.
        )

        print(f"\n{name} set:")

        print("Time range:")
        print(
            current_data[DATETIME_COLUMN].iloc[0],
            "->",
            current_data[DATETIME_COLUMN].iloc[-1],
        )

        print("Total hours:")
        print(len(current_data))

        print("Complete hours:")
        print(valid_hours.sum())

        print("Incomplete hours:")
        print((~valid_hours).sum())

        print("Continuous sequences:")
        print(number_of_sequences)

def print_sequence_summary(data):
    """
    Print information about the continuous hourly sequences.
    """

    valid_data = data.dropna(subset=MODEL_FEATURE_COLUMNS)
    #Elimina una riga se almeno una delle colonne usate dal modello è NaN. possiamo finalmente usarlo, perché abbiamo già conservato l'informazione sulla struttura temporale tramite sequence_id.

    sequence_lengths = (
        valid_data
        .groupby("sequence_id")
        #Raggruppa insieme tutte le righe che appartengono alla stessa sequenza.
        .size()
        #conta le osservazioni
    )

    print("\nContinuous sequences:")

    print("Number of sequences:")
    #verificare che non abbiamo frammentato troppo il dataset
    print(len(sequence_lengths))

    print("\nShortest sequence:")
    print(sequence_lengths.min(), "hours")

    print("\nLongest sequence:")
    print(sequence_lengths.max(), "hours")

    print("\nMedian sequence length:")
    print(sequence_lengths.median(), "hours")

    print("\nMean sequence length:")
    print(f"{sequence_lengths.mean():.2f} hours")

    print("\nSequences shorter than 24 hours:")
    print((sequence_lengths < 24).sum())

    print("\nFive longest sequences:")
    print(sequence_lengths.sort_values(ascending=False).head())

def print_hourly_summary(data):
    """
    Print basic information about the hourly dataset.
    """

    valid_hours = data[MODEL_FEATURE_COLUMNS].notna().all(axis=1)
    #colonna con true e false. True significa: questa ora contiene tutte le feature.

    print("\nHourly dataset:")
    print(data.head().to_string())

    print("\nDataset shape:")
    print(data.shape)

    print("\nNumber of complete hours:")
    print(valid_hours.sum())

    print("\nNumber of incomplete hours:")
    print((~valid_hours).sum())

    print("\nPercentage of complete hours:")
    print(f"{valid_hours.mean() * 100:.2f}%")

    print("\nValid minutes per hour:")
    print(data["valid_minutes"].describe())
    #.describe() calcola automaticamente statistiche riassuntive


def prepare_model_data(data):
    """
    Convert a processed dataset into the format required by the models.

    Returns
    -------
    observations : numpy.ndarray
        Matrix with shape (number_of_observations, number_of_features).

    sequence_lengths : list
        Length of each continuous sequence.
    """

    valid_data = data.dropna(
        subset=MODEL_FEATURE_COLUMNS
    ).copy()

    observations = (
        valid_data[MODEL_FEATURE_COLUMNS]
        .to_numpy()
        #L'HMM e il GMM lavoreranno su questa matrice numerica.
    )

    sequence_lengths = (
        valid_data
        .groupby("sequence_id")
        .size()
        .tolist()
    )
    #prima costruisce una tabella con le lunghezze delle sequenze, poi viene trasformata in una lista python in cui ogni elemento rappresenta la lunghezza di una sequenza.

    if sum(sequence_lengths) != len(observations):
        raise ValueError(
            "Sequence lengths do not match the number of observations."
        )

    return observations, sequence_lengths

def print_model_data_summary(
    train_observations,
    train_lengths,
    validation_observations,
    validation_lengths,
    test_observations,
    test_lengths,
):
    """
    Print the final shapes of the datasets used by the models.
    """

    datasets = [
        (
            "Training",
            train_observations,
            #observations contiene i dati numerici da dare all’HMM
            train_lengths,
            #lengths gli dice dove finiscono e iniziano le diverse sequenze temporali.
        ),
        (
            "Validation",
            validation_observations,
            validation_lengths,
        ),
        (
            "Test",
            test_observations,
            test_lengths,
        ),
    ]

    print("\nModel input data:")

    for name, observations, lengths in datasets:

        print(f"\n{name}:")

        print("Observation matrix shape:")
        print(observations.shape)

        print("Number of sequences:")
        print(len(lengths))

        print("Total observations from sequence lengths:")
        print(sum(lengths))


if __name__ == "__main__":
    raw_data = load_raw_data()
    print_data_summary(raw_data)
    time_data = create_datetime(raw_data)
    print_time_summary(time_data)
    analyze_missing_values(time_data)
    analyze_missing_blocks(time_data)

    hourly_data = aggregate_hourly_data(time_data)
    print_hourly_summary(hourly_data)

    sequence_data = add_sequence_ids(hourly_data)
    print_sequence_summary(sequence_data)

    train_data, validation_data, test_data = (
    split_hourly_data(hourly_data)
    )

    print_split_summary(
        train_data,
        validation_data,
        test_data,
    )
    (
        train_standardized,
        validation_standardized,
        test_standardized,
        feature_means,
        feature_stds,
    ) = standardize_splits(
        train_data,
        validation_data,
        test_data,
    )

    print_standardization_summary(
        train_standardized,
        validation_standardized,
        test_standardized,
        feature_means,
        feature_stds,
    )
    train_observations, train_lengths = (
    prepare_model_data(train_standardized)
    )

    validation_observations, validation_lengths = (
        prepare_model_data(validation_standardized)
    )

    test_observations, test_lengths = (
        prepare_model_data(test_standardized)
    )
    print_model_data_summary(
        train_observations,
        train_lengths,
        validation_observations,
        validation_lengths,
        test_observations,
        test_lengths,
    )
