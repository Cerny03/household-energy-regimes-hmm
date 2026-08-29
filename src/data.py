import pandas as pd

from src.config import (
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
    ]

    data = pd.read_csv(
        RAW_DATA_FILE,
        sep=";",
        usecols=columns_to_load,
        na_values="?",
        low_memory=False,
    )

    return data

def create_datetime(data):
    """
    Combine the Date and Time columns into a single datetime column.
    """

    data = data.copy()

    data[DATETIME_COLUMN] = pd.to_datetime(
        data[DATE_COLUMN] + " " + data[TIME_COLUMN],
        format="%d/%m/%Y %H:%M:%S",
    )

    data = data.sort_values(DATETIME_COLUMN)
    data = data.reset_index(drop=True)

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

    print("\nChronologically ordered:") 
    print(data[DATETIME_COLUMN].is_monotonic_increasing)


def analyze_missing_values(data):
    """
    Analyze how missing values are distributed across the energy features.
    """

    missing = data[RAW_FEATURE_COLUMNS].isna()

    rows_with_any_missing = missing.any(axis=1)
    rows_with_all_missing = missing.all(axis=1)

    number_any_missing = rows_with_any_missing.sum()
    number_all_missing = rows_with_all_missing.sum()

    number_partial_missing = (
        rows_with_any_missing & ~rows_with_all_missing
    ).sum()

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


def analyze_missing_blocks(data):
    """
    Analyze consecutive blocks of missing measurements.
    """

    missing_rows = data[RAW_FEATURE_COLUMNS].isna().all(axis=1)

    missing_times = data.loc[
        missing_rows,
        [DATETIME_COLUMN],
    ].copy()

    time_difference = missing_times[DATETIME_COLUMN].diff()

    new_block = time_difference != pd.Timedelta(minutes=1)

    missing_times["block"] = new_block.cumsum()

    block_lengths = missing_times.groupby("block").size()

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


def aggregate_hourly_data(data):
    """
    Aggregate minute-level measurements into hourly energy consumption.
    Only complete hours are considered valid.
    """

    data = data.copy()

    data = data.set_index(DATETIME_COLUMN)

    hourly_sum = (
        data[RAW_FEATURE_COLUMNS]
        .resample(RESAMPLE_FREQUENCY)
        .sum(min_count=MIN_VALID_MINUTES_PER_HOUR)
    )

    valid_minutes = (
        data["Global_active_power"]
        .resample(RESAMPLE_FREQUENCY)
        .count()
    )

    hourly_data = pd.DataFrame(index=hourly_sum.index)

    hourly_data["global_energy_kwh"] = (    
        hourly_sum["Global_active_power"] / 60
    )

    hourly_data["kitchen_energy_kwh"] = (
        hourly_sum["Sub_metering_1"] / 1000
    )

    hourly_data["laundry_energy_kwh"] = (
        hourly_sum["Sub_metering_2"] / 1000
    )

    hourly_data["water_heater_ac_energy_kwh"] = (
        hourly_sum["Sub_metering_3"] / 1000
    )

    hourly_data["valid_minutes"] = valid_minutes

    hourly_data = hourly_data.reset_index()

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
    )

    new_sequence = valid_hours & ~previous_hour_valid

    sequence_ids = new_sequence.cumsum()

    data["sequence_id"] = sequence_ids
    data.loc[~valid_hours, "sequence_id"] = pd.NA

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

    train_end = int(
        number_of_hours * TRAIN_FRACTION
    )

    validation_end = int(
        number_of_hours
        * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    train_data = data.iloc[:train_end].copy()

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
            ]
            .nunique()
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

    sequence_lengths = (
        valid_data
        .groupby("sequence_id")
        .size()
    )

    print("\nContinuous sequences:")

    print("Number of sequences:")
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
    )

    sequence_lengths = (
        valid_data
        .groupby("sequence_id")
        .size()
        .tolist()
    )

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
            train_lengths,
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
