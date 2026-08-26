from pathlib import Path
from zipfile import ZipFile

from src.main import main


def prepare_dataset():
    """
    Extract the raw dataset if it has not been extracted yet.
    """

    project_root = Path(__file__).resolve().parent

    raw_data_dir = (
        project_root / "data" / "raw"
    )

    zip_file = (
        raw_data_dir
        / "household_power_consumption.zip"
    )

    data_file = (
        raw_data_dir
        / "household_power_consumption.txt"
    )

    if data_file.exists():
        return

    if not zip_file.exists():
        raise FileNotFoundError(
            "Dataset archive not found in data/raw/."
        )

    print("Extracting dataset...")

    with ZipFile(zip_file, "r") as archive:
        archive.extractall(raw_data_dir)

    print("Dataset extracted.")


if __name__ == "__main__":
    prepare_dataset()
    main()