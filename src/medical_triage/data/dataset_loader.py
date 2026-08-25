"""Download and validate the Medical Abstracts TC dataset."""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DATASET_BASE_URL = (
    "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
)

DATASET_FILES = (
    "medical_tc_train.csv",
    "medical_tc_test.csv",
    "medical_tc_labels.csv",
)

DEFAULT_DATA_DIR = Path("data/raw")


class DatasetLoader:
    """Download and manage the Medical Abstracts TC dataset.

    The loader verifies whether the required dataset files already
    exist locally. Missing files are downloaded from the official
    GitHub repository.

    Existing files are preserved and are not downloaded again.
    """

    def __init__(
        self,
        destination: Path = DEFAULT_DATA_DIR,
    ) -> None:
        """Initialize the dataset loader.

        Args:
            destination: Directory where dataset files will be stored.
        """
        self._destination = destination

    def prepare(self) -> None:
        """Ensure that all required dataset files are available locally.

        Existing files are reused. Missing files are downloaded from
        the official Medical Abstracts TC Corpus repository.

        Raises:
            RuntimeError: If one or more files cannot be downloaded.
        """
        self._destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        for filename in DATASET_FILES:
            destination_path = self._destination / filename

            if destination_path.exists():
                print(f"[OK] Dataset file already exists: {destination_path}")
                continue

            self._download_file(
                filename=filename,
                destination_path=destination_path,
            )

    def _download_file(
        self,
        filename: str,
        destination_path: Path,
    ) -> None:
        """Download a single dataset file.

        Args:
            filename: Name of the remote dataset file.
            destination_path: Local path where the file will be saved.

        Raises:
            RuntimeError: If the file cannot be downloaded.
        """
        url = f"{DATASET_BASE_URL}/{filename}"

        print(f"[DOWNLOAD] {filename}")

        try:
            with urlopen(
                url,
                timeout=60,
            ) as response:
                content = response.read()

        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"Failed to download dataset file: {url}") from error

        temporary_path = destination_path.with_suffix(f"{destination_path.suffix}.tmp")

        temporary_path.write_bytes(content)

        temporary_path.replace(destination_path)

        print(f"[OK] Saved: {destination_path}")


def main() -> None:
    """Prepare the Medical Abstracts TC dataset."""
    loader = DatasetLoader()

    loader.prepare()


if __name__ == "__main__":
    main()
