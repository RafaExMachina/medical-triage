"""Unit tests for the medical dataset loader."""

from pathlib import Path

import pytest

from medical_triage.data.dataset_loader import (
    DATASET_FILES,
    DatasetLoader,
)


def test_existing_dataset_files_should_be_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that existing dataset files are not downloaded again."""
    for filename in DATASET_FILES:
        file_path = tmp_path / filename
        file_path.write_text(
            "existing dataset content",
            encoding="utf-8",
        )

    downloaded_files: list[str] = []

    def fake_download(
        self: DatasetLoader,
        filename: str,
        destination_path: Path,
    ) -> None:
        """Record unexpected download attempts."""
        downloaded_files.append(filename)

    monkeypatch.setattr(
        DatasetLoader,
        "_download_file",
        fake_download,
    )

    loader = DatasetLoader(
        destination=tmp_path,
    )

    loader.prepare()

    assert downloaded_files == []

    for filename in DATASET_FILES:
        file_path = tmp_path / filename

        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == "existing dataset content"


def test_missing_dataset_files_should_be_downloaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that missing dataset files trigger a download."""
    downloaded_files: list[str] = []

    def fake_download(
        self: DatasetLoader,
        filename: str,
        destination_path: Path,
    ) -> None:
        """Simulate a successful dataset download."""
        downloaded_files.append(filename)

        destination_path.write_text(
            f"downloaded content for {filename}",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        DatasetLoader,
        "_download_file",
        fake_download,
    )

    loader = DatasetLoader(
        destination=tmp_path,
    )

    loader.prepare()

    assert downloaded_files == list(DATASET_FILES)

    for filename in DATASET_FILES:
        file_path = tmp_path / filename

        assert file_path.exists()

        assert (
            file_path.read_text(encoding="utf-8")
            == f"downloaded content for {filename}"
        )


def test_only_missing_dataset_files_should_be_downloaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that only missing files are downloaded."""
    existing_filename = DATASET_FILES[0]
    existing_file = tmp_path / existing_filename

    existing_file.write_text(
        "existing content",
        encoding="utf-8",
    )

    downloaded_files: list[str] = []

    def fake_download(
        self: DatasetLoader,
        filename: str,
        destination_path: Path,
    ) -> None:
        """Simulate downloading a missing dataset file."""
        downloaded_files.append(filename)

        destination_path.write_text(
            "downloaded content",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        DatasetLoader,
        "_download_file",
        fake_download,
    )

    loader = DatasetLoader(
        destination=tmp_path,
    )

    loader.prepare()

    expected_downloads = list(DATASET_FILES[1:])

    assert downloaded_files == expected_downloads

    assert existing_file.read_text(encoding="utf-8") == "existing content"

    for filename in DATASET_FILES:
        assert (tmp_path / filename).exists()


def test_destination_directory_should_be_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that the destination directory is created automatically."""
    destination = tmp_path / "data" / "raw"

    def fake_download(
        self: DatasetLoader,
        filename: str,
        destination_path: Path,
    ) -> None:
        """Simulate downloading a dataset file."""
        destination_path.write_text(
            "dataset content",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        DatasetLoader,
        "_download_file",
        fake_download,
    )

    assert not destination.exists()

    loader = DatasetLoader(
        destination=destination,
    )

    loader.prepare()

    assert destination.exists()
    assert destination.is_dir()

    for filename in DATASET_FILES:
        assert (destination / filename).exists()
