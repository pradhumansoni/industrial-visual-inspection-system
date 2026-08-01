from pathlib import Path

from src.data.data_auditor import DatasetAuditor


def main() -> None:
    """
    Run the complete dataset auditor.
    """

    dataset_root = Path("data/raw/magnetic_tile")

    auditor = DatasetAuditor(dataset_root)

    auditor.audit()

    print("\n" + "=" * 70)
    print("Dataset Auditor Test Passed")
    print("=" * 70)


if __name__ == "__main__":
    main()