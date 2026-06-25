"""
download.py
-----------
CLI entry point for downloading solar observation datasets.

Currently this is a placeholder module.

Future versions will:
- Download SoLEXS Level-1 data
- Download HEL1OS Level-1 data
- Validate downloaded files
- Store them under ml/data/raw/

Usage:
    python -m ml.data.download --date 20260621 --instrument solexs
"""

from pathlib import Path
import argparse


def download_solexs(date: str) -> None:
    """Placeholder for SoLEXS downloader."""

    output_dir = Path("ml/data/raw/solexs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SoLEXS Downloader")
    print("=" * 60)
    print(f"Requested Date : {date}")
    print(f"Output Folder  : {output_dir.resolve()}")
    print()
    print("Download module not implemented yet.")
    print("Place Level-1 SoLEXS files in:")
    print(output_dir.resolve())
    print("=" * 60)


def download_helios(date: str) -> None:
    """Placeholder for HEL1OS downloader."""

    output_dir = Path("ml/data/raw/helios")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HEL1OS Downloader")
    print("=" * 60)
    print(f"Requested Date : {date}")
    print(f"Output Folder  : {output_dir.resolve()}")
    print()
    print("Download module not implemented yet.")
    print("Place HEL1OS Level-1 files in:")
    print(output_dir.resolve())
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Solar dataset downloader"
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Observation date (YYYYMMDD)"
    )

    parser.add_argument(
        "--instrument",
        required=True,
        choices=["solexs", "helios"],
        help="Instrument name"
    )

    args = parser.parse_args()

    if args.instrument == "solexs":
        download_solexs(args.date)
    else:
        download_helios(args.date)


if __name__ == "__main__":
    main()