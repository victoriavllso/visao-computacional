import sys
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.constants import (
    DATASET_DIR,
    HF_FOOD_FILE,
    FILTER_BRAZIL_OUTPUT_DIR,
    FILTER_BRAZIL_OUTPUT_FILE,
)
from utils.logger import Logger

BRAZIL_SUFFIX = ":brazil"

log = Logger("filter")


def _count_rows(path: Path) -> int:
    """Return the row count of a Parquet file without loading it into memory."""
    return pq.read_metadata(path).num_rows


def _get_matching_row_indices(
    parquet_file: pq.ParquetFile, rg_idx: int, suffix: str
) -> pa.Array:
    """Read only the countries_tags column for a Row Group and return matching row indices."""
    rg_table = parquet_file.read_row_group(rg_idx, columns=["countries_tags"])
    countries_col = rg_table.column("countries_tags").chunk(0)

    flat_tags = pc.list_flatten(countries_col)
    tag_matches = pc.ends_with(flat_tags, pattern=suffix)
    parent_indices = pc.list_parent_indices(countries_col)

    matching_indices = pc.unique(pc.filter(parent_indices, tag_matches))

    return matching_indices


def _process_row_group(
    parquet_file: pq.ParquetFile,
    rg_idx: int,
    writer: pq.ParquetWriter | None,
) -> tuple[pq.ParquetWriter | None, int]:
    """Process a single Row Group, append matching rows to writer, and clean up memory."""
    matching_indices = _get_matching_row_indices(parquet_file, rg_idx, BRAZIL_SUFFIX)
    rows_added = 0

    if len(matching_indices) > 0:
        full_rg_table = parquet_file.read_row_group(rg_idx)
        filtered = full_rg_table.take(matching_indices)
        rows_added = filtered.num_rows

        if writer is None:
            writer = pq.ParquetWriter(FILTER_BRAZIL_OUTPUT_FILE, filtered.schema)
        writer.write_table(filtered)

    return writer, rows_added


def filter_brazil(dataset_dir: Path):
    if FILTER_BRAZIL_OUTPUT_FILE.exists():
        log.skip("Brazil filter already completed")
        log.blank()
        return
    input_file = dataset_dir / HF_FOOD_FILE

    if not input_file.exists():
        log.error(f"Input file not found: {input_file}")
        log.error("Run download_dataset.py first.")
        sys.exit(1)

    log.info(f"Reading {input_file} …")
    total_rows = pq.read_metadata(input_file).num_rows
    log.info(f"Total records in dataset: {total_rows:,}")

    log.blank()
    log.info(
        f'Filtering for products ending with "{BRAZIL_SUFFIX}" in countries_tags …'
    )

    writer: pq.ParquetWriter | None = None
    brazil_rows = 0

    try:
        parquet_file = pq.ParquetFile(input_file)
        FILTER_BRAZIL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for rg_idx in range(parquet_file.num_row_groups):
            writer, rows_added = _process_row_group(parquet_file, rg_idx, writer)
            brazil_rows += rows_added
    finally:
        if writer is not None:
            writer.close()

    log.blank()
    log.info(f"Records before filter : {total_rows:,}")
    log.info(f"Records after filter  : {brazil_rows:,}")
    log.info(f"Reduction             : {total_rows - brazil_rows:,} rows removed")
    log.blank()
    log.info(f"Output written to: {FILTER_BRAZIL_OUTPUT_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the Open Food Facts dataset to products sold in Brazil."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Open Food Facts dataset directory (default: {DATASET_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(" Open Food Facts — Brazil Filter")
    print("=" * 60)

    filter_brazil(args.dataset_dir)

    log.blank()
    log.info("Done.")


if __name__ == "__main__":
    main()
