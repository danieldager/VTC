import polars as pl
from pathlib import Path


def get_task_shard(
    manifest_path: str, array_id: int, array_count: int
) -> tuple[int, int, list[str]]:
    """
    Parses manifest (txt, csv, parquet) and returns (total_files, chunk_size, file_paths)
    for the specific array task.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"{manifest_path} not found")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        # Read all lines
        all_paths = sorted([l.strip() for l in path.read_text().splitlines() if l.strip()])

        total_files = len(all_paths)
        chunk_size = total_files // array_count
        start_idx = array_id * chunk_size
        if array_id == array_count - 1:
            end_idx = total_files
            chunk_size = end_idx - start_idx
        else:
            end_idx = start_idx + chunk_size

        return total_files, chunk_size, all_paths[start_idx:end_idx]

    elif suffix in [".parquet", ".csv"]:
        # Use Polars
        if suffix == ".parquet":
            lf = pl.scan_parquet(manifest_path)
        else:
            lf = pl.scan_csv(manifest_path)

        # Identify path column
        schema = lf.collect_schema()
        if "path" in schema.names():
            col_name = "path"
        elif "audio_filepath" in schema.names():
            col_name = "audio_filepath"
        else:
            raise ValueError(
                f"Manifest {manifest_path} must contain 'path' or 'audio_filepath' column"
            )

        total_files = lf.select(pl.len()).collect().item()
        base_chunk_size = total_files // array_count
        start_idx = array_id * base_chunk_size

        length = base_chunk_size
        if array_id == array_count - 1:
            length = total_files - start_idx

        return (
            total_files,
            length,
            lf.sort(col_name)
            .slice(start_idx, length)
            .select(col_name)
            .collect()
            .get_column(col_name)
            .to_list(),
        )

    else:
        raise ValueError(f"Unsupported manifest extension: {suffix}")
