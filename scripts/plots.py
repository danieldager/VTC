import shutil
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


def get_project_paths():
    """Setup and return relevant project paths."""
    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent
    chunk_dir = project_root / "output" / "chunks5"
    figures_dir = project_root / "figures"
    manifest_path = project_root / "manifests" / "chunks5.csv"

    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "audio_samples").mkdir(parents=True, exist_ok=True)

    return chunk_dir, figures_dir, manifest_path


def load_vad_data(parquet_glob: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load VAD data from parquet files.

    Returns:
        vad_segments: DataFrame with exploded speech segments (start, duration).
        vad_metadata: DataFrame with file-level statistics (speech_ratio, avg_segment_len).
    """
    try:
        # Select Only Essential Columns
        needed_cols = [
            "file_id",
            "duration",
            "speech_ratio",
            "speech_avg",
            "speech_segments",
        ]
        df = pl.read_parquet(parquet_glob, columns=needed_cols)

        # 1. Extract File Metadata
        meta_cols = [c for c in needed_cols if c != "speech_segments" and c in df.columns]
        vad_metadata = df.select(meta_cols).rename({"file_id": "uid"})

        # 2. Extract Segments (Handle potential schema variations in Inner Struct)
        inner_fields = (
            df.select(pl.col("speech_segments").first())
            .schema["speech_segments"]
            .inner.fields
        )
        inner_names = {f.name for f in inner_fields}

        start_key = "start_sec" if "start_sec" in inner_names else "start"
        dur_key = "duration_sec" if "duration_sec" in inner_names else "duration"

        vad_segments = (
            df.explode("speech_segments")
            .drop_nulls("speech_segments")
            .select(
                [
                    pl.col("file_id").alias("uid"),
                    pl.col("speech_segments")
                    .struct.field(start_key)
                    .alias("start_time_s"),
                    pl.col("speech_segments").struct.field(dur_key).alias("duration_s"),
                    pl.col("duration").alias("total"),
                ]
            )
        )

        return vad_segments, vad_metadata

    except Exception as e:
        print(f"Warning: Could not load VAD from parquet: {e}")
        return pl.DataFrame(), pl.DataFrame()


def load_vtc_data(rttm_path: Path, raw_rttm_dir: Path = None) -> pl.DataFrame:
    """Load VTC RTTM data, generating from raw files if CSV is missing."""

    def _read_rttm_csv(path: Path) -> pl.DataFrame:
        """Helper to read our specific RTTM CSV format."""
        return pl.read_csv(path)

    def _parse_single_rttm(path: Path) -> pl.DataFrame:
        """Parse a standard .rttm text file into a DataFrame."""
        try:
            return pl.read_csv(
                source=path,
                has_header=False,
                columns=[1, 3, 4],  # uid, start, duration
                new_columns=("uid", "start_time_s", "duration_s"),
                schema_overrides={
                    "uid": pl.String,
                    "start_time_s": pl.Float64,
                    "duration_s": pl.Float64,
                },
                separator=" ",
            )
        except pl.exceptions.NoDataError:
            return pl.DataFrame(
                schema={
                    "uid": pl.String,
                    "start_time_s": pl.Float64,
                    "duration_s": pl.Float64,
                }
            )

    # 1. Try Loading consolidated CSV
    if rttm_path.exists():
        print("Loading VTC RTTM from CSV...")
        return _read_rttm_csv(rttm_path)

    # 2. Fallback: Generate from raw .rttm files
    if raw_rttm_dir and raw_rttm_dir.exists():
        print("Consolidated VTC CSV not found. Generating from raw RTTM files...")
        files = sorted(list(raw_rttm_dir.glob("*.rttm")))
        print(f"  Found {len(files)} raw files.")

        if files:
            chunks = [_parse_single_rttm(f) for f in files]
            return pl.concat(chunks)

    print("Error: Could not find VTC data (neither CSV nor raw RTTMs).")
    return pl.DataFrame()


def calculate_metrics(df_vad: pl.DataFrame, df_vtc: pl.DataFrame) -> pl.DataFrame:
    """
    Core Logic: Calculate IoU, Precision, Recall between VAD and VTC.
    Uses time-interval intersection math.
    """

    def flatten_intervals(df: pl.LazyFrame, include_total=False):
        """Merges overlapping intervals within the same file (uid)."""
        # 1. Mark where new groups start (gaps between segments)
        lf = df.sort("uid", "start_time_s").with_columns(
            end_time_s=pl.col("start_time_s") + pl.col("duration_s")
        )

        lf = lf.with_columns(
            is_new_group=(
                pl.col("start_time_s")
                > pl.col("end_time_s").shift().over("uid").cum_max().fill_null(0)
            )
        ).with_columns(group_id=pl.col("is_new_group").cum_sum().over("uid"))

        # 2. Aggregate by group to get continuous segments
        aggs = [
            pl.col("start_time_s").min().alias("start_time_s"),
            pl.col("end_time_s").max().alias("end_time_s"),
        ]
        if include_total:
            aggs.append(pl.col("total").first())

        return lf.group_by("uid", "group_id").agg(aggs).with_columns(
            duration_s=pl.col("end_time_s") - pl.col("start_time_s")
        )

    # --- Pipeline Execution ---
    vad_flat = flatten_intervals(df_vad.lazy(), include_total=True)
    vtc_flat = flatten_intervals(df_vtc.lazy())

    # 1. Calculate Statistics (Total Speech Time per File)
    vad_stats = vad_flat.group_by("uid").agg(
        [
            pl.col("duration_s").sum().alias("vad_dur"),
            pl.col("total").first().alias("file_total"),
        ]
    )
    vtc_stats = vtc_flat.group_by("uid").agg(
        pl.col("duration_s").sum().alias("vtc_dur")
    )

    # 2. Calculate Intersection (Where both pipelines agree speech exists)
    intersection = (
        vad_flat.join(vtc_flat, on="uid", suffix="_vtc")
        .filter(  # Filter strictly overlapping segments
            (pl.col("start_time_s") < pl.col("end_time_s_vtc"))
            & (pl.col("end_time_s") > pl.col("start_time_s_vtc"))
        )
        .select(
            [
                "uid",
                pl.max_horizontal("start_time_s", "start_time_s_vtc").alias("s"),
                pl.min_horizontal("end_time_s", "end_time_s_vtc").alias("e"),
            ]
        )
        .group_by("uid")
        .agg((pl.col("e") - pl.col("s")).sum().alias("TP"))
    )

    # 3. Combine & Compute Final Metrics (IoU, precision, recall)
    return (
        vad_stats.join(vtc_stats, on="uid", how="left")
        .join(intersection, on="uid", how="left")
        .fill_null(0)
        .with_columns(
            [
                (pl.col("vtc_dur") - pl.col("TP")).alias("FP"),
                (pl.col("vad_dur") - pl.col("TP")).alias("FN"),
            ]
        )
        .with_columns(
            TN=pl.col("file_total") - (pl.col("TP") + pl.col("FP") + pl.col("FN"))
        )
        .with_columns(
            [
                (pl.col("TP") / (pl.col("TP") + pl.col("FP") + pl.col("FN"))).alias("IoU"),
                (pl.col("TP") / (pl.col("TP") + pl.col("FP"))).alias("Precision"),
                (pl.col("TP") / (pl.col("TP") + pl.col("FN"))).alias("Recall"),
            ]
        )
        .collect()
    )


def print_executive_summary(results_df: pl.DataFrame, low_iou_thresh=0.5) -> dict:
    """Print high-level statistics and failure analysis."""
    
    # 1. Global Aggregates (Summing seconds across entire dataset)
    total = results_df.select(
        [
            pl.sum("TP"), pl.sum("TP").alias("TP_sum"), # Keep alias consistent
            pl.sum("FP"),
            pl.sum("FN"),
            pl.sum("TN"),
            pl.sum("vad_dur"),
            pl.sum("vtc_dur"),
        ]
    ).to_dicts()[0]

    # Calculate Global Metrics
    global_iou = total["TP"] / (total["TP"] + total["FP"] + total["FN"])
    global_prec = total["TP"] / (total["TP"] + total["FP"])
    global_rec = total["TP"] / (total["TP"] + total["FN"])
    
    # 2. Failure Analysis Groups
    df_low = results_df.filter(pl.col("IoU") < low_iou_thresh)
    df_high = results_df.filter(pl.col("IoU") >= low_iou_thresh)

    print("=" * 60)
    print("           PIPELINE SUPERPOSITION ANALYSIS")
    print("=" * 60)
    print(f"Global IoU (Total Agreement): {global_iou:.2%}")
    print(f"Precision (Reliability):      {global_prec:.2%}")
    print(f"Recall (Sensitivity):         {global_rec:.2%}")
    print("-" * 60)
    print(f"Total Speech (VAD):           {total['vad_dur'] / 3600:.1f} hours")
    print(f"Total Speech (VTC):           {total['vtc_dur'] / 3600:.1f} hours")
    print("=" * 60)
    print(f"\n   FAILURE DIAGNOSTICS (Target IoU < {low_iou_thresh:.0%})   ")
    print("-" * 60)
    print(f"High Performance Files:       {len(df_high):5d} ({len(df_high)/len(results_df):.1%})")
    print(f"Low Performance Files:        {len(df_low):5d} ({len(df_low)/len(results_df):.1%})")

    def _print_group_stats(df, label):
        if df.is_empty():
            return
        dur = df["file_total"].mean()
        ratio = df["speech_ratio"].mean() if "speech_ratio" in df.columns else float("nan")
        seg_len = df["speech_avg"].mean() if "speech_avg" in df.columns else float("nan")

        print(f"\n--- {label} ---")
        print(f"  Mean Duration:     {dur:.2f} s")
        print(f"  Mean Speech Ratio: {ratio:.1%}")
        print(f"  Mean Avg Seg Len:  {seg_len:.2f} s")
        
    _print_group_stats(df_high, "High IoU Group")
    _print_group_stats(df_low, "Low IoU Group")
    print("=" * 60)
    
    # Return dictionary for plotting
    return {
        "total_vad_speech_sec": total["vad_dur"],
        "total_vtc_speech_sec": total["vtc_dur"],
    }


def create_dashboard(results: pl.DataFrame, global_stats: dict, output_file: Path):
    """Generate and save 2x3 Plotly dashboard."""
    low_thresh = 0.5
    
    # Pre-calculate Viz Groups
    df_low = results.filter(pl.col("IoU") < low_thresh)
    df_high = results.filter(pl.col("IoU") >= low_thresh)

    # Setup 2x3 Grid
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "<b>1. Volume</b><br>(Total Hours)",
            "<b>2. Consistency</b><br>(Precision vs Recall)",
            "<b>3. IoU Dist</b><br>(Histogram)",
            "<b>4. Duration</b><br>(High vs Low IoU)",
            "<b>5. Speech Ratio</b><br>(High vs Low IoU)",
            "<b>6. Segmentation</b><br>(Seg Len vs IoU)",
        ),
        vertical_spacing=0.15,
    )

    # --- Plots ---

    # 1. Bar: Volume
    vad_h = global_stats["total_vad_speech_sec"] / 3600
    vtc_h = global_stats["total_vtc_speech_sec"] / 3600
    fig.add_trace(
        go.Bar(
            x=["VAD", "VTC"],
            y=[vad_h, vtc_h],
            marker_color=["#3498db", "#e74c3c"],
            text=[f"{vad_h:.1f}h", f"{vtc_h:.1f}h"],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Scatter: P vs R
    fig.add_trace(
        go.Scatter(
            x=results["Recall"],
            y=results["Precision"],
            mode="markers",
            marker=dict(color="#AB63FA", opacity=0.4, size=6),
            name="Files",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Hist: IoU
    fig.add_trace(
        go.Histogram(x=results["IoU"], marker_color="#636EFA", showlegend=False),
        row=1,
        col=3,
    )

    # 4. Hist: Duration (Outliers clipped at P99)
    p99 = results["file_total"].quantile(0.99)
    for df, name, color in [(df_high, "High IoU", "#1f77b4"), (df_low, "Low IoU", "#d62728")]:
        fig.add_trace(
            go.Histogram(
                x=df.filter(pl.col("file_total") < p99)["file_total"],
                name=name,
                marker_color=color,
                opacity=0.75,
                histnorm="percent",
                legendgroup=name,
            ),
            row=2,
            col=1,
        )
    fig.update_xaxes(title_text=f"Duration (0-{p99:.0f}s)", row=2, col=1)

    # 5. Hist: Speech Ratio
    if "speech_ratio" in results.columns:
        for df, name, color in [(df_high, "High IoU", "#1f77b4"), (df_low, "Low IoU", "#d62728")]:
            fig.add_trace(
                go.Histogram(
                    x=df["speech_ratio"],
                    name=name,
                    marker_color=color,
                    opacity=0.75,
                    histnorm="percent",
                    legendgroup=name,
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    # 6. Scatter: Seg Len vs IoU
    if "speech_avg" in results.columns:
        fig.add_trace(
            go.Scatter(
                x=results["speech_avg"],
                y=results["IoU"],
                mode="markers",
                marker=dict(color="gray", opacity=0.3, size=4),
                showlegend=False,
            ),
            row=2,
            col=3,
        )
        fig.update_xaxes(type="log", title_text="Avg Seg Len (Log)", row=2, col=3)

    # --- Styling ---
    fig.update_layout(
        height=900,
        width=1400,
        title_text="<b>Superposition Analysis & Diagnostics</b>",
        barmode="group",
        template="plotly_white",
    )
    fig.write_html(str(output_file))
    
    # Save PNG
    try:
        fig.write_image(str(output_file.with_suffix(".png")), scale=2)
        print(f"Saved plots to {output_file.with_suffix('.png')}")
    except Exception:
        print("Note: Could not save PNG (kaleido missing?)")


def export_audio_samples(results: pl.DataFrame, manifest_path: Path, output_dir: Path, n=10):
    """Copy N files with lowest IoU to output folder for inspection."""
    if n == 0 or not manifest_path.exists():
        return

    print(f"\nExporting top {n} inconsistent audio samples...")
    target_dir = output_dir / "audio_samples"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Join results with manifest to get file paths
    # Match logic: Remove .wav from manifest ID to match result UID
    manifest = pl.read_csv(manifest_path).with_columns(
        uid_match=pl.col("file_id").str.replace(r"\.wav$", "")
    )
    
    worst_files = (
        results.sort("IoU")
        .head(n)
        .join(manifest, left_on="uid", right_on="uid_match", how="left")
    )

    for row in worst_files.iter_rows(named=True):
        if row.get("path") and Path(row["path"]).exists():
            dest = target_dir / f"iou_{row['IoU']:.2f}_{row['uid']}.wav"
            try:
                shutil.copy2(row["path"], dest)
                print(f"  -> {dest.name}")
            except Exception:
                pass


def main():
    chunk_dir, figures_dir, manifest_path = get_project_paths()

    # 1. Load Data
    print("Loading Data...")
    vad_segments, vad_meta = load_vad_data(str(chunk_dir / "pyannote" / "*.parquet"))
    
    if vad_segments.is_empty():
        print("Critical Error: No VAD data found.")
        return

    vtc_df = load_vtc_data(
        rttm_path=chunk_dir / "rttm.csv", 
        raw_rttm_dir=chunk_dir / "rttm"
    )

    if vtc_df.is_empty():
        print("Critical Error: No VTC data found.")
        return

    # 2. Run Analysis
    print("Calculating Metrics...")
    results = calculate_metrics(vad_segments, vtc_df)

    # Join metadata (speech_ratio etc) to results
    if not vad_meta.is_empty():
        results = results.join(vad_meta, on="uid", how="left")

    # 3. Save & Report
    results.write_csv(chunk_dir / "global.csv")
    global_stats = print_executive_summary(results)
    
    create_dashboard(results, global_stats, figures_dir / "superposition_analysis.html")
    export_audio_samples(results, manifest_path, figures_dir, n=0) # Set N>0 to export


if __name__ == "__main__":
    main()


