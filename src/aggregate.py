#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = [16, 37, 47, 214, 1947]

DEFAULT_METRICS = [
    "train_raw_sr",
    "train_raw_mean",
    "train_raw_mdd",
    "test_raw_sr",
    "test_raw_mean",
    "test_raw_mdd",
]


def parse_exp_name_from_folder(folder_name: str) -> dict:
    """
    Fallback parser for folders like:
      euro50_k10_actualk5_alpha0p5_seed16
      euro50_k10_actual5_alpha0p5_seed16

    Values inside the CSV take precedence; the folder name is only a fallback.
    """
    pattern = re.compile(
        r"^(?P<dataset>[^_]+)(?:_(?P<method>PFL|HIST)_(?P<selector>[a-z_]+))?"
        r"_k(?P<k_percent>\d+)_actualk?(?P<actual_k>\d+)_alpha(?P<alpha>[0-9p\.]+)_seed(?P<seed>\d+)$"
    )
    m = pattern.match(folder_name)
    if not m:
        return {}

    out = m.groupdict()
    out["k_percent"] = int(out["k_percent"])
    out["actual_k"] = int(out["actual_k"])
    out["seed"] = int(out["seed"])
    out["alpha"] = float(out["alpha"].replace("p", "."))
    out["method"] = out["method"] or "DFL"
    out["selector"] = out["selector"] or "oscar"
    return out


def read_one_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return df

    folder_meta = parse_exp_name_from_folder(csv_path.parent.name)

    for col, val in folder_meta.items():
        if col not in df.columns:
            df[col] = val
    # older DFL summaries predate the method/selector columns
    if "method" not in df.columns:
        df["method"] = "DFL"
    if "selector" not in df.columns:
        df["selector"] = "oscar"

    df["summary_path"] = str(csv_path)
    return df


def collect_summaries(artifacts_dir: str) -> pd.DataFrame:
    artifacts_path = Path(artifacts_dir)
    paths = sorted(artifacts_path.glob("*/summary_metrics.csv"))

    if not paths:
        raise FileNotFoundError(f"No summary_metrics.csv found under: {artifacts_path.resolve()}")

    frames = []
    for path in paths:
        try:
            frames.append(read_one_summary(path))
        except Exception as e:
            print(f"[WARN] failed to read {path}: {e}")

    if not frames:
        raise RuntimeError("No valid summary_metrics.csv files were loaded.")

    return pd.concat(frames, ignore_index=True)


def aggregate_metrics(
    df: pd.DataFrame,
    seeds: list[int],
    metrics: list[str],
    group_cols: list[str],
    ddof: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = set(group_cols + ["seed"])
    missing_required = sorted(required_cols - set(df.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    available_metrics = [m for m in metrics if m in df.columns]
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        print(f"[WARN] these metric columns were not found and will be skipped: {missing_metrics}")

    if not available_metrics:
        raise ValueError("No metric columns are available to aggregate.")

    df = df.copy()
    df["seed"] = df["seed"].astype(int)

    # historic runs are deterministic (single seed 0) and are always kept
    is_hist = df["method"].eq("HIST") if "method" in df.columns else pd.Series(False, index=df.index)
    df = df[df["seed"].isin(seeds) | is_hist].copy()
    if df.empty:
        raise ValueError(f"No rows matched seeds={seeds}")

    for m in available_metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    rows = []
    pretty_rows = []

    for keys, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))
        seeds_present = sorted(g["seed"].dropna().astype(int).unique().tolist())
        is_hist_group = "method" in g.columns and g["method"].eq("HIST").all()
        missing_seeds = [] if is_hist_group else sorted(set(seeds) - set(seeds_present))

        row = {
            **base,
            "n_seeds": len(seeds_present),
            "seeds_present": ",".join(map(str, seeds_present)),
            "missing_seeds": ",".join(map(str, missing_seeds)),
        }
        pretty_row = row.copy()

        for m in available_metrics:
            vals = g[m].dropna()
            mean_val = vals.mean()
            std_val = vals.std(ddof=ddof)

            row[f"{m}_mean"] = mean_val
            row[f"{m}_std"] = std_val

            if pd.isna(std_val):
                pretty_row[m] = f"{mean_val:.6f} ± nan"
            else:
                pretty_row[m] = f"{mean_val:.6f} ± {std_val:.6f}"

        rows.append(row)
        pretty_rows.append(pretty_row)

    return pd.DataFrame(rows), pd.DataFrame(pretty_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate summary_metrics.csv over seeds by dataset/k/actual_k/alpha."
    )
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--ddof", type=int, default=1, choices=[0, 1],
                        help="1: sample std, 0: population std")
    parser.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--group_cols", type=str, nargs="+",
                        default=["dataset", "method", "selector", "k_percent", "actual_k", "alpha"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = collect_summaries(args.artifacts_dir)

    all_loaded_path = out_dir / "all_loaded_summary_metrics.csv"
    all_df.to_csv(all_loaded_path, index=False)

    agg_df, pretty_df = aggregate_metrics(
        all_df,
        seeds=args.seeds,
        metrics=args.metrics,
        group_cols=args.group_cols,
        ddof=args.ddof,
    )

    agg_path = out_dir / "aggregated_summary_metrics.csv"
    pretty_path = out_dir / "aggregated_summary_metrics_pretty.csv"

    agg_df.to_csv(agg_path, index=False)
    pretty_df.to_csv(pretty_path, index=False)

    print("=" * 80)
    print(f"[SAVE] loaded rows -> {all_loaded_path}")
    print(f"[SAVE] mean/std    -> {agg_path}")
    print(f"[SAVE] pretty      -> {pretty_path}")
    print("=" * 80)

    preview_cols = args.group_cols + ["n_seeds", "missing_seeds"]
    preview_metrics = []
    for m in args.metrics:
        if f"{m}_mean" in agg_df.columns:
            preview_metrics += [f"{m}_mean", f"{m}_std"]

    print(agg_df[preview_cols + preview_metrics].to_string(index=False))


if __name__ == "__main__":
    main()