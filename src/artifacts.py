from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from utils import alpha_str


def make_exp_name(mode: str, dataset: str, k_percent: int, actual_k: int, alpha: float, seed: int, selector: str) -> str:
    if mode == "dfl":
        return f"{dataset}_k{k_percent}_actualk{actual_k}_alpha{alpha_str(alpha)}_seed{seed}"
    tag = {"pfl": "PFL", "historic": "HIST"}[mode]
    return f"{dataset}_{tag}_{selector}_k{k_percent}_actualk{actual_k}_alpha0p0_seed{seed}"


def pfl_model_dir_name(dataset: str, seed: int) -> str:
    return f"{dataset}_PFL_MSE_seed{seed}"


def batch_dir_name(mode: str, dataset: str, alpha: float) -> str:
    if mode == "dfl":
        return f"{dataset}_batch_alpha{alpha_str(alpha)}"
    return {"pfl": f"{dataset}_PFL_baselines_batch", "historic": f"{dataset}_HIST_batch"}[mode]


def write_trace(save_root: Path, split_name: str, trace: dict) -> dict:
    df = (pd.DataFrame.from_dict(trace, orient="index").reset_index()
          .rename(columns={"index": "date"}).sort_values("t"))
    csv_path = save_root / f"{split_name}_trace.csv"
    json_path = save_root / f"{split_name}_trace.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    return {f"{split_name}_trace_csv": str(csv_path), f"{split_name}_trace_json": str(json_path)}


def _weights_frame(asset_names: list[str], vec: torch.Tensor) -> list:
    return asset_names if len(asset_names) == vec.numel() else list(range(vec.numel()))


def save_run(
    save_root: Path,
    summary_row: dict,
    meta_extra: dict,
    traces: dict[str, dict],
    lasts: dict[str, dict],
    asset_names: list[str],
    mode: str,
    selector: str,
    actual_k: int,
    model: nn.Module | None = None,
) -> dict:
    save_root.mkdir(parents=True, exist_ok=True)
    paths: dict = {}

    summary_csv = save_root / "summary_metrics.csv"
    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)
    paths["summary_metrics_csv"] = str(summary_csv)

    for split_name, trace in traces.items():
        paths.update(write_trace(save_root, split_name, trace))

    if model is not None:
        model_path = save_root / "model.pt"
        torch.save(model.state_dict(), model_path)
        paths["model_pt"] = str(model_path)

    test_last = lasts.get("test") or {}
    train_last = lasts.get("train") or {}
    w_key = "w_oscar" if mode == "dfl" else f"w_{selector}"
    pred_key = "combined_mean" if mode == "dfl" else "pred_combined_mean"
    vectors_pt = save_root / "last_vectors.pt"
    torch.save({
        w_key: test_last.get("w"),
        pred_key: test_last.get("pred"),
        "true_combined_mean_raw": test_last.get("true_raw"),
        "gate": test_last.get("gate"),
        "train_last_weight": train_last.get("w"),
        "train_last_pred_combined_mean": train_last.get("pred"),
        "train_last_true_combined_mean_raw": train_last.get("true_raw"),
    }, vectors_pt)
    paths["vectors_pt"] = str(vectors_pt)
    paths["vectors_csv"] = None

    w_last = test_last.get("w")
    if w_last is not None:
        df_vec = pd.DataFrame({
            "asset": _weights_frame(asset_names, w_last),
            w_key: w_last.numpy(),
            pred_key: test_last["pred"].numpy(),
            "true_combined_mean_raw": test_last["true_raw"].numpy(),
        })
        vectors_csv = save_root / "last_vectors.csv"
        df_vec.to_csv(vectors_csv, index=False)
        paths["vectors_csv"] = str(vectors_csv)

        weights_csv = save_root / f"weights_{selector}_k{actual_k}.csv"
        df_vec[["asset", w_key]].rename(columns={w_key: "weight"}).to_csv(weights_csv, index=False)
        torch.save({
            "dataset": summary_row.get("dataset"),
            "method": summary_row.get("method"),
            "selector": selector,
            "k_percent": summary_row.get("k_percent"),
            "actual_k": actual_k,
            "seed": summary_row.get("seed"),
            "weights": w_last,
            "asset_names": asset_names,
        }, save_root / f"weights_{selector}_k{actual_k}.pt")

    meta = {**summary_row, **meta_extra, "paths": paths}
    with open(save_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print(f"[SAVE] {save_root}")
    return paths


def save_pfl_model(save_root: Path, model: nn.Module, summary_row: dict, meta_extra: dict) -> Path:
    save_root.mkdir(parents=True, exist_ok=True)
    model_path = save_root / "model.pt"
    torch.save(model.state_dict(), model_path)
    summary_csv = save_root / "training_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)
    meta = {**summary_row, **meta_extra,
            "paths": {"model_pt": str(model_path), "training_summary_csv": str(summary_csv)}}
    with open(save_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"[SAVE] model -> {model_path}")
    return model_path


def save_batch_summary(root: Path, name: str, rows: list[dict], filename: str = "all_summary_metrics.csv") -> Path:
    """Merges into the existing batch file: rows with the same exp_name are replaced, others kept."""
    batch_root = root / name
    batch_root.mkdir(parents=True, exist_ok=True)
    path = batch_root / filename
    df = pd.DataFrame(rows)
    if path.exists() and "exp_name" in df.columns:
        old = pd.read_csv(path)
        if "exp_name" in old.columns:
            old = old[~old["exp_name"].isin(df["exp_name"])]
            df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"[DONE] {len(rows)} new rows, {len(df)} total -> {path}")
    return path
