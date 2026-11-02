from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

from data import Split
from problem import SparseSharpeProblem
from utils import summarize_returns


@dataclass
class EvalConfig:
    sdr_stride: int = 1
    mdd_invalid_policy: str = "nan"
    print_every: int = 50
    compute_pfl_loss: bool = True


def evaluate_split(
    problem: SparseSharpeProblem,
    split: Split,
    mu_hat_all: torch.Tensor,
    k: int,
    selector: str,
    cfg: EvalConfig,
) -> tuple[dict, dict, dict]:
    """Walk-forward evaluation -> (metrics, trace {date: row}, last)."""
    t0 = perf_counter()
    T = len(split)
    tag = f"[{split.name.upper()}|{selector}|k={k}]"

    if selector == "sdr" and cfg.sdr_stride > 1:
        eval_indices = list(range(0, T, cfg.sdr_stride))
        if (T - 1) not in eval_indices:
            eval_indices.append(T - 1)
    else:
        eval_indices = list(range(T))

    rows: dict[int, dict] = {}
    last = {"w": None, "pred": None, "true_raw": None, "gate": None}

    with torch.no_grad():
        for count, i in enumerate(eval_indices, start=1):
            s = split[i]
            mu_hat = mu_hat_all[i]
            if selector == "oscar":
                mu_hat = mu_hat.to(device=s.cov.device, dtype=s.cov.dtype)
            w, info = problem.get_decision(mu_hat, s, k, selector)

            w64 = w.detach().cpu().to(torch.float64).view(-1)
            cov64 = s.cov.detach().cpu().to(torch.float64)
            true_raw = s.y_raw.detach().cpu().to(torch.float64).view(-1)
            ret_raw = float(torch.dot(true_raw, w64).item())
            den = torch.sqrt((w64 @ (cov64 @ w64)).clamp_min(1e-16))
            step_sr = float((torch.dot(true_raw, w64) / (den + 1e-12)).item())
            gate = info.get("gate")

            rows[i] = {
                "t": int(i),
                "ret_true_raw": ret_raw,
                "step_sr_raw": step_sr,
                "weight_nnz_1e-8": int((w64.abs() > 1e-8).sum().item()),
                "weight_abs_sum": float(w64.abs().sum().item()),
                "weight_sum": float(w64.sum().item()),
                "gate_sum": float(gate.sum().item()) if gate is not None else math.nan,
                "gate_nnz_1e-3": int((gate > 1e-3).sum().item()) if gate is not None else math.nan,
                "interpolated": False,
            }
            last = {
                "w": w64.clone(),
                "pred": mu_hat.detach().cpu().to(torch.float64).view(-1).clone(),
                "true_raw": true_raw.clone(),
                "gate": gate.detach().cpu().clone() if gate is not None else None,
            }

            if count % max(1, cfg.print_every) == 0 or count == len(eval_indices):
                print(f"{tag} {count}/{len(eval_indices)} solved | elapsed={perf_counter() - t0:.1f}s")

    if len(eval_indices) < T:  # sdr_stride > 1: interpolate skipped steps
        x_full = np.arange(T, dtype=np.float64)
        t_known = np.asarray(eval_indices, dtype=np.float64)
        ret_full = np.interp(x_full, t_known, [rows[i]["ret_true_raw"] for i in eval_indices])
        sr_full = np.interp(x_full, t_known, [rows[i]["step_sr_raw"] for i in eval_indices])
        for i in range(T):
            if i not in rows:
                rows[i] = {
                    "t": int(i),
                    "ret_true_raw": float(ret_full[i]),
                    "step_sr_raw": float(sr_full[i]),
                    "weight_nnz_1e-8": math.nan,
                    "weight_abs_sum": math.nan,
                    "weight_sum": math.nan,
                    "gate_sum": math.nan,
                    "gate_nnz_1e-3": math.nan,
                    "interpolated": True,
                }

    trace = {split[i].date: rows[i] for i in range(T)}
    ret_list = [rows[i]["ret_true_raw"] for i in range(T)]
    sr_list = [rows[i]["step_sr_raw"] for i in range(T)]

    metrics = summarize_returns(ret_list, split.name, invalid_policy=cfg.mdd_invalid_policy)
    if cfg.compute_pfl_loss:
        mse = nn.MSELoss()(mu_hat_all.detach().cpu().to(torch.float32),
                           split.y_norm.detach().cpu().to(torch.float32)).item()
    else:
        mse = math.nan
    metrics[f"{split.name}_avg_pfl_loss"] = mse
    metrics[f"{split.name}_avg_step_sr_raw"] = float(np.nanmean(sr_list))

    print(
        f"{tag} Avg DFL loss: {metrics[f'{split.name}_avg_dfl_loss']:.4f} | "
        f"Raw SR: {metrics[f'{split.name}_raw_sr']:.4f} | Raw Mean: {metrics[f'{split.name}_raw_mean']:.6f} | "
        f"Raw MDD: {metrics[f'{split.name}_raw_mdd']:.4f}"
    )
    return metrics, trace, last
