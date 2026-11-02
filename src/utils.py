from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path.cwd(), here.parent, here.parent.parent]
    for c in candidates:
        if (c / "data" / "daily").exists() and (c / "artifacts").exists():
            return c
    for c in candidates:
        if (c / "data" / "daily").exists():
            return c
    return here.parent.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cuda: int | None) -> torch.device:
    if cuda is not None and cuda >= 0 and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if cuda >= num_gpus:
            print(f"[WARN] Requested cuda:{cuda}, but only {num_gpus} GPU(s) available. Falling back to CPU.")
            return torch.device("cpu")
        device = torch.device(f"cuda:{cuda}")
        torch.cuda.set_device(device)
        print(f"[INFO] Using GPU: {device} ({torch.cuda.get_device_name(device)})")
        return device
    if cuda is not None and cuda >= 0 and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Using CPU.")
    else:
        print("[INFO] Using CPU as requested (--cuda -1).")
    return torch.device("cpu")


def resolve_k_from_percent(k_percent: int, n_assets: int) -> int:
    k_value = int(round(n_assets * k_percent / 100.0))
    return max(1, min(k_value, n_assets - 1))


def alpha_str(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def fmt_elapsed(secs: float) -> str:
    hrs = int(secs // 3600)
    mins = int((secs % 3600) // 60)
    return f"{hrs:02d}:{mins:02d}:{secs % 60:05.2f}"


def compute_mdd_from_returns(returns, invalid_policy: str = "nan"):
    if isinstance(returns, list):
        returns = torch.tensor(returns, dtype=torch.float64)
    else:
        returns = returns.detach().to(dtype=torch.float64).cpu()

    gross = 1.0 + returns
    if (gross <= 0).any():
        msg = "MDD undefined: some returns <= -1."
        if invalid_policy == "raise":
            raise ValueError(msg)
        if invalid_policy == "clip":
            gross = torch.clamp(gross, min=1e-12)
        else:
            return float("nan"), None, None

    log_wealth = torch.cumsum(torch.log(gross), dim=0)
    running_peak_log = torch.cummax(log_wealth, dim=0).values
    drawdown = 1.0 - torch.exp(log_wealth - running_peak_log)
    mdd = drawdown.max().item()
    return mdd, log_wealth, drawdown


def summarize_returns(rets: list[float], prefix: str, invalid_policy: str = "nan") -> dict:
    ret_tensor = torch.tensor(rets, dtype=torch.float64)
    mean_raw = ret_tensor.mean().item()
    std_raw = ret_tensor.std(unbiased=True).item() if len(rets) > 1 else float("nan")
    sr_raw = mean_raw / (std_raw + 1e-12)
    mdd_raw, _, _ = compute_mdd_from_returns(rets, invalid_policy=invalid_policy)
    return {
        f"{prefix}_avg_dfl_loss": -mean_raw,
        f"{prefix}_avg_decision_loss": -mean_raw,
        f"{prefix}_raw_sr": sr_raw,
        f"{prefix}_raw_mean": mean_raw,
        f"{prefix}_raw_mdd": mdd_raw,
    }
