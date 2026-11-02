from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


@dataclass
class Sample:
    t: int
    date: str
    x: torch.Tensor        # [W*N] normalized window
    y_norm: torch.Tensor   # [N]   next-day normalized return
    y_raw: torch.Tensor    # [N]   next-day raw return
    cov: torch.Tensor      # [N,N]
    L: torch.Tensor        # [N,N] Cholesky factor of cov


@dataclass
class Split:
    name: str
    X_flat: torch.Tensor   # [T, W*N]
    y_norm: torch.Tensor   # [T, N]
    y_raw: torch.Tensor    # [T, N]
    covs: torch.Tensor     # [T, N, N]
    Ls: torch.Tensor       # [T, N, N]
    dates: list[str]

    def __len__(self) -> int:
        return self.X_flat.shape[0]

    def __getitem__(self, i: int) -> Sample:
        return Sample(
            t=int(i),
            date=str(self.dates[i]) if i < len(self.dates) else str(i),
            x=self.X_flat[i],
            y_norm=self.y_norm[i],
            y_raw=self.y_raw[i],
            cov=self.covs[i],
            L=self.Ls[i],
        )


def load_returns(dataset: str, data_root: str, data_suffix: str, root: Path) -> pd.DataFrame:
    data_path = root / data_root / f"{dataset}_{data_suffix}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"data not found: {data_path}")
    return pd.read_csv(data_path, index_col="Date")


def normalize_returns(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)


# Making time-series data
def prepare_time_series_data(data: pd.DataFrame, window_size: int):
    inputs, outputs = [], []
    for i in range(len(data) - window_size):
        inputs.append(data.iloc[i:i + window_size].values)
        outputs.append(data.iloc[i + window_size].values)
    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)


def _psd_with_cholesky(cov: torch.Tensor, shrinkage: float, eps: float, max_tries: int, what: str):
    cov = 0.5 * (cov + cov.T)
    n = cov.shape[0]
    eye = torch.eye(n, device=cov.device, dtype=cov.dtype)
    trace_avg = torch.trace(cov) / n
    cov = (1.0 - shrinkage) * cov + shrinkage * trace_avg * eye
    cov = 0.5 * (cov + cov.T)
    eigvals = torch.linalg.eigvalsh(cov)
    min_eig = float(eigvals.min().item())
    jitter = max(-min_eig + eps, eps)
    for _ in range(max_tries):
        try:
            trial_cov = cov + jitter * eye
            trial_cov = 0.5 * (trial_cov + trial_cov.T)
            trial_L = torch.linalg.cholesky(trial_cov)
            return trial_cov, trial_L
        except RuntimeError:
            jitter *= 10.0
    raise RuntimeError(
        f"Cholesky failed ({what}): min_eig={min_eig:.3e}, last_jitter={jitter:.3e}, n_assets={n}"
    )


def compute_rolling_covariances(
    X_windows: torch.Tensor,
    shrinkage: float = 0.10,
    eps: float = 1e-4,
    max_tries: int = 8,
    out_dtype: torch.dtype = torch.float32,
):
    covs_psd, chol_factors = [], []
    with torch.no_grad():
        for i in range(X_windows.shape[0]):
            window = X_windows[i].detach().to(dtype=torch.float64)
            cov, L = _psd_with_cholesky(torch.cov(window.T), shrinkage, eps, max_tries,
                                        what=f"sample {i}, window={window.shape[0]}")
            covs_psd.append(cov.to(dtype=out_dtype))
            chol_factors.append(L.to(dtype=out_dtype))
    return torch.stack(covs_psd, dim=0), torch.stack(chol_factors, dim=0)


def compute_fixed_covariance(
    normalized_train: pd.DataFrame,
    shrinkage: float = 0.10,
    eps: float = 1e-4,
    max_tries: int = 8,
    out_dtype: torch.dtype = torch.float32,
):
    X = torch.tensor(normalized_train.values, dtype=torch.float64)
    cov, L = _psd_with_cholesky(torch.cov(X.T), shrinkage, eps, max_tries, what="fixed train cov")
    return cov.to(dtype=out_dtype), L.to(dtype=out_dtype)


def build_splits(
    df_raw: pd.DataFrame,
    window_size: int,
    train_ratio: float,
    cov_mode: str,
    cov_shrinkage: float,
    cov_eps: float,
    cov_max_tries: int,
    device: torch.device,
) -> tuple[Split, Split, list[str]]:
    normalized = normalize_returns(df_raw)
    train_size = int(train_ratio * len(normalized))
    parts = {
        "train": (normalized.iloc[:train_size], df_raw.iloc[:train_size]),
        "test": (normalized.iloc[train_size:], df_raw.iloc[train_size:]),
    }

    fixed = None
    if cov_mode == "fixed":
        fixed = compute_fixed_covariance(parts["train"][0], cov_shrinkage, cov_eps, cov_max_tries)
    elif cov_mode != "rolling":
        raise ValueError(f"Unknown cov_mode: {cov_mode}")

    splits = {}
    for name, (norm_df, raw_df) in parts.items():
        if len(norm_df) <= window_size:
            raise ValueError(f"{name} length {len(norm_df)} is not larger than window_size={window_size}.")
        X, y = prepare_time_series_data(norm_df, window_size)
        _, y_raw = prepare_time_series_data(raw_df, window_size)
        if cov_mode == "rolling":
            covs, Ls = compute_rolling_covariances(X, cov_shrinkage, cov_eps, cov_max_tries)
        else:
            cov, L = fixed
            covs = cov.unsqueeze(0).expand(X.shape[0], -1, -1).contiguous()
            Ls = L.unsqueeze(0).expand(X.shape[0], -1, -1).contiguous()
        splits[name] = Split(
            name=name,
            X_flat=X.view(X.shape[0], -1).to(device),
            y_norm=y.to(device),
            y_raw=y_raw.to(device),
            covs=covs.to(device),
            Ls=Ls.to(device),
            dates=[str(d) for d in norm_df.index[window_size:]],
        )
    return splits["train"], splits["test"], list(df_raw.columns)
