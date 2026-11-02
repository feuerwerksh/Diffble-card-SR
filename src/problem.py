from __future__ import annotations

from pathlib import Path

import torch

from baseline import OSCAR, SDR_revised, mSSRM_PGA, make_SDR_solver
from data import Sample, Split, build_splits, load_returns, prepare_time_series_data
from oscar import DiffbleOSCAR

SELECTORS = ("oscar", "sdr", "ssrm", "oscar_hard")


class SparseSharpeProblem:
    """Cardinality-constrained Sharpe-ratio maximisation on one market (PThenO-style)."""

    def __init__(
        self,
        dataset: str,
        root: Path,
        device: torch.device,
        data_root: str = "data/daily",
        data_suffix: str = "160101_251231",
        window_size: int = 100,
        train_ratio: float = 0.8,
        cov_mode: str = "rolling",
        cov_shrinkage: float = 0.10,
        cov_eps: float = 1e-4,
        cov_max_tries: int = 8,
        topk_alpha: float = 5.0,
        topk_iters: int = 32,
        sdr_solver: str = "SCS",
        sdr_eps: float = 1e-3,
        sdr_max_iters: int = 2000,
    ):
        self.dataset = dataset
        self.device = device
        self.window_size = int(window_size)
        self.train_ratio = float(train_ratio)
        self.cov_mode = cov_mode
        self.topk_alpha = float(topk_alpha)
        self.topk_iters = int(topk_iters)
        self.sdr_kwargs = dict(solver=sdr_solver, eps=sdr_eps, max_iters=sdr_max_iters)

        self.df_raw = load_returns(dataset, data_root, data_suffix, root)
        self.asset_names = list(self.df_raw.columns)
        self.n_assets = self.df_raw.shape[1]
        self.train, self.test, _ = build_splits(
            self.df_raw, self.window_size, self.train_ratio,
            cov_mode, cov_shrinkage, cov_eps, cov_max_tries, device,
        )

        self._oscar_layers: dict[int, DiffbleOSCAR] = {}
        self._sdr_fixed: dict[int, callable] = {}

    def get_train_data(self) -> Split:
        return self.train

    def get_test_data(self) -> Split:
        return self.test

    def get_split(self, name: str) -> Split:
        return {"train": self.train, "test": self.test}[name]

    def get_modelio_shape(self) -> tuple[int, int]:
        return self.window_size * self.n_assets, self.n_assets

    def get_twostageloss(self) -> str:
        return "mse"

    def historic_mu(self, split: Split, kind: str = "normalized") -> torch.Tensor:
        T = len(split)
        if kind == "normalized":
            return split.X_flat.view(T, self.window_size, self.n_assets).mean(dim=1)
        if kind == "raw":
            train_size = int(self.train_ratio * len(self.df_raw))
            raw_df = self.df_raw.iloc[:train_size] if split.name == "train" else self.df_raw.iloc[train_size:]
            X_raw, _ = prepare_time_series_data(raw_df, self.window_size)
            return X_raw.mean(dim=1).to(self.device)
        raise ValueError(f"Unknown hist_mu kind: {kind}")

    def _oscar(self, k: int) -> DiffbleOSCAR:
        if k not in self._oscar_layers:
            self._oscar_layers[k] = DiffbleOSCAR(self.n_assets, k, self.topk_alpha, self.topk_iters).to(self.device)
        return self._oscar_layers[k]

    def _sdr_solver_fixed(self, k: int, cov: torch.Tensor):
        if k not in self._sdr_fixed:
            S_np = cov.detach().cpu().to(torch.float64).numpy()
            self._sdr_fixed[k] = make_SDR_solver(S_np, k, **self.sdr_kwargs)
        return self._sdr_fixed[k]

    def get_decision(self, mu_hat: torch.Tensor, aux: Sample, k: int, selector: str) -> tuple[torch.Tensor, dict]:
        """(w, info); only 'oscar' is differentiable, the baselines return float64 CPU weights."""
        if selector == "oscar":
            w, gate = self._oscar(k).run(mu_hat, aux.cov, aux.L)
            return w, {"gate": gate}

        mu64 = mu_hat.detach().cpu().to(torch.float64).view(-1)
        cov64 = aux.cov.detach().cpu().to(torch.float64)
        if selector == "sdr":
            if self.cov_mode == "fixed":
                sr, w = self._sdr_solver_fixed(k, cov64)(mu64)
            else:
                sr, w = SDR_revised(mu64, cov64, k, **self.sdr_kwargs)
        elif selector == "ssrm":
            sr, w = mSSRM_PGA(mu64, cov64, k)
        elif selector == "oscar_hard":
            last_err = None
            for eps in (1e-8, 1e-6, 1e-4):  # retry with more jitter on solver failure
                try:
                    sr, w = OSCAR(mu64, cov64, k, eps=eps)
                    break
                except Exception as e:
                    last_err = e
            else:
                raise RuntimeError(f"oscar_hard failed at t={aux.t}: {last_err}")
        else:
            raise ValueError(f"Unknown selector: {selector}. Use one of {SELECTORS}.")
        return w, {"sr_est": float(sr)}

    def get_objective(self, y_raw: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return -torch.matmul(y_raw, w)
