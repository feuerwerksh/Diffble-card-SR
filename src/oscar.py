from __future__ import annotations

import cvxpy as cp
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer

from selector import soft_topk_1d


def QCQPlayer(_n: int) -> CvxpyLayer:
    y_var = cp.Variable(_n)
    t_var = cp.Variable()
    mean_param = cp.Parameter(_n)
    L_param = cp.Parameter((_n, _n))

    obj = cp.Maximize(y_var @ mean_param)
    const = [cp.sum(y_var) == t_var,
             cp.norm(L_param @ y_var, 2) <= 1,
             t_var >= 1]

    problem = cp.Problem(obj, const)
    return CvxpyLayer(problem, parameters=[mean_param, L_param], variables=[y_var, t_var])


class DiffbleOSCAR(nn.Module):
    def __init__(self, n_assets: int, k: int, topk_alpha: float = 5.0, topk_iters: int = 32):
        super().__init__()
        self.k = int(k)
        self.n = int(n_assets)
        assert 0 < self.k < self.n, f"soft top-k needs 0 < k < n (got k={self.k}, n={self.n})"
        self.topk_alpha = float(topk_alpha)
        self.topk_iters = int(topk_iters)

        self.layer_O = QCQPlayer(self.n)
        self.layer_AR = QCQPlayer(self.n)

    @torch.enable_grad()
    def run(self, mean: torch.Tensor, cov_psd: torch.Tensor, L: torch.Tensor | None = None):
        if L is None:
            L = torch.linalg.cholesky(cov_psd)
        y_opt, t_opt = self.layer_O(mean, L)
        w = y_opt / (t_opt + 1e-8)
        Lw = L.T @ w
        gate = soft_topk_1d(Lw.abs(), self.k, alpha=self.topk_alpha, iters=self.topk_iters)
        masked_mean = mean * gate
        y_oscar, t_oscar = self.layer_AR(masked_mean, L)
        w_oscar = y_oscar / (t_oscar + 1e-8)
        return w_oscar, gate

    forward = run
