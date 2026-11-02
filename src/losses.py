from __future__ import annotations

import cvxpy as cp
import torch

from data import Sample
from problem import SparseSharpeProblem


def MSE(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (yhat - y).square().mean()


def get_loss_fn(name: str, problem: SparseSharpeProblem, k: int | None = None, alpha: float = 0.5):
    """Returns loss_fn(yhat, y_norm, aux) -> (loss, parts); parts are logged per epoch."""
    if name == "mse":
        def mse_loss(yhat, y, aux):
            loss = MSE(yhat, y)
            return loss, {"pfl": loss.item()}
        return mse_loss

    if name == "dfl":
        if k is None:
            raise ValueError("dfl loss needs k")

        def dfl_loss(yhat, y, aux: Sample):
            try:
                w, _ = problem.get_decision(yhat, aux, k, "oscar")
            except (cp.SolverError, RuntimeError) as e:
                print(f"[WARN] t={aux.t} ({aux.date}) decision layer failed: {e}")
                return yhat.sum() * float("nan"), {}  # dropped by the nan-safe mean in train_model
            l_dfl = problem.get_objective(aux.y_raw, w)
            l_mse = MSE(yhat, y)
            loss = alpha * l_dfl + (1 - alpha) * l_mse
            with torch.no_grad():
                sharpe = torch.dot(y, w) / torch.sqrt(w @ aux.cov @ w + 1e-12)
            return loss, {"dfl": l_dfl.item(), "pfl": l_mse.item(), "sharpe": sharpe.item()}
        return dfl_loss

    raise ValueError(f"Unknown loss: {name} (use 'mse' or 'dfl')")
