import numpy as np
import cvxpy as cp
import torch
from oscar import QCQPlayer

def SDR_revised(mu: torch.Tensor, S: torch.Tensor, k: int,
                solver: str = "SCS", eps: float = 1e-3, max_iters: int = 2000):
    """
    Semidefinite relaxation (P7) for cardinality-constrained Sharpe maximization.
    - SDP: max Tr((mu mu^T) Y) s.t. Tr(Y)=z, sum(abs(Y))<=k z, Tr(S Y)=1, Y >= 0
    - Rounding: leading eigenvector
    - Final scaling: budget (sum-to-1), as in the paper
    Returns:
      SR: float (Sharpe with budget=1 weights)
      w : torch.Tensor [n], float64, cpu, sum(w)=1
    """
    # --- to numpy (float64) ---
    mu_np = mu.detach().cpu().to(torch.float64).numpy().reshape(-1, 1)  # [n,1]
    S_np  = S.detach().cpu().to(torch.float64).numpy()                  # [n,n]
    n = mu_np.shape[0]
    M = mu_np @ mu_np.T  # [n,n]

    # --- SDP (SDR) ---
    Y = cp.Variable((n, n), PSD=True)
    z = cp.Variable(nonneg=True)

    objective = cp.Maximize(cp.trace(M @ Y))
    constraints = [
        cp.trace(Y) == z,               # scale reference
        cp.sum(cp.abs(Y)) <= k * z,     # entrywise ℓ1 sparsity surrogate
        cp.trace(S_np @ Y) == 1         # variance = 1 on lifted var
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, eps=eps, max_iters=max_iters, warm_start=False, verbose=False)

    if Y.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDR failed: status={prob.status}")

    # --- rank-1 rounding: leading eigenvector of symmetrized Y ---
    Y_opt = Y.value
    Y_sym = (Y_opt + Y_opt.T) / 2.0
    evals, evecs = np.linalg.eigh(Y_sym)
    v = evecs[:, np.argmax(evals)]  # leading eigenvector

    # --- budget (sum-to-1) scaling ---
    denom = np.sum(v)
    if abs(denom) < 1e-16:
        raise RuntimeError("SDR rounding failed: leading eigenvector has near-zero sum.")
    w_budget = v / denom                    # sum-to-1
    # Sharpe under budget=1
    num = float(mu_np.ravel().dot(w_budget))
    den = float(np.sqrt(w_budget @ (S_np @ w_budget) + 1e-16))
    SR = num / den

    w = torch.from_numpy(w_budget.astype(np.float64))
    return SR, w

def mSSRM_PGA(p, Q, k):
    """
    Inputs: p (torch.Tensor [n]), Q (torch.Tensor [n,n])
    Returns: SR (float), w (torch.Tensor [n], cpu, float64)
    """
    # --- to numpy ---
    p_t = p.detach().cpu().to(torch.float64)
    Q_t = Q.detach().cpu().to(torch.float64)
    p_np = p_t.numpy()
    Q_np = Q_t.numpy()

    Param = {'eps': 1e-4, 'iternum': 10000, 'tol': 1e-5, 'm': k}
    RE = [100.0]
    N = p_np.shape[0]

    # Step size
    Param['alpha'] = 0.999 / np.linalg.norm(Q_np, 2)
    w = p_np.copy()
    i = 1
    while i <= Param['iternum'] and RE[-1] > Param['tol']:
        w1 = w.copy()
        w_pre = w - Param['alpha'] * (Q_np @ w - p_np)
        itw = np.argsort(-w_pre)
        w = np.zeros(N)
        w[itw[:Param['m']]] = w_pre[itw[:Param['m']]]
        denom = np.linalg.norm(w1, 2)
        RE.append(np.linalg.norm(w - w1, 2) / (denom if denom > 0 else 1.0))
        i += 1

    if np.sum(w) == 0:
        w = np.zeros(N)
    else:
        w = w / np.sum(w)

    SR = float((p_np.T @ w) / np.sqrt(w.T @ Q_np @ w + 1e-16))
    w = torch.from_numpy(w).to(dtype=torch.float64)
    return SR, w


def mSSRM_PGA_revised(p, Qeps, k):
    """
    m-Sparse Sharpe Ratio Maximization via Proximal Gradient Algorithm (Algorithm A1).

    Not used by main.py (problem.get_decision uses mSSRM_PGA).

    Inputs:
        p    : torch.Tensor [N]      -- p = (1/T) Rᵀ 1_T
        Qeps : torch.Tensor [N, N]   -- Qε = QᵀQ + εI  (symmetric PD; caller must add εI)
        k    : int                   -- sparsity level m
    Returns:
        SR : float
        w  : torch.Tensor [N], cpu, float64
    """
    # --- to numpy (float64) ---
    p_np = p.detach().cpu().to(torch.float64).numpy()
    Q_np = Qeps.detach().cpu().to(torch.float64).numpy()

    N = p_np.shape[0]
    m = k
    iternum, tol = 10000, 1e-5

    # Step size: α = 0.999 / λmax(Qε) = 0.999 / ||Qε||_2
    alpha = 0.999 / np.linalg.norm(Q_np, 2)

    w = p_np.copy()          # v^(0) = p
    re = 100.0
    i = 1
    while i <= iternum and re > tol:
        w1 = w.copy()
        w_pre = w - alpha * (Q_np @ w - p_np)   # v - α∇f(v),  ∇f = Qε v - p

        # prox_{ιΩ}: keep ONLY positive components, at most m of them (Prop. 3)
        w = np.zeros(N)
        if (w_pre > 0).sum() > m:
            idx = np.argsort(-w_pre)[:m]        # > m positives → top-m (all positive)
        else:
            idx = np.where(w_pre > 0)[0]        # ≤ m positives → keep all positives
        w[idx] = w_pre[idx]

        denom = np.linalg.norm(w1, 2)
        re = np.linalg.norm(w - w1, 2) / (denom if denom > 0 else 1.0)
        i += 1

    # Normalize to simplex (or stay in risk-free asset if v* = 0)
    s = w.sum()
    if s == 0:               # now safe: all components ≥ 0, so sum==0 ⟺ w==0
        w = np.zeros(N)
    else:
        w = w / s

    SR = float((p_np @ w) / np.sqrt(w @ Q_np @ w + 1e-16))
    w = torch.from_numpy(w).to(dtype=torch.float64)
    return SR, w

_qcqp_cache: dict = {}


def _qcqp(n: int):
    if n not in _qcqp_cache:
        _qcqp_cache[n] = QCQPlayer(n)
    return _qcqp_cache[n]


@torch.no_grad()
def OSCAR(ret: torch.Tensor, cov: torch.Tensor, k: int, eps: float = 1e-8):
    """
    Classic OSCAR with a hard top-k selection (cvxpylayers)
    Inputs: ret μ (torch [n]), cov Σ (torch [n,n]), k (int)
    Returns: SR (float), w (torch [n], cpu, float64)
    """
    ret = ret.detach().to(dtype=torch.float64, device='cpu')
    cov = cov.detach().to(dtype=torch.float64, device='cpu')
    n = ret.numel()

    # Σ = L L^T
    cov = cov + eps * torch.eye(n, dtype=torch.float64)
    L = torch.linalg.cholesky(cov)

    # Step O
    qcqp_full = _qcqp(n)
    y_full, t_full = qcqp_full(ret, L)         # (n,), (1,)
    w_full = y_full / (t_full + eps)

    # Select |L^T w|
    Lw = L.T @ w_full
    idx_topk = torch.topk(Lw.abs(), k=k, largest=True, sorted=True).indices

    # Step AR
    mu_k = ret.index_select(0, idx_topk)
    Sigma_k = cov.index_select(0, idx_topk).index_select(1, idx_topk)
    Sigma_k = Sigma_k + eps * torch.eye(k, dtype=torch.float64)
    # rescale Sigma_k (Sharpe-invariant) so that the budget QCQP stays feasible
    ones_k = torch.ones(k, dtype=torch.float64)
    min_var = 1.0 / float(ones_k @ torch.linalg.solve(Sigma_k, ones_k))
    if min_var > 0.5:
        Sigma_k = Sigma_k * (0.5 / min_var)
    L_k = torch.linalg.cholesky(Sigma_k)

    qcqp_k = _qcqp(k)
    y_k, t_k = qcqp_k(mu_k, L_k)
    w_k = y_k / (t_k + eps)

    # Scatter
    w = torch.zeros(n, dtype=torch.float64)
    w.index_copy_(0, idx_topk, w_k)

    # SR
    num = (ret * w).sum()
    den = torch.sqrt((w @ (cov @ w)).clamp_min(1e-16))
    SR = float((num / den).item()) if den > 0 else 0.0
    return SR, w

def make_SDR_solver(S_np: np.ndarray, k: int,
                    solver: str = "SCS",
                    eps: float = 1e-3,
                    max_iters: int = 2000):
    """
    Builds the SDR SDP once for fixed (S, k) and returns a closure that re-solves it for new mu.
    """
    n = S_np.shape[0]

    Y = cp.Variable((n, n), PSD=True)
    z = cp.Variable(nonneg=True)

    # only M = mu mu^T changes between calls
    M_param = cp.Parameter((n, n))

    objective = cp.Maximize(cp.trace(M_param @ Y))
    constraints = [
        cp.trace(Y) == z,
        cp.sum(cp.abs(Y)) <= k * z,
        cp.trace(S_np @ Y) == 1,
    ]
    prob = cp.Problem(objective, constraints)

    def solve(mu_torch: torch.Tensor):
        mu_np = mu_torch.detach().cpu().to(torch.float64).numpy().reshape(-1, 1)
        M_param.value = mu_np @ mu_np.T

        prob.solve(
            solver=solver,
            eps=eps,
            max_iters=max_iters,
            warm_start=True,
            verbose=False,
        )

        if Y.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"SDR failed: status={prob.status}")

        # rank-1 rounding (same as SDR_revised)
        Y_opt = Y.value
        Y_sym = (Y_opt + Y_opt.T) / 2.0
        evals, evecs = np.linalg.eigh(Y_sym)
        v = evecs[:, np.argmax(evals)]

        w_budget = v / (np.sum(v) + 1e-16)
        num = float(mu_np.ravel().dot(w_budget))
        den = float(np.sqrt(w_budget @ (S_np @ w_budget) + 1e-16))
        SR = num / den

        w = torch.from_numpy(w_budget.astype(np.float64))
        return SR, w

    return solve