"""Differentiable top-k operator (paper Sec. 4.2): sigmoid mask with a bisection-found shift t."""
import torch
from torch.autograd import Function


sigmoid = torch.sigmoid

class TopK1D(Function):
    @staticmethod
    def forward(ctx, xs, k: int, alpha: float = 1.0, iters: int = 64):
        """
        xs: (N,) 1D tensor (may require grad)
        k : int, 0 < k < N
        alpha: float (larger = closer to a hard top-k mask)
        iters: number of bisection iterations
        return: ps (N,), sum(ps)=k
        """
        t, ps = _find_t_1d(xs, k, alpha, iters)
        ctx.save_for_backward(xs, t)
        ctx.alpha = float(alpha)
        return ps

    @staticmethod
    def backward(ctx, grad_output):
        xs, t = ctx.saved_tensors
        alpha = ctx.alpha

        # p = σ(α(x+t))
        p = sigmoid(alpha * (xs + t))
        # v = ∂p/∂x = α * p(1-p)
        v = alpha * (p * (1.0 - p))

        # J = diag(v) - (v v^T) / sum(v)
        s = v.sum().clamp_min(1e-12)
        uv = grad_output * v                      # g ⊙ v
        grad_x = uv - v * (uv.sum() / s)          # g^T J

        # one gradient per forward input: (xs, k, alpha, iters)
        return grad_x, None, None, None


@torch.no_grad()
def _find_t_1d(xs, k: int, alpha: float = 1.0, iters: int = 64):

    n = xs.numel()
    assert 0 < k < n, "k must satisfy 0 < k < len(xs)"
    device, dtype = xs.device, xs.dtype
    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)

    # bracket t so that sigma(alpha*(x+t)) is saturated at 0 / 1 at the two ends
    # α*(xs + lo) <= -10  ⇒ lo <= -xs - 10/α
    # α*(xs + hi) >= +10  ⇒ hi >= -xs + 10/α
    margin = (xs.new_tensor(10.0) / alpha_t)
    lo = -xs.max() - margin
    hi = -xs.min() + margin

    for _ in range(iters):
        mid = (hi + lo) / 2.0
        sums = sigmoid(alpha_t * (xs + mid)).sum()
        if sums < k:
            lo = mid
        else:
            hi = mid

    t = (lo + hi) / 2.0
    ps = sigmoid(alpha_t * (xs + t))
    return t, ps


def soft_topk_1d(xs, k: int, alpha: float = 1.0, iters: int = 64):
    return TopK1D.apply(xs, k, alpha, iters)
