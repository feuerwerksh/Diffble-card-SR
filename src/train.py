from __future__ import annotations

import random
from collections import defaultdict
from time import perf_counter

import torch
import torch.nn as nn

from problem import SparseSharpeProblem
from utils import fmt_elapsed


def train_model(
    model: nn.Module,
    problem: SparseSharpeProblem,
    loss_fn,
    epochs: int,
    batchsize: int,
    lr: float,
    weight_decay: float,
    seed: int,
    log_prefix: str = "",
) -> tuple[nn.Module, list[dict]]:
    py_rng = random.Random(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train = problem.get_train_data()
    n_train = len(train)
    history: list[dict] = []
    t0 = perf_counter()

    for epoch in range(epochs):
        model.train()
        selected_idx = py_rng.sample(range(n_train), min(batchsize, n_train))
        losses, parts_acc = [], defaultdict(list)
        for i in selected_idx:
            s = train[i]
            yhat = model(s.x).squeeze()
            loss, parts = loss_fn(yhat, s.y_norm, s)
            losses.append(loss)
            for key, val in parts.items():
                parts_acc[key].append(val)

        losses = torch.stack(losses)
        n_valid = (len(losses) - losses.isnan().sum()).clamp_min(1)
        loss = losses.nansum() / n_valid
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        row = {"epoch": epoch + 1, "loss": loss.item()}
        row.update({key: sum(vals) / len(vals) for key, vals in parts_acc.items()})
        history.append(row)

        if problem.device.type == "cuda":
            torch.cuda.synchronize(problem.device)
        extra = " | ".join(f"{key.upper()}: {val:.4f}" for key, val in row.items() if key not in ("epoch", "loss"))
        print(
            f"[{log_prefix}] Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}"
            + (f" | {extra}" if extra else "")
            + f" | Time: {fmt_elapsed(perf_counter() - t0)}"
        )

    model.eval()
    return model, history
