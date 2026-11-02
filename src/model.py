from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def dense_nn_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] = [256, 128],
    activation: str = 'tanh',
    dropout: float = 0.1,
):
    if activation == 'relu':
        act = nn.ReLU
    elif activation == 'sigmoid':
        act = nn.Sigmoid
    elif activation == 'tanh':
        act = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def infer_mlp_arch(state_dict: dict) -> tuple[list[int], bool]:
    """(hidden_dims, has_dropout) from a dense_nn_mlp state_dict; activation cannot be inferred."""
    weight_keys = sorted((int(k.split(".")[0]) for k in state_dict if k.endswith(".weight")))
    hidden_dims = [state_dict[f"{i}.weight"].shape[0] for i in weight_keys[:-1]]
    stride = weight_keys[1] - weight_keys[0] if len(weight_keys) > 1 else 2
    return hidden_dims, stride == 3


def load_model(path: Path, input_dim: int, output_dim: int, device: torch.device,
               hidden_dims=None, activation: str = "tanh", dropout: float = 0.1) -> nn.Module:
    state = torch.load(path, map_location=device)
    ckpt_hidden, has_dropout = infer_mlp_arch(state)
    if hidden_dims is not None and list(hidden_dims) != ckpt_hidden:
        print(f"[WARN] checkpoint hidden_dims={ckpt_hidden} differ from requested {list(hidden_dims)}; using checkpoint's")
    model = dense_nn_mlp(input_dim, output_dim, hidden_dims=ckpt_hidden, activation=activation,
                         dropout=dropout if has_dropout else 0.0).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"[LOAD] arch hidden={ckpt_hidden} activation={activation} dropout={'yes' if has_dropout else 'no'}")
    return model


@torch.no_grad()
def predict_all(model: nn.Module, X_flat: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    model.eval()
    preds = [model(X_flat[s:s + batch_size]) for s in range(0, X_flat.shape[0], batch_size)]
    return torch.cat(preds, dim=0)
