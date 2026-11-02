from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from artifacts import batch_dir_name, make_exp_name, pfl_model_dir_name, save_batch_summary, save_pfl_model, save_run
from evaluate import EvalConfig, evaluate_split
from losses import MSE, get_loss_fn
from model import dense_nn_mlp, load_model, predict_all
from problem import SELECTORS, SparseSharpeProblem
from train import train_model
from utils import get_device, get_project_root, resolve_k_from_percent, set_seed

DATASETS = ["euro50", "nikkei225", "ftse100", "kospi200", "snp100"]
HIST_SEED = 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sparse tangent portfolio: historic / PFL / DFL experiments")
    p.add_argument("--mode", type=str, required=True, choices=["historic", "pfl", "dfl"],
                   help="historic: window-mean mu; pfl: MSE-trained model; dfl: decision-focused training")
    p.add_argument("--selector", type=str, nargs="+", default=["oscar"], choices=list(SELECTORS),
                   help="downstream optimizer(s); dfl supports oscar only")
    p.add_argument("--dataset", type=str, default="euro50", choices=DATASETS)
    p.add_argument("--k_list", type=int, nargs="+", default=[10, 15, 20], help="k as %% of n_assets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.5, help="dfl only: weight of the DFL term")
    # training (pfl, dfl)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batchsize", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--activation", type=str, default="tanh", choices=["relu", "sigmoid", "tanh"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--retrain", action="store_true", help="pfl: retrain even if the MSE checkpoint exists")
    p.add_argument("--model_dir", type=str, default=None,
                   help="pfl: where {dataset}_PFL_MSE_seed{s}/model.pt is looked up / saved (default: save_dir)")
    # data / covariance
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--data_root", type=str, default="data/daily")
    p.add_argument("--data_suffix", type=str, default="160101_251231")
    p.add_argument("--cov_mode", type=str, default="rolling", choices=["rolling", "fixed"])
    p.add_argument("--cov_shrinkage", type=float, default=0.10)
    p.add_argument("--cov_eps", type=float, default=1e-4)
    p.add_argument("--cov_max_tries", type=int, default=8)
    # oscar layer
    p.add_argument("--topk_alpha", type=float, default=5.0)
    p.add_argument("--topk_iters", type=int, default=32)
    # sdr
    p.add_argument("--sdr_solver", type=str, default="SCS")
    p.add_argument("--sdr_eps", type=float, default=1e-3)
    p.add_argument("--sdr_max_iters", type=int, default=2000)
    p.add_argument("--sdr_stride", type=int, default=1,
                   help="sdr only: solve every N steps and interpolate the realised-return trace (1 = exact)")
    # historic
    p.add_argument("--hist_mu", type=str, default="normalized", choices=["normalized", "raw"])
    # evaluation / io
    p.add_argument("--splits", type=str, nargs="+", default=["train", "test"], choices=["train", "test"])
    p.add_argument("--mdd_invalid_policy", type=str, default="nan", choices=["raise", "nan", "clip"])
    p.add_argument("--print_every", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="artifacts")
    p.add_argument("--cuda", type=int, default=0, help="GPU index, -1 for CPU")
    p.add_argument("--overwrite", action="store_true", help="re-run even if summary_metrics.csv exists")
    return p


def make_problem(args, root: Path, device: torch.device) -> SparseSharpeProblem:
    return SparseSharpeProblem(
        dataset=args.dataset, root=root, device=device,
        data_root=args.data_root, data_suffix=args.data_suffix,
        window_size=args.window_size, train_ratio=args.train_ratio,
        cov_mode=args.cov_mode, cov_shrinkage=args.cov_shrinkage, cov_eps=args.cov_eps, cov_max_tries=args.cov_max_tries,
        topk_alpha=args.topk_alpha, topk_iters=args.topk_iters,
        sdr_solver=args.sdr_solver, sdr_eps=args.sdr_eps, sdr_max_iters=args.sdr_max_iters,
    )


def mlp_kwargs(args) -> dict:
    return dict(hidden_dims=list(args.hidden_dims), activation=args.activation, dropout=args.dropout)


def new_model(problem: SparseSharpeProblem, args, device: torch.device):
    with torch.device(device):
        return dense_nn_mlp(*problem.get_modelio_shape(), **mlp_kwargs(args))


def base_row(args, mode: str, selector: str, k_percent: int, actual_k: int, seed: int, alpha: float) -> dict:
    return {
        "dataset": args.dataset,
        "method": {"dfl": "DFL", "pfl": "PFL", "historic": "HIST"}[mode],
        "selector": selector,
        "k_percent": k_percent,
        "actual_k": actual_k,
        "alpha": alpha,
        "epochs": args.epochs if mode != "historic" else 0,
        "batchsize": args.batchsize if mode != "historic" else 0,
        "seed": seed,
        "window_size": args.window_size,
        "cov_mode": args.cov_mode,
        "cov_shrinkage": args.cov_shrinkage,
        "cov_eps": args.cov_eps,
        "cov_max_tries": args.cov_max_tries,
        "sdr_solver": args.sdr_solver if selector == "sdr" else "",
        "sdr_eps": args.sdr_eps if selector == "sdr" else "",
        "sdr_max_iters": args.sdr_max_iters if selector == "sdr" else "",
        "sdr_stride": args.sdr_stride if selector == "sdr" else 1,
    }


def meta_extra(args) -> dict:
    return {
        "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay,
        "hidden_dims": list(args.hidden_dims), "activation": args.activation,
        "topk_alpha": args.topk_alpha, "topk_iters": args.topk_iters,
        "data_suffix": args.data_suffix, "train_ratio": args.train_ratio,
        "hist_mu": args.hist_mu, "mdd_invalid_policy": args.mdd_invalid_policy,
        "splits": list(args.splits), "cuda": args.cuda,
    }


def existing_row(save_root: Path, overwrite: bool) -> dict | None:
    summary_csv = save_root / "summary_metrics.csv"
    if summary_csv.exists() and not overwrite:
        print(f"[SKIP] exists: {summary_csv}")
        return pd.read_csv(summary_csv).iloc[0].to_dict()
    return None


def evaluate_and_save(
    args, problem: SparseSharpeProblem, save_root: Path, mode: str, selector: str,
    k_percent: int, actual_k: int, seed: int, alpha: float,
    mu_hat_by_split: dict[str, torch.Tensor], row_extra: dict, model=None,
) -> dict:
    cfg = EvalConfig(sdr_stride=args.sdr_stride, mdd_invalid_policy=args.mdd_invalid_policy,
                     print_every=args.print_every, compute_pfl_loss=(mode != "historic"))
    row = base_row(args, mode, selector, k_percent, actual_k, seed, alpha)
    row.update(row_extra)
    traces, lasts = {}, {}
    for split_name in args.splits:
        split = problem.get_split(split_name)
        metrics, trace, last = evaluate_split(problem, split, mu_hat_by_split[split_name], actual_k, selector, cfg)
        row.update(metrics)
        traces[split_name] = trace
        lasts[split_name] = last
    row["exp_name"] = save_root.name
    save_run(save_root, row, meta_extra(args), traces, lasts, problem.asset_names, mode, selector, actual_k, model=model)
    return row


def run_dfl(args, problem: SparseSharpeProblem, root: Path, device: torch.device) -> list[dict]:
    if args.selector != ["oscar"]:
        raise SystemExit("--mode dfl supports --selector oscar only")
    save_dir = root / args.save_dir
    seed = args.seed
    rows = []
    for run_idx, k_percent in enumerate(args.k_list, start=1):
        actual_k = resolve_k_from_percent(k_percent, problem.n_assets)
        exp_name = make_exp_name("dfl", args.dataset, k_percent, actual_k, args.alpha, seed, "oscar")
        print("=" * 100)
        print(f"[RUN {run_idx}/{len(args.k_list)}] dfl | {exp_name} | n_assets={problem.n_assets}")
        old = existing_row(save_dir / exp_name, args.overwrite)
        if old is not None:
            rows.append(old)
            continue

        set_seed(seed)
        model = new_model(problem, args, device)
        loss_fn = get_loss_fn("dfl", problem, k=actual_k, alpha=args.alpha)
        model, _ = train_model(model, problem, loss_fn, args.epochs, args.batchsize, args.lr,
                               args.weight_decay, seed, log_prefix=f"{args.dataset}|seed={seed}|k={k_percent}%")
        mu_hat = {s: predict_all(model, problem.get_split(s).X_flat) for s in args.splits}
        rows.append(evaluate_and_save(args, problem, save_dir / exp_name, "dfl", "oscar",
                                      k_percent, actual_k, seed, args.alpha, mu_hat, {}, model=model))
    return rows


def get_or_train_pfl_model(args, problem: SparseSharpeProblem, root: Path, device: torch.device, seed: int):
    model_dir = root / (args.model_dir or args.save_dir) / pfl_model_dir_name(args.dataset, seed)
    ckpt = model_dir / "model.pt"
    in_dim, out_dim = problem.get_modelio_shape()
    if ckpt.exists() and not args.retrain:
        print(f"[LOAD] seed={seed} model -> {ckpt}")
        activation = args.activation
        meta_path = model_dir / "meta.json"
        if meta_path.exists():
            activation = json.loads(meta_path.read_text()).get("activation") or activation
        else:
            print(f"[WARN] no meta.json next to {ckpt}; assuming --activation {activation}")
        return load_model(ckpt, in_dim, out_dim, device, activation=activation, dropout=args.dropout), model_dir

    set_seed(seed)
    model = new_model(problem, args, device)
    model, _ = train_model(model, problem, get_loss_fn("mse", problem), args.epochs, args.batchsize, args.lr,
                           args.weight_decay, seed, log_prefix=f"PFL {args.dataset}|seed={seed}")
    with torch.no_grad():
        train_mse = MSE(predict_all(model, problem.train.X_flat), problem.train.y_norm).item()
        test_mse = MSE(predict_all(model, problem.test.X_flat), problem.test.y_norm).item()
    row = {
        "dataset": args.dataset, "method": "PFL", "loss": "MSE", "seed": seed,
        "epochs": args.epochs, "batchsize": args.batchsize, "window_size": args.window_size,
        "activation": args.activation, "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay,
        "data_suffix": args.data_suffix, "train_mse": train_mse, "test_mse": test_mse, "exp_name": model_dir.name,
    }
    save_pfl_model(model_dir, model, row, {"hidden_dims": list(args.hidden_dims), "cuda": args.cuda})
    return model, model_dir


def run_pfl(args, problem: SparseSharpeProblem, root: Path, device: torch.device) -> list[dict]:
    save_dir = root / args.save_dir
    seed = args.seed
    model, model_dir = get_or_train_pfl_model(args, problem, root, device, seed)
    mu_hat = {s: predict_all(model, problem.get_split(s).X_flat) for s in args.splits}
    rows, run_idx, total = [], 0, len(args.k_list) * len(args.selector)
    for k_percent in args.k_list:
        actual_k = resolve_k_from_percent(k_percent, problem.n_assets)
        for selector in args.selector:
            run_idx += 1
            exp_name = make_exp_name("pfl", args.dataset, k_percent, actual_k, 0.0, seed, selector)
            print("=" * 100)
            print(f"[RUN {run_idx}/{total}] pfl | {exp_name} | cov_mode={args.cov_mode}")
            old = existing_row(save_dir / exp_name, args.overwrite)
            if old is not None:
                rows.append(old)
                continue
            rows.append(evaluate_and_save(args, problem, save_dir / exp_name, "pfl", selector,
                                          k_percent, actual_k, seed, 0.0, mu_hat,
                                          {"model_exp_name": model_dir.name}))
    return rows


def run_historic(args, problem: SparseSharpeProblem, root: Path, device: torch.device) -> list[dict]:
    if args.seed != HIST_SEED:
        print(f"[INFO] historic mode has no randomness; ignoring --seed, using seed={HIST_SEED}")
    set_seed(HIST_SEED)
    save_dir = root / args.save_dir
    mu_hat = {s: problem.historic_mu(problem.get_split(s), args.hist_mu) for s in args.splits}
    rows, run_idx, total = [], 0, len(args.k_list) * len(args.selector)
    for k_percent in args.k_list:
        actual_k = resolve_k_from_percent(k_percent, problem.n_assets)
        for selector in args.selector:
            run_idx += 1
            exp_name = make_exp_name("historic", args.dataset, k_percent, actual_k, 0.0, HIST_SEED, selector)
            print("=" * 100)
            print(f"[RUN {run_idx}/{total}] historic | {exp_name} | hist_mu={args.hist_mu}")
            old = existing_row(save_dir / exp_name, args.overwrite)
            if old is not None:
                rows.append(old)
                continue
            rows.append(evaluate_and_save(args, problem, save_dir / exp_name, "historic", selector,
                                          k_percent, actual_k, HIST_SEED, 0.0, mu_hat, {"hist_mu": args.hist_mu}))
    return rows


def main() -> None:
    args = build_argparser().parse_args()
    root = get_project_root()
    device = get_device(args.cuda)
    print(f"[INFO] root={root} | mode={args.mode} | dataset={args.dataset} | selector={args.selector} | seed={args.seed}")

    problem = make_problem(args, root, device)
    print(f"[INFO] n_assets={problem.n_assets} | train={len(problem.train)} | test={len(problem.test)} | cov_mode={args.cov_mode}")

    rows = {"dfl": run_dfl, "pfl": run_pfl, "historic": run_historic}[args.mode](args, problem, root, device)
    save_batch_summary(root / args.save_dir, batch_dir_name(args.mode, args.dataset, args.alpha), rows)


if __name__ == "__main__":
    main()
