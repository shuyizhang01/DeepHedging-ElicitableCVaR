"""
scripts/train_static.py

Entry point for static-risk (CVaR terminal loss) hedging training.

Usage:
    python scripts/train_static.py
    python scripts/train_static.py --config cfgs/configStaticRisk.yaml
    python scripts/train_static.py --config cfgs/configStaticRisk.yaml --device cpu
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_global_seeds(seed: int = 42) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train static-risk hedging agent")
    parser.add_argument("--config", default="cfgs/configStaticRisk.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device_str = args.device or cfg.get("device", "cuda")
    device = device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
    if device != device_str:
        print(f"CUDA not available — falling back to CPU")

    set_global_seeds(cfg.get("seed", 42))

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    from src.data.calibration import (download_returns, fit_t_copula,
                                      compute_garch_residuals, fit_dcc_parameters)
    from src.envs.DCCGARCH import DCCGARCHSimulator
    from src.envs.HedgingEnv import HedgingEnv
    from src.agents.StaticRiskTraining import train_static_hedging
    from src.pricing.basket import BasketOptionValuationSystem

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    market_cfg   = cfg["market"]
    garch_params = {int(k): v for k, v in cfg["garch_params"].items()}
    dcc_cfg      = cfg["dcc"]
    nu_Q         = dcc_cfg["nu_Q"]
    dcc_alpha    = dcc_cfg.get("dcc_alpha")
    dcc_beta     = dcc_cfg.get("dcc_beta")

    from datetime import datetime
    end_date = (datetime.fromisoformat(market_cfg["end_date"])
                if market_cfg.get("end_date") else None)

    returns_matrix, _ = download_returns(
        tickers=market_cfg["tickers"],
        years_back=market_cfg["years_back"],
        end_date=end_date,
    )
    R_Q, nu_Q = fit_t_copula(returns_matrix)
    Q_bar = R_Q

    if dcc_alpha is None or dcc_beta is None:
        print("Re-fitting DCC parameters from historical data...")
        garch_residuals = compute_garch_residuals(returns_matrix, garch_params)
        Q_bar, dcc_alpha, dcc_beta = fit_dcc_parameters(garch_residuals, R_Q, nu_Q)

    print(f"\nCalibration: nu_Q={nu_Q:.4f}, dcc_alpha={dcc_alpha:.6f}, dcc_beta={dcc_beta:.6f}")

    # ------------------------------------------------------------------
    # Pricing system
    # ------------------------------------------------------------------
    system = BasketOptionValuationSystem.load(cfg["pricing"]["system_path"])

    # ------------------------------------------------------------------
    # Simulator + environment
    # ------------------------------------------------------------------
    env_cfg = cfg["env"]

    simulator = DCCGARCHSimulator(
        params=garch_params,
        Q_bar=Q_bar,
        dcc_alpha=dcc_alpha,
        dcc_beta=dcc_beta,
        nu_Q=nu_Q,
        r_daily=market_cfg["r_daily"],
        device=device,
    )

    env = HedgingEnv(
        simulator=simulator,
        system=system,
        K=env_cfg["K"],
        T_days=env_cfg["T_days"],
        S0=env_cfg["S0"],
        r=env_cfg["r"],
        transaction_cost=env_cfg["transaction_cost"],
        basket_weights=env_cfg["basket_weights"],
    )
    print(f"✓ Environment ready | state_dim={env.state_dim} action_dim={env.action_dim} device={env.device}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    tr      = cfg["training"]
    log_dir = Path(tr["log_dir"]) / f"alpha_{tr['alpha']}"
    log_dir.mkdir(parents=True, exist_ok=True)

    actor, results = train_static_hedging(
        env=env,
        alpha             = tr["alpha"],
        n_iter            = tr["n_iter"],
        batch_size        = tr["batch_size"],
        lr                = tr["lr"],
        hidden_dim        = cfg["actor"]["hidden_dim"],
        action_high       = cfg["actor"]["action_high"],
        log_interval      = tr["log_interval"],
        log_dir           = str(log_dir),
        resume_checkpoint = tr.get("resume_checkpoint"),
    )

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    model_path = log_dir / f"static_final_alpha_{tr['alpha']}.pt"
    torch.save({
        "actor":      actor.state_dict(),
        "state_dim":  env.state_dim,
        "action_dim": env.action_dim,
        "alpha":      tr["alpha"],
        "config": {
            "K": env.K, "T_days": env.T_days,
            "n_assets": env.n_assets,
            "transaction_cost": env.transaction_cost,
        },
    }, model_path)
    print(f"✓ Final model saved to {model_path}")

    summary = {"alpha": tr["alpha"], "model_path": str(model_path),
               "log_dir": str(log_dir)}
    with open(Path(tr["log_dir"]) / "static_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
