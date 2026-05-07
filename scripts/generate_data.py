"""
scripts/generate_data.py

One-time data generation script. Run this before any plotting.
Usage
-----
    python scripts/generate_data.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

# ── User config ────────────────────────────────────────────────────────────────
from src.envs.HedgingEnv        import HedgingEnv
from src.envs.DCCGARCH           import DCCGARCHSimulator
from src.agents.DynamicRiskModel import Actor

BATCH_SIZE   = 100_000
SEED         = 42
DATA_DIR     = "content/data"
SCORING_KEYS = ["arcsin", "arcsinh", "arctan", "log", "power03", "rational"]
ALPHA_LABELS = ["alpha925", "alpha95", "alpha975", "alpha99"]
FIXED_ALPHA  = "alpha95"   # shared paths for portfolio delta only generated here

# Build env, all_models, all_static_models here
# env               = HedgingEnv(...)
# all_models        = ...   # all_models["alpha95"]["arcsin"]["actor"]
# all_static_models = ...   # all_static_models["alpha95"]["actor"]
raise NotImplementedError(
    "Fill in env, all_models, and all_static_models construction before running."
)
# ── end user config ────────────────────────────────────────────────────────────


def save_simulation(out, save_dir):
    """Save every tensor returned by rollout_from_paths to save_dir."""
    os.makedirs(save_dir, exist_ok=True)

    (states, actions, log_probs, costs, next_states, dones,
     portval, deriv_prices, PnL, terminal_pnl,
     S_paths, h_paths, R_paths) = out

    np.save(f"{save_dir}/states.npy",           states.cpu().numpy())
    np.save(f"{save_dir}/actions.npy",           actions.cpu().numpy())
    np.save(f"{save_dir}/log_probs.npy",         log_probs.cpu().numpy())
    np.save(f"{save_dir}/costs.npy",             costs.cpu().numpy())
    np.save(f"{save_dir}/portfolio_values.npy",  portval.cpu().numpy())
    np.save(f"{save_dir}/deriv_prices.npy",      deriv_prices.cpu().numpy())
    np.save(f"{save_dir}/PnL.npy",               PnL.cpu().numpy())
    np.save(f"{save_dir}/terminal_pnl.npy",      terminal_pnl.cpu().numpy())
    np.save(f"{save_dir}/S_paths.npy",           S_paths.cpu().numpy())
    np.save(f"{save_dir}/h_paths.npy",           h_paths.cpu().numpy())
    np.save(f"{save_dir}/R_paths.npy",           R_paths.cpu().numpy())

    pnl_np = terminal_pnl.cpu().numpy()
    print(f"    saved to {save_dir}")
    print(f"    terminal P&L: mean={pnl_np.mean():.4f}  CVaR95={-np.percentile(pnl_np, 5):.4f}")


if __name__ == "__main__":

    # ── generate shared fixed paths ONCE (alpha95 only, for portfolio delta) ──
    print(f"\n{'='*60}")
    print(f"  Generating FIXED shared paths for portfolio delta")
    print(f"  alpha={FIXED_ALPHA}  seed={SEED}  batch={BATCH_SIZE}")
    print(f"{'='*60}")

    fixed_S, fixed_h, fixed_R = env._generate_all_paths_gpu(
        BATCH_SIZE, seed=SEED, randomize_init=False
    )
    with torch.no_grad():
        fixed_d = env._price_all_episodes_batched(
            fixed_S, fixed_h, fixed_R, chunk_size=16_384
        )

    shared_dir = os.path.join(DATA_DIR, FIXED_ALPHA, "_shared_paths")
    os.makedirs(shared_dir, exist_ok=True)
    np.save(f"{shared_dir}/S_paths.npy",      fixed_S.cpu().numpy())
    np.save(f"{shared_dir}/h_paths.npy",      fixed_h.cpu().numpy())
    np.save(f"{shared_dir}/R_paths.npy",      fixed_R.cpu().numpy())
    np.save(f"{shared_dir}/deriv_prices.npy", fixed_d.cpu().numpy())
    print(f"    fixed shared paths saved to {shared_dir}")

    # rollout every alpha95 dynamic actor on fixed paths (for portfolio delta plot)
    for sk in SCORING_KEYS:
        print(f"  fixed rollout {sk} ...")
        actor    = all_models[FIXED_ALPHA][sk]["actor"]
        save_dir = os.path.join(DATA_DIR, FIXED_ALPHA, "_shared_paths", sk)
        out = env.rollout_from_paths(
            actor, fixed_S, fixed_h, fixed_R, fixed_d,
            deterministic=True,
        )
        save_simulation(out, save_dir)
        del out
        torch.cuda.empty_cache()

    del fixed_S, fixed_h, fixed_R, fixed_d
    torch.cuda.empty_cache()

    # ── stochastic paths: all 28 models, all 4 alphas, same seed ──────────────
    # Generate fresh stochastic paths per alpha (randomize_init=True),
    # but use the same SEED every time so results are reproducible and
    # comparable across alphas.

    for alpha_label in ALPHA_LABELS:
        print(f"\n{'='*60}")
        print(f"  Generating STOCHASTIC paths")
        print(f"  alpha={alpha_label}  seed={SEED}  batch={BATCH_SIZE}")
        print(f"{'='*60}")

        stoch_S, stoch_h, stoch_R = env._generate_all_paths_gpu(
            BATCH_SIZE, seed=SEED, randomize_init=True
        )
        with torch.no_grad():
            stoch_d = env._price_all_episodes_batched(
                stoch_S, stoch_h, stoch_R, chunk_size=16_384
            )

        # ── dynamic actors ─────────────────────────────────────────────────
        for sk in SCORING_KEYS:
            print(f"\n  {alpha_label} / {sk}")
            actor    = all_models[alpha_label][sk]["actor"]
            save_dir = os.path.join(DATA_DIR, alpha_label, sk)
            out = env.rollout_from_paths(
                actor, stoch_S, stoch_h, stoch_R, stoch_d,
                deterministic=True,
            )
            save_simulation(out, save_dir)
            del out
            torch.cuda.empty_cache()

        # ── static actor ───────────────────────────────────────────────────
        print(f"\n  {alpha_label} / static")
        static_actor = all_static_models[alpha_label]["actor"]
        save_dir     = os.path.join(DATA_DIR, alpha_label, "static")
        out = env.rollout_from_paths(
            static_actor, stoch_S, stoch_h, stoch_R, stoch_d,
            deterministic=True,
        )
        save_simulation(out, save_dir)
        del out
        torch.cuda.empty_cache()

        del stoch_S, stoch_h, stoch_R, stoch_d
        torch.cuda.empty_cache()

    print(f"\nDone. All data saved to {DATA_DIR}/")
