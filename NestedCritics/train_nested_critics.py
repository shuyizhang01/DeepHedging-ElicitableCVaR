"""
train_nested_critics.py
=======================
Training script for nested CVaR critics (CriticCVaR).

Place at: /content/DeepHedging/NestedCritics/train_nested_critics.py

Import and call from your notebook:
    from NestedCritics.train_nested_critics import train_critics_nested
    from src.agents.DynamicRiskModel import CriticCVaR
    critics, targets, schedulers = train_critics_nested(env=env, actor=actor)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.agents.Nestedcriticsmodel import CriticCVaR


# ── simulation helpers ────────────────────────────────────────────────────────

def one_step_nested(env, S_t, h_t, Q_t, positions_t, actions_t, B_account_pre, deriv_t, t, M):
    B   = S_t.shape[0]
    sim = env.simulator

    trade     = actions_t - positions_t
    tc_cost   = env.transaction_cost * torch.sum(torch.abs(trade) * S_t, dim=1)
    B_post    = B_account_pre - torch.sum(trade * S_t, dim=1) - tc_cost
    B_account = B_post * torch.exp(env.r_daily)
    y_t       = B_account_pre + torch.sum(positions_t * S_t, dim=1) - deriv_t

    S_e     = S_t.unsqueeze(1).expand(B, M, -1).reshape(B*M, env.n_assets)
    h_e     = h_t.unsqueeze(1).expand(B, M, -1).reshape(B*M, env.n_assets)
    Q_e     = Q_t.unsqueeze(1).expand(B, M, -1, -1).reshape(B*M, env.n_assets, env.n_assets)
    a_e     = actions_t.unsqueeze(1).expand(B, M, -1).reshape(B*M, env.n_assets)
    B_acc_e = B_account.unsqueeze(1).expand(B, M).reshape(B*M)

    Z      = sim._sample_student_t((B*M, env.n_assets), env.device)
    d      = torch.sqrt(torch.diagonal(Q_e, dim1=-2, dim2=-1))
    R_e    = Q_e / (d.unsqueeze(-1) * d.unsqueeze(-2))
    L      = torch.linalg.cholesky(R_e)
    Z_corr = torch.bmm(L, Z.unsqueeze(-1)).squeeze(-1)

    sqrt_h = torch.sqrt(h_e)
    h_tp1  = torch.clamp(
        sim.omega + sim.beta_garch * h_e
        + sim.alpha_garch * (Z_corr - sim.gamma * sqrt_h)**2,
        min=1e-12,
    )
    S_tp1 = S_e * torch.exp(sim.r_daily + sim.lambda_ * h_e + sqrt_h * Z_corr)
    Q_tp1 = (
        (1 - sim.dcc_alpha - sim.dcc_beta) * sim.Q_bar
        + sim.dcc_alpha * torch.einsum('bi,bj->bij', Z_corr, Z_corr)
        + sim.dcc_beta  * Q_e
    )
    Q_tp1 = (Q_tp1 + Q_tp1.transpose(-1, -2)) / 2.0

    d_tp1     = torch.sqrt(torch.diagonal(Q_tp1, dim1=-2, dim2=-1))
    R_tp1     = Q_tp1 / (d_tp1.unsqueeze(-1) * d_tp1.unsqueeze(-2))
    t_idx     = torch.full((B*M,), t+1, device=env.device, dtype=torch.float32)
    deriv_tp1 = env._price_derivative_batch(S_tp1, h_tp1, R_tp1, t_idx)

    y_next = B_acc_e + torch.sum(a_e * S_tp1, dim=1) - deriv_tp1
    cost   = (y_t.unsqueeze(1).expand(B, M).reshape(B*M) - y_next).reshape(B, M)

    return (
        cost,
        S_tp1.reshape(B, M, env.n_assets),
        h_tp1.reshape(B, M, env.n_assets),
        Q_tp1.reshape(B, M, env.n_assets, env.n_assets),
        B_account,
    )


def build_state_tp1(env, S_tp1, h_tp1, Q_tp1, t, prev_actions, B_account):
    BM        = S_tp1.shape[0]
    tau       = torch.full((BM, 1), (env.T_days - (t+1)) / env.T_days, device=env.device)
    moneyness = S_tp1 / env.K / env.MONEYNESS_MAX
    vol_feat  = torch.sqrt(252 * h_tp1) / env.vol_scale / env.VOL_MAX
    triu      = env.triu_indices
    d         = torch.sqrt(torch.diagonal(Q_tp1, dim1=-2, dim2=-1))
    R         = Q_tp1 / (d.unsqueeze(-1) * d.unsqueeze(-2))
    corr      = R[:, triu[0], triu[1]]
    port      = 2 * torch.sigmoid(
        B_account / torch.abs(
            torch.sum(prev_actions * S_tp1, dim=-1).unsqueeze(-1)
        )
    )
    return torch.cat([moneyness, tau, vol_feat, prev_actions / env.action_high, port, corr], dim=-1)


def precompute_inner_samples(env, actions, portfolio_values, derivative_values,
                              S_paths, h_paths, Q_paths, start_t, end_t, M, B):
    inner_cache = {}
    with torch.no_grad():
        for local_t in range(end_t - start_t):
            t = start_t + local_t
            if t == env.T_days - 1:
                inner_cache[local_t] = None
                continue

            positions_t     = (torch.zeros(B, env.n_assets, device=env.device)
                               if t == 0 else actions[t-1])
            y_t_outer       = (torch.zeros(B, device=env.device)
                               if t == 0 else portfolio_values[:, t-1])
            B_account_pre_t = (y_t_outer
                               - torch.sum(positions_t * S_paths[:, t], dim=1)
                               + derivative_values[:, t])

            cost_inner, S_tp1, h_tp1, Q_tp1, B_account_post = one_step_nested(
                env, S_paths[:, t], h_paths[:, t], Q_paths[:, t],
                positions_t, actions[t], B_account_pre_t, derivative_values[:, t], t, M,
            )

            BM         = B * M
            a_exp      = actions[t].unsqueeze(1).expand(B, M, -1).reshape(BM, env.n_assets)
            B_acc_exp  = B_account_post.unsqueeze(1).expand(B, M).reshape(BM, 1)
            s_tp1_flat = build_state_tp1(
                env,
                S_tp1.reshape(BM, env.n_assets),
                h_tp1.reshape(BM, env.n_assets),
                Q_tp1.reshape(BM, env.n_assets, env.n_assets),
                t, a_exp, B_acc_exp,
            )
            inner_cache[local_t] = (cost_inner, s_tp1_flat)

    return inner_cache


def compute_target_cvars_from_cache(inner_cache, start_t, end_t, costs, targets,
                                     current_group, group_size, n_groups, alpha_f, B, M, env):
    k            = max(1, int(torch.ceil(torch.tensor((1 - alpha_f) * M)).item()))
    target_cvars = torch.zeros(group_size, B, device=env.device)
    t_last       = end_t - 1

    s_list      = [inner_cache[lt][1] for lt in range(group_size - 1)]
    s_tp1_inner = torch.stack([s_list[0]] + s_list)               # (group_size, BM, state_dim)
    cost_inner  = torch.stack(
        [inner_cache[lt][0] for lt in range(group_size - 1)]
    )                                                              # (group_size-1, B, M)

    V_tp1_inner       = targets[current_group].forward(s_tp1_inner).reshape(group_size, B, M)
    target_cvars[:-1] = torch.topk(cost_inner + V_tp1_inner[1:], k, dim=2).values.mean(dim=2)

    if t_last == env.T_days - 1:
        target_cvars[-1] = costs[t_last]
    else:
        next_group       = current_group + 1
        s_tp1_last       = inner_cache[group_size - 1][1]         # (BM, state_dim)
        cost_last        = inner_cache[group_size - 1][0]         # (B, M)
        V_tp1_last       = targets[next_group].forward_single_head(
            s_tp1_last, local_t=0
        ).reshape(B, M)
        target_cvars[-1] = torch.topk(cost_last + V_tp1_last, k, dim=1).values.mean(dim=1)

    return target_cvars


# ── main training function ────────────────────────────────────────────────────

def train_critics_nested(
    env,
    actor,
    n_groups           = 18,
    alpha              = 95,
    B                  = 1024,
    M                  = 1024,
    K_star             = 40_000,
    K_initial          = 40_000,
    Nepochs            = 12,
    critic_reset_lr    = 1e-5,
    target_update_freq = 250,
    B1                 = 256,
    hidden_dim         = 64,
    head_dim           = 64,
    save_dir           = 'NestedCritics',
):
    """
    Train nested CVaR critics for a fixed actor.

    Checkpoints saved every 2 epochs to `save_dir/critic_checkpoint_epoch{N}.pt`.

    Parameters
    ----------
    env                : HedgingEnv
    actor              : Actor (eval mode, weights frozen)
    n_groups           : number of critic groups (T_days must be divisible by this)
    alpha              : CVaR confidence level as integer, e.g. 95
    B                  : outer batch size (number of simulated paths per epoch)
    M                  : inner Monte Carlo samples per outer path
    K_star             : gradient steps per group for all groups except the last
    K_initial          : gradient steps for the terminal group (trained first)
    Nepochs            : number of outer epochs
    critic_reset_lr    : Adam learning rate
    target_update_freq : steps between target network syncs and target recomputation
    B1                 : mini-batch size for critic gradient updates
    hidden_dim         : passed through to CriticCVaR (unused internally)
    head_dim           : hidden width of each per-timestep MLP head
    save_dir           : directory to write checkpoints into

    Returns
    -------
    critics, targets, schedulers
    """
    os.makedirs(save_dir, exist_ok=True)

    alpha_f    = alpha / 100.0
    group_size = env.T_days // n_groups
    assert env.T_days % n_groups == 0, \
        f"T_days ({env.T_days}) must be divisible by n_groups ({n_groups})"

    critics = nn.ModuleList([
        CriticCVaR(env.state_dim, group_size, hidden_dim, head_dim, device=env.device)
        for _ in range(n_groups)
    ])
    targets = nn.ModuleList([
        CriticCVaR(env.state_dim, group_size, hidden_dim, head_dim, device=env.device)
        for _ in range(n_groups)
    ])
    for g in range(n_groups):
        critics[g].copy_to(targets[g])
        for p in targets[g].parameters():
            p.requires_grad_(False)

    optimizers = [
        optim.Adam(critics[g].parameters(), lr=critic_reset_lr)
        for g in range(n_groups)
    ]
    schedulers = [
        optim.lr_scheduler.CosineAnnealingLR(
            optimizers[g], T_max=Nepochs, eta_min=critic_reset_lr * 0.1
        )
        for g in range(n_groups)
    ]

    for epoch in range(Nepochs):
        print(f"\n{'='*50}\nEpoch {epoch}/{Nepochs}\n{'='*50}")

        with torch.no_grad():
            (states, actions, log_probs, costs, next_states, dones,
             portfolio_values, derivative_values, PnL,
             S_paths, h_paths, Q_paths) = env.simulate_batch(actor, B)

        states            = states.detach()
        actions           = actions.detach()
        costs             = costs.detach()
        derivative_values = derivative_values.detach()
        portfolio_values  = portfolio_values.detach()
        S_paths           = S_paths.detach()
        h_paths           = h_paths.detach()
        Q_paths           = Q_paths.detach()

        for group_iter in range(n_groups):
            current_group = n_groups - 1 - group_iter   # train terminal group first
            start_t       = current_group * group_size
            end_t         = (current_group + 1) * group_size
            n_iters       = K_initial if group_iter == 0 else K_star

            print(f"\n  Group {current_group} | t=[{start_t}, {end_t}) | iters={n_iters}")

            inner_cache = precompute_inner_samples(
                env, actions, portfolio_values, derivative_values,
                S_paths, h_paths, Q_paths, start_t, end_t, M, B,
            )

            with torch.no_grad():
                target_cvars = compute_target_cvars_from_cache(
                    inner_cache, start_t, end_t, costs, targets,
                    current_group, group_size, n_groups, alpha_f, B, M, env,
                )

            total_updates = 0
            while total_updates < n_iters:
                iters_this_round = min(target_update_freq, n_iters - total_updates)

                for it in range(iters_this_round):
                    idx          = torch.randint(0, B, (B1,), device=env.device)
                    states_batch = states[start_t:end_t, idx]           # (group_size, B1, state_dim)
                    V_pred       = critics[current_group](states_batch)  # (group_size, B1)
                    loss         = F.mse_loss(V_pred, target_cvars[:, idx].detach())

                    optimizers[current_group].zero_grad()
                    loss.backward()
                    optimizers[current_group].step()

                    if (total_updates + it) % 1000 == 0:
                        print(f"    group={current_group} it={total_updates + it} "
                              f"loss={loss.item():.4f} "
                              f"V_mean={V_pred[0].mean().item():.4f}")

                total_updates += iters_this_round
                critics[current_group].copy_to(targets[current_group])

                if total_updates < n_iters:
                    with torch.no_grad():
                        target_cvars = compute_target_cvars_from_cache(
                            inner_cache, start_t, end_t, costs, targets,
                            current_group, group_size, n_groups, alpha_f, B, M, env,
                        )

            critics[current_group].copy_to(targets[current_group])
            del inner_cache

        for g in range(n_groups):
            schedulers[g].step()

        if (epoch + 1) % 2 == 0:
            ckpt_path = os.path.join(save_dir, f'critic_checkpoint_epoch{epoch+1}.pt')
            torch.save({
                'epoch':      epoch,
                'critics':    [c.state_dict() for c in critics],
                'targets':    [t.state_dict() for t in targets],
                'optimizers': [o.state_dict() for o in optimizers],
                'schedulers': [s.state_dict() for s in schedulers],
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved: {ckpt_path}")

    return critics, targets, schedulers
