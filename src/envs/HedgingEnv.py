import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class HedgingEnv:
    """
    Hedging environment for a basket option under DCC-GARCH dynamics.

    At each timestep the agent observes a state (moneyness, time-to-expiry,
    volatilities, previous action, portfolio value, correlations) and outputs
    a hedge position (shares held in each asset).

    Parameters
    ----------
    simulator        : DCCGARCHSimulator
    system           : BasketOptionValuationSystem
    K                : float  — strike price
    T_days           : int    — number of trading days
    S0               : array-like, shape (n_assets,) — initial stock prices
    r                : float  — annualised risk-free rate
    transaction_cost : float  — proportional transaction cost rate
    basket_weights   : list[float], length n_assets
    """

    # Normalisation caps (match training)
    MONEYNESS_MAX = 2.25
    VOL_MAX       = 2.70

    def __init__(self, simulator, system, K, T_days, S0,
                 r=0.04, transaction_cost=0.0001,
                 basket_weights=None):
        self.simulator        = simulator
        self.system           = system
        self.K                = K
        self.T_days           = T_days
        self.r                = r
        self.transaction_cost = transaction_cost
        self.device           = simulator.device

        if isinstance(S0, torch.Tensor):
            self.S0 = S0.to(self.device)
        else:
            self.S0 = torch.tensor(S0, dtype=torch.float32, device=self.device)

        self.n_assets = len(self.S0)

        if basket_weights is None:
            basket_weights = [1.0 / self.n_assets] * self.n_assets
        self.basket_weights = torch.tensor(
            basket_weights, dtype=torch.float32, device=self.device
        )

        self.action_high = 1.0

        self.system.call_model = self.system.call_model.to(self.device)
        if hasattr(self.system, 'put_model') and self.system.put_model is not None:
            self.system.put_model = self.system.put_model.to(self.device)

        self.triu_indices = torch.triu_indices(
            self.n_assets, self.n_assets, offset=1, device=self.device
        )
        self.n_corr_features = self.n_assets * (self.n_assets - 1) // 2

        self.vol_scale = torch.tensor(
            np.sqrt(252 * np.array([simulator.params[i]['h_unconditional']
                                    for i in range(self.n_assets)])),
            dtype=torch.float32, device=self.device
        )

        self.r_daily = torch.tensor(r / 252.0, dtype=torch.float32, device=self.device)

        # state: moneyness(4) + tau(1) + vol(4) + prev_action(4) + port_value(1) + corr(6) = 20
        self.state_dim  = self.n_assets + 1 + self.n_assets + self.n_assets + 1 + self.n_corr_features
        self.action_dim = self.n_assets

        self._prepare_torch_scalers()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_torch_scalers(self):
        scaler_X     = self.system.scaler_call['X']
        self.X_mean  = torch.tensor(scaler_X.mean_,  dtype=torch.float32, device=self.device)
        self.X_scale = torch.tensor(scaler_X.scale_, dtype=torch.float32, device=self.device)

    def _price_derivative_batch(self, S_batch, h_batch, R_batch, t_batch):
        """
        Price basket call option for a batch of states.
        Intrinsic value is computed exactly; time value via neural network.
        Always runs under no_grad — pricing is never differentiated.
        """
        def _to(x):
            if not isinstance(x, torch.Tensor):
                return torch.as_tensor(x, dtype=torch.float32, device=self.device)
            return x.to(self.device) if x.device != self.device else x

        S_batch = _to(S_batch)
        h_batch = _to(h_batch)
        R_batch = _to(R_batch)
        t_batch = _to(t_batch)

        T_batch       = (self.T_days - t_batch) / 252.0
        basket_values = torch.sum(S_batch * self.basket_weights, dim=1)
        intrinsic     = torch.clamp(basket_values - self.K, min=0.0)

        corr_features = R_batch[:, self.triu_indices[0], self.triu_indices[1]]
        B             = S_batch.shape[0]
        r_col         = torch.full((B, 1), self.r, device=self.device)
        K_col         = torch.full((B, 1), self.K, device=self.device)
        T_col         = T_batch.unsqueeze(1) if T_batch.dim() == 1 else T_batch

        X        = torch.cat([S_batch, h_batch, r_col, T_col, K_col, corr_features], dim=1)
        X_scaled = (X - self.X_mean) / self.X_scale

        self.system.call_model.eval()
        time_value_log = self.system.call_model(X_scaled).squeeze(-1)
        time_value     = torch.clamp(torch.expm1(time_value_log), min=0.0)
        return intrinsic + time_value

    def _price_all_episodes_batched(self, S_tensor, h_tensor, R_tensor, chunk_size=8192):
        """Price derivative at every (path, timestep) pair. Returns [B, T+1]."""
        B        = S_tensor.shape[0]
        T_plus_1 = S_tensor.shape[1]

        S_flat = S_tensor.reshape(B * T_plus_1, self.n_assets)
        h_flat = h_tensor.reshape(B * T_plus_1, self.n_assets)
        R_flat = R_tensor.reshape(B * T_plus_1, self.n_assets, self.n_assets)
        t_idx  = torch.arange(T_plus_1, device=self.device).repeat(B)

        prices_list = []
        for start in range(0, B * T_plus_1, chunk_size):
            end = min(start + chunk_size, B * T_plus_1)
            prices_list.append(
                self._price_derivative_batch(
                    S_flat[start:end], h_flat[start:end],
                    R_flat[start:end], t_idx[start:end]
                )
            )
        return torch.cat(prices_list).reshape(B, T_plus_1)

    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------

    def simulate_batch(self, actor, batch_size, alpha=0.05,
                       deterministic=False, seed=None, DR=True):
        """
        Roll out `batch_size` episodes with the given actor.

        Parameters
        ----------
        DR : bool
            Dynamic Risk mode (default True).
            - DR=True  : actor is called under torch.no_grad(); all trajectory
                         tensors are stored and returned. Use for the RL critic/
                         actor updates in train_rl_hedging.
            - DR=False : actor is called with grad_checkpoint so gradients flow
                         through actions → terminal_pnl. Only terminal_pnl is
                         returned (other outputs are None). Use for static risk
                         training where the loss is directly CVaR(terminal_pnl).

        Returns (DR=True)
        -----------------
        states, actions, log_probs, costs, next_states, dones,
        portfolio_values, derivative_values, PnL, terminal_pnl,
        S_paths, h_paths, R_paths

        Returns (DR=False)
        ------------------
        None, None, None, None, None, None, None, None, None, terminal_pnl
        """
        ACTOR_CHUNK = 10_000
        SIM_CHUNK   = 10_000

        n_chunks = (batch_size + SIM_CHUNK - 1) // SIM_CHUNK

        S_list, h_list, R_list, d_list = [], [], [], []
        for ci in range(n_chunks):
            cs         = min(SIM_CHUNK, batch_size - ci * SIM_CHUNK)
            chunk_seed = None if seed is None else seed + ci

            S, h, R = self._generate_all_paths_gpu(cs, seed=chunk_seed)
            with torch.no_grad():
                d = self._price_all_episodes_batched(S, h, R, chunk_size=16_384)

            S_list.append(S); h_list.append(h)
            R_list.append(R); d_list.append(d)
            torch.cuda.empty_cache()

        all_S = torch.cat(S_list, dim=0)   # [B, T+1, n_assets]
        all_h = torch.cat(h_list, dim=0)   # [B, T+1, n_assets]
        all_R = torch.cat(R_list, dim=0)   # [B, T+1, n_assets, n_assets]
        all_d = torch.cat(d_list, dim=0)   # [B, T+1]
        del S_list, h_list, R_list, d_list
        torch.cuda.empty_cache()

        B = batch_size
        T = self.T_days

        # Precompute normalised features — keep all_h and all_R alive for return
        all_corr = all_R[:, :T, self.triu_indices[0], self.triu_indices[1]].clone()
        all_vol  = (torch.sqrt(252 * all_h[:, :T]) / self.vol_scale / self.VOL_MAX).clone()
        all_mon  = (all_S[:, :T] / self.K / self.MONEYNESS_MAX).clone()
        torch.cuda.empty_cache()

        # Storage — only allocated when DR=True
        if DR:
            states_s  = torch.zeros(B, T, self.state_dim, device=self.device)
            actions_s = torch.zeros(B, T, self.n_assets,  device=self.device)
            logp_s    = torch.zeros(B, T,                 device=self.device)
            costs_s   = torch.zeros(B, T,                 device=self.device)
            portval_s = torch.zeros(B, T,                 device=self.device)
            pnl_s     = torch.zeros(B, T,                 device=self.device)

        prev_actions = torch.full((B, self.n_assets), 0.10, device=self.device)
        positions    = torch.zeros(B, self.n_assets,        device=self.device)
        B_account    = all_d[:, 0].clone()

        dummy = next(actor.parameters()).sum() * 0.0

        terminal_pnl = None

        for t in range(T):
            tau         = torch.full((B, 1), (T - t) / T, device=self.device)
            moneyness_t = all_mon[:, t]
            vol_t       = all_vol[:, t]
            corr_t      = all_corr[:, t]
            S_t         = all_S[:, t]
            S_next      = all_S[:, t + 1]

            port_norm = 2 * torch.sigmoid(
                B_account.unsqueeze(1) /
                torch.abs(torch.sum(prev_actions.detach() * S_t, dim=1).unsqueeze(1))
            )

            state_t = torch.cat([
                moneyness_t, tau, vol_t,
                prev_actions.detach() / self.action_high,
                port_norm, corr_t
            ], dim=-1)

            if DR:
                states_s[:, t] = state_t

            # ---- actor forward ----
            a_chunks, lp_chunks = [], []
            for b0 in range(0, B, ACTOR_CHUNK):
                b1 = min(b0 + ACTOR_CHUNK, B)
                if DR:
                    with torch.no_grad():
                        a_c, lp_c = actor.sample(state_t[b0:b1], deterministic=deterministic)
                else:
                    def _run(s, d):
                        return actor.sample(s, deterministic=deterministic)
                    a_c, lp_c = grad_checkpoint(
                        _run, state_t[b0:b1], dummy, use_reentrant=False
                    )
                a_chunks.append(a_c); lp_chunks.append(lp_c)

            actions_t  = torch.cat(a_chunks,  dim=0)
            logprobs_t = torch.cat(lp_chunks, dim=0)

            if DR:
                actions_s[:, t] = actions_t
                logp_s[:, t]    = logprobs_t

            # ---- portfolio accounting ----
            trade   = actions_t - positions
            tc_cost = self.transaction_cost * torch.sum(torch.abs(trade) * S_t, dim=1)
            B_pre   = B_account
            B_post  = B_account - torch.sum(trade * S_t, dim=1) - tc_cost

            deriv_t    = all_d[:, t]
            deriv_next = all_d[:, t + 1]

            y_t       = B_pre + torch.sum(positions * S_t, dim=1) - deriv_t
            B_account = B_post * torch.exp(self.r_daily)

            if t == T - 1:
                y_next = (B_account
                          + torch.sum(actions_t * S_next, dim=1)
                          - self.transaction_cost * torch.sum(
                              torch.abs(actions_t) * S_next, dim=1)
                          - deriv_next)
                terminal_pnl = y_next
            else:
                y_next = B_account + torch.sum(actions_t * S_next, dim=1) - deriv_next

            if DR:
                costs_s[:, t]   = y_t - y_next
                portval_s[:, t] = y_next
                pnl_s[:, t]     = torch.sum(actions_t * (S_next - S_t), dim=1)

            prev_actions = actions_t
            positions    = actions_t

        if not DR:
            return (None,) * 9 + (terminal_pnl,)

        # Transpose to [T, B, ...] convention
        states    = states_s.transpose(0, 1)
        actions   = actions_s.transpose(0, 1)
        log_probs = logp_s.transpose(0, 1)
        costs     = costs_s.transpose(0, 1)
        PnL       = pnl_s.transpose(0, 1)

        next_states = torch.zeros(B, T, dtype=torch.bool, device=self.device).transpose(0, 1)
        dones       = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        dones[:, -1] = True
        dones       = dones.transpose(0, 1)

        return (states, actions, log_probs, costs,
                next_states, dones,
                portval_s, all_d, PnL, terminal_pnl,
                all_S, all_h, all_R)

    # ------------------------------------------------------------------
    # Rollout from pre-generated paths
    # ------------------------------------------------------------------
    def rollout_from_paths(self, actor, S_paths, h_paths, R_paths, deriv_prices,
                           deterministic=False, seed=None):
        """
        Identical to simulate_batch (DR=True) but uses pre-generated paths
        instead of calling _generate_all_paths_gpu.
    
        Parameters
        ----------
        S_paths      : [B, T+1, n_assets]             torch.Tensor on self.device
        h_paths      : [B, T+1, n_assets]
        R_paths      : [B, T+1, n_assets, n_assets]   normalised correlation matrix
        deriv_prices : [B, T+1]
        seed         : int or None — unused, kept for API compatibility
    
        Returns
        -------
        states, actions, log_probs, costs, next_states, dones,
        portfolio_values, derivative_values, PnL, terminal_pnl,
        S_paths, h_paths, R_paths
        """
        ACTOR_CHUNK = 10_000
    
        B = S_paths.shape[0]
        T = self.T_days
    
        S_paths = S_paths.clone()
        h_paths = h_paths.clone()
    
        # re-price derivatives with the passed-in paths directly
        with torch.no_grad():
            deriv_prices = self._price_all_episodes_batched(
                S_paths, h_paths, R_paths, chunk_size=16_384
            )
    
        # ── precompute normalised features ────────────────────────────────────
        all_corr = R_paths[:, :T, self.triu_indices[0], self.triu_indices[1]].clone()
        all_vol  = (torch.sqrt(252 * h_paths[:, :T]) / self.vol_scale / self.VOL_MAX).clone()
        all_mon  = (S_paths[:, :T] / self.K / self.MONEYNESS_MAX).clone()
        torch.cuda.empty_cache()
    
        # ── storage ───────────────────────────────────────────────────────────
        states_s  = torch.zeros(B, T, self.state_dim, device=self.device)
        actions_s = torch.zeros(B, T, self.n_assets,  device=self.device)
        logp_s    = torch.zeros(B, T,                 device=self.device)
        costs_s   = torch.zeros(B, T,                 device=self.device)
        portval_s = torch.zeros(B, T,                 device=self.device)
        pnl_s     = torch.zeros(B, T,                 device=self.device)
    
        prev_actions = torch.full((B, self.n_assets), 0.10, device=self.device)
        positions    = torch.zeros(B, self.n_assets,        device=self.device)
        B_account    = deriv_prices[:, 0].clone()
    
        terminal_pnl = None
    
        # ── timestep loop ─────────────────────────────────────────────────────
        for t in range(T):
            tau         = torch.full((B, 1), (T - t) / T, device=self.device)
            moneyness_t = all_mon[:, t]
            vol_t       = all_vol[:, t]
            corr_t      = all_corr[:, t]
            S_t         = S_paths[:, t]
            S_next      = S_paths[:, t + 1]
    
            port_norm = 2 * torch.sigmoid(
                B_account.unsqueeze(1) /
                torch.abs(torch.sum(prev_actions.detach() * S_t, dim=1).unsqueeze(1))
            )
    
            state_t = torch.cat([
                moneyness_t, tau, vol_t,
                prev_actions.detach() / self.action_high,
                port_norm, corr_t
            ], dim=-1)
    
            states_s[:, t] = state_t
    
            # ---- actor forward ----
            a_chunks, lp_chunks = [], []
            for b0 in range(0, B, ACTOR_CHUNK):
                b1 = min(b0 + ACTOR_CHUNK, B)
                with torch.no_grad():
                    a_c, lp_c = actor.sample(state_t[b0:b1], deterministic=deterministic)
                a_chunks.append(a_c)
                lp_chunks.append(lp_c)
    
            actions_t  = torch.cat(a_chunks,  dim=0)
            logprobs_t = torch.cat(lp_chunks, dim=0)
    
            actions_s[:, t] = actions_t
            logp_s[:, t]    = logprobs_t
    
            # ---- portfolio accounting ----
            trade   = actions_t - positions
            tc_cost = self.transaction_cost * torch.sum(torch.abs(trade) * S_t, dim=1)
            B_pre   = B_account
            B_post  = B_account - torch.sum(trade * S_t, dim=1) - tc_cost
    
            deriv_t    = deriv_prices[:, t]
            deriv_next = deriv_prices[:, t + 1]
    
            y_t       = B_pre + torch.sum(positions * S_t, dim=1) - deriv_t
            B_account = B_post * torch.exp(self.r_daily)
    
            if t == T - 1:
                y_next = (B_account
                          + torch.sum(actions_t * S_next, dim=1)
                          - self.transaction_cost * torch.sum(
                              torch.abs(actions_t) * S_next, dim=1)
                          - deriv_next)
                terminal_pnl = y_next
            else:
                y_next = B_account + torch.sum(actions_t * S_next, dim=1) - deriv_next
    
            costs_s[:, t]   = y_t - y_next
            portval_s[:, t] = y_next
            pnl_s[:, t]     = torch.sum(actions_t * (S_next - S_t), dim=1)
    
            prev_actions = actions_t
            positions    = actions_t
    
        # ── transpose ─────────────────────────────────────────────────────────
        states    = states_s.transpose(0, 1)
        actions   = actions_s.transpose(0, 1)
        log_probs = logp_s.transpose(0, 1)
        costs     = costs_s.transpose(0, 1)
        PnL       = pnl_s.transpose(0, 1)
    
        next_states = torch.zeros(B, T, dtype=torch.bool, device=self.device).transpose(0, 1)
        dones       = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        dones[:, -1] = True
        dones       = dones.transpose(0, 1)
    
        return (states, actions, log_probs, costs,
                next_states, dones,
                portval_s, deriv_prices, PnL, terminal_pnl,
                S_paths, h_paths, R_paths)

    # ------------------------------------------------------------------
    # GPU path generation (delegates to simulator)
    # ------------------------------------------------------------------

    def _generate_all_paths_gpu(self, batch_size, seed=None, randomize_init=True):
        """Generate price/variance/correlation paths on GPU via simulator."""
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(seed)
    
        CHUNK  = 8192
        device = self.device
    
        all_S = torch.zeros(batch_size, self.T_days + 1, self.n_assets,
                            device=device, dtype=torch.float32)
        all_h = torch.zeros(batch_size, self.T_days + 1, self.n_assets,
                            device=device, dtype=torch.float32)
        all_R = torch.zeros(batch_size, self.T_days + 1, self.n_assets, self.n_assets,
                            device=device, dtype=torch.float32)
    
        sim = self.simulator
    
        if randomize_init:
            log_shock   = torch.randn(batch_size, self.n_assets, device=device, generator=rng) * 0.15
            all_S[:, 0] = self.S0 * torch.exp(log_shock)
            h_noise     = torch.exp(torch.randn(batch_size, self.n_assets,
                                                device=device, generator=rng) * 0.3)
            h_t_batch   = sim.h_unconditional.unsqueeze(0) * h_noise
        else:
            all_S[:, 0] = self.S0.unsqueeze(0).expand(batch_size, -1)
            h_t_batch   = sim.h_unconditional.unsqueeze(0).expand(batch_size, -1).clone()
    
        h_t_batch  = torch.clamp(h_t_batch, min=1e-12)
        all_h[:, 0] = h_t_batch
    
        Q_t_batch = sim.Q_bar.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        d_batch   = torch.sqrt(torch.diagonal(Q_t_batch, dim1=1, dim2=2))
        all_R[:, 0] = Q_t_batch / (d_batch.unsqueeze(-1) * d_batch.unsqueeze(-2))
    
        Z_all = sim._sample_student_t(
            (batch_size, self.T_days, self.n_assets), device, generator=rng
        )
    
        dcc_alpha = sim.dcc_alpha
        dcc_beta  = sim.dcc_beta
        Q_bar     = sim.Q_bar
    
        for day in range(self.T_days):
            d_batch   = torch.sqrt(torch.diagonal(Q_t_batch, dim1=1, dim2=2))
            R_t_batch = Q_t_batch / (d_batch.unsqueeze(-1) * d_batch.unsqueeze(-2))
    
            if day % 10 == 0:
                bad = ~torch.isfinite(R_t_batch).all(dim=(-1, -2))
                if bad.any():
                    R_t_batch[bad] = Q_bar.unsqueeze(0).expand(bad.sum(), -1, -1)
                    Q_t_batch[bad] = Q_bar.unsqueeze(0).expand(bad.sum(), -1, -1)
    
            all_R[:, day] = R_t_batch
    
            L_t_batch = torch.zeros_like(R_t_batch)
            for i in range(0, batch_size, CHUNK):
                j = min(i + CHUNK, batch_size)
                L_t_batch[i:j] = torch.linalg.cholesky(R_t_batch[i:j])
    
            Z_corr = torch.bmm(L_t_batch, Z_all[:, day].unsqueeze(-1)).squeeze(-1)
    
            sqrt_h    = torch.sqrt(h_t_batch)
            r_t_batch = sim.r_daily + sim.lambda_ * h_t_batch + sqrt_h * Z_corr
            all_S[:, day + 1] = all_S[:, day] * torch.exp(r_t_batch)
    
            h_t_batch = (sim.omega + sim.beta_garch * h_t_batch
                         + sim.alpha_garch * (Z_corr - sim.gamma * sqrt_h) ** 2)
            h_t_batch = torch.clamp(h_t_batch, min=1e-12)
            all_h[:, day + 1] = h_t_batch
    
            outer     = torch.einsum('bi,bj->bij', Z_corr, Z_corr)
            Q_t_batch = ((1 - dcc_alpha - dcc_beta) * Q_bar
                         + dcc_alpha * outer
                         + dcc_beta  * Q_t_batch)
    
            if day % 10 == 0:
                bad = ~torch.isfinite(Q_t_batch).all(dim=(-1, -2))
                if bad.any():
                    Q_t_batch[bad] = Q_bar.unsqueeze(0).expand(bad.sum(), -1, -1)
    
            Q_t_batch = (Q_t_batch + Q_t_batch.transpose(-1, -2)) / 2.0
    
            diag_min  = torch.diagonal(Q_t_batch, dim1=-2, dim2=-1).min(dim=-1).values
            needs_fix = diag_min < 1e-6
            if needs_fix.any():
                sub      = Q_t_batch[needs_fix]
                min_eigs = torch.linalg.eigvalsh(sub).min(dim=-1).values
                bad      = min_eigs < 1e-8
                if bad.any():
                    idx   = needs_fix.nonzero(as_tuple=True)[0][bad]
                    nudge = torch.clamp(-min_eigs[bad] + 1e-8, min=0.0)
                    eye   = torch.eye(self.n_assets, device=device).unsqueeze(0)
                    Q_t_batch[idx] += nudge[:, None, None] * eye
    
        d_batch = torch.sqrt(torch.diagonal(Q_t_batch, dim1=1, dim2=2))
        all_R[:, self.T_days] = Q_t_batch / (d_batch.unsqueeze(-1) * d_batch.unsqueeze(-2))
    
        return all_S, all_h, all_R
