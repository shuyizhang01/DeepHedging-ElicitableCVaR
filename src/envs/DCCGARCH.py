import numpy as np
import torch


class DCCGARCHSimulator:
    """
    Simulates multivariate asset paths using a DCC-GARCH model
    with a Student-t copula for tail dependence.

    Parameters
    ----------
    params : dict
        Per-asset GARCH parameters keyed by asset index.
        Each entry: {'omega', 'alpha', 'beta', 'gamma', 'lambda', 'h_unconditional'}
    Q_bar : np.ndarray, shape (n_assets, n_assets)
        Unconditional correlation matrix from DCC fitting.
    dcc_alpha : float
        DCC parameter α.
    dcc_beta : float
        DCC parameter β.
    nu_Q : float
        Student-t degrees of freedom.
    r_daily : float
        Daily risk-free rate (default: 0.04/252).
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(self, params, Q_bar, dcc_alpha, dcc_beta, nu_Q,
                 r_daily=0.04 / 252, device='cuda'):
        self.params = params
        self.n_assets = len(params)
        self.device = device

        # Store all scalars and matrices as tensors on device
        self.Q_bar      = torch.tensor(Q_bar,      dtype=torch.float32, device=device)
        self.dcc_alpha  = torch.tensor(dcc_alpha,  dtype=torch.float32, device=device)
        self.dcc_beta   = torch.tensor(dcc_beta,   dtype=torch.float32, device=device)
        self.nu_Q       = torch.tensor(nu_Q,       dtype=torch.float32, device=device)
        self.r_daily    = torch.tensor(r_daily,    dtype=torch.float32, device=device)

        # Vectorised GARCH parameters
        self.omega       = torch.tensor([params[i]['omega']           for i in range(self.n_assets)], dtype=torch.float32, device=device)
        self.alpha_garch = torch.tensor([params[i]['alpha']           for i in range(self.n_assets)], dtype=torch.float32, device=device)
        self.beta_garch  = torch.tensor([params[i]['beta']            for i in range(self.n_assets)], dtype=torch.float32, device=device)
        self.gamma       = torch.tensor([params[i]['gamma']           for i in range(self.n_assets)], dtype=torch.float32, device=device)
        self.lambda_     = torch.tensor([params[i]['lambda']          for i in range(self.n_assets)], dtype=torch.float32, device=device)
        self.h_unconditional = torch.tensor([params[i]['h_unconditional'] for i in range(self.n_assets)], dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_student_t(self, shape, device, generator=None):
        """
        Sample from a multivariate Student-t via the normal/chi2 ratio.
        shape: (..., n_assets)
        """
        nu     = self.nu_Q.item()
        nu_int = max(1, round(nu))
        normal = torch.randn(*shape, device=device, generator=generator)
        chi2   = torch.randn(*shape[:-1], nu_int, device=device,
                              generator=generator).pow(2).sum(-1)
        return normal / (chi2.unsqueeze(-1) / nu).sqrt()

    # ------------------------------------------------------------------
    # GPU-batched simulation (used by HedgingEnv)
    # ------------------------------------------------------------------

    def simulate_batch(self, batch_size, T_days, S0, generator=None):
        """
        Simulate a batch of price paths entirely on GPU.

        Parameters
        ----------
        batch_size : int
        T_days     : int
        S0         : torch.Tensor, shape (n_assets,)
        generator  : optional torch.Generator

        Returns
        -------
        S_paths : Tensor [batch_size, T_days+1, n_assets]
        h_paths : Tensor [batch_size, T_days+1, n_assets]
        R_paths : Tensor [batch_size, T_days+1, n_assets, n_assets]
        """
        CHOL_CHUNK = 20_000   # max paths per batched Cholesky call
        device = self.device

        S_paths = torch.zeros(batch_size, T_days + 1, self.n_assets, device=device)
        h_paths = torch.zeros(batch_size, T_days + 1, self.n_assets, device=device)
        R_paths = torch.zeros(batch_size, T_days + 1, self.n_assets, self.n_assets, device=device)

        # Randomise initial prices and variances for diversity
        log_shock = torch.randn(batch_size, self.n_assets, device=device,
                                generator=generator) * 0.15
        S_paths[:, 0] = S0 * torch.exp(log_shock)

        h_noise = torch.exp(torch.randn(batch_size, self.n_assets, device=device,
                                         generator=generator) * 0.3)
        h_t = self.h_unconditional.unsqueeze(0) * h_noise
        h_t = torch.clamp(h_t, min=1e-12)
        h_paths[:, 0] = h_t

        Q_t = self.Q_bar.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        d   = torch.sqrt(torch.diagonal(Q_t, dim1=1, dim2=2))
        R_paths[:, 0] = Q_t / (d.unsqueeze(-1) * d.unsqueeze(-2))

        Z_all = self._sample_student_t(
            (batch_size, T_days, self.n_assets), device, generator=generator
        )

        for day in range(T_days):
            # --- correlation matrix ---
            d     = torch.sqrt(torch.diagonal(Q_t, dim1=1, dim2=2))
            R_t   = Q_t / (d.unsqueeze(-1) * d.unsqueeze(-2))

            # Periodic NaN guard
            if day % 10 == 0:
                bad = ~torch.isfinite(R_t).all(dim=(-1, -2))
                if bad.any():
                    R_t[bad]  = self.Q_bar.unsqueeze(0).expand(bad.sum(), -1, -1)
                    Q_t[bad]  = self.Q_bar.unsqueeze(0).expand(bad.sum(), -1, -1)

            R_paths[:, day] = R_t

            # --- chunked Cholesky for numerical stability ---
            L_t = torch.zeros_like(R_t)
            for i in range(0, batch_size, CHOL_CHUNK):
                j = min(i + CHOL_CHUNK, batch_size)
                L_t[i:j] = torch.linalg.cholesky(R_t[i:j])

            Z_corr = torch.bmm(L_t, Z_all[:, day].unsqueeze(-1)).squeeze(-1)

            # --- price update ---
            sqrt_h  = torch.sqrt(h_t)
            r_t     = self.r_daily + self.lambda_ * h_t + sqrt_h * Z_corr
            S_paths[:, day + 1] = S_paths[:, day] * torch.exp(r_t)

            # --- GARCH variance update ---
            h_t = (self.omega
                   + self.beta_garch  * h_t
                   + self.alpha_garch * (Z_corr - self.gamma * sqrt_h) ** 2)
            h_t = torch.clamp(h_t, min=1e-12)
            h_paths[:, day + 1] = h_t

            # --- DCC update ---
            outer = torch.einsum('bi,bj->bij', Z_corr, Z_corr)
            Q_t   = ((1 - self.dcc_alpha - self.dcc_beta) * self.Q_bar
                     + self.dcc_alpha * outer
                     + self.dcc_beta  * Q_t)

            if day % 10 == 0:
                bad = ~torch.isfinite(Q_t).all(dim=(-1, -2))
                if bad.any():
                    Q_t[bad] = self.Q_bar.unsqueeze(0).expand(bad.sum(), -1, -1)

            # Symmetrise and nudge if near-singular
            Q_t = (Q_t + Q_t.transpose(-1, -2)) / 2.0
            diag_min = torch.diagonal(Q_t, dim1=-2, dim2=-1).min(dim=-1).values
            needs_fix = diag_min < 1e-6
            if needs_fix.any():
                sub      = Q_t[needs_fix]
                min_eig  = torch.linalg.eigvalsh(sub).min(dim=-1).values
                bad_sub  = min_eig < 1e-8
                if bad_sub.any():
                    idx   = needs_fix.nonzero(as_tuple=True)[0][bad_sub]
                    nudge = torch.clamp(-min_eig[bad_sub] + 1e-8, min=0.0)
                    eye   = torch.eye(self.n_assets, device=device).unsqueeze(0)
                    Q_t[idx] += nudge[:, None, None] * eye

        # Final R
        d = torch.sqrt(torch.diagonal(Q_t, dim1=1, dim2=2))
        R_paths[:, T_days] = Q_t / (d.unsqueeze(-1) * d.unsqueeze(-2))

        return S_paths, h_paths, R_paths
