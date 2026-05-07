"""
src/data/calibration.py

Downloads historical price data and fits the three-stage model:
  1. Student-t copula  → R_Q, nu_Q
  2. Heston-Nandi GARCH(1,1) per asset → standardised residuals
  3. DCC parameters (alpha, beta) → Q_bar, dcc_alpha, dcc_beta
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import rankdata, t
from scipy.special import gammaln


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_returns(
    tickers: dict[str, str],
    years_back: int = 2,
    end_date: datetime | None = None,
    scale: float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Download adjusted close prices via yfinance and compute log returns.

    Args:
        tickers:   Mapping of display_name → yfinance ticker, e.g. {"JPM": "JPM"}.
        years_back: How many years of history to pull.
        end_date:  End of the download window (defaults to today).
        scale:     Multiplicative scale applied to log returns (usually 1).

    Returns:
        returns_matrix: np.ndarray of shape (T, N) — log returns, one column per asset.
        prices:         pd.DataFrame of raw closing prices.
    """
    import yfinance as yf

    if end_date is None:
        end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years_back)

    raw = yf.download(
        tickers=list(tickers.values()),
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        threads=False,
    )["Close"]

    raw.columns = list(tickers.keys())
    prices = raw.replace([np.inf, -np.inf], np.nan).dropna()

    ratio = prices.values / np.roll(prices.values, 1, axis=0)
    ratio[~np.isfinite(ratio) | (ratio <= 0)] = np.nan
    log_matrix = np.log(ratio)

    log_returns = (
        pd.DataFrame(log_matrix, index=prices.index, columns=prices.columns).dropna()
        * scale
    )

    returns_list = [log_returns[col].values for col in log_returns.columns]
    min_len = min(len(r) for r in returns_list)
    returns_matrix = np.column_stack([r[-min_len:] for r in returns_list])

    print(f"Downloaded returns: shape={returns_matrix.shape}, "
          f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    return returns_matrix, prices


# ============================================================================
# STEP 1 — STUDENT-t COPULA
# ============================================================================

def fit_t_copula(returns_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Fit a Student-t copula to the return matrix.

    Uses rank-based pseudo-observations, optimises the log-likelihood over the
    degrees-of-freedom parameter ν, then recovers the correlation matrix R_Q.

    Args:
        returns_matrix: (T, N) array of log returns.

    Returns:
        R_Q:  (N, N) correlation matrix under the copula.
        nu_Q: Fitted degrees of freedom.
    """
    T, N = returns_matrix.shape

    U = np.zeros_like(returns_matrix)
    for i in range(N):
        U[:, i] = rankdata(returns_matrix[:, i], method="average") / (T + 1)

    def negll(log_nu: float) -> float:
        nu = np.exp(log_nu)
        W = t.ppf(U, nu)
        R = np.corrcoef(W, rowvar=False)
        invR = np.linalg.inv(R)
        _, logdet = np.linalg.slogdet(R)
        quad = np.einsum("ij,jk,ik->i", W, invR, W)
        ll = np.sum(
            gammaln((nu + N) / 2)
            - gammaln(nu / 2)
            - 0.5 * (N * np.log(nu * np.pi) + logdet)
            - 0.5 * (nu + N) * np.log(1 + quad / nu)
            - t.logpdf(W, nu).sum(axis=1)
        )
        return -ll

    res = minimize(negll, x0=np.log(8.0), bounds=[(np.log(2.1), np.log(200))])
    nu_Q = float(np.exp(res.x.item()))

    W = t.ppf(U, nu_Q)
    R_Q = np.corrcoef(W, rowvar=False)

    print(f"✓ Student-t copula fitted: nu_Q={nu_Q:.4f}")
    return R_Q, nu_Q


# ============================================================================
# STEP 2 — HESTON-NANDI GARCH RESIDUALS
# ============================================================================

def compute_garch_residuals(
    returns_matrix: np.ndarray,
    params: dict[int, dict],
) -> np.ndarray:
    """
    Compute standardised GARCH residuals z_t = r_t / sqrt(h_t) for each asset.

    The variance recursion is the Heston-Nandi GARCH(1,1) form:
        h_t = omega + beta * h_{t-1} + alpha * (z_{t-1} - gamma * sqrt(h_{t-1}))^2

    Args:
        returns_matrix: (T, N) log return array.
        params:         Dict keyed by asset index with keys
                        omega, alpha, beta, gamma, lambda, h_unconditional.

    Returns:
        garch_residuals: (T, N) array of standardised residuals.
    """
    T, N = returns_matrix.shape
    garch_residuals = np.zeros((T, N))

    for i in range(N):
        r = returns_matrix[:, i]
        p = params[i]

        h_t = np.zeros(T)
        h_t[0] = p["h_unconditional"]

        for step in range(1, T):
            sqrt_h = np.sqrt(h_t[step - 1])
            eps = r[step - 1] / sqrt_h if sqrt_h > 0 else 0.0
            h_t[step] = (
                p["omega"]
                + p["beta"] * h_t[step - 1]
                + p["alpha"] * (eps - p["gamma"] * sqrt_h) ** 2
            )
            h_t[step] = max(h_t[step], 1e-12)

        garch_residuals[:, i] = r / np.sqrt(h_t)

    print(f"✓ GARCH residuals computed: shape={garch_residuals.shape}")
    return garch_residuals


# ============================================================================
# STEP 3 — DCC PARAMETERS
# ============================================================================

def fit_dcc_parameters(
    garch_residuals: np.ndarray,
    R_Q: np.ndarray,
    nu_Q: float,
) -> tuple[np.ndarray, float, float]:
    """
    Fit the DCC(1,1) parameters alpha and beta via maximum likelihood.

    The DCC update equation is:
        Q_t = (1 - alpha - beta) * Q_bar + alpha * z_{t-1} z_{t-1}' + beta * Q_{t-1}

    The likelihood uses a Student-t copula with the fitted nu_Q.

    Args:
        garch_residuals: (T, N) standardised residuals from GARCH step.
        R_Q:             (N, N) unconditional correlation (from copula fit).
        nu_Q:            Degrees of freedom from copula fit.

    Returns:
        Q_bar:     (N, N) unconditional correlation matrix (== R_Q for consistency).
        dcc_alpha: Fitted DCC alpha.
        dcc_beta:  Fitted DCC beta.
    """
    T, N = garch_residuals.shape
    Q_bar = R_Q.copy()

    def dcc_likelihood(p: np.ndarray) -> float:
        alpha, beta = p
        if alpha <= 0 or beta <= 0 or alpha + beta >= 1:
            return 1e10

        Q_t = Q_bar.copy()
        ll = 0.0

        for step in range(1, T):
            z = garch_residuals[step - 1]
            Q_t = (1 - alpha - beta) * Q_bar + alpha * np.outer(z, z) + beta * Q_t

            if np.any(np.linalg.eigvalsh(Q_t) <= 0):
                return 1e10

            d = np.sqrt(np.diag(Q_t))
            R_t = Q_t / np.outer(d, d)
            invR = np.linalg.inv(R_t)
            _, logdet = np.linalg.slogdet(R_t)

            z_t = garch_residuals[step]
            quad = z_t @ invR @ z_t
            ll += -0.5 * logdet - 0.5 * (nu_Q + N) * np.log(1 + quad / nu_Q)

        return -ll

    res = minimize(
        dcc_likelihood,
        x0=[0.05, 0.90],
        bounds=[(1e-6, 0.3), (0.5, 0.99)],
        method="L-BFGS-B",
    )

    dcc_alpha, dcc_beta = float(res.x[0]), float(res.x[1])

    print(f"✓ DCC fitted: alpha={dcc_alpha:.6f}, beta={dcc_beta:.6f}, "
          f"alpha+beta={dcc_alpha + dcc_beta:.6f}")

    return Q_bar, dcc_alpha, dcc_beta


# ============================================================================
# CONVENIENCE: RUN FULL PIPELINE
# ============================================================================

def calibrate(
    tickers: dict[str, str],
    garch_params: dict[int, dict],
    years_back: int = 2,
    end_date: datetime | None = None,
) -> dict:
    """
    Run the full three-stage calibration and return all fitted quantities.

    Returns a dict with keys:
        returns_matrix, prices, R_Q, nu_Q, garch_residuals,
        Q_bar, dcc_alpha, dcc_beta
    """
    returns_matrix, prices = download_returns(tickers, years_back, end_date)
    R_Q, nu_Q = fit_t_copula(returns_matrix)
    garch_residuals = compute_garch_residuals(returns_matrix, garch_params)
    Q_bar, dcc_alpha, dcc_beta = fit_dcc_parameters(garch_residuals, R_Q, nu_Q)

    return {
        "returns_matrix": returns_matrix,
        "prices": prices,
        "R_Q": R_Q,
        "nu_Q": nu_Q,
        "garch_residuals": garch_residuals,
        "Q_bar": Q_bar,
        "dcc_alpha": dcc_alpha,
        "dcc_beta": dcc_beta,
    }
