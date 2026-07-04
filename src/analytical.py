# analytical.py

import numpy as np
from scipy.stats import norm


def call_price(S0, tau, K, r, sigma):
    S = np.asarray(S0, dtype=float)
    tau = np.asarray(tau, dtype=float)

    S, tau = np.broadcast_arrays(S, tau)

    price = np.zeros_like(S, dtype=float)

    expiry = tau <= 1e-12
    price[expiry] = np.maximum(S[expiry] - K, 0.0)

    valid = (tau > 1e-12) & (S > 0)

    S_valid = S[valid]
    tau_valid = tau[valid]

    d1 = (
        np.log(S_valid / K)
        + (r + 0.5 * sigma**2) * tau_valid
    ) / (sigma * np.sqrt(tau_valid))

    d2 = d1 - sigma * np.sqrt(tau_valid)

    price[valid] = (
        S_valid * norm.cdf(d1)
        - K * np.exp(-r * tau_valid) * norm.cdf(d2)
    )

    return price


def black_scholes_call_grid(S0, tau, K, r, sigma):
    return call_price(S0, tau, K, r, sigma)