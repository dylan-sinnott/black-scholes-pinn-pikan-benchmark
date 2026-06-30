"""
This file includes functions call_price() and put_price() to calculate the price of a European call and put option
using the Black-Scholes formula.
 """

import numpy as np
import scipy.stats

def call_price(S0, K, T, r, sigma):
    """
    Calculate the price of a European call option using the Black-Scholes formula.

    Parameters:
    S0 : float
        Current stock price
    K : float
        Strike price of the option
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying stock (annualized)

    Returns:
    float
        Price of the European call option
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S0 * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)

    return call_price

def put_price(S0, K, T, r, sigma):
    """
    Calculate the price of a European put option using the Black-Scholes formula.

    Parameters:
    S0 : float
        Current stock price
    K : float
        Strike price of the option
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying stock (annualized)

    Returns:
    float
        Price of the European put option
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2) - S0 * scipy.stats.norm.cdf(-d1)

    return put_price