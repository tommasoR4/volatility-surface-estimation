"""
Black-Scholes-Merton pricing with continuous dividend yield.

Pure pricing functions: no data fetching, no plotting. Every function
takes all its inputs explicitly, so the module has no hidden state.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bsm_d1_d2(S, K, r, q, t, sigma):
    """Return (d1, d2) for the dividend-adjusted BSM formula.

    Parameters
    ----------
    S : float   Spot price of the underlying.
    K : float   Strike price.
    r : float   Risk-free rate (annualised, continuously compounded).
    q : float   Dividend yield (annualised, continuously compounded).
    t : float   Time to maturity, in years.
    sigma : float   Volatility (annualised).
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


def bsm_price(S, K, r, q, t, sigma, option_type='call'):
    """Return the BSM price of a European call or put, adjusted for dividends."""
    d1, d2 = bsm_d1_d2(S=S, K=K, r=r, q=q, t=t, sigma=sigma)

    if option_type == 'call':
        price = S * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-q * t) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return round(price, 3)


def implied_vol(market_price, S, K, r, q, t, option_type='call'):
    """Return the implied volatility that reproduces `market_price` under BSM.

    No closed-form inverse exists, since N(d1) and N(d2) are not algebraically
    invertible with respect to sigma. The option price is monotonically
    increasing in sigma (vega is always positive), so a unique solution exists
    and is found with Brent's method.

    Returns NaN if the solver fails to converge, which happens when the observed
    price falls outside the range BSM can produce for any sigma in [1e-6, 5.0].
    """
    def objective(sigma):
        return bsm_price(S=S, K=K, r=r, q=q, t=t,
                         sigma=sigma, option_type=option_type) - market_price

    try:
        return brentq(objective, 1e-6, 5.0)
    except ValueError:
        return np.nan


def vega(S, K, r, q, t, sigma):
    """Return the vega of an option (sensitivity of price to volatility)."""
    d1, _ = bsm_d1_d2(S=S, K=K, r=r, q=q, t=t, sigma=sigma)
    return S * np.exp(-q * t) * np.sqrt(t) * norm.pdf(d1)
