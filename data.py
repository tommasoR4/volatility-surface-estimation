"""
Data download, filtering, and BSM baseline pricing.

Fetches option chains from Yahoo Finance, applies liquidity and moneyness
filters, and computes the baseline mispricing using historical volatility.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from bsm import bsm_price, implied_vol, vega


def option_maturity(opt_maturity='2026-09-18'):
    """Return time to maturity in years, using 252 trading days."""
    expiration_date = pd.to_datetime(opt_maturity)
    today = pd.Timestamp.now()
    days_to_expiry = (expiration_date - today).days
    return days_to_expiry / 252


def hist_vol(ticker='SPY', period='3mo'):
    """Return annualised historical volatility from log-returns over `period`."""
    opt_ticker = yf.Ticker(ticker)
    hist = opt_ticker.history(period=period)
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
    return log_returns.std() * np.sqrt(252)


def get_bsm_opt_df(ticker='SPY', opt_maturity='2026-09-18', opt_type='call',
                    moneyness_range=(0.6, 1.4), min_volume=10,
                    min_open_interest=50, min_price=0.50):
    """Download and filter an option chain, ready for BSM analysis.

    Parameters
    ----------
    opt_maturity : str or list of str
        A single expiration date, or several. Multiple maturities are
        concatenated into one DataFrame.
    moneyness_range : tuple or None
        (low, high) bounds on K/S. Pass None to skip this filter — useful
        when studying vega across the full range of strikes.

    Returns
    -------
    DataFrame with columns: contractSymbol, currentUnderlyingPrice,
    expirationDate, strike, timetoMaturity, moneyness, lastPrice, midPrice.
    """
    opt_ticker = yf.Ticker(ticker)
    current_price = opt_ticker.history(period='1d')['Close'].iloc[-1]

    maturities = [opt_maturity] if isinstance(opt_maturity, str) else opt_maturity
    all_dfs = []

    for mat in maturities:
        opt_chain = opt_ticker.option_chain(mat)

        if opt_type == 'call':
            opt_df = opt_chain.calls
        elif opt_type == 'put':
            opt_df = opt_chain.puts
        else:
            raise ValueError("opt_type must be 'call' or 'put'")

        opt_df = opt_df.copy()
        opt_df.insert(1, 'expirationDate', mat)
        opt_df.insert(2, 'currentUnderlyingPrice', current_price)
        opt_df.insert(4, 'timetoMaturity', option_maturity(mat))
        opt_df.insert(5, 'moneyness', opt_df.strike / opt_df.currentUnderlyingPrice)
        opt_df['midPrice'] = (opt_df.bid + opt_df.ask) / 2

        keep = (
            (opt_df.bid > 0)
            & (opt_df.ask > 0)
            & (opt_df.volume > min_volume)
            & (opt_df.openInterest > min_open_interest)
            & (opt_df.midPrice > min_price)
        )
        if moneyness_range is not None:
            low, high = moneyness_range
            keep &= (opt_df.moneyness > low) & (opt_df.moneyness < high)

        all_dfs.append(opt_df[keep])

    result = pd.concat(all_dfs, ignore_index=True)

    cols = ['contractSymbol', 'currentUnderlyingPrice', 'expirationDate', 'strike',
            'timetoMaturity', 'moneyness', 'lastPrice', 'midPrice']
    return result.loc[:, cols]


def add_bsm_baseline(opt_df, r, q, sigma_hist, opt_type='call'):
    """Add the baseline BSM price and mispricing columns, using historical sigma.

    This is the naive benchmark the models are compared against: a single
    constant volatility applied to every strike and maturity.

    Adds: bsmPrice, bsmDifference, pctError.
    """
    df = opt_df.copy()

    df['bsmPrice'] = df.apply(
        lambda row: bsm_price(S=row['currentUnderlyingPrice'], K=row['strike'],
                              r=r, q=q, t=row['timetoMaturity'],
                              sigma=sigma_hist, option_type=opt_type),
        axis=1,
    )
    df['bsmDifference'] = df['bsmPrice'] - df['midPrice']
    df['pctError'] = df['bsmDifference'] / df['midPrice'] * 100

    return df


def add_implied_vol(opt_df, r, q, opt_type='call', dropna=True):
    """Add the market-implied volatility column, solved numerically per option.

    Rows where the solver fails to converge are dropped by default.
    Adds: impliedVol.
    """
    df = opt_df.copy()

    df['impliedVol'] = df.apply(
        lambda row: implied_vol(row['midPrice'], row['currentUnderlyingPrice'],
                                row['strike'], r, q, row['timetoMaturity'],
                                option_type=opt_type),
        axis=1,
    )

    if dropna:
        df = df.dropna(subset=['impliedVol']).reset_index(drop=True)

    return df


def add_vega(opt_df, r, q):
    """Add the vega column, computed at each option's implied volatility.

    Requires the impliedVol column (see add_implied_vol).
    """
    df = opt_df.copy()

    df['vega'] = df.apply(
        lambda row: vega(S=row['currentUnderlyingPrice'], K=row['strike'],
                         r=r, q=q, t=row['timetoMaturity'],
                         sigma=row['impliedVol']),
        axis=1,
    )

    return df
