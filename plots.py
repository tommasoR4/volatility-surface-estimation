"""
Plotting functions for the volatility surface project.

All figures share a consistent style: serif font, no top/right spines,
light grid, 300 dpi output. Set the style once by calling set_style()
at the top of the notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.spatial import cKDTree

# Colour palette used across the project
COLOR_DEFAULT = '#4a4a8a'
COLOR_RIDGE = '#2e7d5b'
COLOR_RF = '#c9463d'


def set_style():
    """Apply the report-wide matplotlib style. Call once at notebook start."""
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype',
                                   'URW Palladio L', 'DejaVu Serif']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['font.size'] = 11
    plt.rcParams['savefig.dpi'] = 300


def _finish_axis(ax, xlabel, ylabel, title, hline=None, vline=1.0,
                 integer_yticks=False):
    """Apply shared cosmetics to a single axis."""
    if hline is not None:
        ax.axhline(hline, color='grey', linestyle='-', linewidth=0.8)
    if vline is not None:
        ax.axvline(vline, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if integer_yticks:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _shared_ylim(axes, series_list, margin_frac=0.08):
    """Force a common y-range across axes, with a margin so points aren't clipped."""
    y_min = min(s.min() for s in series_list)
    y_max = max(s.max() for s in series_list)
    margin = (y_max - y_min) * margin_frac
    for ax in axes:
        ax.set_ylim(y_min - margin, y_max + margin)


def plot_two_panel_scatter(calls_df, puts_df, y_col, ylabel,
                            color=COLOR_DEFAULT, hline=None,
                            shared_scale=True, integer_yticks=False,
                            filename=None):
    """
    Two-panel scatter (call | put) of `y_col` against moneyness.

    Used for the IV skew, vega, and BSM mispricing figures — the only
    difference between them is the column plotted and its axis label.

    Parameters
    ----------
    calls_df, puts_df : DataFrame
        Must contain 'moneyness' and `y_col`.
    y_col : str
        Column to plot on the y-axis.
    ylabel : str
        Y-axis label, e.g. 'Implied Volatility' or 'Mispricing ($)'.
    hline : float or None
        Draw a horizontal reference line at this value (use 0 for errors).
    shared_scale : bool
        Force the same y-range on both panels.
    filename : str or None
        If given, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].scatter(calls_df['moneyness'], calls_df[y_col],
                    alpha=0.7, s=25, color=color, edgecolors='none')
    _finish_axis(axes[0], 'Moneyness (K/S)', ylabel, '(a) Call options',
                 hline=hline, integer_yticks=integer_yticks)

    axes[1].scatter(puts_df['moneyness'], puts_df[y_col],
                    alpha=0.7, s=25, color=color, edgecolors='none')
    _finish_axis(axes[1], 'Moneyness (K/S)', ylabel, '(b) Put options',
                 hline=hline, integer_yticks=integer_yticks)

    if shared_scale:
        _shared_ylim(axes, [calls_df[y_col], puts_df[y_col]])

    plt.tight_layout()
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def _make_grid(df, n_points=40, mask_threshold=0.05):
    """Build the (moneyness, maturity) grid used to render a fitted surface.

    Returns the meshgrid, the flattened points for prediction, and a mask
    marking grid cells far from any observation (set to NaN when plotting
    to avoid showing extrapolated values).
    """
    m_range = np.linspace(df['moneyness'].min(), df['moneyness'].max(), n_points)
    t_range = np.linspace(df['timetoMaturity'].min(), df['timetoMaturity'].max(), n_points)
    grid_m, grid_t = np.meshgrid(m_range, t_range)
    # DataFrame rather than a bare array, so scikit-learn sees the same
    # feature names it was fitted with and does not emit a warning
    grid_points = pd.DataFrame(
        np.column_stack([grid_m.ravel(), grid_t.ravel()]),
        columns=['moneyness', 'timetoMaturity'],
    )

    tree = cKDTree(df[['moneyness', 'timetoMaturity']].values)
    distances, _ = tree.query(grid_points, k=1)
    mask = (distances < mask_threshold).reshape(grid_m.shape)

    return grid_m, grid_t, grid_points, mask


def plot_surface_pair(df, models, mask_surface=False, filename=None):
    """
    Two 3D panels side by side: polynomial regression and Random Forest,
    each with the observed market IV scattered on top.

    Both panels share the same z-range so the comparison is honest.

    Parameters
    ----------
    df : DataFrame
        Must contain 'moneyness', 'timetoMaturity', 'impliedVol'.
    models : dict
        Output of models.fit_models(): keys 'poly', 'ridge', 'rf'.
    mask_surface : bool
        If True, hide surface regions far from observed data.
    """
    grid_m, grid_t, grid_points, mask = _make_grid(df)

    grid_poly = models['poly'].transform(grid_points)
    z_ridge = models['ridge'].predict(grid_poly).reshape(grid_m.shape)
    z_rf = models['rf'].predict(grid_points).reshape(grid_m.shape)

    if mask_surface:
        z_ridge = np.where(mask, z_ridge, np.nan)
        z_rf = np.where(mask, z_rf, np.nan)

    z_min = min(df['impliedVol'].min(), np.nanmin(z_ridge), np.nanmin(z_rf))
    z_max = max(df['impliedVol'].max(), np.nanmax(z_ridge), np.nanmax(z_rf))

    fig = plt.figure(figsize=(13, 5.5))

    for i, (z, cmap, title) in enumerate([
        (z_ridge, 'viridis', '(a) Polynomial Regression'),
        (z_rf, 'plasma', '(b) Random Forest'),
    ]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.scatter(df['moneyness'], df['timetoMaturity'], df['impliedVol'],
                   color='black', s=15, alpha=0.8, label='Market IV')
        ax.plot_surface(grid_m, grid_t, z, cmap=cmap, alpha=0.5, edgecolor='none')
        ax.set_xlabel('Moneyness (K/S)', fontsize=10)
        ax.set_ylabel('Maturity (years)', fontsize=10)
        ax.set_zlabel('Implied Volatility', labelpad=10, fontsize=10)
        ax.set_zlim(z_min, z_max)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9)

    plt.subplots_adjust(left=0.03, right=0.93, wspace=0.15)
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_mispricing_grid(df, filename=None):
    """
    2x2 grid of residual mispricing after using model-estimated IV.

    Top row: absolute mispricing (Ridge | Random Forest).
    Bottom row: percentage mispricing (Ridge | Random Forest).
    Each row shares its y-scale so the two models are directly comparable.

    Parameters
    ----------
    df : DataFrame
        Must contain the columns produced by models.add_model_predictions():
        'bsmDifferenceRidge', 'bsmDifferenceRf', 'pctErrorRidge', 'pctErrorRf'.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    panels = [
        (axes[0, 0], 'bsmDifferenceRidge', COLOR_RIDGE,
         '(a) Polynomial Regression — Absolute', 'Mispricing ($)'),
        (axes[0, 1], 'bsmDifferenceRf', COLOR_RF,
         '(b) Random Forest — Absolute', 'Mispricing ($)'),
        (axes[1, 0], 'pctErrorRidge', COLOR_RIDGE,
         '(c) Polynomial Regression — Percentage', 'Mispricing (%)'),
        (axes[1, 1], 'pctErrorRf', COLOR_RF,
         '(d) Random Forest — Percentage', 'Mispricing (%)'),
    ]

    for ax, col, color, title, ylabel in panels:
        ax.scatter(df['moneyness'], df[col], alpha=0.7, s=15,
                   color=color, edgecolors='none')
        _finish_axis(ax, 'Moneyness (K/S)', ylabel, title, hline=0)

    _shared_ylim([axes[0, 0], axes[0, 1]],
                 [df['bsmDifferenceRidge'], df['bsmDifferenceRf']])
    _shared_ylim([axes[1, 0], axes[1, 1]],
                 [df['pctErrorRidge'], df['pctErrorRf']])

    plt.tight_layout()
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_put_call_parity(calls_df, puts_df, r, q, filename=None):
    """
    Deviation from dividend-adjusted put-call parity, plotted against moneyness.

    Only options present in both datasets at the same strike and maturity
    are used. For European options the deviation should be zero; the residual
    reflects the early-exercise premium of American-style contracts.

    Returns the merged DataFrame with the parity error column, so the
    summary statistics can be inspected.
    """
    parity = pd.merge(
        calls_df[['strike', 'expirationDate', 'currentUnderlyingPrice',
                  'timetoMaturity', 'midPrice']],
        puts_df[['strike', 'expirationDate', 'midPrice']],
        on=['strike', 'expirationDate'],
        suffixes=('_call', '_put'),
    )

    parity['observed'] = parity['midPrice_call'] - parity['midPrice_put']
    parity['theoretical'] = (
        parity['currentUnderlyingPrice'] * np.exp(-q * parity['timetoMaturity'])
        - parity['strike'] * np.exp(-r * parity['timetoMaturity'])
    )
    parity['parityError'] = parity['observed'] - parity['theoretical']
    parity['moneyness'] = parity['strike'] / parity['currentUnderlyingPrice']

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(parity['moneyness'], parity['parityError'],
               alpha=0.7, s=25, color=COLOR_DEFAULT, edgecolors='none')
    _finish_axis(ax, 'Moneyness (K/S)', 'Parity deviation ($)',
                 'Put-call parity check', hline=0)

    plt.tight_layout()
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return parity