"""
Volatility surface models: regularized polynomial regression and Random Forest.

Both are fitted on (moneyness, maturity) -> implied volatility, using the same
train-test split so their RMSE values are directly comparable.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from bsm import bsm_price

FEATURES = ['moneyness', 'timetoMaturity']
TARGET = 'impliedVol'


def poly_ridge_reg(X_train, X_test, y_train, y_test, degree=3,
                    alphas=(0.001, 0.01, 0.1, 1.0, 10.0)):
    """Fit a polynomial regression with a Ridge penalty term.

    The lambda parameter is selected by cross-validation over `alphas`.

    Returns (predictions on test set, RMSE, fitted model, fitted transformer).
    """
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    ridge_cv = RidgeCV(alphas=list(alphas))
    ridge_cv.fit(X_train_poly, y_train)

    ridge_pred = ridge_cv.predict(X_test_poly)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

    return ridge_pred, ridge_rmse, ridge_cv, poly


def rand_trees(X_train, X_test, y_train, y_test, n_estimators=200,
               max_depth=6, random_state=42):
    """Fit a Random Forest regressor.

    Depth and number of trees are capped to limit overfitting.

    Returns (predictions on test set, RMSE, fitted model).
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                random_state=random_state)
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    return rf_pred, rf_rmse, rf


def fit_models(opt_df, test_size=0.2, random_state=42):
    """Fit both surface models on the same train-test split.

    Requires the impliedVol column (see data.add_implied_vol).

    Returns
    -------
    dict with keys: 'poly', 'ridge', 'ridge_rmse', 'rf', 'rf_rmse'.
    Pass this dict straight to plots.plot_surface_pair().
    """
    X = opt_df[FEATURES]
    y = opt_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    _, ridge_rmse, ridge, poly = poly_ridge_reg(X_train, X_test, y_train, y_test)
    _, rf_rmse, rf = rand_trees(X_train, X_test, y_train, y_test,
                                 random_state=random_state)

    return {'poly': poly, 'ridge': ridge, 'ridge_rmse': ridge_rmse,
            'rf': rf, 'rf_rmse': rf_rmse}


def add_model_predictions(opt_df, models, r, q, opt_type='call'):
    """Apply both fitted models to every option, then re-price with BSM.

    The estimated IV replaces the historical sigma of the baseline, so the
    residual mispricing measures how much of the original error the surface
    models remove.

    Adds: impliedVolRidge, impliedVolRf, bsmPriceRidge, bsmPriceRf,
    bsmDifferenceRidge, pctErrorRidge, bsmDifferenceRf, pctErrorRf.
    """
    df = opt_df.copy()
    X_all = df[FEATURES]

    df['impliedVolRidge'] = models['ridge'].predict(models['poly'].transform(X_all))
    df['impliedVolRf'] = models['rf'].predict(X_all)

    for label, iv_col in [('Ridge', 'impliedVolRidge'), ('Rf', 'impliedVolRf')]:
        price_col = f'bsmPrice{label}'
        df[price_col] = df.apply(
            lambda row: bsm_price(S=row['currentUnderlyingPrice'], K=row['strike'],
                                  r=r, q=q, t=row['timetoMaturity'],
                                  sigma=row[iv_col], option_type=opt_type),
            axis=1,
        )
        df[f'bsmDifference{label}'] = df[price_col] - df['midPrice']
        df[f'pctError{label}'] = df[f'bsmDifference{label}'] / df['midPrice'] * 100

    return df
