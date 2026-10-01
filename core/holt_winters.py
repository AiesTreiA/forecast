"""
Champion Forecasting Engine: Holt-Winters Exponential Smoothing.
Implements additive and multiplicative Holt-Winters with weekly seasonality (period=7),
extracting level, trend, seasonal indices, and residual series.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class HoltWintersResult:
    model_name: str
    alpha: float
    beta: float
    gamma: float
    fitted_values: pd.Series
    forecast_values: pd.Series
    residuals: pd.Series
    level: pd.Series
    trend: pd.Series
    season: pd.Series
    aic: float
    bic: float
    training_metrics: Dict[str, float]


class HoltWintersForecaster:
    """
    Standard Retail Holt-Winters implementation matching supermarket forecasting systems.
    """

    def __init__(
        self,
        seasonal_periods: int = 7,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool = False,
        initialization_method: str = "heuristic",
    ):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.initialization_method = initialization_method
        self.fitted_model_ = None

    def fit_predict(
        self,
        train_series: pd.Series,
        forecast_horizon: int,
        model_label: str = "Holt-Winters Champion",
    ) -> HoltWintersResult:
        """
        Fits Holt-Winters on historical series and forecasts horizon periods ahead.
        """
        y = train_series.astype(float).clip(lower=0.1)

        try:
            model = ExponentialSmoothing(
                y,
                seasonal_periods=self.seasonal_periods,
                trend=self.trend,
                seasonal=self.seasonal,
                damped_trend=self.damped_trend,
                initialization_method=self.initialization_method,
            )
            fitted = model.fit(optimized=True)
        except Exception:
            # Fallback to estimated or simpler init if solver fails
            model = ExponentialSmoothing(
                y,
                seasonal_periods=self.seasonal_periods,
                trend=self.trend,
                seasonal=self.seasonal,
                damped_trend=self.damped_trend,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)

        self.fitted_model_ = fitted

        # Forecast
        forecast = fitted.forecast(forecast_horizon)
        # Avoid negative demand
        forecast = pd.Series(np.maximum(0.0, forecast.values), index=range(len(train_series), len(train_series) + forecast_horizon))
        fitted_vals = pd.Series(np.maximum(0.0, fitted.fittedvalues.values), index=train_series.index)
        residuals = pd.Series(train_series.values - fitted_vals.values, index=train_series.index)

        # Decomposed components
        level = pd.Series(fitted.level, index=train_series.index)
        trend = pd.Series(fitted.slope if hasattr(fitted, "slope") else np.zeros(len(train_series)), index=train_series.index)
        season = pd.Series(fitted.season if hasattr(fitted, "season") else np.zeros(len(train_series)), index=train_series.index)

        # Calculate in-sample training metrics
        actual = train_series.values
        pred = fitted_vals.values
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        sum_actual = float(np.sum(actual))
        wape = float(np.sum(np.abs(actual - pred)) / sum_actual * 100.0) if sum_actual > 0 else 0.0

        return HoltWintersResult(
            model_name=model_label,
            alpha=float(fitted.params.get("smoothing_level", 0.0)),
            beta=float(fitted.params.get("smoothing_trend", 0.0)),
            gamma=float(fitted.params.get("smoothing_seasonal", 0.0)),
            fitted_values=fitted_vals,
            forecast_values=forecast,
            residuals=residuals,
            level=level,
            trend=trend,
            season=season,
            aic=float(fitted.aic) if hasattr(fitted, "aic") else 0.0,
            bic=float(fitted.bic) if hasattr(fitted, "bic") else 0.0,
            training_metrics={
                "MAE": mae,
                "RMSE": rmse,
                "WAPE": wape,
            },
        )
