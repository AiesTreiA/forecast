"""
Challenger Layer: LightGBM Residual Booster.
Trains on Holt-Winters residuals (e_t = y_t - y_hw_t) using retail-specific exogenous features:
promotions, discount depth, paydays (quincenas), holidays, and calendar interactions.
Guarantees graceful fallback to Holt-Winters when exogenous features are inactive.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb


@dataclass
class BoosterResult:
    train_residuals: np.ndarray
    train_predicted_residuals: np.ndarray
    forecast_residuals: np.ndarray
    feature_names: List[str]
    feature_importances: pd.DataFrame
    model: lgb.LGBMRegressor


class ResidualBooster:
    """
    Non-linear residual booster designed specifically to amplify Holt-Winters
    without replacing its baseline trend and seasonality decomposition.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 150,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 10,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.model_: Optional[lgb.LGBMRegressor] = None
        self.feature_names_: List[str] = []

    def extract_features(
        self,
        df_series: pd.DataFrame,
        hw_fitted_or_forecast: np.ndarray,
        is_training: bool = True,
    ) -> pd.DataFrame:
        """
        Builds retail exogenous and calendar features.
        """
        df = df_series.copy()
        
        # Calendar & Payday features
        dates = pd.to_datetime(df["date"])
        dom = dates.dt.day
        dow = dates.dt.dayofweek
        month = dates.dt.month
        
        features = pd.DataFrame(index=df.index)
        features["day_of_week"] = dow
        features["is_weekend"] = (dow >= 4).astype(int)  # Fri, Sat, Sun
        features["month"] = month
        
        # Payday dynamics (Quincena & Fin de mes)
        features["is_payday"] = np.isin(dom, [14, 15, 16, 29, 30, 31]).astype(int)
        
        # Distance to payday
        payday_targets = [15, 30]
        dist_to_payday = np.zeros(len(dom))
        for i, day in enumerate(dom):
            dist_to_payday[i] = min(abs(day - 15), abs(day - 1), abs(day - 30))
        features["dist_to_payday"] = dist_to_payday

        # Holidays
        features["is_holiday"] = df["is_holiday"].astype(int) if "is_holiday" in df.columns else 0

        # Promotional Features
        features["is_promo"] = df["is_promo"].astype(int) if "is_promo" in df.columns else 0
        features["discount_pct"] = df["discount_pct"].astype(float) if "discount_pct" in df.columns else 0.0

        # Promo Type One-Hot Encoding
        if "promo_type" in df.columns:
            for ptype in ["catalogo_ofertas", "cabecera_gondola", "descuento_20pct", "2x1"]:
                features[f"promo_{ptype}"] = (df["promo_type"] == ptype).astype(int)
        else:
            for ptype in ["catalogo_ofertas", "cabecera_gondola", "descuento_20pct", "2x1"]:
                features[f"promo_{ptype}"] = 0

        # Promo interaction with weekend
        features["promo_weekend_boost"] = features["is_promo"] * features["is_weekend"]
        features["promo_payday_boost"] = features["is_promo"] * features["is_payday"]

        # Holt-Winters baseline level as feature
        features["hw_baseline"] = hw_fitted_or_forecast

        return features

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        hw_train_fitted: np.ndarray,
        target_uncensored_sales: np.ndarray,
        test_df: pd.DataFrame,
        hw_forecast: np.ndarray,
    ) -> BoosterResult:
        """
        Fits LightGBM on the Holt-Winters training residuals and forecasts residuals on test set.
        """
        # Calculate target residuals
        y_train_residuals = target_uncensored_sales - hw_train_fitted

        # Build feature matrices
        X_train = self.extract_features(train_df, hw_train_fitted, is_training=True)
        X_test = self.extract_features(test_df, hw_forecast, is_training=False)

        self.feature_names_ = list(X_train.columns)

        # Fit LightGBM with Huber or L1 objective for retail robustness
        model = lgb.LGBMRegressor(
            objective="regression_l1",  # Robust to promo spikes
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
        )

        model.fit(X_train, y_train_residuals)
        self.model_ = model

        # Predict residuals
        train_pred_res = model.predict(X_train)
        test_pred_res = model.predict(X_test)

        # Feature importances
        importance_df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

        return BoosterResult(
            train_residuals=y_train_residuals,
            train_predicted_residuals=train_pred_res,
            forecast_residuals=test_pred_res,
            feature_names=self.feature_names_,
            feature_importances=importance_df,
            model=model,
        )
