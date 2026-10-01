"""
Auditable Explainability Engine for Retail Demand.
Uses SHAP (SHapley Additive exPlanations) on the LightGBM Residual Booster
to break down forecast uplifts into concrete retail drivers (promotions, paydays, seasonality).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb


class ForecastExplainer:
    """
    Computes exact attribution of residual uplifts for retail auditing.
    """

    def __init__(self, model: lgb.LGBMRegressor, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain_dataset(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Computes SHAP values matrix and expected baseline value.
        """
        X = X_df[self.feature_names]
        shap_values = self.explainer.shap_values(X)
        expected_value = float(
            self.explainer.expected_value
            if not isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value[0]
        )
        return shap_values, expected_value

    def explain_single_day(
        self,
        X_row: pd.Series,
        hw_baseline_val: float,
        date_str: str = "",
    ) -> Dict[str, Any]:
        """
        Provides a waterfall breakdown for a specific calendar day:
        HW Baseline -> SHAP Feature Impacts -> Final Challenger Forecast.
        """
        X_df = pd.DataFrame([X_row[self.feature_names]])
        shap_vals = self.explainer.shap_values(X_df)[0]
        base_exp = float(
            self.explainer.expected_value
            if not isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value[0]
        )

        contributions = []
        for feat, sval in zip(self.feature_names, shap_vals):
            raw_val = X_row.get(feat, 0)
            contributions.append({
                "feature": feat,
                "feature_value": raw_val,
                "shap_impact": float(sval),
            })

        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)

        predicted_residual = base_exp + float(np.sum(shap_vals))
        final_forecast = max(0.0, hw_baseline_val + predicted_residual)

        return {
            "date": date_str,
            "hw_baseline": hw_baseline_val,
            "base_expected_residual": base_exp,
            "predicted_residual": predicted_residual,
            "final_forecast": final_forecast,
            "contributions": contributions,
        }

    def get_global_feature_importance(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates mean absolute SHAP value for each feature.
        """
        shap_vals, _ = self.explain_dataset(X_df)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        # Pretty labels for retail executives
        labels_map = {
            "hw_baseline": "Nivel Base Holt-Winters",
            "is_promo": "Flag Promoción Activa",
            "discount_pct": "Profundidad Descuento (%)",
            "promo_weekend_boost": "Sinergia Promo + Fin de Semana",
            "promo_payday_boost": "Sinergia Promo + Quincena",
            "is_payday": "Efecto Quincena / Fin de Mes",
            "dist_to_payday": "Proximidad a Quincena",
            "is_weekend": "Fin de Semana (Viernes/Sábado/Domingo)",
            "day_of_week": "Día de la Semana",
            "month": "Mes del Año",
            "is_holiday": "Feriado / Festividad",
            "promo_2x1": "Mecánica Promo 2x1",
            "promo_cabecera_gondola": "Mecánica Cabecera Góndola",
            "promo_catalogo_ofertas": "Mecánica Catálogo Ofertas",
            "promo_descuento_20pct": "Mecánica Descuento Directo 20%",
        }
        importance_df["feature_label"] = importance_df["feature"].map(labels_map).fillna(importance_df["feature"])
        return importance_df
