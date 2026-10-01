"""
Layer 3: Conformal Prediction & Uncertainty Quantification for Retail Inventory.
Computes distribution-free prediction intervals with finite-sample coverage guarantees (1 - alpha)
and sizes Safety Stock without relying on fragile Gaussian assumptions.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ConformalIntervalResult:
    coverage_target: float
    empirical_coverage_gaussian: float
    empirical_coverage_conformal: float
    gaussian_upper: np.ndarray
    gaussian_lower: np.ndarray
    conformal_upper: np.ndarray
    conformal_lower: np.ndarray
    gaussian_safety_stock: float
    conformal_safety_stock: float
    conformal_quantile_q: float
    gaussian_sigma: float
    interval_efficiency_ratio: float


class ConformalPredictor:
    """
    Split Conformal Prediction engine calibrated for retail demand and safety stock sizing.
    """

    def __init__(self, target_service_level: float = 0.95):
        """
        target_service_level: e.g. 0.95 (meaning 95% cycle service level / coverage).
        alpha = 1 - target_service_level (0.05).
        """
        self.target_service_level = target_service_level
        self.alpha = 1.0 - target_service_level
        self.q_score_: float = 0.0
        self.sigma_: float = 0.0

    def calibrate(
        self,
        y_calibration: np.ndarray,
        y_hat_calibration: np.ndarray,
    ) -> float:
        """
        Computes non-conformity scores on calibration set and extracts the finite-sample quantile.
        """
        residuals = y_calibration - y_hat_calibration
        self.sigma_ = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0

        # Absolute non-conformity score
        scores = np.abs(residuals)
        n = len(scores)

        # Finite-sample adjusted quantile index
        # ceil((n + 1) * (1 - alpha)) / n
        quantile_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_score_ = float(np.quantile(scores, quantile_level, method="higher" if hasattr(np, "quantile") else "linear"))

        return self.q_score_

    def predict_intervals_and_evaluate(
        self,
        y_hat_test: np.ndarray,
        y_true_test: np.ndarray,
        lead_time_days: int = 3,
    ) -> ConformalIntervalResult:
        """
        Generates Gaussian and Conformal intervals on test set, sizing safety stocks and evaluating coverage.
        """
        z_alpha = stats.norm.ppf(self.target_service_level)

        # Gaussian intervals & Safety Stock
        gauss_width = z_alpha * self.sigma_
        gaussian_upper = np.maximum(0.0, y_hat_test + gauss_width)
        gaussian_lower = np.maximum(0.0, y_hat_test - gauss_width)
        gaussian_safety_stock = float(gauss_width * np.sqrt(lead_time_days))

        # Conformal intervals & Safety Stock
        conformal_upper = np.maximum(0.0, y_hat_test + self.q_score_)
        conformal_lower = np.maximum(0.0, y_hat_test - self.q_score_)
        conformal_safety_stock = float(self.q_score_ * np.sqrt(lead_time_days))

        # Empirical test coverage
        cov_gauss = np.mean((y_true_test >= gaussian_lower) & (y_true_test <= gaussian_upper))
        cov_conf = np.mean((y_true_test >= conformal_lower) & (y_true_test <= conformal_upper))

        # Efficiency: average interval width comparison
        avg_gauss_width = np.mean(gaussian_upper - gaussian_lower)
        avg_conf_width = np.mean(conformal_upper - conformal_lower)
        eff_ratio = avg_conf_width / avg_gauss_width if avg_gauss_width > 0 else 1.0

        return ConformalIntervalResult(
            coverage_target=self.target_service_level,
            empirical_coverage_gaussian=float(cov_gauss),
            empirical_coverage_conformal=float(cov_conf),
            gaussian_upper=gaussian_upper,
            gaussian_lower=gaussian_lower,
            conformal_upper=conformal_upper,
            conformal_lower=conformal_lower,
            gaussian_safety_stock=gaussian_safety_stock,
            conformal_safety_stock=conformal_safety_stock,
            conformal_quantile_q=self.q_score_,
            gaussian_sigma=self.sigma_,
            interval_efficiency_ratio=float(eff_ratio),
        )
