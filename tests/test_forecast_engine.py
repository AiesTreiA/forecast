"""
Unit & Integration Tests for Holt-Winters+ Core Engines and Pipeline.
"""

import pytest
import numpy as np
import pandas as pd

from core.data_generator import generate_retail_dataset, SKU_CATALOG
from core.stockout_corrector import StockoutCorrector, audit_stockout_corrections
from core.holt_winters import HoltWintersForecaster
from core.residual_booster import ResidualBooster
from core.conformal_uq import ConformalPredictor
from core.inventory_engine import RetailInventorySimulator, compare_champion_vs_challenger_financials
from pipeline import ForecastingPipeline


@pytest.fixture(scope="module")
def sample_dataset():
    # Fast 1-year dataset for unit testing
    return generate_retail_dataset(start_date="2024-01-01", end_date="2024-12-31", seed=42)


def test_data_generator_schema(sample_dataset):
    df = sample_dataset
    assert not df.empty
    expected_cols = [
        "date", "store_id", "sku_id", "true_demand", "observed_sales",
        "stockout_flag", "lost_sales", "is_promo", "discount_pct", "unit_cost", "retail_price"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col}"

    assert (df["observed_sales"] <= df["true_demand"]).all()
    assert (df["stockout_flag"].isin([0, 1])).all()


def test_stockout_corrector(sample_dataset):
    df = sample_dataset
    sub = df[(df["store_id"] == "TIENDA-01") & (df["sku_id"] == "SKU-204")].copy()
    
    audited_df, metrics = audit_stockout_corrections(sub)
    assert "champion_demand" in audited_df.columns
    assert "challenger_demand" in audited_df.columns
    assert (audited_df["champion_demand"] >= audited_df["observed_sales"]).all()
    assert (audited_df["challenger_demand"] >= audited_df["observed_sales"]).all()
    assert metrics["stockout_days"] > 0
    assert metrics["champion_imputed_units"] > 0
    assert metrics["challenger_imputed_units"] > 0


def test_holt_winters_fit_predict(sample_dataset):
    df = sample_dataset
    sub = df[(df["store_id"] == "TIENDA-01") & (df["sku_id"] == "SKU-101")].copy()
    sub["champion_demand"] = StockoutCorrector.correct_baseline_champion(sub)

    train = sub.iloc[:-30]["champion_demand"]
    hw = HoltWintersForecaster(seasonal_periods=7, trend="add", seasonal="add")
    res = hw.fit_predict(train, forecast_horizon=30)

    assert len(res.forecast_values) == 30
    assert (res.forecast_values >= 0).all()
    assert len(res.fitted_values) == len(train)
    assert res.training_metrics["WAPE"] > 0


def test_residual_booster(sample_dataset):
    df = sample_dataset
    sub = df[(df["store_id"] == "TIENDA-01") & (df["sku_id"] == "SKU-204")].copy()
    sub["challenger_demand"] = StockoutCorrector.correct_advanced_challenger(sub)

    train_df = sub.iloc[:-30].copy()
    test_df = sub.iloc[-30:].copy()

    hw = HoltWintersForecaster(seasonal_periods=7)
    hw_res = hw.fit_predict(train_df["challenger_demand"], forecast_horizon=30)

    booster = ResidualBooster(n_estimators=30)
    boost_res = booster.fit_predict(
        train_df=train_df,
        hw_train_fitted=hw_res.fitted_values.values,
        target_uncensored_sales=train_df["challenger_demand"].values,
        test_df=test_df,
        hw_forecast=hw_res.forecast_values.values,
    )

    assert len(boost_res.forecast_residuals) == 30
    assert not boost_res.feature_importances.empty


def test_conformal_prediction():
    rng = np.random.default_rng(42)
    y_calib = rng.normal(100, 15, size=300)
    y_hat_calib = np.full(300, 100.0)

    cp = ConformalPredictor(target_service_level=0.95)
    q = cp.calibrate(y_calib, y_hat_calib)
    assert q > 0

    y_test_true = rng.normal(100, 15, size=200)
    y_test_hat = np.full(200, 100.0)

    res = cp.predict_intervals_and_evaluate(y_test_hat, y_test_true, lead_time_days=3)
    assert res.empirical_coverage_conformal >= 0.85
    assert res.conformal_safety_stock > 0
    assert res.gaussian_safety_stock > 0


def test_inventory_simulation():
    dates = pd.date_range("2024-01-01", periods=30)
    demands = np.full(30, 50.0)
    forecast = np.full(30, 50.0)

    sim = RetailInventorySimulator(lead_time_days=2, unit_cost=2.0, retail_price=4.0)
    res = sim.simulate_policy(
        policy_label="TestPolicy",
        dates=dates,
        actual_demands=demands,
        forecast_daily=forecast,
        safety_stock=20.0,
        initial_inventory=100.0,
    )

    assert res.fill_rate_pct > 95.0
    assert res.total_gross_profit > 0
    assert res.total_holding_cost_dollars > 0


def test_pipeline_end_to_end(sample_dataset):
    pipe = ForecastingPipeline(test_days=30, target_service_level=0.95)
    res = pipe.run("TIENDA-01", "SKU-204", df=sample_dataset)

    assert res.sku_id == "SKU-204"
    assert not res.metrics_summary_table.empty
    assert res.financial_deltas is not None
    assert len(res.test_df) == 30
