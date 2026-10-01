"""
End-to-End Forecasting Pipeline: Champion vs Challenger Orchestrator.
Executes the full auditable pipeline for any SKU-Store combination:
1. Data ingestion & Stockout Uncensoring
2. Champion Holt-Winters baseline fitting
3. Challenger Residual Boosting (LightGBM)
4. Conformal Prediction & Safety Stock sizing
5. Physical Inventory Simulation & Dollarized Business ROI
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from core.data_generator import load_or_generate_dataset, SKU_CATALOG
from core.stockout_corrector import StockoutCorrector, audit_stockout_corrections
from core.holt_winters import HoltWintersForecaster, HoltWintersResult
from core.residual_booster import ResidualBooster, BoosterResult
from core.conformal_uq import ConformalPredictor, ConformalIntervalResult
from core.inventory_engine import RetailInventorySimulator, PolicySimulationResult, compare_champion_vs_challenger_financials
from core.explainability import ForecastExplainer


@dataclass
class PipelineExecutionResult:
    store_id: str
    store_name: str
    sku_id: str
    sku_name: str
    category: str
    unit_cost: float
    retail_price: float
    shelf_life_days: int
    lead_time_days: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    full_df: pd.DataFrame
    stockout_audit_metrics: Dict[str, Any]
    hw_champion_result: HoltWintersResult
    booster_result: BoosterResult
    conformal_result: ConformalIntervalResult
    sim_champion: PolicySimulationResult
    sim_challenger: PolicySimulationResult
    financial_deltas: Dict[str, Any]
    explainer: ForecastExplainer
    metrics_summary_table: pd.DataFrame


class ForecastingPipeline:
    """
    High-level facade running the entire Champion vs Challenger benchmark.
    """

    def __init__(
        self,
        test_days: int = 60,
        target_service_level: float = 0.95,
        annual_holding_cost_rate: float = 0.22,
    ):
        self.test_days = test_days
        self.target_service_level = target_service_level
        self.annual_holding_cost_rate = annual_holding_cost_rate

    def run(
        self,
        store_id: str,
        sku_id: str,
        df: Optional[pd.DataFrame] = None,
    ) -> PipelineExecutionResult:
        """
        Executes end-to-end forecast, UQ, and inventory simulation.
        """
        if df is None:
            df = load_or_generate_dataset()

        # 1. Filter Store and SKU
        mask = (df["store_id"] == store_id) & (df["sku_id"] == sku_id)
        sub_df = df[mask].sort_values("date").reset_index(drop=True).copy()
        if len(sub_df) == 0:
            raise ValueError(f"No records found for Store '{store_id}' and SKU '{sku_id}'")

        store_name = sub_df["store_name"].iloc[0]
        sku_name = sub_df["sku_name"].iloc[0]
        category = sub_df["category"].iloc[0]
        unit_cost = float(sub_df["unit_cost"].iloc[0])
        retail_price = float(sub_df["retail_price"].iloc[0])
        shelf_life = int(sub_df["shelf_life_days"].iloc[0])
        lead_time = int(sub_df["lead_time_days"].iloc[0])

        # 2. Uncensoring & Stockout Audit
        audited_df, stockout_metrics = audit_stockout_corrections(sub_df)

        # 3. Train / Test Split
        n_total = len(audited_df)
        n_test = min(self.test_days, n_total // 4)
        n_train = n_total - n_test

        train_df = audited_df.iloc[:n_train].copy().reset_index(drop=True)
        test_df = audited_df.iloc[n_train:].copy().reset_index(drop=True)

        # 4. Champion Model: Holt-Winters fitted on Champion Stockout-Corrected Sales
        hw_forecaster = HoltWintersForecaster(
            seasonal_periods=7,
            trend="add",
            seasonal="add",
            initialization_method="heuristic",
        )
        hw_res = hw_forecaster.fit_predict(
            train_series=train_df["champion_demand"],
            forecast_horizon=n_test,
            model_label="Holt-Winters Champion",
        )

        # 5. Challenger Layer 2: Residual Boosting (LightGBM)
        booster = ResidualBooster(
            learning_rate=0.04,
            n_estimators=180,
            max_depth=4,
            num_leaves=15,
        )
        booster_res = booster.fit_predict(
            train_df=train_df,
            hw_train_fitted=hw_res.fitted_values.values,
            target_uncensored_sales=train_df["challenger_demand"].values,
            test_df=test_df,
            hw_forecast=hw_res.forecast_values.values,
        )

        # Build final forecasts on test set
        hw_forecast = hw_res.forecast_values.values
        challenger_forecast = np.maximum(0.0, hw_forecast + booster_res.forecast_residuals)

        test_df["pred_hw_champion"] = hw_forecast
        test_df["pred_challenger"] = challenger_forecast
        test_df["boost_residual"] = booster_res.forecast_residuals

        # 6. Challenger Layer 3: Conformal Prediction & Uncertainty Quantification
        conformal = ConformalPredictor(target_service_level=self.target_service_level)
        # Calibrate using in-sample / out-of-fold residuals from the training booster
        conformal.calibrate(
            y_calibration=train_df["challenger_demand"].values,
            y_hat_calibration=hw_res.fitted_values.values + booster_res.train_predicted_residuals,
        )

        ground_truth_test = test_df["true_demand"].values if "true_demand" in test_df.columns else test_df["observed_sales"].values

        conformal_res = conformal.predict_intervals_and_evaluate(
            y_hat_test=challenger_forecast,
            y_true_test=ground_truth_test,
            lead_time_days=lead_time,
        )

        test_df["conformal_lower"] = conformal_res.conformal_lower
        test_df["conformal_upper"] = conformal_res.conformal_upper
        test_df["gaussian_lower"] = conformal_res.gaussian_lower
        test_df["gaussian_upper"] = conformal_res.gaussian_upper

        # 7. Inventory Replenishment & Financial Simulation
        sim = RetailInventorySimulator(
            lead_time_days=lead_time,
            unit_cost=unit_cost,
            retail_price=retail_price,
            annual_holding_cost_rate=self.annual_holding_cost_rate,
            shelf_life_days=shelf_life,
            order_frequency_days=2,
        )

        initial_stock = float(ground_truth_test[0] * (lead_time + 2))

        # Champion Simulation: HW Forecast + Gaussian Safety Stock
        sim_champion = sim.simulate_policy(
            policy_label="Champion (Holt-Winters + OOS Corrector)",
            dates=test_df["date"],
            actual_demands=ground_truth_test,
            forecast_daily=hw_forecast,
            safety_stock=conformal_res.gaussian_safety_stock,
            initial_inventory=initial_stock,
        )

        # Challenger Simulation: HW+ Boosted Forecast + Conformal Safety Stock
        sim_challenger = sim.simulate_policy(
            policy_label="Challenger (Holt-Winters+ Residual Booster & Conformal)",
            dates=test_df["date"],
            actual_demands=ground_truth_test,
            forecast_daily=challenger_forecast,
            safety_stock=conformal_res.conformal_safety_stock,
            initial_inventory=initial_stock,
        )

        financial_deltas = compare_champion_vs_challenger_financials(sim_champion, sim_challenger)

        # 8. SHAP Explainability Engine
        explainer = ForecastExplainer(
            model=booster_res.model,
            feature_names=booster_res.feature_names,
        )

        # 9. Comprehensive Comparison Scorecard Table
        actuals = ground_truth_test
        hw_f = hw_forecast
        ch_f = challenger_forecast
        sum_actuals = float(np.sum(actuals))

        mae_hw = float(np.mean(np.abs(actuals - hw_f)))
        mae_ch = float(np.mean(np.abs(actuals - ch_f)))
        rmse_hw = float(np.sqrt(np.mean((actuals - hw_f) ** 2)))
        rmse_ch = float(np.sqrt(np.mean((actuals - ch_f) ** 2)))
        wape_hw = float(np.sum(np.abs(actuals - hw_f)) / sum_actuals * 100.0) if sum_actuals > 0 else 0.0
        wape_ch = float(np.sum(np.abs(actuals - ch_f)) / sum_actuals * 100.0) if sum_actuals > 0 else 0.0
        mape_hw = float(np.mean(np.abs((actuals - hw_f) / np.maximum(1.0, actuals))) * 100.0)
        mape_ch = float(np.mean(np.abs((actuals - ch_f) / np.maximum(1.0, actuals))) * 100.0)

        summary_rows = [
            {
                "Dimensión": "🎯 Precisión Pronóstico",
                "Métrica": "WAPE (%)",
                "Champion (HW)": f"{wape_hw:.2f}%",
                "Challenger (HW+)": f"{wape_ch:.2f}%",
                "Mejora Directa": f"{wape_hw - wape_ch:+.2f} pp",
            },
            {
                "Dimensión": "🎯 Precisión Pronóstico",
                "Métrica": "MAE (Unidades / día)",
                "Champion (HW)": f"{mae_hw:.2f}",
                "Challenger (HW+)": f"{mae_ch:.2f}",
                "Mejora Directa": f"{mae_hw - mae_ch:+.2f} u",
            },
            {
                "Dimensión": "🎯 Precisión Pronóstico",
                "Métrica": "RMSE (Unidades)",
                "Champion (HW)": f"{rmse_hw:.2f}",
                "Challenger (HW+)": f"{rmse_ch:.2f}",
                "Mejora Directa": f"{rmse_hw - rmse_ch:+.2f} u",
            },
            {
                "Dimensión": "🛡️ Incertidumbre & Riesgo",
                "Métrica": "Cobertura Empírica (Obj 95%)",
                "Champion (HW)": f"{conformal_res.empirical_coverage_gaussian * 100:.1f}%",
                "Challenger (HW+)": f"{conformal_res.empirical_coverage_conformal * 100:.1f}%",
                "Mejora Directa": f"{(conformal_res.empirical_coverage_conformal - conformal_res.empirical_coverage_gaussian) * 100:+.1f} pp",
            },
            {
                "Dimensión": "🛡️ Incertidumbre & Riesgo",
                "Métrica": "Stock de Seguridad Recomendado",
                "Champion (HW)": f"{conformal_res.gaussian_safety_stock:.1f} u",
                "Challenger (HW+)": f"{conformal_res.conformal_safety_stock:.1f} u",
                "Mejora Directa": f"{conformal_res.conformal_safety_stock - conformal_res.gaussian_safety_stock:+.1f} u",
            },
            {
                "Dimensión": "📦 Operación Retail",
                "Métrica": "Fill Rate (%)",
                "Champion (HW)": f"{sim_champion.fill_rate_pct:.2f}%",
                "Challenger (HW+)": f"{sim_challenger.fill_rate_pct:.2f}%",
                "Mejora Directa": f"{financial_deltas['delta_fill_rate_pct']:+.2f} pp",
            },
            {
                "Dimensión": "📦 Operación Retail",
                "Métrica": "Días con Quiebre (OOS)",
                "Champion (HW)": f"{sim_champion.stockout_days} días",
                "Challenger (HW+)": f"{sim_challenger.stockout_days} días",
                "Mejora Directa": f"{financial_deltas['delta_stockout_days']:+d} días",
            },
            {
                "Dimensión": "📦 Operación Retail",
                "Métrica": "Ventas Perdidas (Unidades)",
                "Champion (HW)": f"{sim_champion.total_lost_sales_units:,.0f} u",
                "Challenger (HW+)": f"{sim_challenger.total_lost_sales_units:,.0f} u",
                "Mejora Directa": f"{-financial_deltas['delta_sales_units']:+,.0f} u",
            },
            {
                "Dimensión": "💰 Impacto Financiero",
                "Métrica": "Margen Bruto Total",
                "Champion (HW)": f"${sim_champion.total_gross_profit:,.2f}",
                "Challenger (HW+)": f"${sim_challenger.total_gross_profit:,.2f}",
                "Mejora Directa": f"${financial_deltas['delta_gross_profit_dollars']:+,.2f}",
            },
            {
                "Dimensión": "💰 Impacto Financiero",
                "Métrica": "Costo de Almacenamiento (Holding)",
                "Champion (HW)": f"${sim_champion.total_holding_cost_dollars:,.2f}",
                "Challenger (HW+)": f"${sim_challenger.total_holding_cost_dollars:,.2f}",
                "Mejora Directa": f"${financial_deltas['delta_holding_cost_dollars']:+,.2f}",
            },
            {
                "Dimensión": "💰 Impacto Financiero",
                "Métrica": "Beneficio Económico Neto",
                "Champion (HW)": f"${sim_champion.net_economic_profit_dollars:,.2f}",
                "Challenger (HW+)": f"${sim_challenger.net_economic_profit_dollars:,.2f}",
                "Mejora Directa": f"${financial_deltas['delta_net_profit_dollars']:+,.2f}",
            },
        ]

        metrics_summary_table = pd.DataFrame(summary_rows)

        return PipelineExecutionResult(
            store_id=store_id,
            store_name=store_name,
            sku_id=sku_id,
            sku_name=sku_name,
            category=category,
            unit_cost=unit_cost,
            retail_price=retail_price,
            shelf_life_days=shelf_life,
            lead_time_days=lead_time,
            train_df=train_df,
            test_df=test_df,
            full_df=audited_df,
            stockout_audit_metrics=stockout_metrics,
            hw_champion_result=hw_res,
            booster_result=booster_res,
            conformal_result=conformal_res,
            sim_champion=sim_champion,
            sim_challenger=sim_challenger,
            financial_deltas=financial_deltas,
            explainer=explainer,
            metrics_summary_table=metrics_summary_table,
        )
