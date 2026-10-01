"""
Stockout & Lost Sales Uncensoring Engine.
Reconstructs latent uncensored demand from truncated observed sales during out-of-stock periods.
Includes:
- Baseline Champion Method: Rolling Day-of-Week historical non-stockout average.
- Challenger Advanced Method: Trend-adjusted rate + Promotional lift compensation.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class StockoutCorrector:
    """
    Implements auditable demand uncensoring methods for retail time series.
    """

    @staticmethod
    def correct_baseline_champion(
        df_series: pd.DataFrame,
        lookback_weeks: int = 8,
    ) -> pd.Series:
        """
        Champion Method:
        When stockout occurs (or sales=0 with inventory=0), replace sales with
        the median of non-stockout sales on the same day-of-week over recent history.
        """
        df = df_series.copy()
        corrected = df["observed_sales"].astype(float).copy()

        # Identify days with stockout
        is_oos = (df["stockout_flag"] == 1) | ((df["observed_sales"] == 0) & (df["on_hand_inventory"] == 0))

        # Calculate day-of-week medians on non-stockout days
        valid_sales = df[~is_oos]
        dow_medians = valid_sales.groupby("day_of_week")["observed_sales"].median()
        global_median = valid_sales["observed_sales"].median()

        for idx in df[is_oos].index:
            dow = df.loc[idx, "day_of_week"]
            current_date = df.loc[idx, "date"]
            
            # Lookback window
            min_date = current_date - pd.Timedelta(weeks=lookback_weeks)
            recent_non_oos = valid_sales[
                (valid_sales["date"] >= min_date)
                & (valid_sales["date"] < current_date)
                & (valid_sales["day_of_week"] == dow)
            ]
            
            if len(recent_non_oos) >= 2:
                imputed = recent_non_oos["observed_sales"].median()
            else:
                imputed = dow_medians.get(dow, global_median)

            # Imputed demand cannot be less than whatever partial sales were observed
            corrected.loc[idx] = max(df.loc[idx, "observed_sales"], imputed)

        return corrected

    @staticmethod
    def correct_advanced_challenger(
        df_series: pd.DataFrame,
        lookback_weeks: int = 6,
    ) -> pd.Series:
        """
        Challenger Method:
        Uncensoring that accounts for:
        1. Local trend / level (EWMA of non-OOS days)
        2. Day of week index
        3. Active promotional uplift (if OOS happened during a promo, naive replacement severely underestimates demand!)
        """
        orig_index = df_series.index
        df = df_series.copy().reset_index(drop=True)
        corrected = df["observed_sales"].astype(float).copy()
        is_oos = (df["stockout_flag"] == 1) | ((df["observed_sales"] == 0) & (df["on_hand_inventory"] == 0))

        # Baseline clean non-OOS series
        valid_mask = ~is_oos
        valid_df = df[valid_mask].copy()

        # Approximate promo lift elasticity on valid days
        if "is_promo" in df.columns and valid_df["is_promo"].sum() > 5:
            promo_mean = valid_df[valid_df["is_promo"] == 1]["observed_sales"].mean()
            non_promo_mean = valid_df[valid_df["is_promo"] == 0]["observed_sales"].mean()
            promo_ratio = max(1.15, (promo_mean / non_promo_mean)) if non_promo_mean > 0 else 1.25
        else:
            promo_ratio = 1.25

        # Rolling EWMA level of non-OOS sales
        valid_df["ewma_level"] = valid_df["observed_sales"].ewm(span=21, adjust=False).mean()
        
        # Day of week seasonality factors
        dow_means = valid_df.groupby("day_of_week")["observed_sales"].mean()
        dow_factors = dow_means / (dow_means.mean() if dow_means.mean() > 0 else 1.0)

        # Merge back level to full index by forward filling
        full_df = df[["date", "day_of_week", "is_promo", "discount_pct", "observed_sales"]].copy()
        full_df = full_df.merge(valid_df[["date", "ewma_level"]], on="date", how="left")
        full_df["ewma_level"] = full_df["ewma_level"].ffill().bfill()

        for idx in df[is_oos].index:
            dow = full_df.loc[idx, "day_of_week"]
            level = full_df.loc[idx, "ewma_level"]
            dow_factor = dow_factors.get(dow, 1.0)
            
            # Base estimate = local level * day of week pattern
            est_demand = level * dow_factor

            # If promo was active during the stockout, scale up
            if full_df.loc[idx, "is_promo"] == 1:
                disc = full_df.loc[idx, "discount_pct"]
                # Additional lift proportional to discount depth
                lift = 1.0 + (disc * 2.2) if disc > 0 else promo_ratio
                est_demand = est_demand * lift

            corrected.loc[idx] = max(df.loc[idx, "observed_sales"], est_demand)

        corrected.index = orig_index
        return corrected


def audit_stockout_corrections(df_series: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes an audit comparing Raw Sales vs Champion Correction vs Challenger Correction.
    When ground-truth true_demand is present, calculates recovery accuracy.
    """
    df = df_series.copy().sort_values("date").reset_index(drop=True)
    
    # Calculate both corrections
    df["champion_demand"] = StockoutCorrector.correct_baseline_champion(df)
    df["challenger_demand"] = StockoutCorrector.correct_advanced_challenger(df)

    oos_mask = df["stockout_flag"] == 1
    n_oos = int(oos_mask.sum())
    total_days = len(df)
    oos_rate = n_oos / total_days if total_days > 0 else 0.0

    raw_sales_oos = df.loc[oos_mask, "observed_sales"].sum()
    champ_sales_oos = df.loc[oos_mask, "champion_demand"].sum()
    chall_sales_oos = df.loc[oos_mask, "challenger_demand"].sum()

    metrics = {
        "total_days": total_days,
        "stockout_days": n_oos,
        "stockout_rate_pct": oos_rate * 100.0,
        "observed_sales_total": float(df["observed_sales"].sum()),
        "champion_demand_total": float(df["champion_demand"].sum()),
        "challenger_demand_total": float(df["challenger_demand"].sum()),
        "champion_imputed_units": float(champ_sales_oos - raw_sales_oos),
        "challenger_imputed_units": float(chall_sales_oos - raw_sales_oos),
    }

    # If true demand exists (benchmark evaluation)
    if "true_demand" in df.columns:
        true_oos = df.loc[oos_mask, "true_demand"].sum()
        metrics["true_demand_total"] = float(df["true_demand"].sum())
        metrics["true_lost_sales_units"] = float(true_oos - raw_sales_oos)
        
        # MAE and WAPE on stockout days
        err_champ = np.abs(df.loc[oos_mask, "true_demand"] - df.loc[oos_mask, "champion_demand"])
        err_chall = np.abs(df.loc[oos_mask, "true_demand"] - df.loc[oos_mask, "challenger_demand"])
        
        metrics["champion_mae_on_oos"] = float(err_champ.mean()) if n_oos > 0 else 0.0
        metrics["challenger_mae_on_oos"] = float(err_chall.mean()) if n_oos > 0 else 0.0
        metrics["champion_wape_on_oos"] = float(err_champ.sum() / true_oos * 100.0) if true_oos > 0 else 0.0
        metrics["challenger_wape_on_oos"] = float(err_chall.sum() / true_oos * 100.0) if true_oos > 0 else 0.0

    return df, metrics
