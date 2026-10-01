"""
Financial & Inventory Simulation Engine for Supermarkets.
Translates forecasting accuracy and safety stock sizing directly into retail dollars:
- Working capital (Average on-hand inventory)
- Holding costs (annual carrying rate)
- Lost sales profit (gross margin lost due to out-of-stocks)
- Perishable waste / merma
- Achieved service level & fill rate
- Net dollarized profit uplift (Champion vs Challenger)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class PolicySimulationResult:
    policy_name: str
    total_sales_units: float
    total_lost_sales_units: float
    total_revenue: float
    total_gross_profit: float
    stockout_days: int
    stockout_rate_pct: float
    fill_rate_pct: float
    cycle_service_level_pct: float
    avg_inventory_units: float
    avg_working_capital_dollars: float
    total_holding_cost_dollars: float
    total_waste_units: float
    total_waste_cost_dollars: float
    net_economic_profit_dollars: float
    daily_trajectory: pd.DataFrame


class RetailInventorySimulator:
    """
    Simulates physical store replenishment over a test horizon with lead times,
    order pipelines, shelf life spoilage, and daily demand fulfillment.
    """

    def __init__(
        self,
        lead_time_days: int = 3,
        unit_cost: float = 2.0,
        retail_price: float = 3.5,
        annual_holding_cost_rate: float = 0.22,  # 22% annual cost of capital & warehousing
        shelf_life_days: int = 30,
        order_frequency_days: int = 2,
    ):
        self.lead_time_days = lead_time_days
        self.unit_cost = unit_cost
        self.retail_price = retail_price
        self.margin_unit = retail_price - unit_cost
        self.daily_holding_cost_rate = annual_holding_cost_rate / 365.25
        self.shelf_life_days = shelf_life_days
        self.order_frequency_days = order_frequency_days

    def simulate_policy(
        self,
        policy_label: str,
        dates: pd.Series,
        actual_demands: np.ndarray,
        forecast_daily: np.ndarray,
        safety_stock: float,
        initial_inventory: float,
    ) -> PolicySimulationResult:
        """
        Runs daily inventory replenishment simulation for the given forecast and safety stock.
        """
        n_days = len(actual_demands)
        current_inv = max(initial_inventory, safety_stock + forecast_daily[0] * self.lead_time_days)
        
        # Track pipeline orders: list of dicts {arrival_day, qty, expiration_day}
        pipeline_orders: List[Dict[str, Any]] = []
        
        # FIFO inventory batches for expiration tracking {day_arrived, qty}
        inv_batches: List[Dict[str, float]] = [{"day_arrived": 0, "qty": current_inv}]

        # Daily trajectory logs
        log_dates = []
        log_demand = []
        log_sales = []
        log_lost = []
        log_inv = []
        log_order = []
        log_waste = []
        log_stockout = []

        stockout_count = 0
        total_waste_units = 0.0

        for t in range(n_days):
            today_date = dates.iloc[t] if hasattr(dates, "iloc") else dates[t]

            # 1. Receive arriving orders
            arrived_orders = [o for o in pipeline_orders if o["arrival_day"] == t]
            for order in arrived_orders:
                inv_batches.append({"day_arrived": t, "qty": order["qty"]})
            
            # Recalculate total current inventory
            current_inv = sum(b["qty"] for b in inv_batches)

            # 2. Check for shelf-life spoilage (merma)
            waste_today = 0.0
            surviving_batches = []
            for b in inv_batches:
                age = t - b["day_arrived"]
                if age >= self.shelf_life_days and self.shelf_life_days < 180:
                    waste_today += b["qty"]
                else:
                    surviving_batches.append(b)
            
            inv_batches = surviving_batches
            total_waste_units += waste_today
            current_inv = sum(b["qty"] for b in inv_batches)

            start_inv = current_inv
            dem = float(actual_demands[t])

            # 3. Fulfill demand FIFO
            if current_inv >= dem:
                sold = dem
                lost = 0.0
                is_so = 0
                # Deduct from batches FIFO
                remaining_demand = dem
                for b in inv_batches:
                    if b["qty"] >= remaining_demand:
                        b["qty"] -= remaining_demand
                        remaining_demand = 0.0
                        break
                    else:
                        remaining_demand -= b["qty"]
                        b["qty"] = 0.0
                inv_batches = [b for b in inv_batches if b["qty"] > 0]
                current_inv -= dem
            else:
                sold = current_inv
                lost = dem - current_inv
                is_so = 1
                stockout_count += 1
                inv_batches = []
                current_inv = 0.0

            # 4. Replenishment logic
            # Review inventory position = on_hand + on_order
            on_order = sum(o["qty"] for o in pipeline_orders if o["arrival_day"] > t)
            inventory_position = current_inv + on_order

            # Dynamic Order-Up-To based on forecast over LeadTime + ReviewPeriod
            horizon = self.lead_time_days + self.order_frequency_days
            future_demand_estimate = float(np.sum(forecast_daily[t : min(t + horizon, n_days)]))
            # If near end, extrapolate
            if min(t + horizon, n_days) - t < horizon:
                future_demand_estimate += forecast_daily[t] * (horizon - (min(t + horizon, n_days) - t))

            order_up_to_level = future_demand_estimate + safety_stock
            reorder_point = (forecast_daily[t] * self.lead_time_days) + safety_stock

            order_qty = 0.0
            if t % self.order_frequency_days == 0 and inventory_position < order_up_to_level:
                order_qty = max(0.0, order_up_to_level - inventory_position)
                pipeline_orders.append({
                    "arrival_day": t + self.lead_time_days,
                    "qty": order_qty,
                })

            # Append logs
            log_dates.append(today_date)
            log_demand.append(dem)
            log_sales.append(sold)
            log_lost.append(lost)
            log_inv.append(start_inv)
            log_order.append(order_qty)
            log_waste.append(waste_today)
            log_stockout.append(is_so)

        trajectory_df = pd.DataFrame({
            "date": log_dates,
            "demand": log_demand,
            "sales": log_sales,
            "lost_sales": log_lost,
            "start_inventory": log_inv,
            "order_placed": log_order,
            "waste_units": log_waste,
            "stockout": log_stockout,
        })

        tot_demand = float(np.sum(log_demand))
        tot_sales = float(np.sum(log_sales))
        tot_lost = float(np.sum(log_lost))
        avg_inv = float(np.mean(log_inv))

        revenue = tot_sales * self.retail_price
        gross_profit = tot_sales * self.margin_unit
        working_capital = avg_inv * self.unit_cost
        holding_cost = np.sum(np.array(log_inv) * self.unit_cost * self.daily_holding_cost_rate)
        waste_cost = total_waste_units * self.unit_cost
        net_profit = gross_profit - holding_cost - waste_cost

        fill_rate = (tot_sales / tot_demand * 100.0) if tot_demand > 0 else 100.0
        csl = ((n_days - stockout_count) / n_days * 100.0) if n_days > 0 else 100.0

        return PolicySimulationResult(
            policy_name=policy_label,
            total_sales_units=tot_sales,
            total_lost_sales_units=tot_lost,
            total_revenue=revenue,
            total_gross_profit=gross_profit,
            stockout_days=stockout_count,
            stockout_rate_pct=(stockout_count / n_days * 100.0),
            fill_rate_pct=fill_rate,
            cycle_service_level_pct=csl,
            avg_inventory_units=avg_inv,
            avg_working_capital_dollars=working_capital,
            total_holding_cost_dollars=float(holding_cost),
            total_waste_units=total_waste_units,
            total_waste_cost_dollars=waste_cost,
            net_economic_profit_dollars=float(net_profit),
            daily_trajectory=trajectory_df,
        )


def compare_champion_vs_challenger_financials(
    res_champion: PolicySimulationResult,
    res_challenger: PolicySimulationResult,
) -> Dict[str, Any]:
    """
    Computes direct executive delta metrics showing the exact dollarized value generated by Holt-Winters+.
    """
    delta_sales_units = res_challenger.total_sales_units - res_champion.total_sales_units
    delta_revenue = res_challenger.total_revenue - res_champion.total_revenue
    delta_gross_profit = res_challenger.total_gross_profit - res_champion.total_gross_profit
    delta_holding_cost = res_challenger.total_holding_cost_dollars - res_champion.total_holding_cost_dollars
    delta_waste_cost = res_challenger.total_waste_cost_dollars - res_champion.total_waste_cost_dollars
    delta_net_profit = res_challenger.net_economic_profit_dollars - res_champion.net_economic_profit_dollars
    delta_fill_rate = res_challenger.fill_rate_pct - res_champion.fill_rate_pct
    delta_stockout_days = res_challenger.stockout_days - res_champion.stockout_days
    delta_working_capital = res_challenger.avg_working_capital_dollars - res_champion.avg_working_capital_dollars

    return {
        "delta_sales_units": delta_sales_units,
        "delta_revenue_dollars": delta_revenue,
        "delta_gross_profit_dollars": delta_gross_profit,
        "delta_holding_cost_dollars": delta_holding_cost,
        "delta_waste_cost_dollars": delta_waste_cost,
        "delta_net_profit_dollars": delta_net_profit,
        "delta_fill_rate_pct": delta_fill_rate,
        "delta_stockout_days": delta_stockout_days,
        "delta_working_capital_dollars": delta_working_capital,
    }
