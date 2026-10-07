"""
Página 5: Capa 4: Simulador Financiero y de Inventario Retail (ROI en Dólares).
Demuestra el impacto económico real en capital de trabajo, merma, quiebres evitados y margen bruto.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app.ui_utils import (
    apply_custom_css,
    render_store_sku_selector,
    get_pipeline_result,
    PLOTLY_LAYOUT_DEFAULTS,
)
from core.inventory_engine import RetailInventorySimulator, compare_champion_vs_challenger_financials

st.set_page_config(page_title="5. Simulador ROI Retail | Holt-Winters+", page_icon="💰", layout="wide")
apply_custom_css()

store_id, sku_id, test_days, service_level, dataset_name = render_store_sku_selector()
result = get_pipeline_result(store_id, sku_id, test_days, service_level, dataset_name=dataset_name)

st.markdown(
    """
    <div class="main-header">
        <span class="challenger-badge">Capa 4: Traducción a P&L de Retail</span>
        <h2 style="margin:8px 0; color:#f8fafc;">Simulador Financiero & ROI de Inventario</h2>
        <p style="color:#94a3b8; margin:0;">
            El gerente comercial de un supermercado no compra WAPE ni RMSE; compra margen bruto recuperado, 
            rotación de capital de trabajo y reducción de mermas.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Interactive Financial Parameters
st.markdown("### 🎛️ Parámetros Financieros & Operacionales del SKU")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)

curr_step = 10.0 if dataset_name == "castano" else 0.1
curr_unit = "CLP" if dataset_name == "castano" else "USD"

with col_p1:
    unit_cost = st.number_input(
        f"Costo Unitario ({curr_unit}):",
        min_value=0.1,
        max_value=500000.0,
        value=float(result.unit_cost),
        step=curr_step,
    )
with col_p2:
    retail_price = st.number_input(
        f"Precio Venta ({curr_unit}):",
        min_value=0.2,
        max_value=1000000.0,
        value=float(result.retail_price),
        step=curr_step,
    )
with col_p3:
    holding_rate = st.slider("Tasa Anual de Almacenamiento / Capital (%):", min_value=10, max_value=35, value=22, step=1) / 100.0
with col_p4:
    n_stores_chain = st.number_input("Escala de Cadena (Nº Tiendas similares):", min_value=1, max_value=500, value=25, step=5)

margin_unit = retail_price - unit_cost
margin_pct = (margin_unit / retail_price * 100.0) if retail_price > 0 else 0.0

formatted_margin = f"${margin_unit:,.0f} CLP" if dataset_name == "castano" else f"${margin_unit:,.2f} USD"
st.caption(f"Margen Bruto Unitario: **{formatted_margin}** ({margin_pct:.1f}%) | Lead Time: **{result.lead_time_days} días** | Vida Útil: **{result.shelf_life_days} días**")

# Run Dynamic Simulation with user parameters
sim = RetailInventorySimulator(
    lead_time_days=result.lead_time_days,
    unit_cost=unit_cost,
    retail_price=retail_price,
    annual_holding_cost_rate=holding_rate,
    shelf_life_days=result.shelf_life_days,
    order_frequency_days=2,
)

test_df = result.test_df
actuals = test_df["true_demand"].values
initial_inv = float(actuals[0] * (result.lead_time_days + 2))

sim_champ = sim.simulate_policy(
    policy_label="Champion (Holt-Winters)",
    dates=test_df["date"],
    actual_demands=actuals,
    forecast_daily=result.hw_champion_result.forecast_values.values,
    safety_stock=result.conformal_result.gaussian_safety_stock,
    initial_inventory=initial_inv,
)

sim_chall = sim.simulate_policy(
    policy_label="Challenger (Holt-Winters+)",
    dates=test_df["date"],
    actual_demands=actuals,
    forecast_daily=test_df["pred_challenger"].values,
    safety_stock=result.conformal_result.conformal_safety_stock,
    initial_inventory=initial_inv,
)

deltas = compare_champion_vs_challenger_financials(sim_champ, sim_chall)

is_clp = (dataset_name == "castano")
curr_code = "CLP" if is_clp else "USD"

def fmt_diff_money(val: float, signed: bool = True) -> str:
    abs_v = abs(val)
    if is_clp:
        formatted = f"${abs_v:,.0f} CLP"
    else:
        formatted = f"${abs_v:,.2f} USD"
    if signed:
        sign = "+" if val > 0 else ("-" if val < 0 else "")
        return f"{sign}{formatted}"
    return formatted

def fmt_diff_units(val: float, signed: bool = True) -> str:
    abs_v = abs(val)
    if signed:
        sign = "+" if val > 0 else ("-" if val < 0 else "")
        return f"{sign}{abs_v:,.0f} u"
    return f"{abs_v:,.0f} u"

# Impact Ribbon for 1 Store
st.markdown(f"### 💵 Impacto Económico en {result.store_name} ({test_days} Días)")

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric(
        "Ventas Adicionales Recuperadas",
        fmt_diff_units(deltas['delta_sales_units']),
        f"Quiebres: {sim_chall.stockout_days} vs {sim_champ.stockout_days} días",
    )

with r2:
    st.metric(
        "Margen Bruto Ganado",
        fmt_diff_money(deltas['delta_gross_profit_dollars']),
        f"{fmt_diff_money(deltas['delta_revenue_dollars'])} en ventas",
    )

with r3:
    st.metric(
        "Delta Costo Almacenamiento",
        fmt_diff_money(deltas['delta_holding_cost_dollars']),
        f"Inventario prom: {sim_chall.avg_inventory_units:.0f} vs {sim_champ.avg_inventory_units:.0f} u",
        delta_color="inverse" if deltas['delta_holding_cost_dollars'] > 0 else "normal",
    )

with r4:
    st.metric(
        "Beneficio Económico Neto",
        fmt_diff_money(deltas['delta_net_profit_dollars']),
        f"Fill Rate: {sim_chall.fill_rate_pct:.1f}% vs {sim_champ.fill_rate_pct:.1f}%",
        delta_color="normal",
    )

# Day-by-Day Inventory Trajectory Comparison
st.markdown("### 📦 Simulación de Trayectoria de Inventario en Góndola")

traj_champ = sim_champ.daily_trajectory
traj_chall = sim_chall.daily_trajectory

fig_traj = go.Figure()

# Champion Inventory
fig_traj.add_trace(
    go.Scatter(
        x=traj_champ["date"],
        y=traj_champ["start_inventory"],
        name="Inventario Champion (HW + Gauss)",
        line=dict(color="#f59e0b", width=1.5, dash="dash"),
    )
)

# Challenger Inventory
fig_traj.add_trace(
    go.Scatter(
        x=traj_chall["date"],
        y=traj_chall["start_inventory"],
        name="Inventario Challenger (HW+ Conformal)",
        line=dict(color="#38bdf8", width=2.5),
    )
)

# Stockout points Champion
oos_champ_pts = traj_champ[traj_champ["stockout"] == 1]
if len(oos_champ_pts) > 0:
    fig_traj.add_trace(
        go.Scatter(
            x=oos_champ_pts["date"],
            y=oos_champ_pts["start_inventory"],
            name="Quiebre de Stock en Champion",
            mode="markers",
            marker=dict(color="#ef4444", size=10, symbol="triangle-down"),
        )
    )

fig_traj.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    title=dict(text="Evolución del Stock Físico Diario en Tienda (Góndola)", x=0.01),
    yaxis_title="Unidades en Inventario",
    xaxis_title="Fecha",
    height=400,
)

st.plotly_chart(fig_traj, use_container_width=True)

# Chain-Wide Projection Extrapolation
st.markdown(f"### 🌐 Extrapolación a Nivel Cadena ({n_stores_chain} Sucursales)")

annual_factor = (365.25 / test_days)
annual_net_1_store = deltas["delta_net_profit_dollars"] * annual_factor
chain_annual_net = annual_net_1_store * n_stores_chain
chain_lost_sales_recovered = deltas["delta_sales_units"] * annual_factor * n_stores_chain

col_c1, col_c2 = st.columns([1, 1])

with col_c1:
    portfolio_estimate = fmt_diff_money(chain_annual_net * 30, signed=False)
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid #38bdf8;">
            <h3 style="color:#38bdf8; margin-top:0;">Impacto Anualizado Proyectado en la Cadena</h3>
            <p style="font-size:1.1rem; color:#f8fafc; font-weight:600; margin-bottom:8px;">
                Beneficio Neto Adicional: <span style="color:#10b981; font-size:1.4rem;">{fmt_diff_money(chain_annual_net)}</span> / año
            </p>
            <p style="font-size:0.95rem; color:#cbd5e1; margin-bottom:4px;">
                • Unidades de venta recuperadas: <strong>{fmt_diff_units(chain_lost_sales_recovered)}</strong> / año para este SKU.
            </p>
            <p style="font-size:0.95rem; color:#cbd5e1; margin-bottom:4px;">
                • Si se implementa sobre un portafolio de <strong>50 SKUs de alta rotación</strong>, el valor creado supera los 
                <strong style="color:#38bdf8;">{portfolio_estimate} anuales</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_c2:
    # Breakdown Bar Chart
    waterfall_df = pd.DataFrame({
        "Concepto": ["Margen Ventas Recuperadas", "Costo Almacenamiento", "Beneficio Neto"],
        "Monto": [
            deltas["delta_gross_profit_dollars"] * annual_factor * n_stores_chain,
            -deltas["delta_holding_cost_dollars"] * annual_factor * n_stores_chain,
            chain_annual_net,
        ],
    })
    
    text_labels = [fmt_diff_money(v) for v in waterfall_df["Monto"]]
    fig_wf = go.Figure(
        go.Bar(
            x=waterfall_df["Concepto"],
            y=waterfall_df["Monto"],
            marker_color=["#10b981", "#ef4444", "#38bdf8"],
            text=text_labels,
            textposition="auto",
        )
    )
    fig_wf.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text=f"Desglose Financiero Anual ({n_stores_chain} Tiendas)", x=0.01),
        yaxis_title=f"Monto ({curr_code})",
        height=260,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# Export Data Button
st.markdown("---")
st.markdown("### 📥 Descargar Reporte de Auditoría")

csv_data = traj_chall.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar Simulación Diaria de Inventario (CSV)",
    data=csv_data,
    file_name=f"simulacion_inventario_{sku_id}_{store_id}.csv",
    mime="text/csv",
)
