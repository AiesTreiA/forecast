"""
Holt-Winters+ | Home & Executive Dashboard.
Main landing page for pitching to retail demand forecasting experts.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from app.ui_utils import (
    apply_custom_css,
    render_store_sku_selector,
    get_pipeline_result,
    PLOTLY_LAYOUT_DEFAULTS,
)

# Page configuration
st.set_page_config(
    page_title="Holt-Winters+ | Retail Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# Sidebar Selectors
store_id, sku_id, test_days, service_level, dataset_name = render_store_sku_selector()

# Run Pipeline
result = get_pipeline_result(
    store_id=store_id,
    sku_id=sku_id,
    test_days=test_days,
    service_level=service_level,
    dataset_name=dataset_name,
)

# Main Title & Executive Pitch Narrative
st.markdown(
    """
    <div class="main-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span class="challenger-badge">Arquitectura Champion vs Challenger</span>
                <h1 style="margin: 8px 0 4px 0; color:#f8fafc; font-size:2.2rem; font-weight:700;">
                    Holt-Winters<span style="color:#38bdf8;">+</span>
                </h1>
                <p style="color:#94a3b8; margin:0; font-size:1.05rem;">
                    Amplificación auditable y no invasiva de modelos de suavizamiento exponencial para grandes cadenas de supermercados.
                </p>
            </div>
            <div style="text-align:right;">
                <span class="champion-badge">Champion: Holt-Winters + Corrección OOS</span><br>
                <span style="font-size:0.85rem; color:#64748b;">Auditoría matemática en tiempo real</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Executive Pitch Context Alert
with st.expander("📌 La Tesis de Holt-Winters+: ¿Por qué NO reemplazar tu modelo actual?", expanded=False):
    st.markdown(
        r"""
        > **El modelo de Holt-Winters es un estándar de la industria por una razón:** es matemáticamente elegante, 
        estable ante ruido, y captura perfectamente el nivel base, la tendencia y la estacionalidad semanal del retail.
        
        Sin embargo, las cadenas de retail modernas se enfrentan a **tres límites matemáticos** que Holt-Winters no puede resolver solo:
        1. **Promociones y Elasticidad**: HW proyecta el patrón estacional fijo; no puede anticipar un descuento del 25% o un 2x1 el próximo fin de semana.
        2. **Efecto Calendario & Quincenas**: Los picos de pago (días 15 y 30) y feriados no siguen un ciclo semanal de 7 días.
        3. **Incertidumbre No-Gaussiana en Colas**: Dimensionar stock de seguridad asumiendo distribución normal (\(Z \cdot \sigma\)) provoca quiebres masivos en picos de demanda y sobrestock en días valle.

        **Nuestra Solución (Challenger):** Mantenemos **tu Holt-Winters intacto como ancla baseline**, y añadimos capas auditables 
        (Uncensoring corregido, Booster de residuos vía LightGBM con SHAP, e Intervalos Conformal). 
        Si no hay promociones activas, el sistema regresa exactamente a tu Holt-Winters.
        """
    )

# Executive KPI Ribbon
st.markdown("### 🏆 Rendimiento Comparativo en Test (Últimos 60 Días)")

col1, col2, col3, col4 = st.columns(4)

wape_hw = float(result.metrics_summary_table.loc[result.metrics_summary_table["Métrica"] == "WAPE (%)", "Champion (HW)"].values[0].replace("%", ""))
wape_ch = float(result.metrics_summary_table.loc[result.metrics_summary_table["Métrica"] == "WAPE (%)", "Challenger (HW+)"].values[0].replace("%", ""))
delta_wape = wape_hw - wape_ch

with col1:
    st.metric(
        label="🎯 Error Global (WAPE)",
        value=f"{wape_ch:.1f}%",
        delta=f"-{delta_wape:.1f} pp error",
        delta_color="normal",
        help="Weighted Absolute Percentage Error en período fuera de muestra. Menor es mejor.",
    )

with col2:
    fill_champ = result.sim_champion.fill_rate_pct
    fill_chall = result.sim_challenger.fill_rate_pct
    delta_fill = fill_chall - fill_champ
    st.metric(
        label="📦 Nivel de Servicio (Fill Rate)",
        value=f"{fill_chall:.1f}%",
        delta=f"+{delta_fill:.1f} pp servicio",
        delta_color="normal",
        help="% de unidades de demanda real satisfechas inmediatamente desde inventario en góndola.",
    )

with col3:
    oos_champ = result.sim_champion.stockout_days
    oos_chall = result.sim_challenger.stockout_days
    delta_oos = oos_chall - oos_champ
    st.metric(
        label="🚫 Días con Quiebre (OOS)",
        value=f"{oos_chall} días",
        delta=f"{delta_oos:d} días",
        delta_color="inverse",
        help="Días en que el inventario físico llegó a 0 y se perdieron ventas.",
    )

with col4:
    net_roi = result.financial_deltas["delta_net_profit_dollars"]
    gross_roi = result.financial_deltas["delta_gross_profit_dollars"]
    unit_label = "CLP" if dataset_name == "castano" else "USD"
    net_val_str = f"+${net_roi:,.0f} {unit_label}" if net_roi >= 0 else f"-${abs(net_roi):,.0f} {unit_label}"
    gross_val_str = f"+${gross_roi:,.0f}" if gross_roi >= 0 else f"-${abs(gross_roi):,.0f}"
    st.metric(
        label=f"💰 Beneficio Neto ({unit_label})",
        value=net_val_str,
        delta=f"{gross_val_str} margen bruto",
        delta_color="normal" if net_roi >= 0 else "inverse",
        help="Margen bruto adicional recuperado descontando costo de capital de inventario y mermas.",
    )

# Interactive Comparison Forecast Chart
st.markdown(f"### 📈 Pronóstico Fuera de Muestra: **{result.sku_name}** ({result.store_name})")

test_df = result.test_df

fig = go.Figure()

# Actual demand
fig.add_trace(
    go.Scatter(
        x=test_df["date"],
        y=test_df["true_demand"],
        name="Demanda Real (Ground Truth)",
        line=dict(color="#f8fafc", width=2.5),
        mode="lines+markers",
        marker=dict(size=4),
    )
)

# Champion Holt-Winters
fig.add_trace(
    go.Scatter(
        x=test_df["date"],
        y=test_df["pred_hw_champion"],
        name="Champion (Holt-Winters Clásico)",
        line=dict(color="#94a3b8", width=2, dash="dash"),
    )
)

# Challenger Holt-Winters+
fig.add_trace(
    go.Scatter(
        x=test_df["date"],
        y=test_df["pred_challenger"],
        name="Challenger (Holt-Winters+ Amplificado)",
        line=dict(color="#38bdf8", width=3),
    )
)

# Highlight active promotions
promo_days = test_df[test_df["is_promo"] == 1]
if len(promo_days) > 0:
    fig.add_trace(
        go.Scatter(
            x=promo_days["date"],
            y=promo_days["true_demand"],
            mode="markers",
            name="Período con Promoción Activa",
            marker=dict(color="#f59e0b", size=9, symbol="star"),
        )
    )

fig.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    title=dict(text="Comparativa Dinámica: Champion vs Challenger en Test Horizon", x=0.01),
    yaxis_title="Unidades de Venta / Día",
    xaxis_title="Fecha",
    height=450,
)

st.plotly_chart(fig, use_container_width=True)

# Full Scorecard Table
st.markdown("### 📋 Cuadro de Mando Comparativo Integral (Champion vs Challenger)")
st.dataframe(
    result.metrics_summary_table,
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# Architectural Stepper Cards
st.markdown("### 🧱 Las 4 Capas de Amplificación Auditables")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
        <div class="metric-card">
            <h4 style="color:#38bdf8; margin-top:0;">1. Uncensoring</h4>
            <p style="font-size:0.9rem; color:#cbd5e1;">
                Detección y corrección de ventas perdidas por quiebres de stock. Compensa la demanda latente en promociones.
            </p>
            <a href="/Datos_y_Stockouts" style="color:#0284c7; font-weight:600; font-size:0.85rem;">Explorar Capa 1 →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="metric-card">
            <h4 style="color:#38bdf8; margin-top:0;">2. Residual Booster</h4>
            <p style="font-size:0.9rem; color:#cbd5e1;">
                LightGBM sobre los residuos de HW. Captura elasticidad de descuento, tipo de promo y quincenas con SHAP 100% auditable.
            </p>
            <a href="/Residual_Boosting" style="color:#0284c7; font-weight:600; font-size:0.85rem;">Explorar Capa 2 →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="metric-card">
            <h4 style="color:#38bdf8; margin-top:0;">3. Conformal UQ</h4>
            <p style="font-size:0.9rem; color:#cbd5e1;">
                Cuantificación de incertidumbre sin asumir distribución normal. Garantía finita de cobertura para dimensionar Safety Stock.
            </p>
            <a href="/Conformal_SafetyStock" style="color:#0284c7; font-weight:600; font-size:0.85rem;">Explorar Capa 3 →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div class="metric-card">
            <h4 style="color:#38bdf8; margin-top:0;">4. Simulador ROI</h4>
            <p style="font-size:0.9rem; color:#cbd5e1;">
                Simulación diaria de inventario, merma por vencimiento, capital de trabajo inmovilizado y ROI proyectado a nivel cadena.
            </p>
            <a href="/Simulador_ROI_Retail" style="color:#0284c7; font-weight:600; font-size:0.85rem;">Explorar Capa 4 →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
