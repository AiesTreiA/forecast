"""
Página 4: Capa 3: Conformal Prediction y Dimensionamiento de Stock de Seguridad.
Demuestra la debilidad de asumir distribución normal en retail y la garantía de cobertura de Conformal Prediction.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy import stats

from app.ui_utils import (
    apply_custom_css,
    render_store_sku_selector,
    get_pipeline_result,
    PLOTLY_LAYOUT_DEFAULTS,
)
from core.conformal_uq import ConformalPredictor

st.set_page_config(page_title="4. Conformal Safety Stock | Holt-Winters+", page_icon="🛡️", layout="wide")
apply_custom_css()

store_id, sku_id, test_days, service_level = render_store_sku_selector()
result = get_pipeline_result(store_id, sku_id, test_days, service_level)
conf_res = result.conformal_result
test_df = result.test_df

st.markdown(
    r"""
    <div class="main-header">
        <span class="challenger-badge">Capa 3: Cuantificación de Incertidumbre sin Supuestos</span>
        <h2 style="margin:8px 0; color:#f8fafc;">Conformal Prediction vs Stock de Seguridad Gaussiano</h2>
        <p style="color:#94a3b8; margin:0;">
            La teoría clásica de inventario asume que los errores de pronóstico siguen una campana de Gauss normal. 
            En el retail real, las promociones generan colas pesadas (kurtosis alta). Conformal Prediction ofrece una garantía matemática de cobertura (\(1 - \alpha\)) sin supuestos de distribución.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Comparison Metric Cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Nivel de Servicio Objetivo",
        f"{conf_res.coverage_target * 100:.1f}%",
        help="Probabilidad exigida de no tener quiebre de stock en el ciclo.",
    )

with c2:
    st.metric(
        "Cobertura Gaussiana Lograda",
        f"{conf_res.empirical_coverage_gaussian * 100:.1f}%",
        f"{(conf_res.empirical_coverage_gaussian - conf_res.coverage_target)*100:.1f} pp desvío",
        delta_color="inverse",
        help="Porcentaje real de días dentro del intervalo gaussiano clásico.",
    )

with c3:
    st.metric(
        "Cobertura Conformal Lograda",
        f"{conf_res.empirical_coverage_conformal * 100:.1f}%",
        f"{(conf_res.empirical_coverage_conformal - conf_res.coverage_target)*100:+.1f} pp meta",
        delta_color="normal",
        help="Porcentaje real de días dentro del intervalo Conformal distribution-free.",
    )

with c4:
    delta_ss = conf_res.conformal_safety_stock - conf_res.gaussian_safety_stock
    st.metric(
        "Stock de Seguridad Recomendado",
        f"{conf_res.conformal_safety_stock:.1f} u",
        f"{delta_ss:+.1f} u vs Gauss",
        help="Unidades de stock de seguridad requeridas para cubrir el Lead Time.",
    )

# The Heavy Tail Explanation
st.markdown("### ⚠️ La Trampa de la Campana de Gauss en Retail")

train_residuals = result.train_df["challenger_demand"].values - (
    result.hw_champion_result.fitted_values.values + result.booster_result.train_predicted_residuals
)

col_chart1, col_chart2 = st.columns([1, 1])

with col_chart1:
    # Histogram of residuals vs normal curve
    fig_hist = go.Figure()
    
    # Histogram
    fig_hist.add_trace(
        go.Histogram(
            x=train_residuals,
            histnorm="probability density",
            name="Residuos Reales Retail",
            marker_color="#38bdf8",
            opacity=0.7,
            nbinsx=40,
        )
    )

    # Theoretical Normal Curve
    mu = np.mean(train_residuals)
    sigma = np.std(train_residuals)
    x_axis = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 200)
    y_norm = stats.norm.pdf(x_axis, mu, sigma)
    
    fig_hist.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_norm,
            mode="lines",
            name="Supuesto Teórico Gaussiano",
            line=dict(color="#ef4444", width=2.5, dash="dash"),
        )
    )

    fig_hist.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text="Distribución Empírica de Residuos vs Supuesto Normal", x=0.01),
        xaxis_title="Error Residual (Unidades)",
        yaxis_title="Densidad",
        height=340,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_chart2:
    kurt = float(stats.kurtosis(train_residuals))
    skew = float(stats.skew(train_residuals))
    st.markdown(
        rf"""
        <div class="metric-card" style="margin-top: 15px;">
            <h4 style="color:#f59e0b; margin-top:0;">Diagnóstico Estadístico de la Demanda</h4>
            <p style="font-size:0.9rem; color:#cbd5e1;">
                Observe las colas extendidas hacia la derecha provocadas por promociones y fines de semana:
            </p>
            <ul style="font-size:0.85rem; color:#94a3b8;">
                <li><strong>Asimetría (Skewness):</strong> {skew:+.2f} (sesgo positivo hacia picos de demanda).</li>
                <li><strong>Kurtosis Excedente:</strong> {kurt:+.2f} (colas significativamente más pesadas que una normal).</li>
            </ul>
            <p style="font-size:0.9rem; color:#f8fafc;">
                <strong>Consecuencia Práctica:</strong> Al dimensionar inventario con la fórmula clásica \(Z_{{0.95}} \cdot \sigma \cdot \sqrt{{L}}\), 
                el supermercado se queda sin stock justamente en los fines de semana de mayor margen. 
                Conformal Prediction computa el cuantil empírico real <strong>\(q = {conf_res.conformal_quantile_q:.1f}\) unidades</strong>, garantizando protección real.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Prediction Intervals in Test
st.markdown("### 📈 Intervalos de Predicción y Protección en Góndola (Horizonte Test)")

fig_int = go.Figure()

# Conformal Interval (Shaded Band)
fig_int.add_trace(
    go.Scatter(
        x=test_df["date"],
        y=test_df["conformal_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    )
)
fig_int.add_trace(
    go.Scatter(
        x=test_df["date"],
        y=test_df["conformal_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(56, 189, 248, 0.15)",
        name=f"Banda Conformal ({conf_res.coverage_target*100:.0f}% Cobertura Garantizada)",
    )
)

# Gaussian Interval (Dashed lines)
fig_int.add_trace(
    go.Scatter(
        x=test_df["date"],
        y=test_df["gaussian_upper"],
        name="Límite Superior Gaussiano Teórico",
        line=dict(color="#ef4444", width=1.5, dash="dot"),
    )
)

# Challenger Forecast
fig_int.add_trace(
    go.Scatter(
        x=test_df["date"],
        y=test_df["pred_challenger"],
        name="Pronóstico Puntual Challenger",
        line=dict(color="#38bdf8", width=2),
    )
)

# Real Demands
fig_int.add_trace(
    go.Scatter(
        x=test_df["date"],
        y=test_df["true_demand"],
        name="Demanda Real Observada",
        mode="lines+markers",
        marker=dict(color="#f8fafc", size=4),
        line=dict(color="#f8fafc", width=1.5),
    )
)

fig_int.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    title=dict(text="Evaluación de Cobertura en Test: Conformal vs Gaussiano", x=0.01),
    yaxis_title="Unidades de Demanda / Día",
    xaxis_title="Fecha",
    height=450,
)

st.plotly_chart(fig_int, use_container_width=True)
