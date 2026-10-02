"""
Página 2: El Modelo Champion: Holt-Winters (Suavizamiento Exponencial).
Auditoría profunda de la formulación matemática de Holt-Winters con estacionalidad semanal (m=7).
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from app.ui_utils import (
    apply_custom_css,
    render_store_sku_selector,
    get_pipeline_result,
    PLOTLY_LAYOUT_DEFAULTS,
)

st.set_page_config(page_title="2. Holt-Winters Champion | Holt-Winters+", page_icon="📈", layout="wide")
apply_custom_css()

store_id, sku_id, test_days, service_level, dataset_name = render_store_sku_selector()
result = get_pipeline_result(store_id, sku_id, test_days, service_level, dataset_name=dataset_name)
hw_res = result.hw_champion_result

st.markdown(
    r"""
    <div class="main-header">
        <span class="champion-badge">El Modelo Champion de Referencia</span>
        <h2 style="margin:8px 0; color:#f8fafc;">Auditoría Matemática: Holt-Winters Clásico</h2>
        <p style="color:#94a3b8; margin:0;">
            Descomposición en Nivel (\(\alpha\)), Tendencia (\(\beta\)) y Estacionalidad Semanal (\(\gamma\), período \(m=7\)).
            Validamos su precisión y demostramos exactamente dónde se producen sus residuos.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mathematical Formulation Expander
with st.expander("📐 Ecuaciones del Modelo Champion (Holt-Winters Aditivo con Estacionalidad Semanal)", expanded=False):
    st.latex(r"\ell_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1}) \quad \text{[Nivel]}")
    st.latex(r"b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta)b_{t-1} \quad \text{[Tendencia]}")
    st.latex(r"s_t = \gamma (y_t - \ell_t) + (1 - \gamma)s_{t-m} \quad \text{[Estacionalidad } m=7\text{]}")
    st.latex(r"\hat{y}_{t+h|t} = \ell_t + h \cdot b_t + s_{t+h-m(k+1)} \quad \text{[Pronóstico } h\text{-pasos]}")

# Parameter KPI Cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Alpha (Nivel)",
        f"{hw_res.alpha:.4f}",
        help="Velocidad de adaptación ante cambios en el nivel de demanda base.",
    )

with c2:
    st.metric(
        "Beta (Tendencia)",
        f"{hw_res.beta:.4f}",
        help="Velocidad de ajuste de la pendiente de crecimiento o decrecimiento.",
    )

with c3:
    st.metric(
        "Gamma (Estacionalidad)",
        f"{hw_res.gamma:.4f}",
        help="Ajuste de los índices estacionales del ciclo semanal (lunes a domingo).",
    )

with c4:
    st.metric(
        "WAPE In-Sample (Train)",
        f"{hw_res.training_metrics['WAPE']:.2f}%",
        f"MAE: {hw_res.training_metrics['MAE']:.1f} u",
        help="Error promedio ponderado sobre los datos de entrenamiento corregidos.",
    )

# Decomposed Components Plot
st.markdown("### 🧩 Descomposición de Componentes de Holt-Winters (Entrenamiento)")

train_df = result.train_df
dates_train = train_df["date"]

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=(
        "Demanda Histórica Corregida vs Ajuste Holt-Winters",
        "Componente de Nivel Base (Level)",
        "Componente de Tendencia (Trend)",
        "Índice Estacional Semanal (Seasonal m=7)",
    ),
)

# 1. Observed vs Fitted
fig.add_trace(
    go.Scatter(x=dates_train, y=train_df["champion_demand"], name="Demanda Corregida", line=dict(color="#64748b", width=1.5)),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=dates_train, y=hw_res.fitted_values, name="Ajuste Holt-Winters", line=dict(color="#f59e0b", width=2)),
    row=1,
    col=1,
)

# 2. Level
fig.add_trace(
    go.Scatter(x=dates_train, y=hw_res.level, name="Nivel (Level)", line=dict(color="#38bdf8", width=1.5)),
    row=2,
    col=1,
)

# 3. Trend
fig.add_trace(
    go.Scatter(x=dates_train, y=hw_res.trend, name="Tendencia (Trend)", line=dict(color="#a855f7", width=1.5)),
    row=3,
    col=1,
)

# 4. Seasonal
fig.add_trace(
    go.Scatter(x=dates_train, y=hw_res.season, name="Estacionalidad Semanal", line=dict(color="#10b981", width=1.5)),
    row=4,
    col=1,
)

fig.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    height=700,
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

# Weekly Seasonal Profile Bar Chart
st.markdown("### 📅 Perfil Semanal Extraído por Holt-Winters")

dow_labels = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
# Extract last 7 seasonal values
seasonal_pattern = hw_res.season.values[-7:]
dow_df = pd.DataFrame({"Día": dow_labels, "Impacto Estacional (u)": seasonal_pattern})

fig_dow = go.Figure(
    go.Bar(
        x=dow_df["Día"],
        y=dow_df["Impacto Estacional (u)"],
        marker_color=["#0284c7" if v >= 0 else "#f97316" for v in dow_df["Impacto Estacional (u)"]],
        text=[f"{v:+.1f} u" for v in dow_df["Impacto Estacional (u)"]],
        textposition="auto",
    )
)
fig_dow.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    title=dict(text="Aporte Aditivo por Día de la Semana respecto al Nivel Base", x=0.01),
    yaxis_title="Delta Unidades vs Nivel Base",
    height=320,
)
st.plotly_chart(fig_dow, use_container_width=True)

# The Residual Analysis (Bridge to Challenger Layer 2)
st.markdown("### 🔍 El Diagnóstico: ¿Por qué quedan residuos en Holt-Winters?")

residuals_train = train_df["champion_demand"].values - hw_res.fitted_values.values

col_res1, col_res2 = st.columns([2, 1])

with col_res1:
    fig_res = go.Figure()
    fig_res.add_trace(
        go.Scatter(
            x=dates_train,
            y=residuals_train,
            name="Residuos (Demanda Real - Pronóstico HW)",
            line=dict(color="#ef4444", width=1),
        )
    )
    fig_res.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig_res.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text="Serie Temporal de Residuos de Holt-Winters en Entrenamiento", x=0.01),
        yaxis_title="Residuo (Unidades)",
        height=320,
    )
    st.plotly_chart(fig_res, use_container_width=True)

with col_res2:
    st.markdown(
        """
        <div class="metric-card" style="margin-top:20px;">
            <h4 style="color:#ef4444; margin-top:0;">Los Picos en los Residuos</h4>
            <p style="font-size:0.9rem; color:#cbd5e1;">
                Observe que los residuos <strong>no son ruido blanco aleatorio</strong>. Presentan picos agudos recurrentes que coinciden con:
            </p>
            <ul style="font-size:0.85rem; color:#94a3b8;">
                <li>Campañas promocionales (2x1, catálogo).</li>
                <li>Días de pago de sueldos (días 15 y 30).</li>
                <li>Vísperas de feriados nacionales.</li>
            </ul>
            <p style="font-size:0.9rem; color:#38bdf8;">
                👉 <strong>Aquí es donde entra la Capa 2:</strong> En lugar de reemplazar HW, entrenamos un modelo que aprende a predecir exactamente estos picos residuales.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
