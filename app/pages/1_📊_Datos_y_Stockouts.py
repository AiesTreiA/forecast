"""
Página 1: Auditoría de Datos y Corrección de Quiebres de Stock (Uncensoring).
Demuestra la reconstrucción de la demanda latente no observada cuando el inventario llega a cero.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
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

st.set_page_config(page_title="1. Datos & Stockouts | Holt-Winters+", page_icon="📊", layout="wide")
apply_custom_css()

store_id, sku_id, test_days, service_level = render_store_sku_selector()
result = get_pipeline_result(store_id, sku_id, test_days, service_level)

st.markdown(
    """
    <div class="main-header">
        <span class="challenger-badge">Capa 1: Reconstrucción de Demanda Latente</span>
        <h2 style="margin:8px 0; color:#f8fafc;">Auditoría de Ventas Perdidas por Quiebre de Stock (OOS)</h2>
        <p style="color:#94a3b8; margin:0;">
            Cuando la góndola se queda sin stock (Inventario = 0), las ventas observadas caen a cero. 
            Alimentar un modelo de series temporales con ceros artificiales destruye el factor estacional.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Narrative & Method Explanation
col_intro1, col_intro2 = st.columns([1, 1])

with col_intro1:
    st.info(
        """
        **Método Champion (Tu Método Actual):**
        - Identifica días con `stockout_flag == 1` o ventas=0 con inventario=0.
        - Reemplaza el valor con la **mediana histórica del mismo día de la semana** en semanas recientes.
        - *Ventaja:* Simple, auditable, elimina los ceros que distorsionan el suavizamiento.
        """
    )

with col_intro2:
    st.success(
        """
        **Método Challenger (Holt-Winters+ Amplificado):**
        - Considera el **nivel local suavizado (EWMA)** + patrón de día de semana.
        - **Compensación por Promoción Activa:** Si el quiebre ocurrió durante un descuento del 20% o un 2x1, 
          la demanda real era significativamente mayor al promedio regular del día.
        """
    )

# Metrics Cards
m = result.stockout_audit_metrics
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Días Analizados", f"{m['total_days']:,} días")
with c2:
    st.metric("Días con Quiebre (OOS)", f"{m['stockout_days']} días", f"{m['stockout_rate_pct']:.2f}% del tiempo")
with c3:
    st.metric("Unidades Imputadas (Champion)", f"+{m['champion_imputed_units']:,.0f} u")
with c4:
    st.metric(
        "Unidades Imputadas (Challenger)",
        f"+{m['challenger_imputed_units']:,.0f} u",
        f"{m['challenger_imputed_units'] - m['champion_imputed_units']:+,.0f} u más detectadas",
    )

# Interactive Chart
st.markdown("### 🔍 Visualización de Reconstrucción de Demanda en Días de Quiebre")

# Show recent 180 days for clarity
df_vis = result.full_df.tail(180).copy()

fig = go.Figure()

# 1. Observed Sales
fig.add_trace(
    go.Scatter(
        x=df_vis["date"],
        y=df_vis["observed_sales"],
        name="Ventas Observadas en POS",
        line=dict(color="#64748b", width=1.5),
        mode="lines",
    )
)

# 2. Champion Imputation
fig.add_trace(
    go.Scatter(
        x=df_vis["date"],
        y=df_vis["champion_demand"],
        name="Demanda Corregida (Champion - Mediana DOW)",
        line=dict(color="#f59e0b", width=2, dash="dash"),
    )
)

# 3. Challenger Imputation
fig.add_trace(
    go.Scatter(
        x=df_vis["date"],
        y=df_vis["challenger_demand"],
        name="Demanda Corregida (Challenger - Trend + Promo Lift)",
        line=dict(color="#38bdf8", width=2.5),
    )
)

# 4. Stockout markers
oos_points = df_vis[df_vis["stockout_flag"] == 1]
if len(oos_points) > 0:
    fig.add_trace(
        go.Scatter(
            x=oos_points["date"],
            y=oos_points["observed_sales"],
            name="Evento de Quiebre de Stock (OOS)",
            mode="markers",
            marker=dict(color="#ef4444", size=10, symbol="x"),
        )
    )

fig.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    title=dict(text="Impacto de la Corrección de Quiebres en la Serie de Entrenamiento (Últimos 180 días)", x=0.01),
    yaxis_title="Unidades de Demanda / Día",
    xaxis_title="Fecha",
    height=480,
)

st.plotly_chart(fig, use_container_width=True)

# Audit Table of Stockout Days
st.markdown("### 📋 Tabla de Auditoría en Días de Quiebre (Muestra de Eventos OOS)")

oos_table = (
    result.full_df[result.full_df["stockout_flag"] == 1][
        [
            "date",
            "day_name",
            "observed_sales",
            "is_promo",
            "discount_pct",
            "champion_demand",
            "challenger_demand",
            "true_demand",
        ]
    ]
    .tail(15)
    .copy()
)

oos_table["diff_vs_champion"] = oos_table["challenger_demand"] - oos_table["champion_demand"]

st.dataframe(
    oos_table.rename(
        columns={
            "date": "Fecha",
            "day_name": "Día",
            "observed_sales": "Venta Observada",
            "is_promo": "En Promo",
            "discount_pct": "Desc. %",
            "champion_demand": "Corrección Champion",
            "challenger_demand": "Corrección Challenger",
            "true_demand": "Demanda Real Latente",
            "diff_vs_champion": "Delta Challenger (u)",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

if "champion_wape_on_oos" in m:
    st.markdown(
        f"""
        > 💡 **Conclusión Clave para el Experto:**
        > En los días de quiebre, el método Champion comete un error WAPE de **{m['champion_wape_on_oos']:.2f}%**, 
        > mientras que la corrección sensible a promociones del Challenger reduce el error a **{m['challenger_wape_on_oos']:.2f}%** 
        > (una mejora directa de **{m['champion_wape_on_oos'] - m['challenger_wape_on_oos']:.2f} puntos porcentuales**).
        """
    )
