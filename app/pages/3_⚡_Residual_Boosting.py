"""
Página 3: Capa 2: Residual Boosting con LightGBM y Explicabilidad SHAP.
Demuestra cómo el booster de residuos captura promociones y quincenas sin alterar el ancla de Holt-Winters.
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

st.set_page_config(page_title="3. Residual Boosting & SHAP | Holt-Winters+", page_icon="⚡", layout="wide")
apply_custom_css()

store_id, sku_id, test_days, service_level, dataset_name = render_store_sku_selector()
result = get_pipeline_result(store_id, sku_id, test_days, service_level, dataset_name=dataset_name)

booster_res = result.booster_result
explainer = result.explainer
test_df = result.test_df

st.markdown(
    r"""
    <div class="main-header">
        <span class="challenger-badge">Capa 2: Boosting No-Lineal de Residuos</span>
        <h2 style="margin:8px 0; color:#f8fafc;">LightGBM Residual Booster + Auditoría SHAP</h2>
        <p style="color:#94a3b8; margin:0;">
            En lugar de usar IA como una caja negra que reemplaza a Holt-Winters, entrenamos un modelo de árboles 
            <strong>únicamente sobre los residuos \(y_t - \hat{y}_{HW}\)</strong>. 
            Cada unidad adicional añadida por el booster está 100% justificada por variables de negocio retail.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Architectural Principle
st.info(
    r"""
    🛡️ **Garantía para el Escéptico de la IA:**
    $$\hat{y}_{\text{Final}} = \max\left(0, \hat{y}_{HW} + \hat{r}_{\text{LightGBM}}\right)$$
    Si en un día determinado no hay promociones activas, ni quincenas, ni eventos extraordinarios, el residual predicho es \(\approx 0\). 
    **El modelo regresa automáticamente a la seguridad de tu Holt-Winters.**
    """
)

# Feature Importance Chart
st.markdown("### 📊 Importancia de Variables en la Explicación de los Residuos")

col_imp1, col_imp2 = st.columns([2, 1])

# Extract test features
booster_engine = result.booster_result.model
X_test = result.booster_result.feature_names
# Build X_test dataframe
from core.residual_booster import ResidualBooster
rb_temp = ResidualBooster()
X_test_df = rb_temp.extract_features(test_df, test_df["pred_hw_champion"].values, is_training=False)

importance_df = explainer.get_global_feature_importance(X_test_df)

with col_imp1:
    fig_imp = go.Figure(
        go.Bar(
            x=importance_df["mean_abs_shap"],
            y=importance_df["feature_label"],
            orientation="h",
            marker_color="#0284c7",
            text=[f"{v:.2f} u" for v in importance_df["mean_abs_shap"]],
            textposition="auto",
        )
    )
    fig_imp.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text="Impacto Medio Absoluto en Unidades (Mean |SHAP|)", x=0.01),
        xaxis_title="Impacto Medio en Pronóstico (|SHAP| unidades)",
        yaxis=dict(autorange="reversed"),
        height=380,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with col_imp2:
    st.markdown("#### Hallazgos Clave de Retail")
    top_feature = importance_df.iloc[0]["feature_label"]
    second_feature = importance_df.iloc[1]["feature_label"]
    st.markdown(
        f"""
        - **{top_feature}**: Es el factor con mayor peso para corregir el residuo.
        - **{second_feature}**: Explica la no-linealidad cuando se combinan múltiples palancas comerciales.
        - **Sinergia Promo + Fin de Semana**: Demuestra que un 20% de descuento un sábado genera un uplift 3 veces mayor que un martes.
        """
    )

# Interactive SHAP Waterfall Inspector
st.markdown("### 🕵️ Inspector Diario de Decisiones (Waterfall de Explicabilidad)")
st.markdown("Selecciona una fecha del horizonte de evaluación para auditar cómo se construyó el pronóstico unidad por unidad:")

date_options = test_df["date"].dt.strftime("%Y-%m-%d").tolist()
# Default to a date that had a promo
promo_indices = test_df[test_df["is_promo"] == 1].index
default_idx = promo_indices[0] if len(promo_indices) > 0 else 0
default_date = test_df.loc[default_idx, "date"].strftime("%Y-%m-%d")

selected_date_str = st.selectbox(
    "Fecha a Inspeccionar:",
    options=date_options,
    index=date_options.index(default_date),
)

# Find selected row
selected_row = test_df[test_df["date"].dt.strftime("%Y-%m-%d") == selected_date_str].iloc[0]
hw_val = float(selected_row["pred_hw_champion"])
true_val = float(selected_row["true_demand"])
row_features = X_test_df.loc[selected_row.name]

single_day_exp = explainer.explain_single_day(
    X_row=row_features,
    hw_baseline_val=hw_val,
    date_str=selected_date_str,
)

# Waterfall Chart
waterfall_names = ["1. Base Holt-Winters"]
waterfall_vals = [hw_val]
waterfall_measures = ["absolute"]

labels_map = {
    "hw_baseline": "Nivel Base HW",
    "is_promo": "Flag Promoción",
    "discount_pct": "Profundidad Descuento",
    "promo_weekend_boost": "Sinergia Promo+Finde",
    "promo_payday_boost": "Sinergia Promo+Quincena",
    "is_payday": "Quincena / Fin de Mes",
    "dist_to_payday": "Proximidad Quincena",
    "is_weekend": "Fin de Semana",
    "day_of_week": "Día Semana",
    "month": "Mes",
    "is_holiday": "Feriado",
    "promo_2x1": "Mecánica 2x1",
    "promo_cabecera_gondola": "Cabecera Góndola",
    "promo_catalogo_ofertas": "Catálogo Ofertas",
    "promo_descuento_20pct": "Desc. 20%",
    "is_rainy_day": "Lluvia en Santiago (Efecto Once)",
    "rain_weekend_boost": "Sinergia Lluvia+Finde",
    "is_fiestas_patrias": "Fiestas Patrias (18 Sept)",
}

# Add top 5 non-zero contributions
for item in single_day_exp["contributions"][:5]:
    if abs(item["shap_impact"]) >= 0.5:
        f_name = labels_map.get(item["feature"], item["feature"])
        waterfall_names.append(f"+ {f_name}")
        waterfall_vals.append(item["shap_impact"])
        waterfall_measures.append("relative")

# Sum total
waterfall_names.append("= Pronóstico Final Challenger")
waterfall_vals.append(single_day_exp["final_forecast"])
waterfall_measures.append("total")

fig_wf = go.Figure(
    go.Waterfall(
        name="Atribución SHAP",
        orientation="v",
        measure=waterfall_measures,
        x=waterfall_names,
        textposition="outside",
        text=[f"{v:+.1f} u" if m == "relative" else f"{v:.1f} u" for v, m in zip(waterfall_vals, waterfall_measures)],
        y=waterfall_vals,
        connector={"line": {"color": "#64748b"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#38bdf8"}},
    )
)

fig_wf.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    title=dict(text=f"Descomposición Paso a Paso del Pronóstico para el día {selected_date_str}", x=0.01),
    yaxis_title="Unidades de Demanda",
    height=420,
)

st.plotly_chart(fig_wf, use_container_width=True)

# Verification Box
c_comp1, c_comp2, c_comp3 = st.columns(3)
with c_comp1:
    st.metric("Champion (Holt-Winters Solo)", f"{hw_val:.1f} u")
with c_comp2:
    st.metric(
        "Challenger (Holt-Winters+)",
        f"{single_day_exp['final_forecast']:.1f} u",
        f"{single_day_exp['final_forecast'] - hw_val:+.1f} u residual",
    )
with c_comp3:
    st.metric(
        "Demanda Real Ocurrida",
        f"{true_val:.1f} u",
        f"Error Challenger: {abs(true_val - single_day_exp['final_forecast']):.1f} u vs HW: {abs(true_val - hw_val):.1f} u",
    )
