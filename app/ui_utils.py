"""
UI Utilities and Styling for Holt-Winters+ Streamlit Application.
Provides cached pipeline execution, custom theme settings, and reusable Plotly styling.
"""

from typing import Dict, Any, Tuple
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from pipeline import ForecastingPipeline, PipelineExecutionResult
from core.data_generator import (
    load_or_generate_dataset,
    SKU_CATALOG,
    STORES,
    CASTANO_CATALOG,
    CASTANO_STORES,
)


# Plotly theme styling matching our sleek dark theme
PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
    font=dict(color="#e2e8f0", family="Inter, system-ui, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.75)",
        bordercolor="#334155",
        borderwidth=1,
    ),
)


def apply_plotly_theme(fig: go.Figure, **kwargs) -> go.Figure:
    """Applies theme and safely configures axes without keyword collision."""
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, **kwargs)
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#475569", linecolor="#475569")
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#475569", linecolor="#475569")
    return fig


def apply_custom_css():
    """Injects high-end enterprise SaaS styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        }

        .metric-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #0284c7;
        }

        .champion-badge {
            background-color: #334155;
            color: #94a3b8;
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .challenger-badge {
            background-color: #0369a1;
            color: #bae6fd;
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Calculando pronósticos y simulaciones en tiempo real...")
def get_pipeline_result(
    store_id: str,
    sku_id: str,
    test_days: int = 60,
    service_level: float = 0.95,
    annual_holding_cost_rate: float = 0.22,
    dataset_name: str = "castano",
) -> PipelineExecutionResult:
    """Cached runner for the end-to-end forecasting pipeline."""
    pipe = ForecastingPipeline(
        test_days=test_days,
        target_service_level=service_level,
        annual_holding_cost_rate=annual_holding_cost_rate,
    )
    df = load_or_generate_dataset(dataset_name=dataset_name)
    return pipe.run(store_id=store_id, sku_id=sku_id, df=df)


def render_store_sku_selector():
    """Renders persistent store and SKU selection in sidebar with client switcher."""
    st.sidebar.markdown("### 🏢 Cliente / Portafolio")
    
    dataset_name = st.sidebar.selectbox(
        "Caso de Negocio:",
        options=["castano", "retail"],
        format_func=lambda x: "🥖 Empresas Castaño (Panadería & Frescos)" if x == "castano" else "🛒 Retail Supermercado General",
        index=0,
    )

    is_castano = (dataset_name == "castano")
    active_stores = CASTANO_STORES if is_castano else STORES
    active_catalog = CASTANO_CATALOG if is_castano else SKU_CATALOG

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏬 Configuración Retail")
    
    store_options = {s["store_id"]: f"{s['store_name']}" for s in active_stores}
    selected_store_id = st.sidebar.selectbox(
        "Sucursal / Tienda:",
        options=list(store_options.keys()),
        format_func=lambda x: store_options[x],
        index=0,
        key=f"store_{dataset_name}",
    )

    sku_keys = list(active_catalog.keys())
    sku_options = {sid: f"{cfg.sku_name} [{cfg.category}]" for sid, cfg in active_catalog.items()}
    if is_castano:
        default_sku_idx = sku_keys.index("CAS-401") if "CAS-401" in sku_keys else 0
    else:
        default_sku_idx = sku_keys.index("SKU-101") if "SKU-101" in sku_keys else 0

    selected_sku_id = st.sidebar.selectbox(
        "Producto (SKU):",
        options=sku_keys,
        format_func=lambda x: sku_options[x],
        index=default_sku_idx,
        key=f"sku_{dataset_name}",
    )

    test_days = st.sidebar.slider("Horizonte de Evaluación (Días Test):", min_value=30, max_value=90, value=60, step=15)
    service_level = st.sidebar.slider("Nivel de Servicio Objetivo (%):", min_value=85, max_value=99, value=95, step=1) / 100.0

    st.sidebar.markdown("---")
    if is_castano:
        st.sidebar.info("🥖 **Caso Castaño Activo**: Precios en CLP, dinámica de lluvia Santiago, Fiestas Patrias y mermas por vida útil (1-14 días).")
    st.sidebar.caption("⚡ **Holt-Winters+** v1.0 | Champion vs Challenger Engine")

    return selected_store_id, selected_sku_id, test_days, service_level, dataset_name
