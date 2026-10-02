"""
Retail Demand & Stockout Data Generator.
Simulates multi-year daily sales, promotions, price elasticity,
inventory dynamics and censored demand (out-of-stock events) for real retail SKUs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class SKUConfig:
    sku_id: str
    sku_name: str
    category: str
    base_demand: float
    trend_annual: float
    weekend_multiplier: float
    promo_elasticity: float
    unit_cost: float
    retail_price: float
    shelf_life_days: int
    lead_time_days: int
    target_service_level: float


SKU_CATALOG: Dict[str, SKUConfig] = {
    "SKU-101": SKUConfig(
        sku_id="SKU-101",
        sku_name="Leche Entera 1L Tetra",
        category="Lácteos y Desayuno",
        base_demand=120.0,
        trend_annual=0.03,
        weekend_multiplier=1.25,
        promo_elasticity=1.4,
        unit_cost=0.95,
        retail_price=1.49,
        shelf_life_days=30,
        lead_time_days=2,
        target_service_level=0.95,
    ),
    "SKU-204": SKUConfig(
        sku_id="SKU-204",
        sku_name="Cerveza Artesanal IPA 473ml",
        category="Bebidas & Licores",
        base_demand=75.0,
        trend_annual=0.08,
        weekend_multiplier=2.60,
        promo_elasticity=2.8,
        unit_cost=1.60,
        retail_price=2.99,
        shelf_life_days=180,
        lead_time_days=3,
        target_service_level=0.92,
    ),
    "SKU-305": SKUConfig(
        sku_id="SKU-305",
        sku_name="Detergente Líquido Concentrado 3L",
        category="Cuidado del Hogar",
        base_demand=45.0,
        trend_annual=0.02,
        weekend_multiplier=1.45,
        promo_elasticity=3.2,
        unit_cost=6.80,
        retail_price=12.90,
        shelf_life_days=365,
        lead_time_days=5,
        target_service_level=0.90,
    ),
    "SKU-402": SKUConfig(
        sku_id="SKU-402",
        sku_name="Yogur Griego Frutos Rojos 150g",
        category="Frescos y Refrigerados",
        base_demand=60.0,
        trend_annual=0.05,
        weekend_multiplier=1.35,
        promo_elasticity=1.8,
        unit_cost=0.70,
        retail_price=1.35,
        shelf_life_days=12,
        lead_time_days=2,
        target_service_level=0.96,
    ),
    "SKU-510": SKUConfig(
        sku_id="SKU-510",
        sku_name="Carne Vacuno Asado de Tira 1kg",
        category="Carnicería",
        base_demand=50.0,
        trend_annual=0.04,
        weekend_multiplier=3.10,
        promo_elasticity=2.2,
        unit_cost=6.50,
        retail_price=10.99,
        shelf_life_days=7,
        lead_time_days=3,
        target_service_level=0.94,
    ),
}

STORES = [
    {"store_id": "TIENDA-01", "store_name": "Hipermercado Urbano Central", "scale": 1.4},
    {"store_id": "TIENDA-02", "store_name": "Supermercado Express Centro", "scale": 0.85},
]


# =======================================================================
# CATÁLOGO Y TIENDAS ESPECÍFICAS DE EMPRESAS CASTAÑO (CHILE)
# =======================================================================

CASTANO_CATALOG: Dict[str, SKUConfig] = {
    "CAS-101": SKUConfig(
        sku_id="CAS-101",
        sku_name="Pan de Molde Blanco Familiar 560g",
        category="Pan Envasado / Molde",
        base_demand=185.0,
        trend_annual=0.03,
        weekend_multiplier=1.45,
        promo_elasticity=2.2,
        unit_cost=1350.0,    # $1.350 CLP
        retail_price=2190.0, # $2.190 CLP (Margen: $840 CLP)
        shelf_life_days=14,
        lead_time_days=1,
        target_service_level=0.96,
    ),
    "CAS-102": SKUConfig(
        sku_id="CAS-102",
        sku_name="Pan de Molde Integral Linaza & Chía 500g",
        category="Pan Saludable & Granos",
        base_demand=95.0,
        trend_annual=0.06,
        weekend_multiplier=1.30,
        promo_elasticity=1.7,
        unit_cost=1480.0,
        retail_price=2590.0, # Margen: $1.110 CLP
        shelf_life_days=14,
        lead_time_days=1,
        target_service_level=0.95,
    ),
    "CAS-201": SKUConfig(
        sku_id="CAS-201",
        sku_name="Bolsa Marraqueta Fresca Selección 500g",
        category="Pan Tradicional Fresco",
        base_demand=240.0,
        trend_annual=0.02,
        weekend_multiplier=2.25, # Pico masivo fin de semana (la once)
        promo_elasticity=1.9,
        unit_cost=890.0,
        retail_price=1590.0,  # Margen: $700 CLP
        shelf_life_days=2,    # Ultra perecedero: merma crítica si sobra
        lead_time_days=1,
        target_service_level=0.98,
    ),
    "CAS-301": SKUConfig(
        sku_id="CAS-301",
        sku_name="Empanada de Pino Horno Tradicional 200g",
        category="Empanadas & Comidas Listas",
        base_demand=115.0,
        trend_annual=0.05,
        weekend_multiplier=2.85,
        promo_elasticity=2.4,
        unit_cost=1290.0,
        retail_price=2390.0, # Margen: $1.100 CLP
        shelf_life_days=3,   # Merma a los 3 días
        lead_time_days=1,
        target_service_level=0.95,
    ),
    "CAS-401": SKUConfig(
        sku_id="CAS-401",
        sku_name="Pack 4 Medialunas Manteca Horneadas",
        category="Bollería & Pastelería",
        base_demand=130.0,
        trend_annual=0.04,
        weekend_multiplier=2.60,
        promo_elasticity=3.1, # Muy elástica a 2x1 y combos
        unit_cost=1150.0,
        retail_price=2490.0, # Margen: $1.340 CLP
        shelf_life_days=2,   # Se endurecen a los 2 días
        lead_time_days=1,
        target_service_level=0.94,
    ),
    "CAS-501": SKUConfig(
        sku_id="CAS-501",
        sku_name="Pan Hamburguesa Brioche con Sésamo 4 un 320g",
        category="Especialidades Panadería",
        base_demand=85.0,
        trend_annual=0.08,
        weekend_multiplier=2.90, # Fuertes picos viernes/sábado/asados
        promo_elasticity=2.5,
        unit_cost=1250.0,
        retail_price=2290.0, # Margen: $1.040 CLP
        shelf_life_days=10,
        lead_time_days=2,
        target_service_level=0.94,
    ),
}

CASTANO_STORES = [
    {
        "store_id": "CANAL-SUPER",
        "store_name": "Jumbo Kennedy (Canal Supermercados)",
        "scale": 1.75,
        "type": "supermercado",
    },
    {
        "store_id": "LOCAL-METRO",
        "store_name": "Tienda Castaño Metro Tobalaba (Canal Tienda Propia)",
        "scale": 1.25,
        "type": "tienda_propia",
    },
]


def generate_retail_dataset(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic daily retail dataset with ground-truth latent demand,
    inventory depletion, out-of-stock events, promotions, and paydays.
    """
    rng = np.random.default_rng(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(date_range)

    records = []

    for store in STORES:
        store_id = store["store_id"]
        store_name = store["store_name"]
        store_scale = store["scale"]

        for sku_id, cfg in SKU_CATALOG.items():
            # 1. Base trend & weekly pattern
            t = np.arange(n_days)
            annual_trend = 1.0 + (cfg.trend_annual * t / 365.25)

            # Weekly seasonality weights [Mon, Tue, Wed, Thu, Fri, Sat, Sun]
            # Standard week curve scaled by SKU weekend multiplier
            base_dow_weights = np.array([0.75, 0.80, 0.85, 0.95, 1.30, 1.55, 1.25])
            if cfg.weekend_multiplier > 1.5:
                # Amplify weekends for beer/meat
                base_dow_weights[4] = 1.45
                base_dow_weights[5] = base_dow_weights[5] * (cfg.weekend_multiplier / 1.55)
                base_dow_weights[6] = base_dow_weights[6] * (cfg.weekend_multiplier / 1.55)
            dow_factors = base_dow_weights / base_dow_weights.mean()

            # 2. Monthly Payday effect (Quincena: 14-16, Fin de mes: 29-31)
            days_of_month = date_range.day.values
            is_payday = np.isin(days_of_month, [14, 15, 16, 29, 30, 31]).astype(int)
            payday_factor = 1.0 + is_payday * rng.uniform(0.12, 0.28, size=n_days)

            # 3. Calendar Holidays
            # Christmas, New Year, Easter, Halloween, etc.
            is_holiday = np.isin(
                date_range.strftime("%m-%d"),
                ["01-01", "05-01", "09-18", "09-19", "10-31", "12-24", "12-25", "12-31"],
            ).astype(int)
            holiday_factor = 1.0 + is_holiday * (0.45 if cfg.category in ["Carnicería", "Bebidas & Licores"] else 0.20)

            # 4. Promotions (Flyer, Gondola End, Price Cut)
            # Promotions happen randomly every 2-4 weeks, lasting 3 to 7 days
            is_promo = np.zeros(n_days, dtype=int)
            discount_pct = np.zeros(n_days, dtype=float)
            promo_type = ["ninguna"] * n_days

            day_idx = 10
            while day_idx < n_days - 7:
                if rng.random() < 0.22:
                    promo_duration = rng.integers(3, 8)
                    p_type = rng.choice(["catalogo_ofertas", "cabecera_gondola", "descuento_20pct", "2x1"])
                    disc = 0.15 if p_type == "catalogo_ofertas" else 0.20 if p_type == "cabecera_gondola" else 0.25 if p_type == "descuento_20pct" else 0.40
                    
                    end_idx = min(day_idx + promo_duration, n_days)
                    is_promo[day_idx:end_idx] = 1
                    discount_pct[day_idx:end_idx] = disc
                    for k in range(day_idx, end_idx):
                        promo_type[k] = p_type
                    day_idx = end_idx + rng.integers(12, 28)
                else:
                    day_idx += 7

            promo_lift = 1.0 + (discount_pct * cfg.promo_elasticity)

            # 5. True Uncensored Latent Demand (Poisson / NegBinomial distributed around expectation)
            expected_demand = (
                cfg.base_demand
                * store_scale
                * annual_trend
                * dow_factors[date_range.dayofweek]
                * payday_factor
                * holiday_factor
                * promo_lift
            )

            # Daily random noise
            noise = rng.gamma(shape=25.0, scale=1.0 / 25.0, size=n_days)
            true_demand = np.maximum(0, rng.poisson(lam=np.maximum(1.0, expected_demand * noise)))

            # 6. Physical Inventory & Stockout Simulation
            # Retail inventory with periodic replenishment review and lead time
            lead_time = cfg.lead_time_days
            on_hand_inv = np.zeros(n_days, dtype=float)
            observed_sales = np.zeros(n_days, dtype=float)
            stockout_flag = np.zeros(n_days, dtype=int)
            lost_sales = np.zeros(n_days, dtype=float)

            # Safety stock sizing roughly for baseline
            cycle_demand = cfg.base_demand * store_scale * (lead_time + 4)
            current_inventory = cycle_demand * 1.2
            pipeline_orders: List[Dict[str, float]] = []

            for i in range(n_days):
                # Receive inbound orders arriving today
                arriving = [o["qty"] for o in pipeline_orders if o["arrival_day"] == i]
                current_inventory += sum(arriving)

                start_inv = current_inventory
                dem = true_demand[i]

                if current_inventory >= dem:
                    sold = dem
                    current_inventory -= dem
                    so = 0
                    lost = 0
                else:
                    # Stockout occurred!
                    sold = max(0.0, current_inventory)
                    lost = dem - sold
                    current_inventory = 0.0
                    so = 1

                on_hand_inv[i] = start_inv
                observed_sales[i] = sold
                stockout_flag[i] = so
                lost_sales[i] = lost

                # Reorder policy: if inventory + on-order < reorder_point, place order
                on_order_qty = sum(o["qty"] for o in pipeline_orders if o["arrival_day"] > i)
                inventory_position = current_inventory + on_order_qty

                # Occasional supplier delay or out of stock creates a gap
                reorder_point = cfg.base_demand * store_scale * (lead_time + 1.5)
                order_up_to = cfg.base_demand * store_scale * (lead_time + 4.5)

                if inventory_position < reorder_point:
                    order_qty = max(0.0, order_up_to - inventory_position)
                    # 4% chance of supplier stockout or delivery delay causing a severe retail stockout
                    extra_delay = rng.integers(1, 4) if rng.random() < 0.05 else 0
                    pipeline_orders.append({
                        "arrival_day": i + lead_time + extra_delay,
                        "qty": order_qty,
                    })

            # Assemble into records
            for i, d in enumerate(date_range):
                records.append({
                    "date": d,
                    "store_id": store_id,
                    "store_name": store_name,
                    "sku_id": sku_id,
                    "sku_name": cfg.sku_name,
                    "category": cfg.category,
                    "unit_cost": cfg.unit_cost,
                    "retail_price": cfg.retail_price,
                    "margin_unit": cfg.retail_price - cfg.unit_cost,
                    "shelf_life_days": cfg.shelf_life_days,
                    "lead_time_days": cfg.lead_time_days,
                    "true_demand": int(true_demand[i]),
                    "observed_sales": int(observed_sales[i]),
                    "stockout_flag": int(stockout_flag[i]),
                    "lost_sales": int(lost_sales[i]),
                    "on_hand_inventory": float(on_hand_inv[i]),
                    "is_promo": int(is_promo[i]),
                    "discount_pct": float(discount_pct[i]),
                    "promo_type": promo_type[i],
                    "is_payday": int(is_payday[i]),
                    "is_holiday": int(is_holiday[i]),
                    "day_of_week": d.dayofweek,
                    "day_name": d.strftime("%A"),
                    "month": d.month,
                    "year": d.year,
                })

    df = pd.DataFrame(records)
    return df


def generate_castano_dataset(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic daily demand, promotions, weather dynamics (Santiago rain effect),
    Fiestas Patrias surges, and censored demand for Empresas Castaño.
    """
    rng = np.random.default_rng(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(date_range)

    # Santiago Winter Rain Simulator (May to August)
    is_rainy_day = np.zeros(n_days, dtype=int)
    for i, d in enumerate(date_range):
        if d.month in [5, 6, 7, 8]:
            # ~18% probability of rain during Santiago winter
            if rng.random() < 0.18:
                is_rainy_day[i] = 1

    records = []

    for store in CASTANO_STORES:
        store_id = store["store_id"]
        store_name = store["store_name"]
        store_scale = store["scale"]

        for sku_id, cfg in CASTANO_CATALOG.items():
            t = np.arange(n_days)
            annual_trend = 1.0 + (cfg.trend_annual * t / 365.25)

            # Weekly seasonality (Chilean bakery cycle: Sunday once is peak, Monday breakfast prep)
            base_dow_weights = np.array([0.90, 0.85, 0.85, 0.95, 1.35, 1.65, 1.50])
            if cfg.weekend_multiplier > 2.0:
                base_dow_weights[4] = 1.45
                base_dow_weights[5] = base_dow_weights[5] * (cfg.weekend_multiplier / 1.65)
                base_dow_weights[6] = base_dow_weights[6] * (cfg.weekend_multiplier / 1.50)
            dow_factors = base_dow_weights / base_dow_weights.mean()

            # Chilean Payday (Quincena: 14-16, Fin de mes: 29-31)
            days_of_month = date_range.day.values
            is_payday = np.isin(days_of_month, [14, 15, 16, 29, 30, 31]).astype(int)
            payday_factor = 1.0 + is_payday * (0.22 if cfg.category in ["Bollería & Pastelería", "Empanadas & Comidas Listas"] else 0.10)

            # Chilean Calendar & Fiestas Patrias (18 y 19 de Septiembre)
            is_holiday = np.isin(
                date_range.strftime("%m-%d"),
                ["01-01", "05-01", "05-21", "07-16", "08-15", "09-18", "09-19", "09-20", "10-31", "11-01", "12-25"],
            ).astype(int)

            is_fiestas_patrias = np.isin(
                date_range.strftime("%m-%d"),
                ["09-16", "09-17", "09-18", "09-19", "09-20"],
            ).astype(int)

            # Fiestas Patrias massive lift on Empanadas and BBQ Buns
            fp_lift = np.ones(n_days)
            if sku_id == "CAS-301":  # Empanadas
                fp_lift = 1.0 + is_fiestas_patrias * 3.8  # ~4.8x explosion!
            elif sku_id in ["CAS-201", "CAS-501"]:  # Marraqueta & Hamburguesa
                fp_lift = 1.0 + is_fiestas_patrias * 1.5  # ~2.5x

            # Santiago Rain Effect (El clásico "lluvia = once con pan caliente y medialunas")
            rain_lift = np.ones(n_days)
            if cfg.category in ["Pan Tradicional Fresco", "Bollería & Pastelería", "Empanadas & Comidas Listas"]:
                rain_lift = 1.0 + is_rainy_day * rng.uniform(0.28, 0.45, size=n_days)
            else:
                rain_lift = 1.0 + is_rainy_day * 0.12

            # Promotions (Lunes de Pan, 2x1 en Medialunas, etc.)
            is_promo = np.zeros(n_days, dtype=int)
            discount_pct = np.zeros(n_days, dtype=float)
            promo_type = ["ninguna"] * n_days

            day_idx = 8
            while day_idx < n_days - 7:
                if rng.random() < 0.25:
                    promo_duration = rng.integers(3, 7)
                    if cfg.category == "Bollería & Pastelería":
                        p_type = rng.choice(["2x1", "catalogo_ofertas", "descuento_20pct"])
                    elif cfg.category == "Pan Envasado / Molde":
                        p_type = rng.choice(["cabecera_gondola", "catalogo_ofertas", "descuento_20pct"])
                    else:
                        p_type = rng.choice(["catalogo_ofertas", "descuento_20pct"])

                    disc = 0.15 if p_type == "catalogo_ofertas" else 0.20 if p_type in ["cabecera_gondola", "descuento_20pct"] else 0.40
                    end_idx = min(day_idx + promo_duration, n_days)
                    is_promo[day_idx:end_idx] = 1
                    discount_pct[day_idx:end_idx] = disc
                    for k in range(day_idx, end_idx):
                        promo_type[k] = p_type
                    day_idx = end_idx + rng.integers(10, 24)
                else:
                    day_idx += 5

            promo_lift = 1.0 + (discount_pct * cfg.promo_elasticity)

            # True Uncensored Latent Demand
            expected_demand = (
                cfg.base_demand
                * store_scale
                * annual_trend
                * dow_factors[date_range.dayofweek]
                * payday_factor
                * fp_lift
                * rain_lift
                * promo_lift
            )

            noise = rng.gamma(shape=25.0, scale=1.0 / 25.0, size=n_days)
            true_demand = np.maximum(0, rng.poisson(lam=np.maximum(1.0, expected_demand * noise)))

            # Physical Inventory & Stockout Simulation
            lead_time = cfg.lead_time_days
            on_hand_inv = np.zeros(n_days, dtype=float)
            observed_sales = np.zeros(n_days, dtype=float)
            stockout_flag = np.zeros(n_days, dtype=int)
            lost_sales = np.zeros(n_days, dtype=float)

            # Bakery inventory orders
            cycle_demand = cfg.base_demand * store_scale * (lead_time + 2)
            current_inventory = cycle_demand * 1.15
            pipeline_orders: List[Dict[str, float]] = []

            for i in range(n_days):
                arriving = [o["qty"] for o in pipeline_orders if o["arrival_day"] == i]
                current_inventory += sum(arriving)

                start_inv = current_inventory
                dem = true_demand[i]

                if current_inventory >= dem:
                    sold = dem
                    current_inventory -= dem
                    so = 0
                    lost = 0
                else:
                    sold = max(0.0, current_inventory)
                    lost = dem - sold
                    current_inventory = 0.0
                    so = 1

                on_hand_inv[i] = start_inv
                observed_sales[i] = sold
                stockout_flag[i] = so
                lost_sales[i] = lost

                on_order_qty = sum(o["qty"] for o in pipeline_orders if o["arrival_day"] > i)
                inventory_position = current_inventory + on_order_qty

                # Daily replenishment for fresh bread
                reorder_point = cfg.base_demand * store_scale * (lead_time + 0.8)
                order_up_to = cfg.base_demand * store_scale * (lead_time + 2.4)

                if inventory_position < reorder_point:
                    order_qty = max(0.0, order_up_to - inventory_position)
                    pipeline_orders.append({
                        "arrival_day": i + lead_time,
                        "qty": order_qty,
                    })

            for i, d in enumerate(date_range):
                records.append({
                    "date": d,
                    "store_id": store_id,
                    "store_name": store_name,
                    "sku_id": sku_id,
                    "sku_name": cfg.sku_name,
                    "category": cfg.category,
                    "unit_cost": cfg.unit_cost,
                    "retail_price": cfg.retail_price,
                    "margin_unit": cfg.retail_price - cfg.unit_cost,
                    "shelf_life_days": cfg.shelf_life_days,
                    "lead_time_days": cfg.lead_time_days,
                    "true_demand": int(true_demand[i]),
                    "observed_sales": int(observed_sales[i]),
                    "stockout_flag": int(stockout_flag[i]),
                    "lost_sales": int(lost_sales[i]),
                    "on_hand_inventory": float(on_hand_inv[i]),
                    "is_promo": int(is_promo[i]),
                    "discount_pct": float(discount_pct[i]),
                    "promo_type": promo_type[i],
                    "is_payday": int(is_payday[i]),
                    "is_holiday": int(is_holiday[i]),
                    "is_rainy_day": int(is_rainy_day[i]),
                    "is_fiestas_patrias": int(is_fiestas_patrias[i]),
                    "day_of_week": d.dayofweek,
                    "day_name": d.strftime("%A"),
                    "month": d.month,
                    "year": d.year,
                })

    df = pd.DataFrame(records)
    return df


def get_dataset_path(dataset_name: str = "castano") -> Path:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = "castano_demand.parquet" if dataset_name == "castano" else "retail_demand.parquet"
    return data_dir / filename


def get_default_dataset_path() -> Path:
    return get_dataset_path("castano")


def load_or_generate_dataset(dataset_name: str = "castano", force_regenerate: bool = False) -> pd.DataFrame:
    """Loads existing Parquet dataset or generates fresh realistic retail dataset."""
    target_path = get_dataset_path(dataset_name)
    if target_path.exists() and not force_regenerate:
        return pd.read_parquet(target_path)

    if dataset_name == "castano":
        df = generate_castano_dataset()
    else:
        df = generate_retail_dataset()

    df.to_parquet(target_path, index=False)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generador de datasets de demanda retail")
    parser.add_argument("--dataset", choices=["castano", "retail"], default="castano", help="Dataset a generar")
    args = parser.parse_args()

    print(f"Generando dataset: '{args.dataset}'...")
    df = load_or_generate_dataset(dataset_name=args.dataset, force_regenerate=True)
    print(f"Dataset generado exitosamente con {len(df):,} filas.")
    print(f"Período: {df['date'].min().date()} a {df['date'].max().date()}")
    print(f"SKUs: {df['sku_id'].nunique()} | Tiendas/Canales: {df['store_id'].nunique()}")
    print(f"Tasa global de quiebre de stock (OOS): {df['stockout_flag'].mean():.2%}")
    print(f"Ventas totales observadas: {df['observed_sales'].sum():,}")
    print(f"Ventas perdidas totales por quiebre: {df['lost_sales'].sum():,}")
    if "is_rainy_day" in df.columns:
        print(f"Días de lluvia simulados en Santiago: {df['is_rainy_day'].sum() // (df['sku_id'].nunique() * df['store_id'].nunique())} días/año")

