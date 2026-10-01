# 📈 Holt-Winters+ | Enterprise Retail Demand Forecasting

> **Plataforma SaaS de predicción de demanda retail diseñada bajo la estrategia Champion vs Challenger.**
> Construida específicamente para auditar, respetar y amplificar modelos tradicionales de Holt-Winters con corrección de quiebres de stock (out-of-stock) mediante capas no invasivas de Machine Learning, Conformal Prediction y Simulación de Inventario.

---

## 🎯 Contexto y Estrategia del Pitch ("Champion vs Challenger")

El estándar tradicional en grandes cadenas de retail y supermercados es el **suavizamiento exponencial de Holt-Winters con estacionalidad semanal (\(m=7\)) + corrección de ventas perdidas por quiebres de stock**. 

Este sistema es matemáticamente sólido, altamente estable y cuenta con la confianza absoluta de directores de abastecimiento. **Reemplazarlo por una "caja negra" de IA genera rechazo justificado.**

### La Tesis de Holt-Winters+
No reemplazamos el modelo Champion: **lo convertimos en nuestra piedra angular y lo amplificamos con 4 capas auditables**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │                   DEMANDA REAL EN POS                  │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                ┌──────────────────────────────────┴──────────────────────────────────┐
                │                                                                     │
                ▼                                                                     ▼
┌───────────────────────────────┐                                     ┌───────────────────────────────┐
│     CHAMPION (Línea Base)     │                                     │     CHALLENGER (Ampliación)   │
├───────────────────────────────┤                                     ├───────────────────────────────┤
│ • Imputación OOS por mediana  │                                     │ 1. Uncensoring Sensible Promo │
│   día de semana histórico.    │                                     │    (recupera demanda latente) │
│ • Holt-Winters Clásico:       │                                     │ 2. Residual Booster (LightGBM)│
│   Nivel (α) + Tendencia (β)   │                                     │    únicamente sobre errores yt│
│   + Estacionalidad m=7 (γ).   │                                     │    de HW usando promociones.  │
│ • Stock de Seguridad          │                                     │ 3. Conformal Prediction       │
│   Gaussiano (Z * σ * √L).     │                                     │    (garantía 1-α en colas).   │
│                               │                                     │ 4. Simulador Financiero ROI   │
│                               │                                     │    (Margen vs Holding vs Merma│
└───────────────────────────────┘                                     └───────────────────────────────┘
```

---

## 🧱 Las 4 Capas Auditables

### Capa 0: El Champion (Holt-Winters Clásico + Corrección OOS)
- Descomposición aditiva/multiplicativa con período estacional \(m=7\) (semana comercial).
- Suavizamiento exponencial de Nivel (\(\alpha\)), Tendencia (\(\beta\)) y Estacionalidad (\(\gamma\)).
- Corrección de demanda censurada mediante mediana histórica de ventas del mismo día de la semana.

### Capa 1: Uncensoring Avanzado (Reconstrucción de Demanda Latente)
- Si un producto sufre quiebre de stock un sábado con un **descuento del 20% o promoción 2x1**, la mediana de un sábado normal subestima la demanda real hasta en un 50%.
- El un-censoring del Challenger ajusta la demanda esperada combinando el nivel suavizado (EWMA), el índice de día de la semana y la elasticidad promocional del evento activo.

### Capa 2: Residual Boosting con LightGBM + SHAP
- **Principio de Seguridad:** El modelo de gradient boosting **NO predice la demanda directamente**, sino exclusivamente el residuo del Champion:
  $$\hat{y}_{\text{Final}} = \max\left(0, \hat{y}_{HW} + \hat{r}_{\text{LightGBM}}\right)$$
- **Fallback Automático:** Si no hay promociones, ni efectos de quincena (días 15 y 30), ni feriados, \(\hat{r} \approx 0\) y el pronóstico regresa de forma 100% idéntica al Holt-Winters del cliente.
- **Auditabilidad SHAP:** Cada unidad adicional agregada por el booster se descompone en un gráfico Waterfall: cuántas unidades aporta el descuento, cuántas el fin de semana y cuántas el día de cobro.

### Capa 3: Conformal Prediction & Incertidumbre No-Paramétrica
- La teoría clásica asume errores gaussianos \(\mathcal{N}(0, \sigma^2)\). En retail, los picos promocionales generan **colas pesadas (alta kurtosis y skewness)**.
- El stock de seguridad gaussiano (\(Z_{0.95} \cdot \sigma \cdot \sqrt{L}\)) produce un servicio real de solo 75%-85% en picos de demanda.
- Conformal Prediction calcula el cuantil empírico no-paramétrico \(q_{1-\alpha}\) sobre un conjunto de calibración, ofreciendo una **garantía matemática de cobertura finita (\(1-\alpha\))**.

### Capa 4: Simulación de Inventario & ROI en Dólares de Retail
- Motor discreto día a día que simula la operación real de la tienda:
  - Lead time del proveedor y órdenes de reposición.
  - Vencimiento y merma por vida útil (productos frescos).
  - Costo de almacenamiento y capital de trabajo inmovilizado (\(\approx 22\%\) anual).
  - Unidades de venta recuperadas por evitar quiebres de stock.
  - Beneficio económico neto en dólares reales (\(\Delta \text{Margen Bruto} - \Delta \text{Holding Cost} - \Delta \text{Mermas}\)).

---

## 🚀 Inicio Rápido (Un Solo Comando)

### Prerrequisitos
- Python 3.11 o superior.
- `make` (opcional, pero recomendado).

### 1. Clonar y ejecutar con Make
```bash
# Iniciar la aplicación Streamlit directamente (puerto 8501)
make run
```
Si es la primera vez que se ejecuta, crea el entorno virtual `.venv`, instala dependencias y lanza la aplicación.

### 2. O ejecutar directamente con Streamlit
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Home.py
```

La aplicación estará disponible inmediatamente en `http://localhost:8501`.

---

## 🧪 Pruebas Automatizadas

El proyecto cuenta con una suite completa de pruebas unitarias y de integración para validar la consistencia matemática:

```bash
make test
# o bien:
.venv/bin/python -m pytest tests/ -v
```

---

## 📁 Estructura del Código

Diseñada modularmente para permitir desacoplar el core matemático y exponerlo fácilmente en una API REST (FastAPI) o worker de procesamiento batch:

```
forecast/
├── Makefile                          # Automatización de instalación, tests y ejecución
├── README.md                         # Documentación técnica y comercial de la solución
├── requirements.txt                  # Dependencias versionadas
├── data/
│   └── retail_demand.parquet         # Dataset sintético multi-año con quiebres y promociones
├── core/
│   ├── __init__.py
│   ├── data_generator.py             # Generador de datos SKU-Tienda con dinámica de inventario
│   ├── stockout_corrector.py         # Imputación Champion (DOW) vs Challenger (Promo-aware)
│   ├── holt_winters.py               # Champion Holt-Winters (Statsmodels) + extracción componentes
│   ├── residual_booster.py           # Challenger Layer 2: LightGBM sobre residuos de HW
│   ├── conformal_uq.py               # Challenger Layer 3: Split Conformal Prediction & Safety Stock
│   ├── inventory_engine.py           # Challenger Layer 4: Simulación P&L, Fill Rate, Merma y ROI
│   └── explainability.py             # SHAP TreeExplainer + Waterfalls interactivos
├── pipeline.py                       # Orquestador end-to-end (ejecución completa cacheable)
├── tests/
│   └── test_forecast_engine.py       # Pruebas unitarias de todos los motores
└── app/
    ├── Home.py                       # Dashboard Ejecutivo: Scorecard, KPIs y pitch principal
    ├── ui_utils.py                   # Tema visual enterprise, Plotly dark mode y selectores
    └── pages/
        ├── 1_📊_Datos_y_Stockouts.py     # Auditoría de Ventas Perdidas por OOS
        ├── 2_📈_Holt_Winters_Champion.py # Auditoría matemática del modelo Champion
        ├── 3_⚡_Residual_Boosting.py     # LightGBM Residual Booster + SHAP Waterfall
        ├── 4_🛡️_Conformal_SafetyStock.py # Conformal Prediction vs Gaussiano
        └── 5_💰_Simulador_ROI_Retail.py  # Simulador financiero y extrapolación a cadena
```

---

## 🔄 Hoja de Ruta para Migración a Producción (FastAPI + Web)

El diseño desacoplado de `core/` y `pipeline.py` permite migrar directamente a una arquitectura de microservicios:

```
[ Frontend React / Next.js / Vue ]
                │
                ▼ REST API / gRPC
┌──────────────────────────────────────────────────┐
│             FastAPI Backend Service              │
│  - POST /api/v1/forecast/champion                │
│  - POST /api/v1/forecast/challenger              │
│  - POST /api/v1/inventory/simulate               │
│  - GET  /api/v1/explain/shap/{date}              │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│      core/ (Holt-Winters, Booster, Conformal)    │
│  - Storage en PostgreSQL / ClickHouse / Parquet  │
│  - Workers asíncronos con Celery / Redis         │
└──────────────────────────────────────────────────┘
```

---

## 📊 Catálogo de SKUs Retail Incluidos para la Demostración

| SKU ID | Nombre del Producto | Categoría | Dinámica Particular | Elasticidad Promo |
| :--- | :--- | :--- | :--- | :--- |
| **SKU-101** | Leche Entera 1L Tetra | Lácteos y Desayuno | Alto volumen básico, demanda estable | Moderada |
| **SKU-204** | Cerveza Artesanal IPA 473ml | Bebidas & Licores | Pico masivo en fines de semana y 2x1 | **Muy Alta** |
| **SKU-305** | Detergente Concentrado 3L | Cuidado del Hogar | Compra de reposición periódica y voluminosa | Alta |
| **SKU-402** | Yogur Griego Frutos Rojos | Frescos y Refrigerados | Producto perecedero (12 días vida útil) | Media |
| **SKU-510** | Asado de Tira Vacuno 1kg | Carnicería | Picos en feriados, asados y domingos | Alta |
