import warnings
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# IMPORTACIÓN DE MÓDULOS LOCALES
# =========================================================
from datos import generar_demanda_sintetica, convertir_a_mensual
from generar_pronosticos import (
    METODOS_PRONOSTICO,
    generar_forecast,
    generar_forecast_mejor_por_producto,
)
from simulacion_inventario import (
    simular_producto,
    calcular_kpis,
    optimizar_stock_seguridad,
    obtener_parametros_producto,
)
from visualizacion import (
    grafico_forecast,
    grafico_inventario,
    grafico_tradeoff,
    formatear_comparacion,
)

warnings.filterwarnings("ignore")


# =========================================================
# FUNCIONES AUXILIARES - FORECAST COMERCIAL Y DASHBOARD
# =========================================================
def normalizar_product_id(serie: pd.Series) -> pd.Series:
    return (
        serie.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def leer_forecast_comercial_opcional(xls: pd.ExcelFile) -> pd.DataFrame:
    """
    Lee el forecast comercial si existe en el Excel.
    Prioriza una hoja llamada Forecast_Comercial, pero si no existe,
    busca automáticamente una hoja que tenga columnas equivalentes a:
    date, product_id y forecast_company.
    """
    columnas_salida = ["date", "product_id", "forecast_company"]

    alias = {
        "fecha": "date",
        "mes": "date",
        "periodo": "date",
        "period": "date",
        "date": "date",
        "producto": "product_id",
        "sku": "product_id",
        "product_id": "product_id",
        "id_producto": "product_id",
        "codigo": "product_id",
        "código": "product_id",
        "grupo de demanda": "product_id",
        "grupo_demanda": "product_id",
        "grupo": "product_id",
        "forecast": "forecast_company",
        "forecast comercial": "forecast_company",
        "forecast_comercial": "forecast_company",
        "forecast empresa": "forecast_company",
        "forecast_empresa": "forecast_company",
        "forecast_company": "forecast_company",
        "pronostico": "forecast_company",
        "pronóstico": "forecast_company",
        "pronostico comercial": "forecast_company",
        "pronóstico comercial": "forecast_company",
        "pronostico_comercial": "forecast_company",
        "pronostico_empresa": "forecast_company",
        "pronóstico_empresa": "forecast_company",
    }

    # Primero intentamos con nombres de hoja esperados.
    hojas_prioritarias = [
        "Forecast_Comercial",
        "Forecast Comercial",
        "forecast_comercial",
        "forecast comercial",
        "Pronostico_Comercial",
        "Pronóstico_Comercial",
        "Pronostico Comercial",
        "Pronóstico Comercial",
    ]

    hojas_a_revisar = []
    for h in hojas_prioritarias:
        if h in xls.sheet_names and h not in hojas_a_revisar:
            hojas_a_revisar.append(h)

    # Luego revisamos todas las demás hojas, por si el nombre no coincide.
    for h in xls.sheet_names:
        if h not in hojas_a_revisar:
            hojas_a_revisar.append(h)

    for hoja in hojas_a_revisar:
        try:
            df = pd.read_excel(xls, sheet_name=hoja)
        except Exception:
            continue

        if df.empty:
            continue

        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

        if not all(c in df.columns for c in columnas_salida):
            continue

        df = df[columnas_salida].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["product_id"] = normalizar_product_id(df["product_id"])
        df["forecast_company"] = pd.to_numeric(
            df["forecast_company"], errors="coerce"
        ).fillna(0)

        df = df.dropna(subset=["date"])
        if df.empty:
            continue

        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

        df = (
            df.groupby(["product_id", "date"], as_index=False)["forecast_company"]
            .sum()
            .sort_values(["product_id", "date"])
            .reset_index(drop=True)
        )

        return df

    return pd.DataFrame(columns=columnas_salida)

def obtener_costos_unitarios(df_parametros: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae costo unitario desde la hoja Datos.
    Usa unit_value si existe; si no, intenta unit_cost, costo_unitario o aliases similares.
    """
    if df_parametros is None or df_parametros.empty:
        return pd.DataFrame(columns=["product_id", "unit_cost"])

    df = df_parametros.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    alias = {
        "grupo de demanda": "product_id",
        "grupo_demanda": "product_id",
        "sku": "product_id",
        "producto": "product_id",
        "product_id": "product_id",
        "codigo": "product_id",
        "código": "product_id",
        "unit_value": "unit_cost",
        "unit value": "unit_cost",
        "valor unitario": "unit_cost",
        "valor_unitario": "unit_cost",
        "unit_cost": "unit_cost",
        "unit cost": "unit_cost",
        "costo unitario": "unit_cost",
        "costo_unitario": "unit_cost",
        "costo": "unit_cost",
        "valor": "unit_cost",
    }

    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    if "product_id" not in df.columns:
        return pd.DataFrame(columns=["product_id", "unit_cost"])

    if "unit_cost" not in df.columns:
        df["unit_cost"] = 0

    out = df[["product_id", "unit_cost"]].copy()
    out["product_id"] = normalizar_product_id(out["product_id"])
    out["unit_cost"] = pd.to_numeric(out["unit_cost"], errors="coerce").fillna(0)
    out = out.drop_duplicates("product_id")

    return out

def calcular_ahorro_forecast_2025(
    df_forecast_auto: pd.DataFrame,
    df_forecast_empresa: pd.DataFrame,
    df_parametros: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Calcula el ahorro potencial comparando:
    ventas reales 2025 vs forecast empresa 2025 vs forecast propuesto 2025.

    Ahorro potencial S/ = Error empresa S/ - Error propuesta S/.
    Si el resultado es negativo, significa que la propuesta no mejora a la empresa en ese SKU.
    """
    kpis_cero = {
        "ahorro_total": 0.0,
        "error_empresa": 0.0,
        "error_propuesta": 0.0,
        "skus_comparados": 0,
    }

    if (
        df_forecast_auto is None
        or df_forecast_auto.empty
        or df_forecast_empresa is None
        or df_forecast_empresa.empty
    ):
        return pd.DataFrame(), kpis_cero

    prop = df_forecast_auto[
        (df_forecast_auto["tipo_periodo"] == "Histórico")
        & (pd.to_datetime(df_forecast_auto["date"]).dt.year == 2025)
    ].copy()

    emp = df_forecast_empresa[
        pd.to_datetime(df_forecast_empresa["date"]).dt.year == 2025
    ].copy()

    if prop.empty or emp.empty:
        return pd.DataFrame(), kpis_cero

    prop["date"] = pd.to_datetime(prop["date"]).dt.to_period("M").dt.to_timestamp()
    emp["date"] = pd.to_datetime(emp["date"]).dt.to_period("M").dt.to_timestamp()
    prop["product_id"] = normalizar_product_id(prop["product_id"])
    emp["product_id"] = normalizar_product_id(emp["product_id"])

    costos = obtener_costos_unitarios(df_parametros)

    df = prop.merge(emp, on=["product_id", "date"], how="inner")
    df = df.merge(costos, on="product_id", how="left")
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0)

    if df.empty:
        return pd.DataFrame(), kpis_cero

    filas = []

    for producto, sub in df.groupby("product_id"):
        real = pd.to_numeric(sub["demand_real"], errors="coerce").fillna(0)
        empresa = pd.to_numeric(sub["forecast_company"], errors="coerce").fillna(0)
        propuesta = pd.to_numeric(sub["demand_forecast"], errors="coerce").fillna(0)
        costo = pd.to_numeric(sub["unit_cost"], errors="coerce").fillna(0)

        error_empresa_s = ((empresa - real).abs() * costo).sum()
        error_propuesta_s = ((propuesta - real).abs() * costo).sum()
        ahorro = error_empresa_s - error_propuesta_s

        suma_real = real.sum()
        wmape_empresa = ((empresa - real).abs().sum() / suma_real) if suma_real > 0 else 0
        wmape_propuesta = ((propuesta - real).abs().sum() / suma_real) if suma_real > 0 else 0
        bias_empresa = ((empresa - real).sum() / suma_real) if suma_real > 0 else 0
        bias_propuesta = ((propuesta - real).sum() / suma_real) if suma_real > 0 else 0

        metodo = sub["method_used"].iloc[0] if "method_used" in sub.columns else ""

        filas.append(
            {
                "Producto": producto,
                "Mejor método": metodo,
                "Error empresa S/": float(error_empresa_s),
                "Error propuesta S/": float(error_propuesta_s),
                "Ahorro potencial S/": float(ahorro),
                "wMAPE empresa": float(wmape_empresa),
                "wMAPE propuesta": float(wmape_propuesta),
                "Bias empresa": float(bias_empresa),
                "Bias propuesta": float(bias_propuesta),
            }
        )

    resumen = pd.DataFrame(filas)

    error_empresa_total = resumen["Error empresa S/"].sum()
    error_propuesta_total = resumen["Error propuesta S/"].sum()
    ahorro_total = error_empresa_total - error_propuesta_total
    kpis = {
        "ahorro_total": float(ahorro_total),
        "error_empresa": float(error_empresa_total),
        "error_propuesta": float(error_propuesta_total),
        "skus_comparados": int(resumen["Producto"].nunique()),
    }

    return resumen, kpis

def grafico_ahorro_forecast(df_ahorro: pd.DataFrame):
    top = df_ahorro.copy()
    top = top.sort_values("Ahorro potencial S/", ascending=False).head(10)

    fig = px.bar(
        top,
        x="Ahorro potencial S/",
        y="Producto",
        orientation="h",
        title="Top 10 SKUs con mayor ahorro potencial",
        labels={
            "Ahorro potencial S/": "Ahorro potencial (S/)",
            "Producto": "SKU",
        },
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def preparar_comparacion_economica_sku(
    df_forecast_auto: pd.DataFrame,
    df_forecast_empresa: pd.DataFrame,
    df_parametros: pd.DataFrame,
    producto: str,
) -> pd.DataFrame:
    if (
        df_forecast_auto is None
        or df_forecast_auto.empty
        or df_forecast_empresa is None
        or df_forecast_empresa.empty
    ):
        return pd.DataFrame()

    producto_norm = normalizar_product_id(pd.Series([producto])).iloc[0]

    prop = df_forecast_auto[
        (df_forecast_auto["tipo_periodo"] == "Histórico")
        & (pd.to_datetime(df_forecast_auto["date"]).dt.year == 2025)
    ].copy()
    emp = df_forecast_empresa[
        pd.to_datetime(df_forecast_empresa["date"]).dt.year == 2025
    ].copy()

    if prop.empty or emp.empty:
        return pd.DataFrame()

    prop["date"] = pd.to_datetime(prop["date"]).dt.to_period("M").dt.to_timestamp()
    emp["date"] = pd.to_datetime(emp["date"]).dt.to_period("M").dt.to_timestamp()
    prop["product_id"] = normalizar_product_id(prop["product_id"])
    emp["product_id"] = normalizar_product_id(emp["product_id"])

    prop = prop[prop["product_id"] == producto_norm]
    emp = emp[emp["product_id"] == producto_norm]

    if prop.empty or emp.empty:
        return pd.DataFrame()

    costos = obtener_costos_unitarios(df_parametros)
    df = prop.merge(emp, on=["product_id", "date"], how="inner")
    df = df.merge(costos, on="product_id", how="left")

    if df.empty:
        return pd.DataFrame()

    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0)
    df["Venta Real"] = pd.to_numeric(df["demand_real"], errors="coerce").fillna(0)
    df["Forecast Empresa"] = pd.to_numeric(df["forecast_company"], errors="coerce").fillna(0)
    df["Forecast Propuesto"] = pd.to_numeric(df["demand_forecast"], errors="coerce").fillna(0)
    df["Error Empresa (S/)"] = (df["Forecast Empresa"] - df["Venta Real"]).abs() * df["unit_cost"]
    df["Error Propuesta (S/)"] = (df["Forecast Propuesto"] - df["Venta Real"]).abs() * df["unit_cost"]
    df["Ahorro Potencial (S/)"] = df["Error Empresa (S/)"] - df["Error Propuesta (S/)"]
    df["Mes"] = pd.to_datetime(df["date"]).dt.strftime("%b %Y").str.upper()

    return df.sort_values("date").reset_index(drop=True)


def grafico_comparacion_economica_sku(df_economico_sku: pd.DataFrame):
    df_plot = df_economico_sku[[
        "date",
        "Venta Real",
        "Forecast Empresa",
        "Forecast Propuesto",
    ]].copy()
    df_plot = df_plot.melt(
        id_vars="date",
        value_vars=["Venta Real", "Forecast Empresa", "Forecast Propuesto"],
        var_name="Serie",
        value_name="Unidades",
    )

    fig = px.line(
        df_plot,
        x="date",
        y="Unidades",
        color="Serie",
        markers=True,
        title="Ventas reales vs forecast comercial vs forecast propuesto",
        labels={"date": "Mes", "Unidades": "Unidades"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig

def grafico_modelos_ganadores(df_comparacion: pd.DataFrame):
    mejores = df_comparacion[df_comparacion["Es mejor"]].copy()
    conteo = mejores["Método"].value_counts().reset_index()
    conteo.columns = ["Método", "Cantidad de SKUs"]

    fig = px.pie(
        conteo,
        names="Método",
        values="Cantidad de SKUs",
        hole=0.45,
        title="Distribución de modelos ganadores",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig



# =========================================================
# FUNCIONES AUXILIARES - TVU POR LOTE
# =========================================================
def _normalizar_nombre_columna(columna) -> str:
    return (
        str(columna)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("_", " ")
        .replace("-", " ")
    )


def _mapear_columnas_tvu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza columnas de la hoja TVU aunque vengan con espacios,
    mayúsculas/minúsculas o nombres alternativos.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    alias = {
        "lote": "lote",
        "lot": "lote",
        "batch": "lote",
        "batch number": "lote",
        "n lote": "lote",
        "nro lote": "lote",
        "numero lote": "lote",
        "número lote": "lote",

        "demanda agrupada final": "producto",
        "producto": "producto",
        "sku": "producto",
        "product id": "producto",
        "product": "producto",
        "id producto": "producto",
        "codigo": "producto",
        "código": "producto",
        "grupo de demanda": "producto",
        "grupo demanda": "producto",

        "tvu": "tvu",
        "tvu months": "tvu",
        "tvu meses": "tvu",
        "vida util": "tvu",
        "vida útil": "tvu",
        "meses vencimiento": "tvu",

        "costo uni": "costo_unitario",
        "costo unitario": "costo_unitario",
        "costo unit": "costo_unitario",
        "costo": "costo_unitario",
        "costo_unitario": "costo_unitario",
        "unit cost": "costo_unitario",
        "unit value": "costo_unitario",
        "unit_cost": "costo_unitario",
        "unit_value": "costo_unitario",
        "valor unitario": "costo_unitario",

        "warehouse": "warehouse",
        "almacen": "warehouse",
        "almacén": "warehouse",
        "bodega": "warehouse",
        "subinventario": "warehouse",

        "stock": "stock",
        "stock actual": "stock",
        "stock_actual": "stock",
        "initial stock": "stock",
        "initial_stock": "stock",
        "inventario": "stock",
    }

    df = df.copy()
    columnas_mapeadas = {}
    for c in df.columns:
        c_norm = _normalizar_nombre_columna(c)
        columnas_mapeadas[c] = alias.get(c_norm, c_norm)

    return df.rename(columns=columnas_mapeadas)


def leer_tvu_desde_excel(xls: pd.ExcelFile) -> pd.DataFrame:
    """
    Lee la hoja TVU del Excel. Si no existe, devuelve DataFrame vacío.
    """
    if xls is None or "TVU" not in xls.sheet_names:
        return pd.DataFrame()

    return pd.read_excel(xls, sheet_name="TVU")


def clasificar_riesgo_tvu(tvu):
    """
    Clasificación por TVU.
    Se usa < 5 para que valores decimales como 4.5 no queden sin clasificar.
    """
    valor = pd.to_numeric(tvu, errors="coerce")

    if pd.isna(valor) or valor <= 0:
        return "Sin dato"
    if 0 <= valor < 5:
        return "🔴 Alto"
    if 5 <= valor <= 10:
        return "🟡 Medio"
    if valor > 10:
        return "🟢 Bajo"

    return "Sin dato"


def preparar_tvu_lotes(df_tvu_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el TVU por lote usando la hoja TVU.
    Cada fila representa un lote con su propio TVU, stock y costo.
    """
    columnas_salida = [
        "lote",
        "producto",
        "warehouse",
        "tvu",
        "stock",
        "costo_unitario",
        "valor_en_riesgo",
        "riesgo_tvu",
        "orden_riesgo",
    ]

    if df_tvu_raw is None or df_tvu_raw.empty:
        return pd.DataFrame(columns=columnas_salida)

    df = _mapear_columnas_tvu(df_tvu_raw)

    for col in ["lote", "producto", "warehouse", "tvu", "stock", "costo_unitario"]:
        if col not in df.columns:
            df[col] = None

    df = df[["lote", "producto", "warehouse", "tvu", "stock", "costo_unitario"]].copy()

    df["lote"] = df["lote"].astype(str).str.strip()
    df["producto"] = normalizar_product_id(df["producto"])
    df["warehouse"] = df["warehouse"].astype(str).str.strip()

    df["tvu"] = pd.to_numeric(df["tvu"], errors="coerce")
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0)
    df["costo_unitario"] = pd.to_numeric(df["costo_unitario"], errors="coerce").fillna(0)

    df["valor_en_riesgo"] = df["stock"] * df["costo_unitario"]
    df["riesgo_tvu"] = df["tvu"].apply(clasificar_riesgo_tvu)

    orden = {
        "🔴 Alto": 1,
        "🟡 Medio": 2,
        "🟢 Bajo": 3,
        "Sin dato": 4,
    }
    df["orden_riesgo"] = df["riesgo_tvu"].map(orden).fillna(4).astype(int)

    df = df.sort_values(
        ["orden_riesgo", "tvu", "valor_en_riesgo"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)

    return df[columnas_salida]


def resumen_tvu_lotes(df_tvu: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Genera resumen por riesgo y KPIs superiores del módulo TVU.
    """
    if df_tvu is None or df_tvu.empty:
        resumen = pd.DataFrame(
            {
                "riesgo_tvu": ["🔴 Alto", "🟡 Medio", "🟢 Bajo", "Sin dato"],
                "cantidad_lotes": [0, 0, 0, 0],
                "stock": [0, 0, 0, 0],
                "valor_en_riesgo": [0.0, 0.0, 0.0, 0.0],
            }
        )
        kpis = {
            "lotes_alto": 0,
            "lotes_medio": 0,
            "stock_riesgo": 0.0,
            "valor_riesgo": 0.0,
            "lote_critico": "Sin datos",
        }
        return resumen, kpis

    orden_categorias = ["🔴 Alto", "🟡 Medio", "🟢 Bajo", "Sin dato"]

    resumen = (
        df_tvu.groupby("riesgo_tvu", as_index=False)
        .agg(
            cantidad_lotes=("lote", "count"),
            stock=("stock", "sum"),
            valor_en_riesgo=("valor_en_riesgo", "sum"),
        )
    )

    resumen = (
        pd.DataFrame({"riesgo_tvu": orden_categorias})
        .merge(resumen, on="riesgo_tvu", how="left")
        .fillna({"cantidad_lotes": 0, "stock": 0, "valor_en_riesgo": 0})
    )

    resumen["cantidad_lotes"] = resumen["cantidad_lotes"].astype(int)

    en_riesgo = df_tvu[df_tvu["riesgo_tvu"].isin(["🔴 Alto", "🟡 Medio"])].copy()

    kpis = {
        "lotes_alto": int((df_tvu["riesgo_tvu"] == "🔴 Alto").sum()),
        "lotes_medio": int((df_tvu["riesgo_tvu"] == "🟡 Medio").sum()),
        "stock_riesgo": float(en_riesgo["stock"].sum()) if not en_riesgo.empty else 0.0,
        "valor_riesgo": float(en_riesgo["valor_en_riesgo"].sum()) if not en_riesgo.empty else 0.0,
        "lote_critico": str(en_riesgo.iloc[0]["lote"]) if not en_riesgo.empty else "Sin datos",
    }

    return resumen, kpis


def resumen_tvu_por_producto(df_tvu: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por DEMANDA AGRUPADA FINAL / producto.
    """
    columnas = [
        "Producto",
        "Cantidad de lotes",
        "Stock total",
        "Valor total en riesgo",
        "TVU mínimo",
        "Riesgo más crítico",
    ]

    if df_tvu is None or df_tvu.empty:
        return pd.DataFrame(columns=columnas)

    df = df_tvu.copy()
    resumen = (
        df.groupby("producto", as_index=False)
        .agg(
            cantidad_lotes=("lote", "count"),
            stock_total=("stock", "sum"),
            valor_total_en_riesgo=("valor_en_riesgo", "sum"),
            tvu_minimo=("tvu", "min"),
            orden_mas_critico=("orden_riesgo", "min"),
        )
    )

    mapa_riesgo = {
        1: "🔴 Alto",
        2: "🟡 Medio",
        3: "🟢 Bajo",
        4: "Sin dato",
    }
    resumen["riesgo_mas_critico"] = resumen["orden_mas_critico"].map(mapa_riesgo)

    resumen = resumen.sort_values(
        ["orden_mas_critico", "valor_total_en_riesgo", "tvu_minimo"],
        ascending=[True, False, True],
        na_position="last",
    )

    resumen = resumen.rename(
        columns={
            "producto": "Producto",
            "cantidad_lotes": "Cantidad de lotes",
            "stock_total": "Stock total",
            "valor_total_en_riesgo": "Valor total en riesgo",
            "tvu_minimo": "TVU mínimo",
            "riesgo_mas_critico": "Riesgo más crítico",
        }
    )

    return resumen[columnas]


def grafico_tvu_lotes_riesgo(df_resumen: pd.DataFrame):
    fig = px.bar(
        df_resumen,
        x="riesgo_tvu",
        y="cantidad_lotes",
        title="Cantidad de lotes por nivel de riesgo",
        labels={
            "riesgo_tvu": "Riesgo",
            "cantidad_lotes": "Cantidad de lotes",
        },
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def grafico_tvu_lotes_valor(df_resumen: pd.DataFrame):
    df = df_resumen[df_resumen["riesgo_tvu"].isin(["🔴 Alto", "🟡 Medio", "🟢 Bajo"])].copy()

    fig = px.pie(
        df,
        names="riesgo_tvu",
        values="valor_en_riesgo",
        hole=0.45,
        title="Valor económico por riesgo",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def grafico_tvu_alto_medio(resumen_vencimientos: pd.DataFrame):
    df = resumen_vencimientos.copy()
    df = df[df["riesgo_tvu"].isin(["🔴 Alto", "🟡 Medio"])]

    fig = px.pie(
        df,
        names="riesgo_tvu",
        values="valor_en_riesgo",
        hole=0.45,
        title="Valor en riesgo por vencimiento",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def formatear_tvu_lotes(df_tvu: pd.DataFrame) -> pd.DataFrame:
    if df_tvu is None or df_tvu.empty:
        return pd.DataFrame(
            columns=[
                "Lote",
                "Producto",
                "Warehouse",
                "TVU",
                "Stock",
                "Costo unitario",
                "Valor en riesgo",
                "Riesgo",
            ]
        )

    df = df_tvu.copy()
    df = df.rename(
        columns={
            "lote": "Lote",
            "producto": "Producto",
            "warehouse": "Warehouse",
            "tvu": "TVU",
            "stock": "Stock",
            "costo_unitario": "Costo unitario",
            "valor_en_riesgo": "Valor en riesgo",
            "riesgo_tvu": "Riesgo",
        }
    )

    return df[
        [
            "Lote",
            "Producto",
            "Warehouse",
            "TVU",
            "Stock",
            "Costo unitario",
            "Valor en riesgo",
            "Riesgo",
        ]
    ]


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Inventory Intelligence Framework",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Framework de Optimización de Inventarios")
st.caption(
    "Pronóstico mensual + selección automática del mejor método por producto + simulación + optimización de inventarios"
)

# =========================================================
# SIDEBAR - SELECCIÓN DE MÓDULO
# =========================================================
st.sidebar.header("Módulo")

modulo = st.sidebar.radio(
    "Seleccione una herramienta",
    [
        "📊 Vista General Ejecutiva",
        "📈 Pronósticos e Inventarios",
        "⚠️ TVU - Productos próximos a vencer",
    ],
)

# =========================================================
# SIDEBAR - CARGA DE DATOS ÚNICA
# =========================================================
st.sidebar.header("1. Carga de datos")

modo_datos = st.sidebar.radio(
    "Modo de datos",
    ["Generar datos sintéticos", "Subir Excel (Pestañas: Demanda y Datos)"],
)

if modo_datos == "Generar datos sintéticos":
    n_productos = st.sidebar.slider("Número de productos", 1, 50, 5)
    meses = st.sidebar.slider("Meses de historial", 12, 84, 36)
    seed = st.sidebar.number_input("Semilla", min_value=1, max_value=9999, value=42)

    df_real = generar_demanda_sintetica(
        n_productos=n_productos,
        meses=meses,
        seed=seed,
    )
    df_parametros = pd.DataFrame()
    df_forecast_empresa = pd.DataFrame(columns=["date", "product_id", "forecast_company"])
    df_tvu_raw = pd.DataFrame()
    
else:
    archivo = st.sidebar.file_uploader(
        "Sube tu archivo Excel unificado",
        type=["xlsx", "xls"],
    )

    if archivo is None:
        st.info(
            "Sube un archivo Excel que contenga al menos dos pestañas:\n"
            "1. 'Demanda': historial con date, product_id, demand_real.\n"
            "2. 'Datos': maestro de artículos.\n"
            "Opcional: 'Forecast_Comercial' con date, product_id, forecast_company."
        )
        st.stop()

    try:
        xls = pd.ExcelFile(archivo)

        if "Demanda" in xls.sheet_names:
            df_demanda_raw = pd.read_excel(xls, sheet_name="Demanda")
        else:
            df_demanda_raw = pd.read_excel(xls, sheet_name=0)

        df_demanda_raw.columns = [
            str(c).strip().lower() for c in df_demanda_raw.columns
        ]

        alias = {
            "fecha": "date",
            "mes": "date",
            "periodo": "date",
            "día": "date",
            "dia": "date",
            "producto": "product_id",
            "sku": "product_id",
            "id_producto": "product_id",
            "codigo": "product_id",
            "código": "product_id",
            "demanda": "demand_real",
            "venta": "demand_real",
            "ventas": "demand_real",
            "cantidad": "demand_real",
            "unidades": "demand_real",
        }

        df_demanda_raw = df_demanda_raw.rename(
            columns={c: alias.get(c, c) for c in df_demanda_raw.columns}
        )

        df_real = convertir_a_mensual(df_demanda_raw)

        if "Datos" in xls.sheet_names:
            df_parametros = pd.read_excel(xls, sheet_name="Datos")
        else:
            st.error(
                "⚠️ El archivo Excel no tiene una pestaña llamada 'Datos'. "
                "Por favor, agrégala y vuelve a subir el archivo."
            )
            st.stop()

        df_forecast_empresa = leer_forecast_comercial_opcional(xls)
        df_tvu_raw = leer_tvu_desde_excel(xls)
        
    except Exception as e:
        st.error(f"Error procesando el archivo: {str(e)}")
        st.stop()

# =========================================================
# TVU - RIESGO DE VENCIMIENTO POR LOTE
# =========================================================
df_tvu = preparar_tvu_lotes(df_tvu_raw)
resumen_vencimientos, kpis_tvu = resumen_tvu_lotes(df_tvu)
resumen_productos_tvu = resumen_tvu_por_producto(df_tvu)

valor_tvu_riesgo = kpis_tvu["valor_riesgo"]
total_lotes_tvu = len(df_tvu)
total_skus_tvu = df_tvu["producto"].nunique() if not df_tvu.empty else 0

# =========================================================
# MÓDULO TVU INDEPENDIENTE
# =========================================================
if modulo == "⚠️ TVU - Productos próximos a vencer":
    st.subheader("⚠️ Infografía TVU: Productos próximos a vencer")

    st.write(
        "Clasificación por lote usando la hoja **TVU**. "
        "Riesgo alto: TVU desde 1 hasta antes de 5; riesgo medio: TVU de 5 a 10; "
        "riesgo bajo: TVU mayor a 10."
    )

    if df_tvu.empty:
        st.warning(
            "No se pudo construir la infografía TVU. Verifica que el Excel tenga una hoja llamada "
            "'TVU' con columnas: Lote, DEMANDA AGRUPADA FINAL, TVU, COSTO UNI, WAREHOUSE y STOCK."
        )
    else:
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)

        col_t1.metric("🔴 Lotes riesgo alto", f"{kpis_tvu['lotes_alto']:,}")
        col_t2.metric("🟡 Lotes riesgo medio", f"{kpis_tvu['lotes_medio']:,}")
        col_t3.metric("Stock en riesgo", f"{kpis_tvu['stock_riesgo']:,.0f}")
        col_t4.metric("Valor en riesgo", f"S/ {kpis_tvu['valor_riesgo']:,.2f}")

        st.info(f"Lote más crítico: **{kpis_tvu['lote_critico']}**")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.plotly_chart(
                grafico_tvu_lotes_riesgo(resumen_vencimientos),
                use_container_width=True,
            )

        with col_g2:
            st.plotly_chart(
                grafico_tvu_lotes_valor(resumen_vencimientos),
                use_container_width=True,
            )

        st.markdown("### 🚨 Top 10 lotes más críticos")

        top_10 = df_tvu.copy().head(10)

        st.dataframe(
            formatear_tvu_lotes(top_10),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 📦 Resumen por producto agrupado")

        st.dataframe(
            resumen_productos_tvu.style.format({
                "Stock total": "{:,.0f}",
                "Valor total en riesgo": "S/ {:,.2f}",
                "TVU mínimo": "{:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 📋 Detalle completo TVU")

        st.dataframe(
            formatear_tvu_lotes(df_tvu).style.format({
                "TVU": "{:,.2f}",
                "Stock": "{:,.0f}",
                "Costo unitario": "S/ {:,.2f}",
                "Valor en riesgo": "S/ {:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="📥 Descargar TVU por lote (CSV)",
            data=df_tvu.to_csv(index=False).encode("utf-8"),
            file_name="reporte_tvu_lotes.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.stop()

# =========================================================
# PRONÓSTICO MENSUAL
# =========================================================
st.sidebar.header("2. Pronóstico mensual")

modo_pronostico = st.sidebar.selectbox(
    "Selección del método",
    ["Automático: mejor método por producto", "Manual: elegir un método"],
)

ultima_fecha_historica = pd.to_datetime(df_real["date"].max()).to_period("M").to_timestamp()

fecha_fin_pronostico = st.sidebar.date_input(
    "Pronosticar hasta",
    value=pd.Timestamp("2026-12-01"),
    min_value=ultima_fecha_historica.date(),
)

fecha_fin_pronostico = pd.to_datetime(fecha_fin_pronostico).to_period("M").to_timestamp()

df_forecast_auto, df_comparacion = generar_forecast_mejor_por_producto(
    df_real,
    fecha_fin_pronostico=fecha_fin_pronostico,
)

# =========================================================
# RESUMEN FORECAST PARA VISTA GENERAL
# =========================================================
total_skus_forecast = df_real["product_id"].nunique()

df_ahorro_forecast, kpis_forecast = calcular_ahorro_forecast_2025(
    df_forecast_auto=df_forecast_auto,
    df_forecast_empresa=df_forecast_empresa,
    df_parametros=df_parametros,
)

ahorro_total = kpis_forecast["ahorro_total"]
skus_comparados_forecast = kpis_forecast["skus_comparados"]

resumen_mejores_exec = df_comparacion[df_comparacion["Es mejor"]].copy()
modelo_mas_usado = (
    resumen_mejores_exec["Método"].mode().iloc[0]
    if not resumen_mejores_exec.empty
    else "Sin datos"
)

# =========================================================
# MÓDULO VISTA GENERAL EJECUTIVA
# =========================================================
if modulo == "📊 Vista General Ejecutiva":
    st.title("📊 Dashboard Ejecutivo")
    st.caption("Vista general del desempeño del portafolio: forecast, riesgo de vencimiento y modelos ganadores.")

    total_skus = max(total_skus_tvu, total_skus_forecast)
    impacto_identificado = ahorro_total + valor_tvu_riesgo

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("SKU evaluados", f"{total_skus:,}")
    c2.metric("Ahorro Potencial Total", f"S/ {ahorro_total:,.0f}")
    c3.metric("Valor en Riesgo TVU", f"S/ {valor_tvu_riesgo:,.0f}")
    c4.metric("Impacto Económico Identificado", f"S/ {impacto_identificado:,.0f}")
    c5.metric("Modelo más utilizado", modelo_mas_usado)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📈 Ahorro potencial por forecast")

        if df_ahorro_forecast.empty:
            st.info(
                "No se calculó ahorro potencial. Para activarlo, el Excel debe incluir "
                "Forecast_Comercial con date, product_id y forecast_company, y la hoja Datos debe tener unit_value o unit_cost."
            )
        else:
            st.plotly_chart(
                grafico_ahorro_forecast(df_ahorro_forecast),
                use_container_width=True,
            )

    with col_b:
        st.subheader("⚠️ Valor en riesgo por vencimiento")

        if resumen_vencimientos.empty or valor_tvu_riesgo <= 0:
            st.info("No hay productos en riesgo alto o medio.")
        else:
            st.plotly_chart(
                grafico_tvu_alto_medio(resumen_vencimientos),
                use_container_width=True,
            )

    st.divider()

    st.subheader("🧠 Distribución de modelos ganadores")

    if resumen_mejores_exec.empty:
        st.info("No hay métodos ganadores disponibles.")
    else:
        st.plotly_chart(
            grafico_modelos_ganadores(df_comparacion),
            use_container_width=True,
        )

    st.stop()

# =========================================================
# CONFIGURACIÓN DEL MÓDULO PRONÓSTICOS E INVENTARIOS
# =========================================================
if modo_pronostico == "Manual: elegir un método":
    metodo_manual = st.sidebar.selectbox("Método manual", METODOS_PRONOSTICO)
    df_forecast = generar_forecast(
        df_real,
        metodo_manual,
        fecha_fin_pronostico=fecha_fin_pronostico,
    )
else:
    metodo_manual = None
    df_forecast = df_forecast_auto

productos = sorted(df_forecast["product_id"].unique())
producto_sel = st.sidebar.selectbox("Producto a visualizar", productos)

sub_comparacion_producto = df_comparacion[
    df_comparacion["Producto"] == producto_sel
].copy()

mejor_metodo_producto = sub_comparacion_producto.loc[
    sub_comparacion_producto["Es mejor"],
    "Método",
].iloc[0]

mejor_wmape_producto = sub_comparacion_producto.loc[
    sub_comparacion_producto["Es mejor"],
    "wMAPE",
].iloc[0]

if modo_pronostico == "Automático: mejor método por producto":
    st.sidebar.success(f"Método elegido para {producto_sel}: {mejor_metodo_producto}")
else:
    st.sidebar.info(f"Mejor método para {producto_sel}: {mejor_metodo_producto}")

# =========================================================
# POLÍTICA DE INVENTARIO
# =========================================================
st.sidebar.header("3. Política de Inventario")

politica = st.sidebar.selectbox(
    "Política (Modo Simulación)",
    [
        "RS - revisión periódica",
        "sS - punto de reorden y nivel máximo",
        "sQ - punto de reorden y cantidad fija",
    ],
)

ss_max = st.sidebar.slider("Máximo SS para optimizar (meses)", 1, 24, 6)

parametros_del_producto = obtener_parametros_producto(df_parametros, producto_sel)

# =========================================================
# CONTENIDO PRINCIPAL
# =========================================================
sub_forecast = df_forecast[df_forecast["product_id"] == producto_sel].copy()
metodo_usado = sub_forecast["method_used"].iloc[0]

sub_sim = simular_producto(sub_forecast, politica, parametros_del_producto)
kpis = calcular_kpis(sub_sim, parametros_del_producto)
sub_opt = optimizar_stock_seguridad(
    sub_forecast,
    politica,
    parametros_del_producto,
    ss_max=ss_max,
)
mejor = sub_opt.loc[sub_opt["total_cost"].idxmin()]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Método usado", metodo_usado)
col2.metric("Fill rate", f"{kpis['fill_rate']:.2%}")
col3.metric("Inventario promedio", f"{kpis['avg_inventory']:.1f}")
col4.metric("Ventas perdidas", f"{kpis['lost_sales_units']:.0f}")
col5.metric("Costo total", f"S/ {kpis['total_cost']:,.2f}")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Mejor método",
    "📊 Datos y pronóstico",
    "💰 Comparación económica",
    "📦 Simulación",
    "🎯 Optimización",
    "📋 Tablas",
])

# =========================================================
# TAB 1: MEJOR MÉTODO
# =========================================================
with tab1:
    st.subheader("🏆 Análisis Estratégico: Mejor Método por Producto")
    st.write(
        "El framework evalúa todos los modelos mediante Validación Cruzada y selecciona el ganador "
        "basado en el menor wMAPE, utilizando el RMSE y el Bias como criterios de desempate."
    )

    resumen_mejores = (
        df_comparacion[df_comparacion["Es mejor"]]
        .copy()
        .sort_values("Producto")
    )

    resumen_mejores = resumen_mejores[[
        "Producto", "Método", "wMAPE", "Bias", "MAE",
    ]].rename(columns={"Método": "Mejor método"})

    col_graf, col_tabla = st.columns([1.2, 1])

    conteo_metodos = resumen_mejores["Mejor método"].value_counts().reset_index()
    conteo_metodos.columns = ["Método", "Cantidad de Productos"]
    conteo_metodos["Porcentaje"] = (
        conteo_metodos["Cantidad de Productos"] / len(resumen_mejores)
    ) * 100

    with col_graf:
        fig_donut = px.pie(
            conteo_metodos,
            names="Método",
            values="Cantidad de Productos",
            hole=0.45,
            title="Distribución de Métodos Ganadores",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        fig_donut.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True, key="chart_donut_metodos")

    with col_tabla:
        st.write("<br>", unsafe_allow_html=True)
        st.markdown("**Resumen de Asignación de Modelos**")
        st.dataframe(
            conteo_metodos,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Cantidad de Productos": st.column_config.ProgressColumn(
                    "Cantidad",
                    format="%d",
                    min_value=0,
                    max_value=int(conteo_metodos["Cantidad de Productos"].max()),
                ),
                "Porcentaje": st.column_config.NumberColumn(
                    "% del Portafolio",
                    format="%.1f %%",
                ),
            },
        )

    st.divider()
    st.subheader("🔎 Detalle por Producto")

    metodos_disponibles = conteo_metodos["Método"].tolist()
    filtro_metodos = st.multiselect(
        "Filtra la tabla por Método Ganador:",
        options=metodos_disponibles,
        default=metodos_disponibles,
    )

    df_mostrar = resumen_mejores[
        resumen_mejores["Mejor método"].isin(filtro_metodos)
    ].copy()

    df_mostrar["wMAPE"] = df_mostrar["wMAPE"] * 100
    df_mostrar["Bias"] = df_mostrar["Bias"] * 100

    st.dataframe(
        df_mostrar,
        hide_index=True,
        use_container_width=True,
        column_config={
            "wMAPE": st.column_config.NumberColumn(
                "wMAPE (%)",
                help="Error Porcentual Absoluto Medio Ponderado",
                format="%.2f %%",
            ),
            "Bias": st.column_config.NumberColumn(
                "Bias (%)",
                help="Sesgo del pronóstico (Positivo = Sobrepronóstico, Negativo = Subpronóstico)",
                format="%.2f %%",
            ),
            "MAE": st.column_config.NumberColumn("MAE (Unidades)", format="%.2f"),
        },
    )

    st.write("<br>", unsafe_allow_html=True)

    csv_mejores = resumen_mejores.to_csv(index=False).encode("utf-8")
    
    st.download_button(
        label="📥 Descargar detalle completo en CSV",
        data=resumen_mejores.to_csv(index=False).encode("utf-8"),
        file_name="mejor_metodo_por_producto.csv",
        mime="text/csv",
    )

# =========================================================
# TAB 2: DATOS Y PRONÓSTICO
# =========================================================
with tab2:
    st.subheader("📊 Análisis de Demanda y Proyección")
    st.write(
        f"Visualización del comportamiento histórico frente al modelo seleccionado: **{metodo_usado}**."
    )

    col_g1, col_g2 = st.columns([3, 1])

    with col_g1:
        fig = grafico_forecast(sub_forecast)
        st.plotly_chart(fig, use_container_width=True, key="chart_pronostico_sku")

    with col_g2:
        st.markdown("### 🎯 Resumen del Modelo")
        st.metric("Método Seleccionado", metodo_usado)
        st.metric("wMAPE (Error)", f"{mejor_wmape_producto:.2%}")

        st.markdown("---")
        st.markdown("**Insights clave:**")

        if mejor_wmape_producto < 0.20:
            st.success("Modelo de alta precisión. Apto para compras automáticas.")
        elif mejor_wmape_producto < 0.50:
            st.warning("Modelo con precisión moderada. Se recomienda revisión manual.")
        else:
            st.error("Precisión baja. Posible demanda errática o quiebre de stock.")

    st.markdown("### 📋 Comparativa de Métodos (Validación Cruzada)")
    df_comp = formatear_comparacion(sub_comparacion_producto)

    def highlight_best(row):
        return ["background-color: #d4edda" if "✅" in str(val) else "" for val in row]

    st.dataframe(
        df_comp.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# TAB 3: COMPARACIÓN ECONÓMICA Y POLÍTICA LOGÍSTICA
# =========================================================
with tab3:
    st.subheader("💰 Evaluación Económica Integral del SKU")
    st.write(
        "Análisis financiero conjunto: parámetros óptimos de reposición de inventario, "
        "riesgo financiero por quiebre de stock y ahorro en precisión de pronóstico."
    )

    # -----------------------------------------------------------------
    # 1. POLÍTICA DE REPOSICIÓN Y COSTO DE QUIEBRE (Lo solicitado)
    # -----------------------------------------------------------------
    st.markdown("### 📦 Configuración Logística Óptima y Riesgo de Quiebre")
    
    q_opt_val = mejor.get("q_optimo", parametros_del_producto.q_fixed)
    r_opt_val = mejor.get("r_optimo", parametros_del_producto.review_period_months)
    
    if politica == "sQ - punto de reorden y cantidad fija":
        etiqueta_param = "Lote de Compra (Q)"
        valor_param = f"{q_opt_val:,.0f} unds"
    else:
        etiqueta_param = "Periodo de Revisión (R)"
        valor_param = f"{r_opt_val:.0f} meses"

    col_inv1, col_inv2, col_inv3, col_inv4 = st.columns(4)
    col_inv1.metric("Método de Revisión", politica.split(" - ")[0])
    col_inv2.metric("Stock Seguridad (SS)", f"{int(mejor['ss_months'])} meses")
    col_inv3.metric(etiqueta_param, valor_param)
    col_inv4.metric(
        "🚨 Costo de Quiebre Proyectado", 
        f"S/ {mejor['stockout_cost']:,.2f}",
        help=f"Penalidad total por {mejor['lost_sales_units']:,.0f} unidades de venta perdida calculadas a S/ {parametros_del_producto.cost_stockout:,.2f}/und."
    )

    st.divider()

    # -----------------------------------------------------------------
    # 2. COMPARATIVA COMERCIAL DE PRONÓSTICO (Ahorro Potencial)
    # -----------------------------------------------------------------
    st.markdown("### 🎯 Ahorro Potencial por Precisión de Pronóstico")

    df_economico_sku = preparar_comparacion_economica_sku(
        df_forecast_auto=df_forecast_auto,
        df_forecast_empresa=df_forecast_empresa,
        df_parametros=df_parametros,
        producto=producto_sel,
    )

    if df_economico_sku.empty:
        st.info(
            "No se pudo calcular la comparativa comercial para este SKU. "
            "Asegúrate de que tenga datos en la pestaña 'Forecast_Comercial'."
        )
    else:
        error_empresa_sku = df_economico_sku["Error Empresa (S/)"].sum()
        error_propuesta_sku = df_economico_sku["Error Propuesta (S/)"].sum()
        ahorro_sku = df_economico_sku["Ahorro Potencial (S/)"].sum()

        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("Error Forecast Empresa", f"S/ {error_empresa_sku:,.2f}")
        col_e2.metric("Error Forecast Propuesto", f"S/ {error_propuesta_sku:,.2f}")
        col_e3.metric(
            "💡 Ahorro Potencial (S/)", 
            f"S/ {ahorro_sku:,.2f}",
            delta=f"S/ {ahorro_sku:,.2f} de ahorro" if ahorro_sku >= 0 else f"S/ {ahorro_sku:,.2f}"
        )

        st.plotly_chart(
            grafico_comparacion_economica_sku(df_economico_sku),
            use_container_width=True,
            key="chart_comp_economica"
        )

        st.markdown("#### Detalle Mensual Económico (S/)")
        tabla_economica = df_economico_sku[[
            "Mes", "Venta Real", "Forecast Empresa", "Forecast Propuesto",
            "Error Empresa (S/)", "Error Propuesta (S/)", "Ahorro Potencial (S/)",
        ]].copy()

        st.dataframe(
            tabla_economica.style.format({
                "Venta Real": "{:,.0f}",
                "Forecast Empresa": "{:,.0f}",
                "Forecast Propuesto": "{:,.0f}",
                "Error Empresa (S/)": "{:,.2f}",
                "Error Propuesta (S/)": "{:,.2f}",
                "Ahorro Potencial (S/)": "{:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        
# =========================================================
# TAB 4: SIMULACIÓN (Recuperada)
# =========================================================
with tab4:
    st.subheader("📦 Simulación Dinámica de Inventario")
    st.write("Evolución del stock físico frente a la demanda y generación de órdenes de compra según la política seleccionada.")

    st.plotly_chart(grafico_inventario(sub_sim), use_container_width=True, key="chart_simulacion_inventario")

    st.markdown("---")
    st.subheader("📊 Indicadores de Desempeño (KPIs) del Escenario Actual")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.markdown("**Nivel de Servicio**")
        fill_rate_val = kpis['fill_rate']
        st.metric("Fill Rate", f"{fill_rate_val:.2%}")
        st.progress(min(fill_rate_val, 1.0))
        if fill_rate_val < 0.90:
            st.error(f"¡Atención! Ventas perdidas: {int(kpis['lost_sales_units'])} unds.")
        else:
            st.success("Nivel de servicio óptimo.")

    with col_kpi2:
        st.markdown("**Operaciones de Almacén**")
        st.metric("Inventario Promedio", f"{kpis['avg_inventory']:,.0f} unds")
        st.metric("Órdenes Emitidas", f"{kpis['orders']} pedidos")
        st.metric("Meses con Quiebre", f"{kpis['stockout_months']} meses")

    with col_kpi3:
        st.markdown("**Análisis Financiero**")
        st.metric("Costo de Mantener", f"S/ {kpis['holding_cost']:,.2f}")
        st.metric("Costo de Quiebre (Penalidad)", f"S/ {kpis['stockout_cost']:,.2f}")
        st.metric("Costo de Ordenar", f"S/ {kpis['ordering_cost']:,.2f}")
        
    st.info(f"**Costo Total de la Política Actual:** S/ {kpis['total_cost']:,.2f}")

# =========================================================
# TAB 5: OPTIMIZACIÓN (2D Co-Optimización)
# =========================================================
with tab5:
    st.subheader("🎯 Optimización Financiera del Stock de Seguridad y Parámetros de Pedido")
    st.write(
        "Análisis de co-optimización (Grid Search 2D) para encontrar el equilibrio exacto entre "
        "el Stock de Seguridad (SS) y los parámetros logísticos de reposición (Lote Q o Periodo R)."
    )
    
    q_opt_val = mejor.get("q_optimo", parametros_del_producto.q_fixed)
    r_opt_val = mejor.get("r_optimo", parametros_del_producto.review_period_months)

    if politica == "sQ - punto de reorden y cantidad fija":
        param_extra_texto = f"un Lote de Compra Fijo (Q) óptimo de **{q_opt_val:,.0f} unidades**"
    else:
        param_extra_texto = f"un Periodo de Revisión (R) óptimo de **{r_opt_val:.0f} meses**"

    st.success(
        f"**Recomendación Logística Integral:** Para el producto {producto_sel} bajo la política *{politica}*, "
        f"la configuración óptima es tener un Stock de Seguridad de **{int(mejor['ss_months'])} meses** combinada con {param_extra_texto}.\n\n"
        f"Esta estrategia alcanza el Costo Total mínimo de **S/ {mejor['total_cost']:,.2f}** "
        f"con un Nivel de Servicio (Fill Rate) proyectado del **{mejor['fill_rate']:.2%}**."
    )

    st.plotly_chart(grafico_tradeoff(sub_opt), use_container_width=True, key="chart_tradeoff_optimizacion")

    st.markdown("---")
    st.markdown("### 📋 Tabla de Sensibilidad de Escenarios Co-Optimizados")
    
    df_sensibilidad = sub_opt.copy()
    
    if "q_optimo" not in df_sensibilidad.columns:
        df_sensibilidad["q_optimo"] = parametros_del_producto.q_fixed
    if "r_optimo" not in df_sensibilidad.columns:
        df_sensibilidad["r_optimo"] = parametros_del_producto.review_period_months
    
    df_sensibilidad = df_sensibilidad[[
        "ss_months", "q_optimo", "r_optimo", "fill_rate", 
        "lost_sales_units", "holding_cost", "stockout_cost", "total_cost"
    ]]
    df_sensibilidad.columns = [
        "Meses SS", "Lote Q (Unds)", "Revisión R (Meses)", "Fill Rate", 
        "Ventas Perdidas", "Costo Mantener (S/)", "Costo Quiebre (S/)", "Costo Total (S/)"
    ]
    
    def highlight_optimo(row):
        is_optimo = row["Meses SS"] == int(mejor["ss_months"])
        return ['background-color: #d4edda; font-weight: bold' if is_optimo else '' for _ in row]

    st.dataframe(
        df_sensibilidad.style.apply(highlight_optimo, axis=1)
        .format({
            "Lote Q (Unds)": "{:,.0f}",
            "Revisión R (Meses)": "{:.0f}",
            "Fill Rate": "{:.2%}",
            "Ventas Perdidas": "{:,.0f}",
            "Costo Mantener (S/)": "{:,.2f}",
            "Costo Quiebre (S/)": "{:,.2f}",
            "Costo Total (S/)": "{:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# TAB 6: TABLAS Y REPORTES
# =========================================================
with tab6:
    st.subheader("📋 Tablas de Datos y Reportes")
    st.write("Registros detallados de las proyecciones y simulaciones, formateados para exportación y análisis externo.")

    # -------------------------------------------------------------
    # 1. Tabla: Histórico y Pronóstico
    # -------------------------------------------------------------
    st.markdown("#### 📅 Datos Históricos y Pronóstico Futuro")

    df_fore_disp = sub_forecast.copy()
    df_fore_disp["date"] = pd.to_datetime(df_fore_disp["date"]).dt.strftime("%b %Y").str.upper()
    df_fore_disp["demand_real"] = df_fore_disp["demand_real"].apply(
        lambda x: f"{x:,.0f}" if pd.notnull(x) else ""
    )
    df_fore_disp["demand_forecast"] = df_fore_disp["demand_forecast"].apply(lambda x: f"{x:,.0f}")
    df_fore_disp["method_wmape"] = df_fore_disp["method_wmape"].apply(lambda x: f"{x:.2%}")
    df_fore_disp["method_bias"] = df_fore_disp["method_bias"].apply(lambda x: f"{x:.2%}")

    # RENAME nunca falla por cantidad de columnas
    df_fore_disp = df_fore_disp.rename(columns={
        "date": "Fecha",
        "product_id": "Producto",
        "demand_real": "Demanda Real",
        "demand_forecast": "Pronóstico",
        "method_used": "Método Usado",
        "method_wmape": "wMAPE",
        "method_bias": "Bias",
        "tipo_periodo": "Tipo de Período",
    })
    st.dataframe(df_fore_disp, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 2. Tabla: Simulación Mes a Mes
    # -------------------------------------------------------------
    st.markdown("#### 📦 Registro Mensual de la Simulación de Inventario")

    df_sim_disp = sub_sim.copy()
    df_sim_disp["date"] = pd.to_datetime(df_sim_disp["date"]).dt.strftime("%b %Y").str.upper()

    cols_sim = [c for c in ["date", "demand_real", "demand_forecast", "inventory_level", "order_placed", "arrivals", "sales_lost", "reorder_point_s"] if c in df_sim_disp.columns]
    df_sim_disp = df_sim_disp[cols_sim]

    df_sim_disp = df_sim_disp.rename(columns={
        "date": "Mes",
        "demand_real": "Demanda Real",
        "demand_forecast": "Pronóstico",
        "inventory_level": "Inventario Final",
        "order_placed": "Pedido Generado",
        "arrivals": "Llegadas (Recepción)",
        "sales_lost": "Ventas Perdidas",
        "reorder_point_s": "Punto Reorden (s)",
    })

    for col in df_sim_disp.columns:
        if col != "Mes":
            df_sim_disp[col] = pd.to_numeric(df_sim_disp[col], errors="coerce").fillna(0).apply(lambda x: f"{x:,.0f}")

    st.dataframe(df_sim_disp, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 3. Tabla: Escenarios de Optimización
    # -------------------------------------------------------------
    st.markdown("#### 🎯 Resultados de la Optimización de Stock de Seguridad")

    df_opt_disp = sub_opt.copy()
    
    if "q_optimo" not in df_opt_disp.columns:
        df_opt_disp["q_optimo"] = parametros_del_producto.q_fixed
    if "r_optimo" not in df_opt_disp.columns:
        df_opt_disp["r_optimo"] = parametros_del_producto.review_period_months

    cols_deseadas = [c for c in ["ss_months", "q_optimo", "r_optimo", "fill_rate", "avg_inventory", "lost_sales_units", "stockout_months", "orders", "ordering_cost", "holding_cost", "stockout_cost", "total_cost"] if c in df_opt_disp.columns]
    df_opt_disp = df_opt_disp[cols_deseadas]

    # RENAME seguro: renombra lo que existe sin colapsar
    df_opt_disp = df_opt_disp.rename(columns={
        "ss_months": "Meses SS",
        "q_optimo": "Lote Q (Unds)",
        "r_optimo": "Revisión R (Meses)",
        "fill_rate": "Fill Rate",
        "avg_inventory": "Inv. Promedio",
        "lost_sales_units": "Ventas Perdidas",
        "stockout_months": "Meses Quiebre",
        "orders": "Total Órdenes",
        "ordering_cost": "Costo Órdenes (S/)",
        "holding_cost": "Costo Almacenaje (S/)",
        "stockout_cost": "Costo Quiebre (S/)",
        "total_cost": "Costo Total (S/)",
    })

    st.dataframe(
        df_opt_disp.style.format({
            "Lote Q (Unds)": "{:,.0f}",
            "Revisión R (Meses)": "{:.0f}",
            "Fill Rate": "{:.2%}",
            "Inv. Promedio": "{:,.0f}",
            "Ventas Perdidas": "{:,.0f}",
            "Meses Quiebre": "{:.0f}",
            "Total Órdenes": "{:.0f}",
            "Costo Órdenes (S/)": "{:,.2f}",
            "Costo Almacenaje (S/)": "{:,.2f}",
            "Costo Quiebre (S/)": "{:,.2f}",
            "Costo Total (S/)": "{:,.2f}",
        }, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.download_button(
            label="📥 Descargar Pronóstico (CSV)",
            data=sub_forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"pronostico_{producto_sel}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_d2:
        st.download_button(
            label="📥 Descargar Simulación (CSV)",
            data=sub_sim.to_csv(index=False).encode("utf-8"),
            file_name=f"simulacion_{producto_sel}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_d3:
        st.download_button(
            label="📥 Descargar Comparativa Métodos (CSV)",
            data=df_comparacion.to_csv(index=False).encode("utf-8"),
            file_name="comparacion_metodos_portafolio.csv",
            mime="text/csv",
            use_container_width=True,
        )
