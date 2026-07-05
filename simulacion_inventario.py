import math
import pandas as pd
from dataclasses import dataclass

@dataclass
class ParametrosInventario:
    initial_stock: int
    lead_time_months: int
    review_period_months: int
    ss_months: int
    q_fixed: int
    lot_size: int
    cost_order: float
    cost_holding_month: float
    cost_stockout: float

def obtener_parametros_producto(df_params: pd.DataFrame, producto_id: str, demanda_promedio: float = 100.0) -> ParametrosInventario:
    """
    Busca el producto en el dataframe maestro de parámetros y extrae sus valores específicos.
    """
    if df_params is None or df_params.empty:
        return ParametrosInventario(
            initial_stock=0, lead_time_months=1, review_period_months=1,
            ss_months=1, q_fixed=int(demanda_promedio), lot_size=1,
            cost_order=100.0, cost_holding_month=1.0, cost_stockout=100.0
        )

    df_params = df_params.copy()
    df_params.columns = [str(c).strip().lower() for c in df_params.columns]
    
    columna_producto = "grupo de demanda" if "grupo de demanda" in df_params.columns else df_params.columns[0]
    df_filtrado = df_params[df_params[columna_producto].astype(str).str.strip().str.upper() == str(producto_id).strip().upper()]

    if df_filtrado.empty:
        return ParametrosInventario(
            initial_stock=0, lead_time_months=1, review_period_months=1,
            ss_months=1, q_fixed=int(demanda_promedio), lot_size=1,
            cost_order=100.0, cost_holding_month=1.0, cost_stockout=100.0
        )
    
    fila = df_filtrado.iloc[0]

    q_val = int(pd.to_numeric(fila.get("q_fixed", 0)))
    if q_val <= 0:
        q_val = max(1, int(demanda_promedio))

    # Extraer SS tal cual viene en el Excel (sea unidades o meses)
    ss_val = int(pd.to_numeric(fila.get("ss", fila.get("ss_months", 0))))

    return ParametrosInventario(
        initial_stock=int(pd.to_numeric(fila.get("initial_stock", fila.get("initial_stoc", 0)))),
        lead_time_months=int(math.ceil(pd.to_numeric(fila.get("lead_time_months", fila.get("lead_time_mo", 1))))),
        review_period_months=int(pd.to_numeric(fila.get("review_period", 1))),
        ss_months=ss_val,
        q_fixed=q_val,
        lot_size=max(1, int(pd.to_numeric(fila.get("lot_size", 1)))),
        cost_order=float(pd.to_numeric(fila.get("cost_order", 0.0))),
        cost_holding_month=float(pd.to_numeric(fila.get("cost_holding", fila.get("cost_holding_month", 0.0)))),
        cost_stockout=float(pd.to_numeric(fila.get("cost_stockout", 0.0)))
    )
    
def redondear_lote(cantidad: float, lote: int) -> int:
    if cantidad <= 0:
        return 0
    lote = max(1, int(lote))
    return int(math.ceil(cantidad / lote) * lote)

def simular_producto(df_producto: pd.DataFrame, politica: str, p: ParametrosInventario) -> pd.DataFrame:
    df_producto = df_producto.sort_values("date").reset_index(drop=True).copy()
    stock_fisico = float(p.initial_stock)
    pipeline = {}
    resultados = []
    demanda_promedio_mensual = max(0.01, df_producto["demand_forecast"].mean())

    for t, fila in df_producto.iterrows():
        llegada = pipeline.pop(t, 0)
        stock_fisico += llegada
        demanda_durante_lead_time = demanda_promedio_mensual * p.lead_time_months
        
        # 🔴 INTELIGENCIA LOGÍSTICA: Detectar si el SS está en UNIDADES directas o en MESES
        if p.ss_months > 36:
            # Si el número es grande (ej. 1,954,505), ya son UNIDADES exactas de tu Excel
            stock_seguridad = float(p.ss_months)
        else:
            # Si el número es pequeño (ej. 1, 2, 3), son MESES y lo multiplicamos por la demanda
            stock_seguridad = demanda_promedio_mensual * p.ss_months

        punto_reorden = demanda_durante_lead_time + stock_seguridad
        
        if p.ss_months > 36:
            nivel_objetivo = demanda_promedio_mensual * (p.lead_time_months + p.review_period_months) + float(p.ss_months)
        else:
            nivel_objetivo = demanda_promedio_mensual * (p.lead_time_months + p.review_period_months + p.ss_months)

        posicion_inventario = stock_fisico + sum(pipeline.values())
        orden = 0
        
        if politica == "RS - revisión periódica":
            if t % p.review_period_months == 0:
                orden = max(0, nivel_objetivo - posicion_inventario)
        elif politica == "sS - punto de reorden y nivel máximo":
            if posicion_inventario <= punto_reorden:
                orden = max(0, nivel_objetivo - posicion_inventario)
        elif politica == "sQ - punto de reorden y cantidad fija":
            if posicion_inventario <= punto_reorden:
                orden = p.q_fixed

        orden = redondear_lote(orden, p.lot_size)
        if orden > 0:
            mes_llegada = t + p.lead_time_months
            pipeline[mes_llegada] = pipeline.get(mes_llegada, 0) + orden

        demanda_real = float(fila["demand_real"])
        venta_real = min(stock_fisico, demanda_real)
        venta_perdida = max(0, demanda_real - stock_fisico)
        stock_fisico -= venta_real

        resultados.append(
            {
                "date": fila["date"],
                "product_id": fila["product_id"],
                "method_used": fila.get("method_used", ""),
                "demand_real": demanda_real,
                "demand_forecast": fila["demand_forecast"],
                "inventory_level": stock_fisico,
                "inventory_position": posicion_inventario,
                "order_placed": orden,
                "arrivals": llegada,
                "sales_real": venta_real,
                "sales_lost": venta_perdida,
                "reorder_point_s": punto_reorden,
                "target_level_S": nivel_objetivo,
                "is_stockout": int(venta_perdida > 0),
            }
        )
    return pd.DataFrame(resultados)

def calcular_kpis(df_sim: pd.DataFrame, p: ParametrosInventario) -> dict:
    demanda_total = df_sim["demand_real"].sum()
    ventas_perdidas = df_sim["sales_lost"].sum()
    ordenes = (df_sim["order_placed"] > 0).sum()
    inventario_promedio = df_sim["inventory_level"].mean()
    fill_rate = 1 - ventas_perdidas / demanda_total if demanda_total > 0 else 1
    costo_ordenar = ordenes * p.cost_order
    costo_mantener = df_sim["inventory_level"].sum() * p.cost_holding_month
    costo_quiebre = ventas_perdidas * p.cost_stockout
    costo_total = costo_ordenar + costo_mantener + costo_quiebre
    return {
        "fill_rate": fill_rate,
        "avg_inventory": inventario_promedio,
        "lost_sales_units": ventas_perdidas,
        "stockout_months": int(df_sim["is_stockout"].sum()),
        "orders": int(ordenes),
        "ordering_cost": costo_ordenar,
        "holding_cost": costo_mantener,
        "stockout_cost": costo_quiebre,
        "total_cost": costo_total,
    }

def optimizar_stock_seguridad(df_producto: pd.DataFrame, politica: str, p_base: ParametrosInventario, ss_max: int) -> pd.DataFrame:
    filas = []
    demanda_promedio = max(1.0, df_producto["demand_forecast"].mean())
    
    if politica == "sQ - punto de reorden y cantidad fija":
        multiplos_q = [0.5, 1, 1.5, 2, 3, 4, 6]
        valores_q = [redondear_lote(demanda_promedio * m, p_base.lot_size) for m in multiplos_q]
        valores_q = sorted(list(set([q for q in valores_q if q > 0])))
        if not valores_q:
            valores_q = [max(1, p_base.q_fixed)]
        valores_r = [p_base.review_period_months]
    else:
        valores_r = [1, 2, 3, 4, 6]
        valores_q = [max(1, p_base.q_fixed)]

    for ss in range(0, ss_max + 1):
        mejor_escenario_ss = None
        menor_quiebre_ss = float('inf')
        menor_costo_ss = float('inf')
        
        for r_test in valores_r:
            for q_test in valores_q:
                p = ParametrosInventario(
                    initial_stock=p_base.initial_stock,
                    lead_time_months=p_base.lead_time_months,
                    review_period_months=r_test,
                    ss_months=ss,
                    q_fixed=q_test,
                    lot_size=p_base.lot_size,
                    cost_order=p_base.cost_order,
                    cost_holding_month=p_base.cost_holding_month,
                    cost_stockout=p_base.cost_stockout,
                )
                sim = simular_producto(df_producto, politica, p)
                kpis = calcular_kpis(sim, p)
                
                # 🔴 NUEVO CRITERIO: Priorizar menor costo de quiebre. Si empatan en quiebre, gana el de menor costo total.
                if (kpis["stockout_cost"] < menor_quiebre_ss) or (kpis["stockout_cost"] == menor_quiebre_ss and kpis["total_cost"] < menor_costo_ss):
                    menor_quiebre_ss = kpis["stockout_cost"]
                    menor_costo_ss = kpis["total_cost"]
                    mejor_escenario_ss = {
                        "ss_months": ss, 
                        "q_optimo": q_test, 
                        "r_optimo": r_test, 
                        **kpis
                    }

        filas.append(mejor_escenario_ss)

    return pd.DataFrame(filas)

def evaluar_campeon_politicas(df_producto: pd.DataFrame, p_base: ParametrosInventario, ss_max: int) -> dict:
    """
    Torneo Global: Evalúa las 3 políticas basándose estrictamente en cuál genera el MENOR COSTO DE QUIEBRE.
    """
    politicas = [
        "RS - revisión periódica",
        "sS - punto de reorden y nivel máximo",
    ]
    mejor_global = None

    filas_torneo = []
    for pol in politicas:
        df_opt = optimizar_stock_seguridad(df_producto, pol, p_base, ss_max)
        # Elegir el mejor escenario para esta política según menor costo de quiebre
        mejor_pol = df_opt.sort_values(["stockout_cost", "total_cost"]).iloc[0].to_dict()
        mejor_pol["politica_ganadora"] = pol
        filas_torneo.append(mejor_pol)

    df_torneo = pd.DataFrame(filas_torneo)
    # Elegir la política campeona global por menor costo de quiebre
    campeon = df_torneo.sort_values(["stockout_cost", "total_cost"]).iloc[0].to_dict()
    
    return campeon
