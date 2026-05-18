import math

import pandas as pd
import plotly.graph_objects as go
import pulp
import streamlit as st

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Herramienta de dimensionamiento", layout="wide")

INTERVALOS_POR_TURNO = 12          # 12 intervalos de 30 min = 6 horas conectadas
BREAK_OFFSET_MIN = 4               # break entre el intervalo 4 y 7 dentro del turno
BREAK_OFFSET_MAX = 8
HORA_APERTURA = '09:00:00'
HORA_CIERRE = '20:30:00'
MAX_AGENTES = 2000
MERMA_ESTRUCTURAL = 10 / 330.0     # merma fija (breaks/auxiliares) sobre la jornada

COL_LLAMADAS = 'Llam Recibidas'
COL_INTERVALO = 'Intervalo'
COL_SEMANA = 'Semana'
COLUMNAS_REQUERIDAS = [COL_SEMANA, COL_INTERVALO, COL_LLAMADAS]


# --- 2. MOTOR MATEMÁTICO WFM ---
def erlang_c_prob_espera(agentes: int, trafico: float) -> float:
    """Probabilidad de espera (Erlang C) dado N agentes y tráfico ofrecido en Erlangs."""
    if agentes <= trafico:
        return 1.0
    eb_inv = 1.0
    for i in range(1, int(agentes) + 1):
        eb_inv = 1.0 + eb_inv * i / trafico
    eb = 1.0 / eb_inv
    return min(1.0, agentes * eb / (agentes - trafico * (1 - eb)))


@st.cache_data(show_spinner=False)
def calcular_requeridos(llamadas: float, aht: int, abd_obj: float, ocu_max: float) -> int:
    """Número mínimo de agentes para cumplir abandono y ocupación objetivo."""
    if llamadas <= 0:
        return 0
    trafico = (llamadas * aht) / 1800.0
    agentes = math.ceil(trafico) + 1
    while True:
        p_esp = erlang_c_prob_espera(agentes, trafico)
        ocup = (trafico / agentes) * 100
        # se aproxima abandono ≈ Pw * 0.5 (regla práctica)
        if (p_esp * 0.5 * 100) <= abd_obj and ocup <= ocu_max:
            break
        agentes += 1
        if agentes > MAX_AGENTES:
            break
    return agentes


def optimizar_malla(df_curva_semana: pd.DataFrame, ausentismo: float) -> pd.DataFrame:
    """Minimiza la cantidad de agentes en malla cubriendo los requeridos por intervalo."""
    merma = (ausentismo / 100.0) + MERMA_ESTRUCTURAL
    factor_cobertura = max(1 - merma, 0.01)

    intervalos = list(df_curva_semana[COL_INTERVALO])
    req_dict = dict(zip(df_curva_semana[COL_INTERVALO], df_curva_semana['Requeridos']))

    turnos = [
        (i, j)
        for i in range(len(intervalos) - (INTERVALOS_POR_TURNO - 1))
        for j in range(i + BREAK_OFFSET_MIN, i + BREAK_OFFSET_MAX)
    ]
    if not turnos:
        return pd.DataFrame()

    prob = pulp.LpProblem("Opt", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("Ag", turnos, lowBound=0, cat='Integer')
    prob += pulp.lpSum([x[t] for t in turnos])

    for t_idx, t_str in enumerate(intervalos):
        activos = [
            x[(i, j)]
            for (i, j) in turnos
            if i <= t_idx < i + INTERVALOS_POR_TURNO and t_idx != j
        ]
        if activos:
            prob += pulp.lpSum(activos) >= (req_dict[t_str] / factor_cobertura)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != 'Optimal':
        return pd.DataFrame()

    filas = []
    for (i, j) in turnos:
        valor = x[(i, j)].varValue
        if valor and valor > 0:
            inicio_ult = pd.to_datetime(intervalos[i + INTERVALOS_POR_TURNO - 1], format='%H:%M:%S')
            salida = (inicio_ult + pd.Timedelta(minutes=30)).strftime('%H:%M:%S')
            filas.append({
                "Ingreso": intervalos[i],
                "Break": intervalos[j],
                "Salida": salida,
                "Agentes": int(valor),
            })
    return pd.DataFrame(filas)


def calcular_cobertura(df_malla: pd.DataFrame, intervalos: list[str], ausentismo: float) -> list[float]:
    """Para cada intervalo, suma agentes activos (descontando breaks) ajustados por merma."""
    factor = 1 - ((ausentismo / 100.0) + MERMA_ESTRUCTURAL)
    cobertura = []
    for t in intervalos:
        en_silla = 0
        for _, turno in df_malla.iterrows():
            if turno['Ingreso'] <= t < turno['Salida'] and t != turno['Break']:
                en_silla += turno['Agentes']
        cobertura.append(round(en_silla * factor, 1))
    return cobertura


# --- 3. CARGA Y PREPARACIÓN DE DATOS ---
def cargar_datos(archivo) -> pd.DataFrame:
    if archivo.name.lower().endswith('.csv'):
        df = pd.read_csv(archivo, sep=None, engine='python')
    else:
        df = pd.read_excel(archivo)
    df.columns = df.columns.str.strip()
    return df


def preparar_datos_semanales(df_raw: pd.DataFrame) -> pd.DataFrame:
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df_raw.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el archivo: {', '.join(faltantes)}")

    df = df_raw[~df_raw[COL_INTERVALO].astype(str).str.contains('Total', case=False, na=False)].copy()
    df[COL_INTERVALO] = df[COL_INTERVALO].fillna('').astype(str).str.strip()
    df[COL_INTERVALO] = df[COL_INTERVALO].apply(lambda x: x + ':00' if len(x) == 5 else x)
    df[COL_LLAMADAS] = pd.to_numeric(df[COL_LLAMADAS], errors='coerce').fillna(0)

    df_sem = df.groupby([COL_SEMANA, COL_INTERVALO])[COL_LLAMADAS].mean().reset_index()
    df_sem = df_sem[(df_sem[COL_INTERVALO] >= HORA_APERTURA) & (df_sem[COL_INTERVALO] <= HORA_CIERRE)]
    return df_sem


# --- 4. VISTAS ---
def mostrar_volumen_semanal(df_semanal: pd.DataFrame) -> None:
    st.subheader("📊 Volumen Promedio por Semana")
    df_pivot = df_semanal.pivot(index=COL_SEMANA, columns=COL_INTERVALO, values=COL_LLAMADAS).round(1)
    df_pivot.columns = [str(c)[:5] for c in df_pivot.columns]
    st.dataframe(df_pivot.reset_index().astype(str), hide_index=True)


def procesar_semana(df_sem: pd.DataFrame, aht: int, abd: float, ocu: float, aus: float) -> dict | None:
    df_sem = df_sem.copy()
    df_sem['Requeridos'] = df_sem[COL_LLAMADAS].apply(
        lambda x: calcular_requeridos(x, aht, abd, ocu)
    )
    df_malla = optimizar_malla(df_sem, aus)
    if df_malla.empty:
        return None
    df_sem['Cobertura Real'] = calcular_cobertura(df_malla, list(df_sem[COL_INTERVALO]), aus)
    return {"malla": df_malla, "balance": df_sem}


def mostrar_resultados_semana(semana, resultado: dict) -> None:
    df_malla = resultado['malla']
    df_balance = resultado['balance']

    with st.expander(f"📅 Gestión: {semana}"):
        total_programado = int(df_malla['Agentes'].sum())
        max_requerido = int(df_balance['Requeridos'].max())
        prom_requerido = round(df_balance['Requeridos'].mean(), 1)

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Programado", f"{total_programado} pers.")
        k2.metric("Pico Requerido Erlang", f"{max_requerido} pers.")
        k3.metric("Promedio Requerido", f"{prom_requerido} pers.")

        st.write("**Malla de Horarios**")
        df_m_viz = df_malla.copy()
        for c in ["Ingreso", "Break", "Salida"]:
            df_m_viz[c] = df_m_viz[c].str[:5]
        st.dataframe(df_m_viz.reset_index(drop=True).astype(str), hide_index=True)

        st.write("**Balance de Cobertura**")
        df_bal = df_balance[[COL_INTERVALO, 'Requeridos', 'Cobertura Real']].copy()
        df_bal['Diferencia'] = (df_bal['Cobertura Real'] - df_bal['Requeridos']).round(1)
        df_bal[COL_INTERVALO] = df_bal[COL_INTERVALO].str[:5]
        st.dataframe(
            df_bal.set_index(COL_INTERVALO).T.reset_index().round(1).astype(str),
            hide_index=True,
        )

        colores = ['#EF553B' if v < 0 else '#00CC96' for v in df_bal['Diferencia']]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_bal[COL_INTERVALO], y=df_bal['Diferencia'],
            name='Gap', marker_color=colores,
        ))
        fig.add_trace(go.Scatter(
            x=df_bal[COL_INTERVALO], y=df_bal['Requeridos'],
            name='Req Erlang', line=dict(color='#636EFA', width=3),
        ))
        fig.add_trace(go.Scatter(
            x=df_bal[COL_INTERVALO], y=df_bal['Cobertura Real'],
            name='Cobertura Real', line=dict(color='#333333', dash='dot'),
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.2),
        )
        st.plotly_chart(fig, use_container_width=True)


# --- 5. INTERFAZ PRINCIPAL ---
def main() -> None:
    st.title("Dimensionamiento Semanal")
    archivo = st.file_uploader("Reporte de volumen", type=['csv', 'xlsx'])
    if not archivo:
        st.info("Sube un archivo de volumen para comenzar (CSV o XLSX).")
        return

    try:
        df_raw = cargar_datos(archivo)
        df_semanal = preparar_datos_semanales(df_raw)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        return

    if df_semanal.empty:
        st.warning("El archivo no contiene datos dentro del rango horario configurado.")
        return

    mostrar_volumen_semanal(df_semanal)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        aht = st.number_input("AHT (seg)", value=420, min_value=1)
        aus = st.number_input("Ausentismo %", value=9.0, min_value=0.0, max_value=100.0)
    with c2:
        abd = st.number_input("Abandono %", value=10.0, min_value=0.0, max_value=100.0)
        ocu = st.number_input("Ocupación %", value=80.0, min_value=1.0, max_value=100.0)

    if not st.button("Calcular Escenarios", type="primary"):
        return

    resultados = {}
    with st.spinner("Calculando mallas y cobertura..."):
        for semana in df_semanal[COL_SEMANA].unique():
            df_sem = df_semanal[df_semanal[COL_SEMANA] == semana]
            resultado = procesar_semana(df_sem, aht, abd, ocu, aus)
            if resultado is None:
                st.warning(f"No se encontró solución óptima para la semana {semana}.")
                continue
            resultados[semana] = resultado

    if not resultados:
        st.error("No se generaron mallas. Revisa los parámetros de entrada.")
        return

    for semana, resultado in resultados.items():
        mostrar_resultados_semana(semana, resultado)


if __name__ == "__main__":
    main()
