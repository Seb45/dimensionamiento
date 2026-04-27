import streamlit as st
import pandas as pd
import pulp
import math
import openpyxl  # Asegura la compatibilidad con archivos Excel .xlsx
from supabase import create_client, Client

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN DE SUPABASE ---
st.set_page_config(page_title="Proyecto Horizonte", layout="wide")

@st.cache_resource
def init_connection():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None

supabase = init_connection()

# --- 2. FUNCIONES MATEMÁTICAS (WFM CORE) ---
def erlang_c_prob_espera(agentes, trafico):
    if agentes <= trafico:
        return 1.0
    erlang_b_inv = 1.0
    for i in range(1, int(agentes) + 1):
        erlang_b_inv = 1.0 + erlang_b_inv * i / trafico
    erlang_b = 1.0 / erlang_b_inv
    prob_espera = agentes * erlang_b / (agentes - trafico * (1 - erlang_b))
    return min(1.0, prob_espera)

def calcular_requeridos_erlang_a_aprox(llamadas, aht, abandono_obj, ocupacion_max):
    """Aproximación iterativa: Utiliza Erlang C ajustado por abandono."""
    if llamadas == 0:
        return 0
    
    trafico = (llamadas * aht) / 1800.0  # Erlangs para intervalos de 30 min
    agentes = math.ceil(trafico)
    
    while True:
        ocupacion = trafico / agentes if agentes > 0 else 1.0
        prob_espera = erlang_c_prob_espera(agentes, trafico)
        
        # Estimación simplificada de abandono
        abandono_estimado = prob_espera * 0.5 * 100 
        
        if abandono_estimado <= abandono_obj and (ocupacion * 100) <= ocupacion_max:
            break
        agentes += 1
        
        if agentes > trafico + 50:
            break
            
    return agentes

# --- 3. OPTIMIZADOR DE TURNOS (PuLP) ---
def optimizar_turnos_6hs(df_curva, ausentismo_pct):
    # Merma: 10 min baño en 330 netos = 3.03% + ausentismo
    merma_bano = 10 / 330.0
    merma_total = (ausentismo_pct / 100.0) + merma_bano
    
    prob = pulp.LpProblem("Horizonte_Scheduling", pulp.LpMinimize)
    intervalos = list(df_curva['Intervalo'])
    requeridos = dict(zip(df_curva['Intervalo'], df_curva['Requeridos']))
    
    turnos_posibles = []
    for i in range(len(intervalos) - 11):
        # Break entre la 2da (i+4) y 4ta hora (i+7)
        for j in range(i + 4, i + 8):
            turnos_posibles.append((i, j))
            
    x = pulp.LpVariable.dicts("Agentes", turnos_posibles, lowBound=0, cat='Integer')
    prob += pulp.lpSum([x[t] for t in turnos_posibles])
    
    for t_idx, t_str in enumerate(intervalos):
        agentes_en_t = []
        for (i, j) in turnos_posibles:
            if i <= t_idx < i + 12 and t_idx != j:
                agentes_en_t.append(x[(i, j)])
        if agentes_en_t:
            req_neto = requeridos[t_str] / (1 - merma_total)
            prob += pulp.lpSum(agentes_en_t) >= req_neto
            
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    resultados = []
    for (i, j) in turnos_posibles:
        if x[(i, j)].varValue > 0:
            resultados.append({
                "Ingreso": intervalos[i],
                "Break": intervalos[j],
                "Salida": intervalos[i+11],
                "Agentes": int(x[(i, j)].varValue)
            })
    return pd.DataFrame(resultados)

# --- 4. INTERFAZ PRINCIPAL ---
st.title("Proyecto Horizonte | Dimensionamiento y Scheduling")

uploaded_file = st.file_uploader("Sube el volumen por intervalo (.csv o .xlsx)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()
        
    # Limpieza de nombres de columnas
    df_raw.columns = df_raw.columns.str.strip()
    
    if 'Intervalo' not in df_raw.columns:
        st.error(f"No se encontró la columna 'Intervalo'.")
        st.stop()

    # Limpieza de filas de Total
    df_clean = df_raw[~df_raw['Intervalo'].astype(str).str.contains('Total', case=False, na=False)].copy()
    
    # Blindaje contra nulos y tipos float en la columna Intervalo
    df_clean['Intervalo'] = df_clean['Intervalo'].fillna('').astype(str)
    df_clean['Intervalo'] = df_clean['Intervalo'].apply(
        lambda x: str(x).strip() + ':00' if len(str(x).strip()) == 5 else str(x).strip()
    )
    df_clean = df_clean[df_clean['Intervalo'] != '']
    
    if 'Llam Recibidas' in df_clean.columns:
        col_llamadas = 'Llam Recibidas'
    else:
        st.error("No se encontró la columna 'Llam Recibidas'.")
        st.stop()

    df_clean[col_llamadas] = pd.to_numeric(df_clean[col_llamadas], errors='coerce').fillna(0)
    df_curva = df_clean.groupby('Intervalo')[col_llamadas].mean().reset_index()
    df_curva = df_curva.sort_values('Intervalo')
    
    # Ventana operativa 09:00 a 21:00
    df_curva = df_curva[(df_curva['Intervalo'] >= '09:00:00') & (df_curva['Intervalo'] <= '20:30:00')].reset_index(drop=True)
    
    st.write("Curva de llamadas proyectada (Promedio por intervalo):")
    # Blindaje para evitar ArrowTypeError: forzar visualización como string
    st.dataframe(df_curva.set_index('Intervalo').T.astype(str)) 

    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Parámetros Operativos")
        aht = st.number_input("AHT (segundos)", value=250, min_value=1)
        ausentismo = st.number_input("Ausentismo (%)", value=9.0, min_value=0.0)
    with col2:
        st.subheader("Objetivos (Erlang A)")
        abandono_obj = st.number_input("Abandono Objetivo (%)", value=5.0, min_value=0.1)
        ocupacion_max = st.number_input("Ocupación de Diseño (%)", value=85.0, min_value=1.0)
        
    if st.button("Generar Malla Óptima"):
        with st.spinner("Calculando requerimientos y optimizando programación..."):
            df_curva['Requeridos'] = df_curva[col_llamadas].apply(
                lambda v: calcular_requeridos_erlang_a_aprox(v, aht, abandono_obj, ocupacion_max)
            )
            
            df_malla = optimizar_turnos_6hs(df_curva, ausentismo)
            
            st.success(f"Optimización completada. Headcount necesario: {df_malla['Agentes'].sum()} asesores.")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("Requerimientos")
                st.dataframe(df_curva[['Intervalo', col_llamadas, 'Requeridos']])
            with c2:
                st.write("Malla de Turnos")
                st.dataframe(df_malla)
                
            if supabase:
                try:
                    payload = {
                        "usuario_email": "sebastian@horizonte.com", # Identificador profesional
                        "parametros": {"aht": aht, "ausentismo": ausentismo, "abandono": abandono_obj},
                        "volumen_ingresado": df_curva.to_dict(orient="records"),
                        "malla_generada": df_malla.to_dict(orient="records")
                    }
                    supabase.table("escenarios_horizonte").insert(payload).execute()
                    st.info("Escenario guardado en Supabase.")
                except Exception as e:
                    st.error(f"Error en BD: {e}")
