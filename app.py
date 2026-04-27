import streamlit as st
import pandas as pd
import pulp
import math
from supabase import create_client, Client

# --- CONFIGURACIÓN SUPABASE ---
# En tu entorno local, crea un archivo .streamlit/secrets.toml con estas variables
# SUPABASE_URL = "tu_url"
# SUPABASE_KEY = "tu_anon_key"
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase = init_connection()
except Exception as e:
    st.warning("Configura los secrets de Supabase para habilitar el guardado en BD.")

# --- FUNCIONES MATEMÁTICAS (WFM CORE) ---
def erlang_c_prob_espera(agentes, trafico):
    if agentes <= trafico:
        return 1.0
    # Cálculo de Erlang C
    erlang_b_inv = 1.0
    for i in range(1, agentes + 1):
        erlang_b_inv = 1.0 + erlang_b_inv * i / trafico
    erlang_b = 1.0 / erlang_b_inv
    prob_espera = agentes * erlang_b / (agentes - trafico * (1 - erlang_b))
    return min(1.0, prob_espera)

def calcular_requeridos_erlang_a_aprox(llamadas, aht, abandono_obj, ocupacion_max):
    """
    Aproximación iterativa: Utiliza Erlang C ajustado por abandono.
    (La función de Erlang A pura requiere integrales complejas de la impaciencia).
    """
    if llamadas == 0:
        return 0
    
    trafico = (llamadas * aht) / 1800.0  # Erlangs para intervalos de 30 min (1800 seg)
    agentes = math.ceil(trafico)
    
    while True:
        ocupacion = trafico / agentes if agentes > 0 else 1.0
        prob_espera = erlang_c_prob_espera(agentes, trafico)
        
        # Estimación simplificada de abandono basada en probabilidad de espera
        abandono_estimado = prob_espera * 0.5 * 100 # Factor empírico ajustable
        
        if abandono_estimado <= abandono_obj and (ocupacion * 100) <= ocupacion_max:
            break
        agentes += 1
        
        # Freno de seguridad
        if agentes > trafico + 50:
            break
            
    return agentes

# --- OPTIMIZADOR DE TURNOS (PuLP) ---
def optimizar_turnos_6hs(df_curva, ausentismo_pct):
    # Cálculo de merma: 10 min de baño en 330 netos = 3.03%. Más ausentismo.
    merma_bano = 10 / 330.0
    merma_total = (ausentismo_pct / 100.0) + merma_bano
    
    prob = pulp.LpProblem("Impulso_Scheduling", pulp.LpMinimize)
    
    intervalos = list(df_curva['Intervalo'])
    requeridos = dict(zip(df_curva['Intervalo'], df_curva['Requeridos']))
    
    turnos_posibles = []
    # Turnos de 6hs = 12 intervalos de 30 min.
    for i in range(len(intervalos) - 11):
        # Break de 1 intervalo (30min) ubicado entre la 2da hora (i+4) y la 4ta hora (i+7)
        for j in range(i + 4, i + 8):
            turnos_posibles.append((i, j))
            
    x = pulp.LpVariable.dicts("Agentes", turnos_posibles, lowBound=0, cat='Integer')
    
    # Función Objetivo: Minimizar el total de agentes programados
    prob += pulp.lpSum([x[t] for t in turnos_posibles])
    
    # Restricciones de cobertura (considerando la merma)
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
                "Salida": intervalos[i+11], # Inicio del último intervalo (fin del turno 30 min después)
                "Agentes": int(x[(i, j)].varValue)
            })
    return pd.DataFrame(resultados)

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Proyecto Horizonte", layout="wide")
st.title("Proyecto Horizonte | Dimensionamiento y Scheduling")

uploaded_file = st.file_uploader("Sube el volumen por intervalo (.csv o .xlsx)", type=['csv', 'xlsx'])

if uploaded_file:
    # 1. Ingesta y limpieza del archivo
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
        
    # Limpiar filas de 'Total' comunes en las exportaciones de reportes
    df_clean = df_raw[~df_raw['Intervalo'].astype(str).str.contains('Total', case=False, na=False)].copy()
    
    # --- LA SOLUCIÓN: Forzar que toda la columna sea texto para evitar conflictos ---
    df_clean['Intervalo'] = df_clean['Intervalo'].astype(str)
    
    # Algunos formatos de Excel cortan los segundos, esto asegura que el formato sea HH:MM:SS
    df_clean['Intervalo'] = df_clean['Intervalo'].apply(lambda x: x + ':00' if len(x) == 5 else x)
    # -------------------------------------------------------------------------------
    
    # Agrupar por intervalo para obtener el volumen promedio
    df_clean['Llam Recibidas'] = pd.to_numeric(df_clean['Llam Recibidas'], errors='coerce').fillna(0)
    # Agrupar por intervalo para obtener el volumen promedio (útil si hay varios días en el archivo)
    df_clean['Llam Recibidas'] = pd.to_numeric(df_clean['Llam Recibidas'], errors='coerce').fillna(0)
    df_curva = df_clean.groupby('Intervalo')['Llam Recibidas'].mean().reset_index()
    df_curva = df_curva.sort_values('Intervalo')
    
    # Filtrar solo la ventana operativa (09:00 a 21:00)
    df_curva = df_curva[(df_curva['Intervalo'] >= '09:00:00') & (df_curva['Intervalo'] <= '20:30:00')].reset_index(drop=True)
    
    st.write("Curva de llamadas proyectada (Promedio por intervalo):")
    st.dataframe(df_curva.T) # Mostrar transpuesta para que sea más compacta

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
        with st.spinner("Calculando requerimientos y optimizando programación lineal..."):
            
            # Paso 1: Dimensionamiento
            df_curva['Requeridos'] = df_curva['Llam Recibidas'].apply(
                lambda v: calcular_requeridos_erlang_a_aprox(v, aht, abandono_obj, ocupacion_max)
            )
            
            # Paso 2: Optimización de Turnos
            df_malla = optimizar_turnos_6hs(df_curva, ausentismo)
            hc_total = df_malla['Agentes'].sum()
            
            st.success(f"Optimización completada. Headcount necesario: {hc_total} asesores.")
            
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.write("Requerimientos por Intervalo")
                st.dataframe(df_curva[['Intervalo', 'Llamadas', 'Requeridos']])
            with col_res2:
                st.write("Malla de Turnos Optimizada")
                st.dataframe(df_malla)
                
            # Paso 3: Guardar en Supabase
            if 'supabase' in locals():
                try:
                    payload = {
                        "usuario_email": "admin@horizonte.com", # Reemplazar con el token auth real en producción
                        "parametros": {"aht": aht, "ausentismo": ausentismo, "abandono": abandono_obj, "ocupacion": ocupacion_max},
                        "volumen_ingresado": df_curva.to_dict(orient="records"),
                        "malla_generada": df_malla.to_dict(orient="records")
                    }
                    supabase.table("escenarios_horizonte").insert(payload).execute()
                    st.info("El escenario ha sido persistido en la base de datos de Supabase correctamente.")
                except Exception as e:
                    st.error(f"Error registrando en base de datos: {e}")
