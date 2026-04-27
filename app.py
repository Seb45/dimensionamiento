import streamlit as st
import pandas as pd
import pulp
import math
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

# --- 2. AUTENTICACIÓN CON GOOGLE ---
def login_con_google():
    st.warning("Debes iniciar sesión para generar escenarios.")
    # Intenta obtener la URL de redirección de los secrets, o usa localhost por defecto
    app_url = st.secrets.get("REDIRECT_URL", "http://localhost:8501") 
    
    if supabase:
        res = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {"redirect_to": app_url}
        })
        st.link_button("Iniciar sesión con Google", res.url)
    else:
        st.error("Conexión a Supabase no configurada. Revisa los Secrets.")

# Comprobación simple de sesión (ajustar según el manejo de tokens de Supabase en Streamlit)
# Para pruebas locales sin Auth estricto, puedes comentar estas 3 líneas de abajo.
# if not st.session_state.get("usuario_autenticado", False) and supabase:
#     login_con_google()
#     st.stop()


# --- 3. FUNCIONES MATEMÁTICAS (WFM CORE) ---
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
    
    trafico = (llamadas * aht) / 1800.0  # Erlangs para intervalos de 30 min (1800 seg)
    agentes = math.ceil(trafico)
    
    while True:
        ocupacion = trafico / agentes if agentes > 0 else 1.0
        prob_espera = erlang_c_prob_espera(agentes, trafico)
        
        # Estimación simplificada de abandono basada en probabilidad de espera
        abandono_estimado = prob_espera * 0.5 * 100 
        
        if abandono_estimado <= abandono_obj and (ocupacion * 100) <= ocupacion_max:
            break
        agentes += 1
        
        # Freno de seguridad
        if agentes > trafico + 50:
            break
            
    return agentes

# --- 4. OPTIMIZADOR DE TURNOS (PuLP) ---
def optimizar_turnos_6hs(df_curva, ausentismo_pct):
    # Cálculo de merma: 10 min de baño en 330 netos = 3.03%. Más ausentismo.
    merma_bano = 10 / 330.0
    merma_total = (ausentismo_pct / 100.0) + merma_bano
    
    prob = pulp.LpProblem("Horizonte_Scheduling", pulp.LpMinimize)
    
    intervalos = list(df_curva['Intervalo'])
    requeridos = dict(zip(df_curva['Intervalo'], df_curva['Requeridos']))
    
    turnos_posibles = []
    # Turnos de 6hs = 12 intervalos de 30 min.
    for i in range(len(intervalos) - 11):
        # Break de 1 intervalo (30min) ubicado estrictamente entre la 2da hora (i+4) y la 4ta hora (i+7)
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
                "Salida": intervalos[i+11], # Inicio del último intervalo
                "Agentes": int(x[(i, j)].varValue)
            })
    return pd.DataFrame(resultados)


# --- 5. INTERFAZ PRINCIPAL ---
st.title("Proyecto Horizonte | Dimensionamiento y Scheduling")

uploaded_file = st.file_uploader("Sube el volumen por intervalo (.csv o .xlsx)", type=['csv', 'xlsx'])

if uploaded_file:
    # 1. Ingesta inteligente del archivo
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()
        
    # --- BLINDAJE 1: Limpiar espacios ocultos en los nombres de las columnas ---
    df_raw.columns = df_raw.columns.str.strip()
    
    # Verificar si la columna existe después de limpiar
    if 'Intervalo' not in df_raw.columns:
        st.error(f"Error: No se encontró la columna 'Intervalo'. Columnas detectadas: {', '.join(df_raw.columns)}")
        st.stop()

    # 2. Limpiar filas de 'Total' comunes en las exportaciones de reportes
    df_clean = df_raw[~df_raw['Intervalo'].astype(str).str.contains('Total', case=False, na=False)].copy()

    # --- BLINDAJE 2: Forzar que toda la columna sea texto para evitar conflictos ---
    # Llenar nulos con texto vacío y convertir a string
    df_clean['Intervalo'] = df_clean['Intervalo'].fillna('').astype(str)
    
    # Asegurar que el formato sea HH:MM:SS usando str() explícito para evitar fallos de 'float'
    df_clean['Intervalo'] = df_clean['Intervalo'].apply(
        lambda x: str(x).strip() + ':00' if len(str(x).strip()) == 5 else str(x).strip()
    )
    
    # Eliminar posibles filas que hayan quedado sin intervalo (vacías) tras la limpieza
    df_clean = df_clean[df_clean['Intervalo'] != '']

    

    # 4. Agrupar por intervalo para obtener el volumen promedio
    if 'Llam Recibidas' in df_clean.columns:
        col_llamadas = 'Llam Recibidas'
    else:
        st.error(f"No se encontró la columna 'Llam Recibidas'. Columnas actuales: {', '.join(df_clean.columns)}")
        st.stop()

    df_clean[col_llamadas] = pd.to_numeric(df_clean[col_llamadas], errors='coerce').fillna(0)
    df_curva = df_clean.groupby('Intervalo')[col_llamadas].mean().reset_index()
    df_curva = df_curva.sort_values('Intervalo')
    
    # Filtrar solo la ventana operativa (09:00 a 21:00)
    df_curva = df_curva[(df_curva['Intervalo'] >= '09:00:00') & (df_curva['Intervalo'] <= '20:30:00')].reset_index(drop=True)
    
    st.write("Curva de llamadas proyectada (Promedio por intervalo):")
    # Convertimos el Intervalo en el índice antes de transponer para evitar mezclar tipos de datos
    st.dataframe(df_curva.set_index('Intervalo').T)
    
    
    
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
            df_curva['Requeridos'] = df_curva[col_llamadas].apply(
                lambda v: calcular_requeridos_erlang_a_aprox(v, aht, abandono_obj, ocupacion_max)
            )
            
            # Paso 2: Optimización de Turnos
            df_malla = optimizar_turnos_6hs(df_curva, ausentismo)
            hc_total = df_malla['Agentes'].sum()
            
            st.success(f"Optimización completada. Headcount necesario: {hc_total} asesores.")
            
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.write("Requerimientos por Intervalo")
                st.dataframe(df_curva[['Intervalo', col_llamadas, 'Requeridos']])
            with col_res2:
                st.write("Malla de Turnos Optimizada")
                st.dataframe(df_malla)
                
            # Paso 3: Guardar en Supabase
            if supabase:
                try:
                    # En producción, reemplazar con el email del token JWT real
                    email_operador = st.session_state.get("usuario_email", "lider@horizonte.com") 
                    
                    payload = {
                        "usuario_email": email_operador, 
                        "parametros": {"aht": aht, "ausentismo": ausentismo, "abandono": abandono_obj, "ocupacion": ocupacion_max},
                        "volumen_ingresado": df_curva.to_dict(orient="records"),
                        "malla_generada": df_malla.to_dict(orient="records")
                    }
                    supabase.table("escenarios_horizonte").insert(payload).execute()
                    st.info("El escenario ha sido persistido en la base de datos de Supabase correctamente.")
                except Exception as e:
                    st.error(f"Error registrando en base de datos: {e}")
