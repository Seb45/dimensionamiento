import streamlit as st
import pandas as pd
import pulp
import math
import openpyxl
from supabase import create_client, Client

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
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

# --- 2. LÓGICA DE AUTENTICACIÓN (BREAK THE LOOP) ---
def requerir_autenticacion():
    if not supabase:
        st.warning("⚠️ Configuración de Supabase no detectada. Modo local activo.")
        return "usuario_test@horizonte.com"

    # PASO 1: Detectar si volvemos de Google con un código en la URL
    if "code" in st.query_params:
        try:
            codigo_auth = st.query_params["code"]
            # Intercambiamos el código por una sesión real
            res_sesion = supabase.auth.exchange_code_for_session({"auth_code": codigo_auth})
            st.session_state["usuario"] = res_sesion.user.email
            # Limpiamos la URL y recargamos para entrar a la app limpia
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error al procesar el retorno de Google: {e}")

    # PASO 2: Verificar si ya existe una sesión en el estado de la app
    if st.session_state.get("usuario"):
        return st.session_state["usuario"]

    # PASO 3: Si no hay nada, mostramos la pantalla de bloqueo
    st.title("Proyecto Horizonte | Acceso")
    st.write("Bienvenido. Para acceder al dimensionamiento de fuerza laboral, por favor identifícate.")
    
    app_url = st.secrets.get("REDIRECT_URL", "http://localhost:8501")
    
    try:
        res = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {"redirect_to": app_url}
        })
        st.link_button("🌐 Iniciar sesión con Google", res.url, type="primary")
        
        # Opcional: Bypass para desarrollo
        if st.button("Entrar como Invitado (Modo Desarrollo)"):
            st.session_state["usuario"] = "lider_bpo@horizonte.com"
            st.rerun()
            
        st.stop()
    except Exception as e:
        st.error(f"Error de conexión con Auth: {e}")
        st.stop()

# Ejecución del bloqueo
email_usuario = requerir_autenticacion()

# --- 3. MOTOR MATEMÁTICO WFM ---
def erlang_c_prob_espera(agentes, trafico):
    if agentes <= trafico: return 1.0
    eb_inv = 1.0
    for i in range(1, int(agentes) + 1):
        eb_inv = 1.0 + eb_inv * i / trafico
    eb = 1.0 / eb_inv
    prob = agentes * eb / (agentes - trafico * (1 - eb))
    return min(1.0, prob)

def calcular_requeridos(llamadas, aht, abandono_obj, ocupacion_max):
    if llamadas <= 0: return 0
    trafico = (llamadas * aht) / 1800.0
    agentes = math.ceil(trafico) + 1
    while True:
        p_espera = erlang_c_prob_espera(agentes, trafico)
        ocup = (trafico / agentes) * 100
        # Estimación de abandono simplificada para Erlang A
        abd_est = p_espera * 0.5 * 100 
        if abd_est <= abandono_obj and ocup <= ocupacion_max:
            break
        agentes += 1
        if agentes > 2000: break
    return agentes

def optimizar_malla(df_curva, ausentismo):
    # Merma: Ausentismo + 10 min de baño sobre 330 min operativos
    merma_total = (ausentismo / 100.0) + (10 / 330.0)
    
    prob = pulp.LpProblem("Horizonte_Opt", pulp.LpMinimize)
    intervalos = list(df_curva['Intervalo'])
    req_dict = dict(zip(df_curva['Intervalo'], df_curva['Requeridos']))
    
    # Generar combinaciones de turnos (Ingreso i, Break j)
    turnos = []
    for i in range(len(intervalos) - 11):
        for j in range(i + 4, i + 8): # Ventana de break
            turnos.append((i, j))
            
    x = pulp.LpVariable.dicts("Ag", turnos, lowBound=0, cat='Integer')
    prob += pulp.lpSum([x[t] for t in turnos])
    
    for t_idx, t_str in enumerate(intervalos):
        activos = [x[(i, j)] for (i, j) in turnos if i <= t_idx < i + 12 and t_idx != j]
        if activos:
            prob += pulp.lpSum(activos) >= (req_dict[t_str] / (1 - merma_total))
            
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] != 'Optimal': return pd.DataFrame()
    
    res = []
    for (i, j) in turnos:
        if x[(i, j)].varValue > 0:
            res.append({
                "Ingreso": intervalos[i],
                "Break": intervalos[j],
                "Salida": intervalos[i+11],
                "Agentes": int(x[(i, j)].varValue)
            })
    return pd.DataFrame(res)

# --- 4. UI PRINCIPAL ---
st.title("Impulso | Dimensionamiento & Scheduling")
st.caption(f"Sesión activa: {email_usuario}")

archivo = st.file_uploader("Sube el reporte de volumen", type=['csv', 'xlsx'])

if archivo:
    try:
        # Carga y limpieza de nombres
        df_raw = pd.read_csv(archivo, sep=None, engine='python') if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df_raw.columns = df_raw.columns.str.strip()
        
        # Procesamiento de la columna Intervalo
        df_clean = df_raw[~df_raw['Intervalo'].astype(str).str.contains('Total', case=False)].copy()
        df_clean['Intervalo'] = df_clean['Intervalo'].fillna('').astype(str).str.strip()
        df_clean['Intervalo'] = df_clean['Intervalo'].apply(lambda x: x + ':00' if len(x) == 5 else x)
        df_clean = df_clean[df_clean['Intervalo'] != '']
        
        # Agrupación por curva de llamadas
        col_l = 'Llam Recibidas'
        df_clean[col_l] = pd.to_numeric(df_clean[col_l], errors='coerce').fillna(0)
        df_curva = df_clean.groupby('Intervalo')[col_l].mean().reset_index().sort_values('Intervalo')
        
        # Filtro de horario 09:00 a 21:00
        df_curva = df_curva[(df_curva['Intervalo'] >= '09:00:00') & (df_curva['Intervalo'] <= '20:30:00')].reset_index(drop=True)
        
        st.write("Curva de Llamadas (Vista Horizontal):")
        # BLINDAJE PYARROW: reset_index + astype(str) + hide_index
        st.dataframe(df_curva.set_index('Intervalo').T.reset_index().astype(str), hide_index=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            aht = st.number_input("AHT (seg)", value=250)
            aus = st.number_input("Ausentismo %", value=9.0)
        with c2:
            abd = st.number_input("Abandono Objetivo %", value=5.0)
            ocu = st.number_input("Ocupación Máxima %", value=85.0)

        if st.button("Calcular Malla Óptima"):
            with st.spinner("Optimizando recursos..."):
                df_curva['Requeridos'] = df_curva[col_l].apply(lambda x: calcular_requeridos(x, aht, abd, ocu))
                df_malla = optimizar_malla(df_curva, aus)
                
                if not df_malla.empty:
                    st.success(f"Malla generada. Total Headcount: {df_malla['Agentes'].sum()}")
                    
                    # BLINDAJE FINAL: Soluciona el error del Agente 10 y tipos mixtos
                    st.dataframe(df_malla.reset_index(drop=True).astype(str), hide_index=True)
                    

                    if supabase:
                        try:
                            # Preparamos el payload exacto
                            payload = {
                                "usuario_email": str(email_usuario),
                                "parametros": {
                                    "aht": float(aht), 
                                    "aus": float(aus), 
                                    "abd": float(abd), 
                                    "ocu": float(ocu)
                                },
                                "volumen_ingresado": df_curva.to_dict(orient="records"),
                                "malla_generada": df_malla.to_dict(orient="records") # <-- ¡CORRECCIÓN AQUÍ!
                            }
                            
                            # Intentamos la inserción
                            respuesta = supabase.table("escenarios_horizonte").insert(payload).execute()
                            
                            st.info("✅ Escenario persistido correctamente en Supabase.")
                            
                        except Exception as e:
                            st.error(f"❌ Error de base de datos: {str(e)}")                
                else:
                    st.error("No se pudo hallar una solución óptima con estos parámetros.")
                    

    except Exception as e:
        st.error(f"Error técnico: {e}")
