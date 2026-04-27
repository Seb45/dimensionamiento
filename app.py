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

# --- 2. AUTENTICACIÓN CON GOOGLE (SUPABASE) ---
def requerir_autenticacion():
    # Si no hay conexión a Supabase configurada en los secrets, permitimos el uso local/pruebas
    if not supabase:
        st.warning("⚠️ Supabase no está configurado. Ejecutando en modo local sin guardado en BD.")
        return "usuario_local@horizonte.com"

    # Verificar si ya hay una sesión activa en el estado de Streamlit
    if "usuario" not in st.session_state:
        st.session_state["usuario"] = None

    # En un flujo real de Streamlit en la nube, aquí se capturaría el token de redirección.
    # Para la UI, mostramos el botón de login si no hay sesión.
    if not st.session_state["usuario"]:
        st.subheader("🔒 Acceso Restringido")
        st.write("Por favor, inicia sesión con tu cuenta de Google para utilizar Proyecto Horizonte.")
        
        try:
            # URL de redirección (debe coincidir con la configurada en Google Cloud y Supabase)
            app_url = st.secrets.get("REDIRECT_URL", "http://localhost:8501")
            
            res = supabase.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {"redirect_to": app_url}
            })
            
            st.link_button("🌐 Iniciar sesión con Google", res.url, type="primary")
            
            # Botón de simulación para entorno de desarrollo (puedes borrarlo en producción)
            if st.button("Simular Login (Solo Desarrollo)"):
                st.session_state["usuario"] = "lider@horizonte.com"
                st.rerun()
                
            st.stop() # Detiene la ejecución de la app hasta que el usuario se loguee
            
        except Exception as e:
            st.error(f"Error al conectar con el proveedor de autenticación: {e}")
            st.stop()
            
    return st.session_state["usuario"]

# Ejecutar el bloqueo de autenticación
usuario_actual = requerir_autenticacion()


# --- 3. MOTOR WFM (ERLANG A PROX) ---
def erlang_c_prob_espera(agentes, trafico):
    if agentes <= trafico: return 1.0
    erlang_b_inv = 1.0
    for i in range(1, int(agentes) + 1):
        erlang_b_inv = 1.0 + erlang_b_inv * i / trafico
    erlang_b = 1.0 / erlang_b_inv
    prob_espera = agentes * erlang_b / (agentes - trafico * (1 - erlang_b))
    return min(1.0, prob_espera)

def calcular_requeridos(llamadas, aht, abandono_obj, ocupacion_max):
    if llamadas <= 0: return 0
    trafico = (llamadas * aht) / 1800.0
    agentes = math.ceil(trafico) + 1
    while True:
        prob_espera = erlang_c_prob_espera(agentes, trafico)
        ocupacion = (trafico / agentes) * 100
        abandono_est = prob_espera * 0.5 * 100 
        if abandono_est <= abandono_obj and ocupacion <= ocupacion_max:
            break
        agentes += 1
        if agentes > 1000: break # Freno de seguridad
    return agentes

# --- 4. OPTIMIZADOR DE TURNOS (PuLP) ---
def optimizar_turnos_6hs(df_curva, ausentismo_pct):
    merma_total = (ausentismo_pct / 100.0) + (10 / 330.0)
    prob = pulp.LpProblem("Horizonte_Minimization", pulp.LpMinimize)
    
    intervalos = list(df_curva['Intervalo'])
    requeridos = dict(zip(df_curva['Intervalo'], df_curva['Requeridos']))
    
    turnos_posibles = []
    for i in range(len(intervalos) - 11):
        for j in range(i + 4, i + 8): # Break entre 2da y 4ta hora
            turnos_posibles.append((i, j))
            
    x = pulp.LpVariable.dicts("Ag", turnos_posibles, lowBound=0, cat='Integer')
    prob += pulp.lpSum([x[t] for t in turnos_posibles])
    
    for t_idx, t_str in enumerate(intervalos):
        agentes_en_t = [x[(i, j)] for (i, j) in turnos_posibles if i <= t_idx < i + 12 and t_idx != j]
        if agentes_en_t:
            prob += pulp.lpSum(agentes_en_t) >= (requeridos[t_str] / (1 - merma_total))
            
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return pd.DataFrame()

    resultados = []
    for (i, j) in turnos_posibles:
        val = x[(i, j)].varValue
        if val and val > 0:
            resultados.append({
                "Ingreso": intervalos[i],
                "Break": intervalos[j],
                "Salida": intervalos[i+11],
                "Agentes": int(val)
            })
    return pd.DataFrame(resultados)


# --- 5. INTERFAZ Y VISUALIZACIÓN SEGURA ---
st.title("Proyecto Horizonte | Optimización de Staffing")
st.caption(f"👤 Operador: {usuario_actual}")

uploaded_file = st.file_uploader("Cargar volumen por intervalo", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Ingesta
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip()
        
        # Limpieza de datos
        df_clean = df_raw[~df_raw['Intervalo'].astype(str).str.contains('Total', case=False)].copy()
        df_clean['Intervalo'] = df_clean['Intervalo'].fillna('').astype(str).str.strip()
        df_clean['Intervalo'] = df_clean['Intervalo'].apply(lambda x: x + ':00' if len(x) == 5 else x)
        df_clean = df_clean[df_clean['Intervalo'] != ''].sort_values('Intervalo')
        
        col_llam = 'Llam Recibidas'
        df_clean[col_llam] = pd.to_numeric(df_clean[col_llam], errors='coerce').fillna(0)
        df_curva = df_clean.groupby('Intervalo')[col_llam].mean().reset_index()
        df_curva = df_curva[(df_curva['Intervalo'] >= '09:00:00') & (df_curva['Intervalo'] <= '20:30:00')].reset_index(drop=True)
        
        st.write("Curva de Entrada Proyectada:")
        # Blindaje para la visualización transpuesta
        df_display_curva = df_curva.set_index('Intervalo').T.reset_index().astype(str)
        st.dataframe(df_display_curva, hide_index=True)

        st.divider()
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            aht_val = st.number_input("AHT (seg)", value=250)
            aus_val = st.number_input("Ausentismo %", value=9.0)
        with col_p2:
            abd_val = st.number_input("Abandono Objetivo %", value=5.0)
            ocu_val = st.number_input("Ocupación Máxima %", value=85.0)

        if st.button("Ejecutar Optimización"):
            with st.spinner("Procesando motor matemático..."):
                df_curva['Requeridos'] = df_curva[col_llam].apply(lambda x: calcular_requeridos(x, aht_val, abd_val, ocu_val))
                df_malla = optimizar_turnos_6hs(df_curva, aus_val)
                
                if not df_malla.empty:
                    st.success(f"Malla generada con éxito. Headcount total: {df_malla['Agentes'].sum()}")
                    
                    # Blindaje total de la tabla de resultados para evitar fallos de ArrowTypeError
                    st.dataframe(df_malla.reset_index(drop=True).astype(str), hide_index=True)
                    
                    if supabase:
                        try:
                            payload = {
                                "usuario_email": usuario_actual,
                                "parametros": {"aht": aht_val, "aus": aus_val, "abd": abd_val, "ocu": ocu_val},
                                "malla_generada": df_malla.to_dict(orient="records")
                            }
                            supabase.table("escenarios_horizonte").insert(payload).execute()
                            st.info("Escenario guardado en base de datos bajo el usuario actual.")
                        except Exception as e:
                            st.error(f"Error al persistir en BD: {e}")
                else:
                    st.error("No se encontró una solución óptima para los parámetros ingresados.")
                    
    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
