import streamlit as st
import pandas as pd
import pulp
import math
import openpyxl
from supabase import create_client, Client

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
st.set_page_config(page_title="Dimensionamiento", layout="wide")

@st.cache_resource
def init_connection():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None

supabase = init_connection()

# --- 2. LÓGICA DE AUTENTICACIÓN (GOOGLE AUTH) ---
def requerir_autenticacion():
    if not supabase:
        st.warning("⚠️ Configuración de Supabase no detectada. Modo local activo.")
        return "usuario_test@horizonte.com"

    if "code" in st.query_params:
        try:
            codigo_auth = st.query_params["code"]
            res_sesion = supabase.auth.exchange_code_for_session({"auth_code": codigo_auth})
            st.session_state["usuario"] = res_sesion.user.email
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error al procesar el retorno de Google: {e}")

    if st.session_state.get("usuario"):
        return st.session_state["usuario"]

    st.title("Dimensionamiento | Acceso")
    app_url = st.secrets.get("REDIRECT_URL", "https://workforcemanagement.streamlit.app/")
    
    try:
        res = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {"redirect_to": app_url}
        })
        st.link_button("🌐 Iniciar sesión con Google", res.url, type="primary")
        st.stop()
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        st.stop()

email_usuario = requerir_autenticacion()

# --- 3. MOTOR MATEMÁTICO WFM ---
def erlang_c_prob_espera(agentes, trafico):
    if agentes <= trafico: return 1.0
    eb_inv = 1.0
    for i in range(1, int(agentes) + 1):
        eb_inv = 1.0 + eb_inv * i / trafico
    eb = 1.0 / eb_inv
    return min(1.0, agentes * eb / (agentes - trafico * (1 - eb)))

def calcular_requeridos(llamadas, aht, abd_obj, ocu_max):
    if llamadas <= 0: return 0
    trafico = (llamadas * aht) / 1800.0
    agentes = math.ceil(trafico) + 1
    while True:
        p_esp = erlang_c_prob_espera(agentes, trafico)
        ocup = (trafico / agentes) * 100
        if (p_esp * 0.5 * 100) <= abd_obj and ocup <= ocu_max: break
        agentes += 1
        if agentes > 2000: break
    return agentes

def optimizar_malla(df_curva_semana, ausentismo):
    merma = (ausentismo / 100.0) + (10 / 330.0)
    prob = pulp.LpProblem("Opt", pulp.LpMinimize)
    intervalos = list(df_curva_semana['Intervalo'])
    req_dict = dict(zip(df_curva_semana['Intervalo'], df_curva_semana['Requeridos']))
    
    turnos = [(i, j) for i in range(len(intervalos) - 11) for j in range(i + 4, i + 8)]
    x = pulp.LpVariable.dicts("Ag", turnos, lowBound=0, cat='Integer')
    prob += pulp.lpSum([x[t] for t in turnos])
    
    for t_idx, t_str in enumerate(intervalos):
        activos = [x[(i, j)] for (i, j) in turnos if i <= t_idx < i + 12 and t_idx != j]
        if activos:
            prob += pulp.lpSum(activos) >= (req_dict[t_str] / (1 - merma))
            
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != 'Optimal': return pd.DataFrame()


    res = []
    for (i, j) in turnos:
        if x[(i, j)].varValue > 0:
            # Calculamos la salida real sumando 30 min al inicio del 12vo intervalo
            inicio_ult_intervalo = pd.to_datetime(intervalos[i+11], format='%H:%M:%S')
            salida_real = (inicio_ult_intervalo + pd.Timedelta(minutes=30)).strftime('%H:%M:%S')
            
            res.append({
                "Ingreso": intervalos[i], 
                "Break": intervalos[j], 
                "Salida": salida_real, 
                "Agentes": int(x[(i, j)].varValue)
            })
    return pd.DataFrame(res)

# --- 4. INTERFAZ Y PROCESAMIENTO ---
st.title("Dimensionamiento Semanal")
st.caption(f"👤 {email_usuario}")

archivo = st.file_uploader("Reporte de volumen", type=['csv', 'xlsx'])

if archivo:
    try:
        df_raw = pd.read_csv(archivo, sep=None, engine='python') if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df_raw.columns = df_raw.columns.str.strip()
        
        # Limpieza y detección de Semana
        df_clean = df_raw[~df_raw['Intervalo'].astype(str).str.contains('Total', case=False)].copy()
        df_clean['Intervalo'] = df_clean['Intervalo'].fillna('').astype(str).str.strip()
        df_clean['Intervalo'] = df_clean['Intervalo'].apply(lambda x: x + ':00' if len(x) == 5 else x)
        
        col_l = 'Llam Recibidas'
        df_clean[col_l] = pd.to_numeric(df_clean[col_l], errors='coerce').fillna(0)
        
        # Agrupación por Semana e Intervalo
        df_semanal = df_clean.groupby(['Semana', 'Intervalo'])[col_l].mean().reset_index()
        df_semanal = df_semanal[(df_semanal['Intervalo'] >= '09:00:00') & (df_semanal['Intervalo'] <= '20:30:00')]

        st.subheader("📊 Promedios Semanales (Vista Horizontal)")
        # Pivotar para vista horizontal con 1 decimal
        df_pivot = df_semanal.pivot(index='Semana', columns='Intervalo', values=col_l).round(1)
        
        # --- AJUSTE VISUAL: Dejar las columnas en formato HH:MM ---
        df_pivot.columns = [str(col)[:5] for col in df_pivot.columns]
        
        st.dataframe(df_pivot.reset_index().astype(str), hide_index=True)

        st.subheader("📈 Tendencia por Semana")
        # Ajustamos también el gráfico para que el eje X se vea limpio
        df_chart = df_semanal.copy()
        df_chart['Intervalo'] = df_chart['Intervalo'].str[:5]
        st.line_chart(df_chart.pivot(index='Intervalo', columns='Semana', values=col_l))

        

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            aht = st.number_input("AHT (seg)", value=420)
            aus = st.number_input("Ausentismo %", value=9.0)
        with c2:
            abd = st.number_input("Abandono %", value=10.0)
            ocu = st.number_input("Ocupación %", value=70.0)

        if st.button("Calcular Nómina Semanal"):
            with st.spinner("Optimizando mallas..."):
                mallas_totales = []
                for semana in df_semanal['Semana'].unique():
                    df_sem = df_semanal[df_semanal['Semana'] == semana].copy()
                    df_sem['Requeridos'] = df_sem[col_l].apply(lambda x: calcular_requeridos(x, aht, abd, ocu))
                    
                    df_res = optimizar_malla(df_sem, aus)
                    if not df_res.empty:
                        df_res['Semana'] = semana
                        mallas_totales.append(df_res)
                
                if mallas_totales:
                    df_final = pd.concat(mallas_totales)
                    st.success(f"Nómina generada para {len(mallas_totales)} semanas.")
                    
                    # Mostrar nómina separada por semana
                    for sem in df_final['Semana'].unique():
                        with st.expander(f"📅 Nómina: {sem}"):
                            df_sem_viz = df_final[df_final['Semana'] == sem].drop(columns='Semana')
                            st.write(f"**Total agentes necesarios:** {df_sem_viz['Agentes'].sum()}")
                            st.dataframe(df_sem_viz.reset_index(drop=True).astype(str), hide_index=True)

                    if supabase:
                        try:
                            supabase.table("escenarios_horizonte").insert({
                                "usuario_email": email_usuario,
                                "parametros": {"aht": aht, "aus": aus, "abd": abd},
                                "malla_generada": df_final.to_dict(orient="records")
                            }).execute()
                            st.info("✅ Escenarios semanales guardados en BD.")
                        except Exception as e:
                            st.error(f"Error en BD: {e}")
                else:
                    st.error("No se halló solución óptima.")
    except Exception as e:
        st.error(f"Error técnico: {e}")
