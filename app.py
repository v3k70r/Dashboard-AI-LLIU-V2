import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import re
import unicodedata
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 1) CONFIG + BRANDING
# =========================================================
st.set_page_config(page_title="Dashboard AI‑LLIU | MASXXI", page_icon="🦋", layout="wide")

DATA_DIR = "data"
TASA_USD_CLP = 950

URL_LOGO = "https://i.ibb.co/YqVrvwS/Recurso-2.png"

COLOR_PRIMARIO = "#0E116A"
COLOR_ACENTO = "#CEDF74"
COLOR_TEXTO = "#1A1A1A"
COLOR_FONDO = "#F4F5F2"
COLOR_CARD_BG = "#FFFFFF"
COLOR_ESCALA = ["#0E116A", "#1D2396", "#3B44B5", "#626AD1", "#8F96E8", "#CEDF74"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&display=swap');

html, body, [class*="st-"], .stMarkdown, p, span, label, div[data-testid="stMetricValue"] {{
  font-family: 'Montserrat', sans-serif !important;
  color: {COLOR_TEXTO} !important;
}}
.stApp {{ background-color: {COLOR_FONDO} !important; }}

.logo-container {{
  display: flex; justify-content: center; align-items: center;
  padding: 10px; margin-bottom: 12px;
}}

div[data-testid="metric-container"] {{
  background-color: {COLOR_CARD_BG} !important;
  border: 1px solid #EAEAEA !important;
  padding: 16px 16px 14px 16px; border-radius: 16px !important;
  box-shadow: 0 4px 15px rgba(14, 17, 106, 0.04) !important;
  border-left: 8px solid {COLOR_ACENTO} !important;
}}
/* Reduce un poco el tamaño del valor para que no se corte */
div[data-testid="stMetricValue"] {{
  font-size: 2.2rem !important;
  line-height: 1.05 !important;
}}
hr {{ border-color: #DEE2DE !important; }}
.small-muted {{ font-size: 12px; opacity: .75; }}

/* === AI-LLIU Branding: MultiSelect/Select tags (fix contraste) === */
div[data-baseweb="tag"] {{
  background-color: #CEDF74 !important;   /* lima */
  border: 1px solid rgba(14,17,106,0.25) !important;
  border-radius: 999px !important;
}}
/* fuerza color en TODO el contenido del chip */
div[data-baseweb="tag"] * {{
  color: #0E116A !important;             /* azul */
  fill: #0E116A !important;
}}
/* borde/focus del select */
div[data-baseweb="select"] > div {{
  border-color: rgba(14,17,106,0.22) !important;
  box-shadow: none !important;
}}
div[data-baseweb="select"] > div:focus-within {{
  border-color: #0E116A !important;
  box-shadow: 0 0 0 3px rgba(206,223,116,0.35) !important;
}}

/* === Sidebar (AI-LLIU UI polish) === */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(14,17,106,0.06) 0%, rgba(206,223,116,0.10) 100%) !important;
  border-right: 1px solid rgba(14,17,106,0.10) !important;
}}
section[data-testid="stSidebar"] > div {{
  padding-top: 18px !important;
}}
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown h2 {{
  color: #0E116A !important;
  letter-spacing: 0.2px;
}}
section[data-testid="stSidebar"] .stMarkdown hr {{
  border-color: rgba(14,17,106,0.10) !important;
}}
section[data-testid="stSidebar"] label {{
  font-weight: 600 !important;
  color: rgba(26,26,26,0.92) !important;
}}

/* Inputs look */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
section[data-testid="stSidebar"] div[data-baseweb="textarea"] > div {{
  background-color: rgba(255,255,255,0.92) !important;
  border: 1px solid rgba(14,17,106,0.16) !important;
  border-radius: 14px !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {{
  border-color: rgba(14,17,106,0.28) !important;
}}

/* Chips/tags spacing a bit nicer */
section[data-testid="stSidebar"] div[data-baseweb="tag"] {{
  border-radius: 999px !important;
  padding: 4px 8px !important;
}}

/* Buttons */
section[data-testid="stSidebar"] .stButton>button,
section[data-testid="stSidebar"] .stDownloadButton>button {{
  border-radius: 14px !important;
  border: 1px solid rgba(14,17,106,0.20) !important;
  padding: 10px 12px !important;
}}
section[data-testid="stSidebar"] .stButton>button:hover,
section[data-testid="stSidebar"] .stDownloadButton>button:hover {{
  border-color: rgba(14,17,106,0.35) !important;
  box-shadow: 0 6px 18px rgba(14,17,106,0.08) !important;
}}

/* Radio / checkbox accent */
section[data-testid="stSidebar"] input[type="checkbox"],
section[data-testid="stSidebar"] input[type="radio"] {{
  accent-color: #0E116A !important;
}}


/* Override de alta especificidad para chips en sidebar (evita que queden negros/invisibles) */
section[data-testid="stSidebar"] div[data-baseweb="tag"] * {{
  color: #0E116A !important;
  fill: #0E116A !important;
}}


/* === HOTFIX: fuerza chips legibles en Chrome/mac (BaseWeb) === */
section[data-testid="stSidebar"] [data-baseweb="tag"],
section[data-testid="stSidebar"] div[data-baseweb="tag"],
section[data-testid="stSidebar"] span[data-baseweb="tag"] {{
  background-color: #CEDF74 !important;
  background: #CEDF74 !important;
  border: 1px solid rgba(14,17,106,0.25) !important;
  border-radius: 999px !important;
}}
section[data-testid="stSidebar"] [data-baseweb="tag"] * {{
  color: #0E116A !important;
  fill: #0E116A !important;
}}
/* A veces el color viene en un span interno "Tag" */
section[data-testid="stSidebar"] [data-baseweb="tag"] > span,
section[data-testid="stSidebar"] [data-baseweb="tag"] > div {{
  background-color: #CEDF74 !important;
  background: #CEDF74 !important;
}}
/* Si Streamlit aplica primaryColor a los tags, lo anulamos en el contenedor */
section[data-testid="stSidebar"] div[data-baseweb="select"] [data-baseweb="tag"] {{
  background-color: #CEDF74 !important;
}}

</style>
""", unsafe_allow_html=True)

layout_dict = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Montserrat", color=COLOR_TEXTO),
    margin=dict(l=0, r=0, t=35, b=0),
)

# =========================================================
# 2) HELPERS
# =========================================================
def normalizar_cargos(serie: pd.Series) -> np.ndarray:
    s = serie.astype(str).str.lower().str.strip()
    condiciones = [
        s.str.contains(r'utp|técnico|tecnico|curricular|evaluador', regex=True),
        s.str.contains(r'director|directora|rector|rectora', regex=True),
        s.str.contains(r'profesor|docente|educador|maestro', regex=True),
        s.str.contains(r'coordinador|coordinadora', regex=True),
    ]
    return np.select(
        condiciones,
        ['Jefatura UTP', 'Equipo Directivo', 'Docente de Aula', 'Coordinación'],
        default='Otro Profesional'
    )

def inferir_asignatura(titulo: str) -> str:
    t = str(titulo).lower()
    if re.search(r'matem[aá]tica|c[aá]lculo|geometr[ií]a|n[uú]mero|[aá]lgebra|fracci[oó]n|ecuaci[oó]n|datos', t):
        return 'Matemática'
    if re.search(r'ciencia.*natural|biolog[ií]a|naturaleza|medio.*ambiente|ecosistema|c[eé]lula|universo', t):
        return 'Ciencias Naturales'
    if re.search(r'lenguaje|comunicaci[oó]n|literatura|lectura|comprensi[oó]n|escritura|poes[ií]a|cuento|texto', t):
        return 'Lenguaje y Comunicación'
    if re.search(r'historia|geograf[ií]a|sociales|ciudadan[ií]a|c[ií]vica|civica|pueblos originarios|grecia|roma', t):
        return 'Historia y Cs. Sociales'
    if re.search(r'f[ií]sica|qu[ií]mica|termodin[aá]mica|fuerza|energ[ií]a|materia|luz', t):
        return 'Física y Química'
    if re.search(r'arte|m[uú]sica|visuales|danza|pintura|escultura', t):
        return 'Artes y Música'
    if re.search(r'ed.*f[ií]sica|deporte|motricidad|entrenamiento|saludable', t):
        return 'Educación Física'
    if re.search(r'ingl[eé]s|idioma|english', t):
        return 'Inglés'
    if re.search(r'parvularia|k[ií]nder|preb[aá]sica|transici[oó]n|p[aá]rvulo', t):
        return 'Educación Parvularia'
    if re.search(r'arduino|scratch|programaci[oó]n|rob[oó]tica|software|inform[aá]tica|tecnolog[ií]a', t):
        return 'Tecnología e Informática'
    return 'Planificación Transversal'

def limpiar_institucion(inst) -> str:
    if pd.isna(inst) or str(inst).strip() == '' or str(inst).lower() == 'nan':
        return 'NO ESPECIFICADO'
    s = ''.join(c for c in unicodedata.normalize('NFD', str(inst)) if unicodedata.category(c) != 'Mn')
    s = s.upper().strip()
    if "RANCAGUA" in s and ("BICENTENARIO" in s or "TECNICO" in s):
        return 'LICEO BICENTENARIO TÉCNICO DE RANCAGUA'
    return s

def safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except Exception:
        return 0

def fmt_compact(n: float) -> str:
    try:
        n = float(n)
    except Exception:
        return "0"
    absn = abs(n)
    if absn >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B".replace(".", ",")
    if absn >= 1_000_000:
        return f"{n/1_000_000:.1f}M".replace(".", ",")
    if absn >= 1_000:
        return f"{n/1_000:.1f}K".replace(".", ",")
    return f"{int(n):,}".replace(",", ".")

def fmt_clp(n: float) -> str:
    try:
        return f"${int(n):,} CLP".replace(",", ".")
    except Exception:
        return "$0 CLP"

def ensure_full_json(data_dir: str) -> str:
    """
    Reconstruye data/full_conversations.json desde partes <=100MB para evitar subir el archivo completo a GitHub.
    Soporta 2 formatos:
    - data/full_conversations.json.part001, part002...
    - data/*ConversationTable*.part1, part2...
    """
    os.makedirs(data_dir, exist_ok=True)
    out_file = os.path.join(data_dir, "full_conversations.json")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return out_file

    parts = sorted(glob.glob(os.path.join(data_dir, "full_conversations.json.part*")))
    if not parts:
        parts = sorted(glob.glob(os.path.join(data_dir, "*ConversationTable*.part*")))

    if not parts:
        return out_file  # no existe; el caller manejará

    with open(out_file, "wb") as outfile:
        for part in parts:
            with open(part, "rb") as infile:
                outfile.write(infile.read())
    return out_file

# =========================================================
# 3) LOAD
# =========================================================
@st.cache_data(show_spinner="Consolidando datos…")
def load_data():
    full_json_path = ensure_full_json(DATA_DIR)

    # Usuarios
    users_path = os.path.join(DATA_DIR, "cleaned_cognito_users.csv")
    try:
        df_users = pd.read_csv(users_path)
        df_users['jobTitle_norm'] = normalizar_cargos(df_users.get('jobTitle', pd.Series([''] * len(df_users))))
        df_users['inst_clean'] = df_users.get('institution', pd.Series([''] * len(df_users))).apply(limpiar_institucion)
        if 'UserCreateDate' in df_users.columns:
            df_users['UserCreateDate'] = pd.to_datetime(df_users['UserCreateDate'], utc=True, errors='coerce') \
                                           .dt.tz_convert('America/Santiago').dt.tz_localize(None)
    except Exception:
        df_users = pd.DataFrame()

    # Conversaciones
    records = []
    if os.path.exists(full_json_path) and os.path.getsize(full_json_path) > 0:
        with open(full_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            sk = str(item.get("SK", ""))
            uuid_match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', sk)
            if uuid_match and "BOT_ALIAS" not in sk:
                costo_clp = float(item.get("TotalPrice", 0)) * TASA_USD_CLP
                tokens = item.get("TotalTokens", None)
                tokens = int(tokens) if tokens is not None else int(costo_clp * 100)

                records.append({
                    "UserId": uuid_match.group(0),
                    "Fecha": pd.to_datetime(int(item.get("CreateTime")), unit='ms', utc=True) if item.get("CreateTime") else pd.NaT,
                    "Titulo": item.get("Title", ""),
                    "Asignatura": inferir_asignatura(item.get("Title", "")),
                    "Costo_CLP": float(costo_clp),
                    "Tokens": int(tokens),
                })

    df_conv = pd.DataFrame(records)
    if not df_conv.empty:
        df_conv['Fecha_Local'] = pd.to_datetime(df_conv['Fecha'], utc=True, errors='coerce') \
                                    .dt.tz_convert('America/Santiago').dt.tz_localize(None)
        df_conv = df_conv.dropna(subset=['Fecha_Local']).sort_values('Fecha_Local')
        df_conv['DiaSemana'] = df_conv['Fecha_Local'].dt.day_name().map({
            'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles', 'Thursday': 'Jueves',
            'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        })
        df_conv['FranjaHoraria'] = df_conv['Fecha_Local'].dt.hour.apply(
            lambda x: 'Mañana' if 5 <= x < 13 else 'Tarde' if 13 <= x < 19 else 'Noche'
        )
        df_conv['Fecha_Dia'] = df_conv['Fecha_Local'].dt.floor('D')
        df_conv['Semana'] = df_conv['Fecha_Local'].dt.to_period('W').astype(str)
        df_conv['Mes'] = df_conv['Fecha_Local'].dt.to_period('M').astype(str)

    # Master
    if not df_conv.empty and not df_users.empty:
        df_master = pd.merge(df_conv, df_users, left_on="UserId", right_on="sub", how="left")
        df_master['region'] = df_master.get('region', 'Desconocida').fillna('Desconocida')
        df_master['inst_clean'] = df_master.get('inst_clean', 'NO ESPECIFICADO')
        df_master['jobTitle_norm'] = df_master.get('jobTitle_norm', 'Otro Profesional')
    else:
        df_master = df_conv.copy()
        if df_master.empty:
            df_master = pd.DataFrame(columns=['UserId','Fecha_Local','Fecha_Dia','Semana','Mes','Titulo','Asignatura','Costo_CLP','Tokens','DiaSemana','FranjaHoraria','region','inst_clean','jobTitle_norm'])
        df_master['region'] = df_master.get('region', 'Desconocida')
        df_master['inst_clean'] = df_master.get('inst_clean', 'NO ESPECIFICADO')
        df_master['jobTitle_norm'] = df_master.get('jobTitle_norm', 'Otro Profesional')

    return df_master, df_users

df_master, df_users = load_data()

# =========================================================
# 4) HEADER
# =========================================================
st.markdown(f"""
<div class="logo-container">
  <img src="{URL_LOGO}" width="250">
</div>
""", unsafe_allow_html=True)

st.markdown("## Dashboard AI‑LLIU (histórico)")

if df_master.empty:
    st.error("No se cargaron datos. Revisa la carpeta /data (partes .part### + cleaned_cognito_users.csv).")
    st.stop()

# =========================================================
# 5) SIDEBAR — FILTROS
# =========================================================
with st.sidebar:
    st.markdown("### Filtros")
    min_date = df_master['Fecha_Local'].min()
    max_date = df_master['Fecha_Local'].max()

    date_range = st.date_input(
        "Rango de fechas",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )

    roles = sorted(df_master['jobTitle_norm'].dropna().unique().tolist())
    asignaturas = sorted(df_master['Asignatura'].dropna().unique().tolist())
    regiones = sorted(df_master['region'].dropna().unique().tolist())
    insts = sorted(df_master['inst_clean'].dropna().unique().tolist())

    sel_roles = st.multiselect("Rol", roles, default=roles)
    sel_asig = st.multiselect("Asignatura (inferida)", asignaturas, default=asignaturas)
    sel_reg = st.multiselect("Región", regiones, default=regiones)

    top_inst = df_master['inst_clean'].value_counts().head(50).index.tolist()
    sel_inst_mode = st.radio("Institución", ["Todas", "Elegir (top 50)"], horizontal=True)
    if sel_inst_mode == "Elegir (top 50)":
        sel_inst = st.multiselect("Top 50 instituciones", top_inst, default=top_inst)
    else:
        sel_inst = insts

    st.markdown("---")
    show_outliers = st.checkbox("Recortar outliers (p99 tokens/costo)", value=True)

# aplicar filtros
start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (
    (df_master['Fecha_Local'] >= start_d) &
    (df_master['Fecha_Local'] <= (end_d + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))) &
    (df_master['jobTitle_norm'].isin(sel_roles)) &
    (df_master['Asignatura'].isin(sel_asig)) &
    (df_master['region'].isin(sel_reg)) &
    (df_master['inst_clean'].isin(sel_inst))
)
df = df_master.loc[mask].copy()

if df.empty:
    st.warning("Con estos filtros no hay registros. Ajusta rango/selección en la barra lateral.")
    st.stop()

if show_outliers:
    for col in ['Tokens','Costo_CLP']:
        if col in df.columns and df[col].notna().any():
            cap = df[col].quantile(0.99)
            df[col] = np.minimum(df[col], cap)

# =========================================================
# 6) KPIs — ahora en 2 filas (para que no se corten)
# =========================================================
total_cost = float(df['Costo_CLP'].sum())
total_tokens = float(df['Tokens'].sum())
total_sessions = int(len(df))
active_users = safe_nunique(df['UserId'])
time_saved_hours = int(total_sessions * 1.5)

dau = df.groupby('Fecha_Dia')['UserId'].nunique()
wau = df.groupby(pd.Grouper(key='Fecha_Dia', freq='W'))['UserId'].nunique()
mau = df.groupby(pd.Grouper(key='Fecha_Dia', freq='M'))['UserId'].nunique()
dau_avg = dau.mean() if len(dau) else 0
wau_avg = wau.mean() if len(wau) else 0
mau_avg = mau.mean() if len(mau) else 0

meses = max(1, round((df['Fecha_Local'].max() - df['Fecha_Local'].min()).days / 30.41))

st.write("---")

r1c1, r1c2, r1c3 = st.columns(3)
r2c1, r2c2, r2c3 = st.columns(3)

r1c1.metric("Usuarios", fmt_compact(active_users), help="Usuarios únicos en la vista filtrada")
r1c2.metric("Sesiones", fmt_compact(total_sessions), help="Cantidad de planificaciones/sesiones")
r1c3.metric("Ahorro (hrs)", fmt_compact(time_saved_hours), help="Estimación: 1,5h por planificación")

r2c1.metric("Tokens", fmt_compact(total_tokens), help="Suma de tokens (si falta, se estima por costo)")
r2c2.metric("Costo (CLP)", fmt_clp(total_cost), f"Prom/mes: {fmt_clp(total_cost/meses)}")
r2c3.metric("DAU / WAU / MAU", f"{dau_avg:.1f} / {wau_avg:.1f} / {mau_avg:.1f}", help="Promedios en el rango filtrado")

st.caption(
    f"<span class='small-muted'>Vista: {date_range[0]} → {date_range[1]} · "
    f"Roles: {len(sel_roles)} · Asignaturas: {len(sel_asig)} · Regiones: {len(sel_reg)}</span>",
    unsafe_allow_html=True
)

# =========================================================
# 7) TABS
# =========================================================
tab_exec, tab_eng, tab_ret, tab_geo, tab_data = st.tabs([
    "📌 Resumen ejecutivo",
    "📈 Engagement y eficiencia",
    "🔁 Retención",
    "🗺️ Territorio e instituciones",
    "⬇️ Datos"
])

with tab_exec:
    st.markdown("### Distribución curricular e institucional")

    c1, c2 = st.columns([1.25, 1])
    with c1:
        df_asig = df[df['Asignatura'] != 'Planificación Transversal']['Asignatura'].value_counts().reset_index()
        df_asig.columns = ['Asignatura', 'Sesiones']
        fig = px.treemap(df_asig, path=['Asignatura'], values='Sesiones', color='Sesiones',
                         color_continuous_scale=COLOR_ESCALA)
        fig.update_layout(**layout_dict)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        df_inst = df[df['inst_clean'] != 'NO ESPECIFICADO']['inst_clean'].value_counts().head(10).reset_index()
        df_inst.columns = ['Institución', 'Sesiones']
        fig = px.bar(df_inst, x='Sesiones', y='Institución', orientation='h',
                     color_discrete_sequence=[COLOR_PRIMARIO])
        fig.update_layout(**layout_dict, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.markdown("### Evolución: sesiones, usuarios activos y costo")

    daily = df.groupby('Fecha_Dia').agg(
        Sesiones=('Titulo','count'),
        Usuarios=('UserId','nunique'),
        Tokens=('Tokens','sum'),
        Costo_CLP=('Costo_CLP','sum'),
    ).reset_index()

    c3, c4 = st.columns([1.4, 1])
    with c3:
        fig = px.line(daily, x='Fecha_Dia', y=['Sesiones', 'Usuarios'], markers=False)
        fig.update_layout(**layout_dict, legend_title_text="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.area(daily, x='Fecha_Dia', y='Costo_CLP')
        fig.update_layout(**layout_dict, yaxis_title="CLP")
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.markdown("### Concentración de uso (Pareto)")

    by_user = df.groupby('UserId').size().sort_values(ascending=False)
    pareto = (by_user.cumsum() / by_user.sum()).reset_index()
    pareto.columns = ['UserId', 'Cumulativo']
    pareto['Usuarios'] = np.arange(1, len(pareto)+1)
    fig = px.line(pareto, x='Usuarios', y='Cumulativo')
    fig.update_layout(**layout_dict, yaxis_tickformat=".0%", xaxis_title="Usuarios (ordenados por uso)",
                      yaxis_title="Sesiones acumuladas")
    st.plotly_chart(fig, use_container_width=True)

with tab_eng:
    st.markdown("### Hábitos de uso")

    c1, c2, c3 = st.columns([1.05, 1.05, 1.4])
    with c1:
        order = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
        df_dias = df['DiaSemana'].value_counts().reindex(order).fillna(0).reset_index()
        df_dias.columns = ['Día', 'Sesiones']
        fig = px.bar(df_dias, x='Día', y='Sesiones', color_discrete_sequence=[COLOR_PRIMARIO])
        fig.update_layout(**layout_dict)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        df_h = df['FranjaHoraria'].value_counts().reindex(['Mañana','Tarde','Noche']).fillna(0).reset_index()
        df_h.columns = ['Franja', 'Sesiones']
        fig = go.Figure(data=[go.Pie(labels=df_h['Franja'], values=df_h['Sesiones'], hole=.55,
                                     marker_colors=[COLOR_PRIMARIO, COLOR_ACENTO, "#AAB4BE"])])
        fig.update_layout(**layout_dict, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        df_heat = df.copy()
        df_heat['Hora'] = df_heat['Fecha_Local'].dt.hour
        order_map = {d:i for i,d in enumerate(['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'])}
        df_heat['DiaIdx'] = df_heat['DiaSemana'].map(order_map)
        pivot = pd.pivot_table(df_heat, index='DiaIdx', columns='Hora', values='Titulo', aggfunc='count').fillna(0)
        pivot = pivot.reindex(range(7))
        pivot.index = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues")
        fig.update_layout(**layout_dict, coloraxis_showscale=False, xaxis_title="Hora del día", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.markdown("### Eficiencia: tokens y costo")

    c4, c5, c6 = st.columns(3)
    with c4:
        fig = px.histogram(df, x='Tokens', nbins=40)
        fig.update_layout(**layout_dict, xaxis_title="Tokens por sesión", yaxis_title="Sesiones")
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        fig = px.box(df, x="Asignatura", y="Tokens")
        fig.update_layout(**layout_dict, xaxis_title="", yaxis_title="Tokens")
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        agg = df.groupby('UserId').agg(Sesiones=('Titulo','count'), Tokens=('Tokens','sum'), Costo=('Costo_CLP','sum')).reset_index()
        fig = px.scatter(agg, x='Tokens', y='Costo', size='Sesiones', hover_name='UserId')
        fig.update_layout(**layout_dict, xaxis_title="Tokens (acumulados)", yaxis_title="Costo CLP (acumulado)")
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.markdown("### Flujo (Sankey): Rol → Asignatura → Franja")

    s_df = df[['jobTitle_norm','Asignatura','FranjaHoraria']].copy()
    s_df['value'] = 1
    links1 = s_df.groupby(['jobTitle_norm','Asignatura'])['value'].sum().reset_index()
    links2 = s_df.groupby(['Asignatura','FranjaHoraria'])['value'].sum().reset_index()

    nodes = pd.Index(pd.concat([links1['jobTitle_norm'], links1['Asignatura'], links2['FranjaHoraria']]).unique())
    node_map = {k:i for i,k in enumerate(nodes)}
    sources = [node_map[x] for x in links1['jobTitle_norm']] + [node_map[x] for x in links2['Asignatura']]
    targets = [node_map[x] for x in links1['Asignatura']] + [node_map[x] for x in links2['FranjaHoraria']]
    values  = links1['value'].tolist() + links2['value'].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=nodes.tolist(), pad=14, thickness=14),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(**layout_dict)
    st.plotly_chart(fig, use_container_width=True)

with tab_ret:
    st.markdown("### Cohortes semanales (retención de usuarios)")
    st.caption("Cohorte = semana del primer uso; celdas = % de esa cohorte que vuelve en semanas siguientes.")

    tmp = df[['UserId','Fecha_Local']].copy()
    tmp['Week'] = tmp['Fecha_Local'].dt.to_period('W').apply(lambda p: p.start_time)
    first = tmp.groupby('UserId')['Week'].min().rename('Cohort')
    tmp = tmp.join(first, on='UserId')
    tmp['WeekIndex'] = ((tmp['Week'] - tmp['Cohort']) / np.timedelta64(1, 'W')).astype(int)

    cohort = tmp.groupby(['Cohort','WeekIndex'])['UserId'].nunique().reset_index()
    cohort_sizes = cohort[cohort['WeekIndex'] == 0].set_index('Cohort')['UserId']
    cohort['Retention'] = cohort.apply(lambda r: r['UserId'] / cohort_sizes.get(r['Cohort'], np.nan), axis=1)

    heat = cohort.pivot(index='Cohort', columns='WeekIndex', values='Retention').sort_index().fillna(0.0)

    if heat.shape[0] > 0:
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="Blues", zmin=0, zmax=1)
        fig.update_layout(**layout_dict, xaxis_title="Semanas desde 1er uso", yaxis_title="Cohorte (semana)")
        fig.update_traces(customdata=np.round(heat.values*100,1), hovertemplate="Retención: %{customdata}%<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay suficiente información para cohortes con los filtros actuales.")

    st.write("---")
    st.markdown("### Retorno de usuarios (rolling)")

    daily_users = df.groupby('Fecha_Dia')['UserId'].nunique().reset_index(name='Usuarios')
    daily_users['Rolling_7d'] = daily_users['Usuarios'].rolling(7, min_periods=1).mean()
    fig = px.line(daily_users, x='Fecha_Dia', y=['Usuarios','Rolling_7d'])
    fig.update_layout(**layout_dict, legend_title_text="", yaxis_title="Usuarios únicos/día")
    st.plotly_chart(fig, use_container_width=True)

with tab_geo:
    st.markdown("### Territorio e instituciones")

    c1, c2 = st.columns([1.1, 1.2])
    with c1:
        reg = df['region'].value_counts().head(12).reset_index()
        reg.columns = ['Región','Sesiones']
        fig = px.bar(reg, x='Sesiones', y='Región', orientation='h', color_discrete_sequence=[COLOR_ACENTO])
        fig.update_layout(**layout_dict, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if not df_users.empty and 'UserCreateDate' in df_users.columns:
            u = df_users.dropna(subset=['UserCreateDate']).copy()
            u = u.set_index('UserCreateDate').resample('W').size().cumsum().reset_index(name='Acumulado')
            fig = px.line(u, x='UserCreateDate', y='Acumulado', color_discrete_sequence=[COLOR_ACENTO])
            fig.update_layout(**layout_dict, xaxis_title="Semana", yaxis_title="Usuarios acumulados")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay UserCreateDate disponible en la tabla de usuarios para graficar crecimiento.")

    st.write("---")
    st.markdown("### Instituciones: top y evolución")

    topN = 12
    top_inst = df[df['inst_clean'] != 'NO ESPECIFICADO']['inst_clean'].value_counts().head(topN).index.tolist()
    inst_ts = df[df['inst_clean'].isin(top_inst)].groupby(['Mes','inst_clean']).size().reset_index(name='Sesiones')
    if len(inst_ts):
        fig = px.line(inst_ts, x='Mes', y='Sesiones', color='inst_clean')
        fig.update_layout(**layout_dict, legend_title_text="Institución", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay instituciones suficientes con los filtros actuales.")

with tab_data:
    st.markdown("### Exportar vista filtrada")

    c1, c2 = st.columns([1,1])
    with c1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV (vista filtrada)", data=csv, file_name="ai_lliu_filtrado.csv", mime="text/csv")

    with c2:
        monthly = df.groupby('Mes').agg(
            Sesiones=('Titulo','count'),
            Usuarios=('UserId','nunique'),
            Tokens=('Tokens','sum'),
            Costo_CLP=('Costo_CLP','sum')
        ).reset_index()
        st.download_button("⬇️ Descargar CSV (resumen mensual)", data=monthly.to_csv(index=False).encode("utf-8"),
                           file_name="ai_lliu_resumen_mensual.csv", mime="text/csv")

    st.write("---")
    st.markdown("### Vista rápida (muestra)")
    st.dataframe(
        df[['Fecha_Local','UserId','Titulo','Asignatura','jobTitle_norm','inst_clean','region','Tokens','Costo_CLP']].head(200),
        use_container_width=True
    )
