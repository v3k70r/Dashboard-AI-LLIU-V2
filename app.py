import os
import re
import glob
import json
import unicodedata
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="Dashboard AI-LLIU | MASXXI",
    page_icon="",
    layout="wide",
)

DATA_DIR = "data"
DEFAULT_USDCLP = 950.0

# Branding (AI-LLIU)
URL_LOGO = "https://i.ibb.co/YqVrvwS/Recurso-2.png"
COLOR_PRIMARIO = "#0E116A"
COLOR_ACENTO = "#CEDF74"
COLOR_TEXTO = "#1A1A1A"
COLOR_FONDO = "#F4F5F2"

PLOT_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Montserrat, system-ui, -apple-system, Segoe UI, Roboto", color=COLOR_TEXTO),
    margin=dict(l=0, r=0, t=40, b=0),
)


# =========================================================
# ESTILOS (ENCADRE + CONTRASTE TAGS EN CHROME/MAC)
# =========================================================
CSS = """
<style>
/* Base */
html, body, [class*="css"] { font-family: Montserrat, system-ui, -apple-system, Segoe UI, Roboto !important; }
.stApp { background: #F4F5F2; }

/* Evita overflow horizontal */
[data-testid="stAppViewContainer"] { overflow-x: hidden !important; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
@media (max-width: 1200px){ .block-container{ max-width: 1000px; } }
@media (max-width: 900px){ .block-container{ max-width: 92vw; } }

/* Header */
.ai-header{
  display:flex; align-items:center; gap:14px;
  margin: 4px 0 10px 0;
}
.ai-header img{ height:40px; }
.ai-header .title{
  font-size: 22px; font-weight: 800; color:#0E116A;
}
.ai-sub{ color: rgba(26,26,26,.75); margin-top:-6px; }

/* KPI grid (responsivo + anti-desborde) */
.kpi-grid{
  display:grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
  margin: 8px 0 10px 0;
}
@media (max-width: 1200px){ .kpi-grid{ grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 700px){ .kpi-grid{ grid-template-columns: repeat(1, minmax(0, 1fr)); } }

.kpi{
  background:#FFFFFF; border:1px solid rgba(14,17,106,0.10);
  border-radius:16px; padding:14px 14px;
  min-width: 0; /* clave: evita desborde dentro del grid */
}
.kpi .label{
  color: rgba(26,26,26,.70);
  font-weight:650; font-size:13px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.kpi .value{
  color:#0E116A; font-weight:900; font-size:32px; line-height:1.05;
  margin-top:2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.kpi .hint{
  color: rgba(26,26,26,.68); font-size:12px; margin-top:6px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(14,17,106,0.06) 0%, rgba(206,223,116,0.10) 100%) !important;
  border-right: 1px solid rgba(14,17,106,0.10) !important;
}
section[data-testid="stSidebar"] > div{ padding-top: 14px !important; }
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3{ color: #0E116A !important; }

/* Inputs */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.92) !important;
  border: 1px solid rgba(14,17,106,0.16) !important;
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within{
  border-color: #0E116A !important;
  box-shadow: 0 0 0 3px rgba(206,223,116,0.35) !important;
}

/* Tags: fuerza contraste (Chrome/Mac) */
section[data-testid="stSidebar"] [data-baseweb="tag"]{
  background-color: #CEDF74 !important;
  background: #CEDF74 !important;
  border: 1px solid rgba(14,17,106,0.25) !important;
  border-radius: 999px !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] *{
  color: #0E116A !important;
  fill: #0E116A !important;
}

/* Evita que plotly “se salga” */
.js-plotly-plot, .plot-container { max-width: 100% !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
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
        return f"${int(round(float(n))):,} CLP".replace(",", ".")
    except Exception:
        return "$0 CLP"


def limpiar_institucion(inst) -> str:
    if pd.isna(inst) or str(inst).strip() == "" or str(inst).lower() == "nan":
        return "NO ESPECIFICADO"
    s = "".join(c for c in unicodedata.normalize("NFD", str(inst)) if unicodedata.category(c) != "Mn")
    s = s.upper().strip()
    if "RANCAGUA" in s and ("BICENTENARIO" in s or "TECNICO" in s):
        return "LICEO BICENTENARIO TÉCNICO DE RANCAGUA"
    return s


def normalizar_cargos(serie: pd.Series) -> np.ndarray:
    s = serie.astype(str).str.lower().str.strip()
    condiciones = [
        s.str.contains(r"utp|técnico|tecnico|curricular|evaluador", regex=True),
        s.str.contains(r"director|directora|rector|rectora", regex=True),
        s.str.contains(r"profesor|docente|educador|maestro", regex=True),
        s.str.contains(r"coordinador|coordinadora", regex=True),
    ]
    return np.select(
        condiciones,
        ["Jefatura UTP", "Equipo Directivo", "Docente de Aula", "Coordinación"],
        default="Otro Profesional",
    )


def inferir_asignatura(titulo: str) -> str:
    t = str(titulo).lower()
    if re.search(r"matem[aá]tica|c[aá]lculo|geometr[ií]a|n[uú]mero|[aá]lgebra|fracci[oó]n|ecuaci[oó]n|datos", t):
        return "Matemática"
    if re.search(r"ciencia.*natural|biolog[ií]a|naturaleza|medio.*ambiente|ecosistema|c[eé]lula|universo", t):
        return "Ciencias Naturales"
    if re.search(r"lenguaje|comunicaci[oó]n|literatura|lectura|comprensi[oó]n|escritura|poes[ií]a|cuento|texto", t):
        return "Lenguaje y Comunicación"
    if re.search(r"historia|geograf[ií]a|sociales|ciudadan[ií]a|c[ií]vica|civica|pueblos originarios|grecia|roma", t):
        return "Historia y Cs. Sociales"
    if re.search(r"f[ií]sica|qu[ií]mica|termodin[aá]mica|fuerza|energ[ií]a|materia|luz", t):
        return "Física y Química"
    if re.search(r"arte|m[uú]sica|visuales|danza|pintura|escultura", t):
        return "Artes y Música"
    if re.search(r"ed.*f[ií]sica|deporte|motricidad|entrenamiento|saludable", t):
        return "Educación Física"
    if re.search(r"ingl[eé]s|idioma|english", t):
        return "Inglés"
    if re.search(r"parvularia|k[ií]nder|preb[aá]sica|transici[oó]n|p[aá]rvulo", t):
        return "Educación Parvularia"
    if re.search(r"arduino|scratch|programaci[oó]n|rob[oó]tica|software|inform[aá]tica|tecnolog[ií]a", t):
        return "Tecnología e Informática"
    return "Planificación Transversal"


def ensure_full_json(data_dir: str) -> str:
    """
    Reconstruye data/full_conversations.json desde partes .part### (<=100MB).
    Mantiene compatibilidad si las partes vienen con otro prefijo (ConversationTable...partX).
    """
    os.makedirs(data_dir, exist_ok=True)
    out_file = os.path.join(data_dir, "full_conversations.json")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return out_file

    parts = sorted(glob.glob(os.path.join(data_dir, "full_conversations.json.part*")))
    if not parts:
        parts = sorted(glob.glob(os.path.join(data_dir, "*ConversationTable*.part*")))
    if not parts:
        return out_file

    with open(out_file, "wb") as outfile:
        for part in parts:
            with open(part, "rb") as infile:
                outfile.write(infile.read())
    return out_file


def is_test_email(email: str) -> bool:
    e = str(email or "").lower().strip()
    if not re.match(r"[^@]+@[^@]+\.[^@]+", e):
        return True
    bad = ["test", "demo", "fake", "prueba", "asdf", "example", "temporal", "mailinator", "yopmail"]
    return any(b in e for b in bad)


# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data(show_spinner="Consolidando datos…")
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_json_path = ensure_full_json(DATA_DIR)

    # Cognito users
    users_path = os.path.join(DATA_DIR, "cleaned_cognito_users.csv")
    try:
        df_users = pd.read_csv(users_path)
        df_users["jobTitle_norm"] = normalizar_cargos(df_users.get("jobTitle", pd.Series([""] * len(df_users))))
        df_users["inst_clean"] = df_users.get("institution", pd.Series([""] * len(df_users))).apply(limpiar_institucion)
    except Exception:
        df_users = pd.DataFrame()

    # Conversations
    records = []
    if os.path.exists(full_json_path) and os.path.getsize(full_json_path) > 0:
        with open(full_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            sk = str(item.get("SK", ""))
            if "BOT_ALIAS" in sk:
                continue

            uuid_match = re.search(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", sk
            )
            if not uuid_match:
                continue

            total_price_usd = item.get("TotalPrice", 0)
            try:
                total_price_usd = float(total_price_usd)
            except Exception:
                total_price_usd = 0.0

            tokens = item.get("TotalTokens", None)
            try:
                tokens = int(tokens) if tokens not in (None, "", "null") else np.nan
            except Exception:
                tokens = np.nan

            create_ms = item.get("CreateTime")
            fecha_utc = pd.to_datetime(int(create_ms), unit="ms", utc=True) if create_ms else pd.NaT
            title = item.get("Title", "") or ""

            records.append(
                {
                    "UserId": uuid_match.group(0),
                    "Fecha": fecha_utc,
                    "Titulo": title,
                    "Asignatura": inferir_asignatura(title),
                    "TotalPriceUSD": total_price_usd,
                    "Tokens": tokens,
                }
            )

    df_conv = pd.DataFrame(records)

    if not df_conv.empty:
        df_conv["Fecha_Local"] = (
            pd.to_datetime(df_conv["Fecha"], utc=True, errors="coerce")
            .dt.tz_convert("America/Santiago")
            .dt.tz_localize(None)
        )
        df_conv = df_conv.dropna(subset=["Fecha_Local"]).sort_values("Fecha_Local")
        df_conv["Fecha_Dia"] = df_conv["Fecha_Local"].dt.floor("D")
        df_conv["Mes"] = df_conv["Fecha_Local"].dt.to_period("M").astype(str)

        df_conv["DiaSemana"] = df_conv["Fecha_Local"].dt.day_name().map(
            {"Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles", "Thursday": "Jueves",
             "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"}
        )
        df_conv["FranjaHoraria"] = df_conv["Fecha_Local"].dt.hour.apply(
            lambda x: "Mañana" if 5 <= x < 13 else "Tarde" if 13 <= x < 19 else "Noche"
        )

    # Master merge
    if not df_conv.empty and not df_users.empty:
        df_master = pd.merge(df_conv, df_users, left_on="UserId", right_on="sub", how="left")
        df_master["region"] = df_master.get("region", "Desconocida").fillna("Desconocida")
        df_master["inst_clean"] = df_master.get("inst_clean", "NO ESPECIFICADO")
        df_master["jobTitle_norm"] = df_master.get("jobTitle_norm", "Otro Profesional")
        df_master["email"] = df_master.get("email", np.nan)
        df_master["email_verified"] = df_master.get("email_verified", np.nan)
        df_master["Enabled"] = df_master.get("Enabled", np.nan)
    else:
        df_master = df_conv.copy()
        if df_master.empty:
            df_master = pd.DataFrame(columns=["UserId","Fecha_Local","Fecha_Dia","Mes","Titulo","Asignatura","TotalPriceUSD","Tokens"])
        df_master["region"] = df_master.get("region", "Desconocida")
        df_master["inst_clean"] = df_master.get("inst_clean", "NO ESPECIFICADO")
        df_master["jobTitle_norm"] = df_master.get("jobTitle_norm", "Otro Profesional")
        df_master["email"] = df_master.get("email", np.nan)
        df_master["email_verified"] = df_master.get("email_verified", np.nan)
        df_master["Enabled"] = df_master.get("Enabled", np.nan)

    return df_master, df_users


df_master, df_users = load_data()

# =========================================================
# ENCABEZADO
# =========================================================
st.markdown(
    f"""
<div class="ai-header">
  <img src="{URL_LOGO}" alt="AI-LLIU"/>
  <div>
    <div class="title">Dashboard AI-LLIU (histórico)</div>
    <div class="ai-sub">Indicadores de uso, adopción y costos (API + infraestructura).</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if df_master.empty:
    st.error("No se cargaron datos. Verifica /data (partes .part### + cleaned_cognito_users.csv).")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### Filtros")

    min_date = df_master["Fecha_Local"].min()
    max_date = df_master["Fecha_Local"].max()

    date_range = st.date_input(
        "Rango de fechas",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    roles = sorted(df_master["jobTitle_norm"].dropna().unique().tolist())
    asignaturas = sorted(df_master["Asignatura"].dropna().unique().tolist())
    regiones = sorted(df_master["region"].dropna().unique().tolist())
    insts = sorted(df_master["inst_clean"].dropna().unique().tolist())

    sel_roles = st.multiselect("Rol (normalizado)", roles, default=roles)
    sel_asig = st.multiselect("Asignatura (inferida desde el título)", asignaturas, default=asignaturas)
    sel_reg = st.multiselect("Región", regiones, default=regiones)

    top_inst = df_master["inst_clean"].value_counts().head(50).index.tolist()
    sel_inst_mode = st.radio("Institución", ["Todas", "Seleccionar (top 50)"], horizontal=True)
    if sel_inst_mode == "Seleccionar (top 50)":
        sel_inst = st.multiselect("Top 50 instituciones", top_inst, default=top_inst)
    else:
        sel_inst = insts

    st.markdown("---")
    st.markdown("### Parámetros de costo")
    usd_clp = st.number_input("Tipo de cambio USD→CLP", min_value=1.0, value=float(DEFAULT_USDCLP), step=10.0)
    infra_usd_mes = st.number_input("Costo infraestructura (USD/mes)", min_value=0.0, value=350.0, step=10.0)
    st.caption("Costo API estimado desde TotalPrice (USD). Infraestructura modelada como componente fijo mensual.")


start_d = pd.to_datetime(date_range[0])
end_d = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

mask = (
    (df_master["Fecha_Local"] >= start_d)
    & (df_master["Fecha_Local"] <= end_d)
    & (df_master["jobTitle_norm"].isin(sel_roles))
    & (df_master["Asignatura"].isin(sel_asig))
    & (df_master["region"].isin(sel_reg))
    & (df_master["inst_clean"].isin(sel_inst))
)
df = df_master.loc[mask].copy()

if df.empty:
    st.warning("Con estos filtros no hay registros.")
    st.stop()

df["Costo_API_CLP"] = df["TotalPriceUSD"].fillna(0.0).astype(float) * float(usd_clp)

# =========================================================
# USUARIOS (CUENTAS VS ACTIVOS VS HUMANOS)
# =========================================================
user_stats = (
    df.groupby("UserId")
    .agg(conv=("Titulo", "size"), tokens=("Tokens", "sum"), costo_api=("Costo_API_CLP", "sum"))
    .reset_index()
)

cols_possible = ["UserId", "email", "email_verified", "Enabled", "UserStatus", "jobTitle_norm", "inst_clean"]
present_cols = [c for c in cols_possible if c in df.columns]
if present_cols:
    user_attrs = df[present_cols].drop_duplicates("UserId")
    user_stats = user_stats.merge(user_attrs, on="UserId", how="left")

user_stats["humano_validado"] = True
if "email" in user_stats.columns:
    user_stats.loc[user_stats["email"].apply(is_test_email), "humano_validado"] = False
if "email_verified" in user_stats.columns:
    user_stats.loc[user_stats["email_verified"].astype(str).str.lower() != "true", "humano_validado"] = False

# Umbral mínimo de actividad (reduce cuentas de prueba / intentos)
user_stats.loc[(user_stats["conv"] < 2) & ((user_stats["tokens"].fillna(0)) < 800), "humano_validado"] = False

# Anti-automatización simple (IDs con volumen extremo)
user_stats.loc[user_stats["conv"] > 60, "humano_validado"] = False

usuarios_activos = int(df["UserId"].nunique())
usuarios_humanos = int(user_stats.loc[user_stats["humano_validado"], "UserId"].nunique())

# =========================================================
# COSTOS MENSUALES
# =========================================================
monthly = (
    df.groupby("Mes")
    .agg(
        interacciones=("Titulo", "size"),
        usuarios_activos=("UserId", "nunique"),
        costo_api=("Costo_API_CLP", "sum"),
        tokens=("Tokens", "sum"),
    )
    .reset_index()
    .sort_values("Mes")
)
monthly["costo_infra"] = float(infra_usd_mes) * float(usd_clp)
monthly["costo_total"] = monthly["costo_api"] + monthly["costo_infra"]

hum_month = (
    df.merge(user_stats[["UserId", "humano_validado"]], on="UserId", how="left")
    .loc[lambda x: x["humano_validado"] == True]
    .groupby("Mes")["UserId"]
    .nunique()
    .reset_index()
    .rename(columns={"UserId": "usuarios_humanos"})
)
monthly = monthly.merge(hum_month, on="Mes", how="left")
monthly["usuarios_humanos"] = monthly["usuarios_humanos"].fillna(0).astype(int)

monthly["costo_por_humano"] = monthly["costo_total"] / monthly["usuarios_humanos"].replace(0, np.nan)
monthly["costo_por_humano"] = monthly["costo_por_humano"].fillna(0.0)

costo_api_total = float(df["Costo_API_CLP"].sum())
meses_en_rango = int(monthly["Mes"].nunique())
costo_infra_total = float(infra_usd_mes) * float(usd_clp) * max(1, meses_en_rango)
costo_total_rango = costo_api_total + costo_infra_total

prom_mensual_total = costo_total_rango / max(1, meses_en_rango)
prom_mensual_api = costo_api_total / max(1, meses_en_rango)
costo_por_humano_rango = costo_total_rango / max(1, usuarios_humanos)


# =========================================================
# KPIs (TÉCNICOS + ENCUADRE SEGURO)
# =========================================================
st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="label">Usuarios humanos validados</div>
    <div class="value">{fmt_compact(usuarios_humanos)}</div>
    <div class="hint">Depuración por calidad de cuenta y umbral de actividad</div>
  </div>
  <div class="kpi">
    <div class="label">Usuarios activos (IDs únicos)</div>
    <div class="value">{fmt_compact(usuarios_activos)}</div>
    <div class="hint">Actividad dentro del rango filtrado</div>
  </div>
  <div class="kpi">
    <div class="label">Interacciones (conversaciones)</div>
    <div class="value">{fmt_compact(len(df))}</div>
    <div class="hint">Registros del datadump en el rango</div>
  </div>
  <div class="kpi">
    <div class="label">Costo mensual promedio (total)</div>
    <div class="value">{fmt_clp(prom_mensual_total)}</div>
    <div class="hint">API (variable) + infraestructura (fijo)</div>
  </div>
  <div class="kpi">
    <div class="label">Costo mensual promedio (API)</div>
    <div class="value">{fmt_clp(prom_mensual_api)}</div>
    <div class="hint">Estimación desde TotalPrice (USD)</div>
  </div>
  <div class="kpi">
    <div class="label">Costo por usuario humano (período)</div>
    <div class="value">{fmt_clp(costo_por_humano_rango)}</div>
    <div class="hint">Costo total del período / usuarios humanos</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# COSTOS
# =========================================================
st.markdown("### Costos mensuales (API + infraestructura)")

c1, c2 = st.columns([1.25, 1.0], gap="large")
with c1:
    fig = go.Figure()
    fig.add_bar(x=monthly["Mes"], y=monthly["costo_api"], name="Costo API (CLP)")
    fig.add_bar(x=monthly["Mes"], y=monthly["costo_infra"], name="Costo infraestructura (CLP)")
    fig.update_layout(barmode="stack", **PLOT_LAYOUT, legend=dict(orientation="h", y=1.15))
    fig.update_yaxes(title="CLP")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = px.line(monthly, x="Mes", y="costo_total", markers=True, title="Costo total mensual (CLP)")
    fig2.update_layout(**PLOT_LAYOUT)
    fig2.update_yaxes(title="CLP")
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns([1.0, 1.0], gap="large")
with c3:
    fig3 = px.bar(monthly, x="Mes", y="costo_por_humano", title="Costo mensual por usuario humano (CLP)")
    fig3.update_layout(**PLOT_LAYOUT)
    fig3.update_yaxes(title="CLP / usuario humano")
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    fig4 = px.line(monthly, x="Mes", y="usuarios_humanos", markers=True, title="Usuarios humanos por mes")
    fig4.update_layout(**PLOT_LAYOUT)
    fig4.update_yaxes(title="Usuarios")
    st.plotly_chart(fig4, use_container_width=True)

# =========================================================
# USO Y PATRONES
# =========================================================
st.markdown("### Uso y patrones temporales")

u1, u2 = st.columns([1.0, 1.0], gap="large")
with u1:
    daily = df.groupby("Fecha_Dia").agg(interacciones=("Titulo", "size"), usuarios=("UserId", "nunique")).reset_index()
    figu = px.line(
        daily,
        x="Fecha_Dia",
        y=["interacciones", "usuarios"],
        title="Evolución diaria (interacciones y usuarios)",
    )
    figu.update_layout(**PLOT_LAYOUT, legend=dict(orientation="h", y=1.15))
    st.plotly_chart(figu, use_container_width=True)

with u2:
    heat = df.groupby(["DiaSemana", "FranjaHoraria"]).size().reset_index(name="conteo")
    orden_dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    orden_franjas = ["Mañana", "Tarde", "Noche"]
    heat["DiaSemana"] = pd.Categorical(heat["DiaSemana"], categories=orden_dias, ordered=True)
    heat["FranjaHoraria"] = pd.Categorical(heat["FranjaHoraria"], categories=orden_franjas, ordered=True)
    pivot = heat.pivot_table(index="DiaSemana", columns="FranjaHoraria", values="conteo", fill_value=0).loc[orden_dias, orden_franjas]
    fig_h = px.imshow(pivot, text_auto=True, aspect="auto", title="Intensidad de uso (día vs franja horaria)")
    fig_h.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig_h, use_container_width=True)

# =========================================================
# TABLA RESUMEN
# =========================================================
st.markdown("### Resumen mensual")

table = monthly.copy()
for c in ["costo_api", "costo_infra", "costo_total", "costo_por_humano"]:
    table[c] = table[c].round(0)

table = table.rename(
    columns={
        "Mes": "Mes",
        "interacciones": "Interacciones",
        "usuarios_activos": "Usuarios activos",
        "usuarios_humanos": "Usuarios humanos",
        "costo_api": "Costo API (CLP)",
        "costo_infra": "Costo infraestructura (CLP)",
        "costo_total": "Costo total (CLP)",
        "costo_por_humano": "CLP por usuario humano",
    }
)
st.dataframe(table, use_container_width=True)

st.caption(
    "Definiciones: 'Costo API' se estima desde TotalPrice (USD) convertido a CLP. "
    "'Costo infraestructura' corresponde a un componente fijo mensual parametrizable. "
    "'Usuarios humanos validados' aplica reglas de depuración (emails de prueba, verificación y umbrales de actividad)."
)