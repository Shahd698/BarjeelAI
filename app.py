import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Barjeel AI — Precision Cooling Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --sand:   #E8C97A;
    --amber:  #D4842A;
    --ember:  #B85C1C;
    --teal:   #00CED1;
    --cyan:   #7FFFD4;
    --ocean:  #006A7A;
    --ink:    #080C10;
    --slate:  #111820;
    --glass:  #161E28;
    --mist:   #1E2A38;
    --fog:    #243040;
    --text:   #D0DDE8;
    --dim:    #6A8090;
    --border: rgba(0,206,209,0.12);
    --glow:   rgba(0,206,209,0.06);
}

* { box-sizing: border-box; margin: 0; }
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text);
}
h1,h2,h3,h4,h5 { font-family: 'Playfair Display', serif !important; }

.stApp {
    background: var(--ink);
    background-image:
        radial-gradient(ellipse 80% 60% at 50% -20%, rgba(212,132,42,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(0,206,209,0.05) 0%, transparent 60%);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 0 0 48px 0 !important; max-width: 100% !important; }

/* ── Topbar ── */
.topbar {
    position: sticky; top: 0; left: 0; right: 0; z-index: 1000;
    background: rgba(8,12,16,0.97);
    backdrop-filter: blur(24px);
    border-bottom: 1px solid var(--border);
    padding: 0 48px;
    height: 64px;
    display: flex; align-items: center; justify-content: space-between;
}
.topbar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem; font-weight: 900;
    background: linear-gradient(135deg, var(--sand) 0%, var(--teal) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.topbar-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem; color: var(--dim);
    letter-spacing: 1.5px;
}

/* ── Page wrapper ── */
.page-wrap { padding: 24px 48px 64px; }

/* ── Section ── */
.section-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; color: var(--teal); letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 10px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem; font-weight: 700; color: var(--text);
    line-height: 1.1; margin-bottom: 14px;
}
.section-desc {
    font-size: 0.95rem; color: var(--dim); line-height: 1.7;
    max-width: 560px; margin-bottom: 36px;
}

/* ── Cards ── */
.card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    transition: all 0.25s;
    height: 100%;
}
.card:hover {
    border-color: rgba(0,206,209,0.3);
    background: var(--mist);
    transform: translateY(-2px);
}
.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem; color: var(--dim); letter-spacing: 2.5px;
    text-transform: uppercase; margin-bottom: 12px;
}
.card-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem; font-weight: 900; color: var(--teal);
    line-height: 1; margin-bottom: 8px;
}
.card-value.amber { color: var(--sand); }
.card-value.red   { color: #FF6B6B; }
.card-value.green { color: var(--cyan); }
.card-title  { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.card-body   { font-size: 0.82rem; color: var(--dim); line-height: 1.6; }

/* ── KPI strip ── */
.kpi-strip {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 32px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 28px;
}
.kpi-item { text-align: center; flex: 1; }
.kpi-divider { width: 1px; height: 44px; background: var(--border); }
.kpi-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 900; color: var(--teal);
}
.kpi-lbl { font-size: 0.70rem; color: var(--dim); letter-spacing: 0.4px; margin-top: 4px; }
.kpi-sub { font-size: 0.62rem; color: rgba(0,206,209,0.5); margin-top: 3px; font-family: 'Space Mono', monospace; }

/* ── AI Panel ── */
.ai-panel {
    background: linear-gradient(135deg, var(--slate) 0%, var(--glass) 100%);
    border: 1px solid rgba(0,206,209,0.2);
    border-radius: 12px;
    padding: 24px;
    position: relative; overflow: hidden;
}
.ai-mode-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 13px; border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem; letter-spacing: 1.5px; font-weight: 700;
    text-transform: uppercase; margin-bottom: 14px;
}
.badge-green { background: rgba(127,255,212,0.1); border: 1px solid rgba(127,255,212,0.3); color: var(--cyan); }
.badge-amber { background: rgba(212,132,42,0.1);  border: 1px solid rgba(212,132,42,0.3);  color: var(--sand); }
.badge-red   { background: rgba(255,107,107,0.1); border: 1px solid rgba(255,107,107,0.3); color: #FF6B6B; }
.badge-blue  { background: rgba(0,206,209,0.08);  border: 1px solid rgba(0,206,209,0.25);  color: var(--teal); }
.ai-decision { font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 10px; line-height: 1.4; }
.ai-why      { font-size: 0.83rem; color: var(--dim); line-height: 1.65; }
.ai-why strong { color: var(--text); }

/* ── Zone cells ── */
.zone-cell {
    border-radius: 8px; padding: 10px 6px; text-align: center;
}
.zone-cool { background: rgba(0,206,209,0.12); border: 1px solid rgba(0,206,209,0.2); }
.zone-warm { background: rgba(232,201,122,0.10); border: 1px solid rgba(232,201,122,0.2); }
.zone-hot  { background: rgba(184,92,28,0.15);  border: 1px solid rgba(184,92,28,0.25); }
.zone-name { font-family: 'Space Mono', monospace; font-size: 0.58rem; color: var(--dim); letter-spacing: 1px; }
.zone-val  { font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 4px 0; }
.zone-unit { font-size: 0.62rem; color: var(--dim); }

/* ── Problem vs Solution ── */
.vs-table { display: grid; grid-template-columns: 1fr auto 1fr; gap: 20px; align-items: start; }
.vs-col-bad {
    background: rgba(184,92,28,0.06); border: 1px solid rgba(184,92,28,0.2);
    border-radius: 12px; padding: 22px;
}
.vs-col-good {
    background: rgba(0,206,209,0.05); border: 1px solid rgba(0,206,209,0.2);
    border-radius: 12px; padding: 22px;
}
.vs-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem; letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid var(--border);
}
.vs-label.bad  { color: var(--amber); }
.vs-label.good { color: var(--teal); }
.vs-item { display: flex; gap: 10px; margin-bottom: 11px; font-size: 0.86rem; color: var(--dim); line-height: 1.5; }
.vs-mid  { display: flex; flex-direction: column; align-items: center; justify-content: center;
           font-family: 'Playfair Display', serif; font-size: 1.6rem; color: var(--dim); }

/* ── Impact card ── */
.impact-card {
    background: linear-gradient(135deg, rgba(0,206,209,0.08) 0%, rgba(0,106,122,0.05) 100%);
    border: 1px solid rgba(0,206,209,0.2);
    border-radius: 12px; padding: 24px; text-align: center;
}
.impact-num   { font-family: 'Playfair Display', serif; font-size: 2.6rem; font-weight: 900; color: var(--cyan); line-height: 1; }
.impact-label { font-size: 0.72rem; color: var(--dim); letter-spacing: 0.5px; margin-top: 8px; text-transform: uppercase; }

/* ── ROI summary ── */
.roi-summary { background: var(--glass); border: 1px solid var(--border); border-radius: 12px; padding: 24px; }
.roi-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 11px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.88rem;
}
.roi-row:last-child { border-bottom: none; }
.roi-key { color: var(--dim); }
.roi-val { font-weight: 600; color: var(--text); }
.roi-val.teal { color: var(--teal); }
.roi-val.sand { color: var(--sand); }
.roi-val.cyan { color: var(--cyan); }

/* ── Architecture ── */
.arch-row   { display: flex; gap: 12px; }
.arch-box   {
    flex: 1; background: var(--glass);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 18px;
}
.arch-box-label { font-family:'Space Mono',monospace; font-size:0.58rem; color:var(--teal); letter-spacing:2px; margin-bottom:4px; }
.arch-box-title { font-size:0.86rem; font-weight:600; color:var(--text); margin-bottom:4px; }
.arch-box-desc  { font-size:0.74rem; color:var(--dim); line-height:1.5; }
.arch-arrow     { text-align:center; color:var(--dim); font-size:1.1rem; padding:7px 0; }
.arch-center    {
    background: linear-gradient(135deg, rgba(0,206,209,0.10), rgba(0,106,122,0.06));
    border: 1px solid rgba(0,206,209,0.3);
    border-radius: 8px; padding: 18px 22px; text-align: center;
}
.arch-center-title { font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:700; color:var(--teal); margin-bottom:4px; }
.arch-center-desc  { font-size:0.78rem; color:var(--dim); }

/* ── Carbon banner ── */
.carbon-banner {
    background: linear-gradient(135deg, rgba(127,255,212,0.06) 0%, rgba(0,106,122,0.04) 100%);
    border: 1px solid rgba(127,255,212,0.18);
    border-radius: 12px; padding: 22px 28px;
    display: flex; gap: 40px; align-items: center; flex-wrap: wrap;
    margin-bottom: 28px;
}
.carbon-stat { text-align: center; }
.carbon-num  { font-family:'Playfair Display',serif; font-size:2rem; font-weight:900; color:#7FFFD4; }
.carbon-lbl  { font-size:0.7rem; color:var(--dim); letter-spacing:0.5px; margin-top:4px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--ocean), var(--teal)) !important;
    color: var(--ink) !important; border: none !important;
    font-weight: 700 !important; font-family: 'Outfit', sans-serif !important;
    border-radius: 6px !important; letter-spacing: 0.5px !important;
    padding: 0.55rem 1.3rem !important; transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 24px rgba(0,206,209,0.25) !important; }
[data-testid="stDownloadButton"] > button {
    background: transparent !important; border: 1px solid var(--border) !important;
    color: var(--dim) !important; font-size: 0.82rem !important;
}

/* ── Misc ── */
[data-testid="stMetric"] { background: var(--glass); border: 1px solid var(--border); border-radius: 10px; padding: 16px !important; }
[data-testid="stMetricLabel"] p { color: var(--dim) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"]   { color: var(--teal) !important; }
.stNumberInput input    { background: var(--glass) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 8px !important; }
.stSelectbox [data-baseweb="select"] { background: var(--glass) !important; border: 1px solid var(--border) !important; }
.divider { height: 1px; background: var(--border); margin: 36px 0; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.pulse-dot { width:6px; height:6px; border-radius:50%; background:var(--teal); animation:pulse 2s infinite; display:inline-block; margin-right:6px; }

/* ── Footer ── */
.footer {
    border-top: 1px solid var(--border);
    padding: 28px 48px;
    display: flex; justify-content: space-between; align-items: center;
}
.footer-left  { font-size: 0.83rem; color: var(--dim); }
.footer-right { font-family: 'Space Mono', monospace; font-size: 0.62rem; color: rgba(0,206,209,0.4); letter-spacing: 1px; }

/* ── Nav ── */
/* Hide the outer radio group label */
div[data-testid="stRadio"] > label { display: none !important; }

/* Nav container row */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stRadio"]) {
    background: rgba(8,12,16,0.97) !important;
    backdrop-filter: blur(24px) !important;
    border-bottom: 1px solid rgba(0,206,209,0.18) !important;
    position: sticky !important; top: 0 !important; z-index: 999 !important;
    padding: 10px 48px !important;
    margin: 0 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
}

/* Radio option row */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    flex-wrap: wrap !important;
    align-items: center !important;
}

/* Each pill label */
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    display: inline-flex !important;
    align-items: center !important;
    padding: 6px 18px !important;
    border-radius: 20px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    color: #9BAEBB !important;
    cursor: pointer !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.05) !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    color: #D0DDE8 !important;
    border-color: rgba(0,206,209,0.4) !important;
    background: rgba(0,206,209,0.08) !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    color: #00CED1 !important;
    border-color: rgba(0,206,209,0.7) !important;
    background: rgba(0,206,209,0.15) !important;
    box-shadow: 0 0 14px rgba(0,206,209,0.2) !important;
}

/* Hide radio circle but keep text visible */
div[data-testid="stRadio"] input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
}

/* Make sure the text span inside label is visible */
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div {
    display: flex !important;
    align-items: center !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label p {
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    margin: 0 !important;
    line-height: 1 !important;
    visibility: visible !important;
    opacity: 1 !important;
}

@media (max-width: 768px) {
    .page-wrap { padding: 100px 20px 40px; }
    .topbar { padding: 0 20px; }
    .vs-table { grid-template-columns: 1fr; }
    .vs-mid { display: none; }
    .kpi-strip { flex-wrap: wrap; gap: 14px; }
    .footer { flex-direction: column; gap: 8px; text-align: center; }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════
PLOT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit, sans-serif", color="#6A8090"),
    xaxis=dict(gridcolor="rgba(0,206,209,0.06)", zerolinecolor="rgba(0,206,209,0.06)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(0,206,209,0.06)", zerolinecolor="rgba(0,206,209,0.06)", tickfont=dict(size=11)),
    margin=dict(t=24, b=40, l=10, r=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)
TEAL  = "#00CED1"
SAND  = "#E8C97A"
CYAN  = "#7FFFD4"
EMBER = "#B85C1C"

# ═══════════════════════════════════════════════════════════
# DATA & MODEL
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    np.random.seed(42)
    days   = 90
    N      = days * 24
    hours  = np.tile(np.arange(24), days)
    day_of = np.repeat(np.arange(days), 24)

    solar   = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, 1)
    t_base  = 38 + 8 * solar - 4 * np.cos(2 * np.pi * day_of / 365)
    t_out   = np.clip(t_base + np.random.normal(0, 1.5, N), 24, 50)

    weekday  = (day_of % 7) < 5
    occ_base = np.where(
        weekday,
        np.clip(50 + 900 * np.sin(np.pi * np.clip(hours - 8, 0, 10) / 10)**2 * (hours > 7) * (hours < 20), 0, 1200),
        np.clip(20 + 100 * np.sin(np.pi * np.clip(hours - 10, 0, 6) / 6)**2 * (hours > 9) * (hours < 18), 0, 200),
    )
    occupancy  = np.clip(occ_base + np.random.normal(0, 30, N), 0, 1400).astype(int)
    humidity   = np.clip(50 + 20 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 5, N), 28, 88)
    radiation  = np.clip(1050 * solar + np.random.normal(0, 50, N), 0, 1100)
    cooling    = np.clip(
        38 + 0.9 * t_out + 0.03 * occupancy - 0.3 * humidity + 0.02 * radiation
        - 12 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 3, N),
        22, 130
    )
    return pd.DataFrame({
        "hour": hours, "day": day_of, "outdoor_temp": t_out,
        "occupancy": occupancy, "humidity": humidity,
        "solar_radiation": radiation, "cooling_load": cooling,
        "is_weekday": weekday.astype(int),
    })

@st.cache_resource
def train_model(data):
    feats = ["hour","outdoor_temp","occupancy","humidity","solar_radiation","is_weekday"]
    X, y  = data[feats], data["cooling_load"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
    m.fit(Xtr, ytr)
    yp   = m.predict(Xte)
    mae  = mean_absolute_error(yte, yp)
    rmse = float(np.sqrt(mean_squared_error(yte, yp)))
    r2   = r2_score(yte, yp)
    mape = float(np.mean(np.abs((yte - yp) / np.where(yte == 0, 1e-9, yte))) * 100)
    return m, feats, mae, rmse, r2, mape

data  = load_data()
model, features, mae, rmse, r2, mape = train_model(data)

# ── Helpers ────────────────────────────────────────────────
def predict(hour, temp, occ, hum, rad=None, weekday=1):
    if rad is None:
        rad = max(0, 1050 * np.sin(np.pi * max(hour - 6, 0) / 12)) if 6 < hour < 18 else 0
    return float(model.predict(pd.DataFrame([[hour, temp, occ, hum, rad, weekday]], columns=features))[0])

def zone_matrix(pred, rate, seed=0):
    np.random.seed(seed % 999)
    base = np.array([[1.18,1.05,1.22],[1.00,0.92,1.08],[0.88,0.80,0.95]])
    noise = np.random.uniform(0.95, 1.05, (3,3))
    mat = pred * base * noise
    opt = mat * (1 - rate * np.random.uniform(0.85, 1.1, (3,3)))
    return mat, opt

def get_ai_decision(hour, temp, occ, hum, pred):
    q75 = data["cooling_load"].quantile(0.75)
    if occ == 0:
        return dict(mode="Standby Protocol", badge="badge-blue",
            decision="Minimum airflow only — building unoccupied",
            why="Zero occupancy detected. System enters energy-minimum standby, maintaining only equipment-safety airflow and base humidity control.",
            rate=0.48, icon="🌙")
    if temp >= 46:
        return dict(mode="Extreme Heat Response", badge="badge-red",
            decision="Critical cooling active — outdoor temperature exceeds 46°C",
            why="Temperature at hazard threshold. AI prioritises occupied zones and server rooms while suspending non-critical areas.",
            rate=0.10, icon="🌡️")
    if occ >= 1100 and temp >= 38:
        return dict(mode="Peak Demand Guard", badge="badge-amber",
            decision="Redistribute airflow to high-density zones",
            why="High occupancy with extreme heat. Cooling directed proportionally to occupied areas — low-density perimeter zones throttled.",
            rate=0.15, icon="⚡")
    if hum >= 72:
        return dict(mode="Dehumidification Priority", badge="badge-amber",
            decision="Latent cooling mode — reduce humidity before sensible cooling",
            why="Elevated humidity raises perceived temperature and mold risk. AI targets dew-point reduction first.",
            rate=0.16, icon="💧")
    if pred >= q75:
        return dict(mode="Demand Response Mode", badge="badge-amber",
            decision="Active load-shedding — demand in top quartile",
            why="AI applies zone-level load shedding, deferring low-occupancy areas and pre-cooling during off-peak tariff windows.",
            rate=0.22, icon="📊")
    if hour < 6 or hour > 22:
        return dict(mode="Night Setback", badge="badge-blue",
            decision="Setpoint raised 3°C — unoccupied night mode",
            why="Low-usage hours detected. AI reduces fan speed and raises temperature setpoints for minimum-cost overnight conditioning.",
            rate=0.38, icon="🌒")
    return dict(mode="Balanced Efficiency", badge="badge-green",
        decision="Zone-matched precision cooling — optimal comfort per watt",
        why="Normal operating conditions. AI micro-adjusts airflow zone by zone, eliminating overcooling in low-density areas.",
        rate=0.26, icon="✅")

# ═══════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════
PAGES = ["Home", "Dashboard", "Digital Twin", "Solution", "ROI", "Data"]
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown("""
<div class="topbar">
    <div class="topbar-logo">Barjeel AI</div>
    <div class="topbar-tag">PRECISION COOLING · DUBAI</div>
</div>
""", unsafe_allow_html=True)

page = st.radio("nav", PAGES, index=PAGES.index(st.session_state.page), horizontal=True, label_visibility="collapsed")
st.session_state.page = page


# ═══════════════════════════════════════════════════════════════════════════
# ██  HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "Home":

    # ── Hero ─────────────────────────────────────────────
    components.html("""
<!DOCTYPE html><html><head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Space+Mono&family=Outfit:wght@300;400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box;}
body{
  background:linear-gradient(135deg,#080C10 0%,#0b1622 60%,#080C10 100%);
  font-family:'Outfit',sans-serif; color:#D0DDE8;
  min-height:100vh; padding:48px 48px 56px;
  position:relative; overflow:hidden;
}
body::before{
  content:''; position:absolute; top:-100px; left:50%; transform:translateX(-50%);
  width:900px; height:400px;
  background:radial-gradient(ellipse,rgba(212,132,42,0.08) 0%,transparent 70%);
  pointer-events:none;
}
.tower{
  position:absolute; right:6%; top:50%; transform:translateY(-50%);
  opacity:0.06; width:200px; pointer-events:none;
}
.eyebrow{
  font-family:'Space Mono',monospace; font-size:0.66rem; color:#00CED1;
  letter-spacing:3px; text-transform:uppercase; margin-bottom:22px;
  display:flex; align-items:center; gap:12px;
}
.eyebrow::before{content:''; display:block; width:40px; height:1px; background:#00CED1;}
.title{
  font-family:'Playfair Display',serif;
  font-size:clamp(3rem,7vw,6rem); font-weight:900;
  line-height:0.95; letter-spacing:-2px; margin-bottom:26px;
}
.t1{display:block; color:#E8C97A;}
.t2{display:block; color:transparent; -webkit-text-stroke:1px rgba(0,206,209,0.45);}
.subtitle{
  font-size:1rem; color:#6A8090; line-height:1.75;
  max-width:560px; margin-bottom:44px; font-weight:300;
}
.subtitle em{color:#E8C97A; font-style:normal; font-weight:500;}
.scroll{
  font-family:'Space Mono',monospace; font-size:0.62rem;
  color:#6A8090; letter-spacing:2px; display:flex; align-items:center; gap:8px;
}
.scroll-line{width:1px; height:36px; background:linear-gradient(to bottom,#6A8090,transparent);}
</style></head>
<body>
<div class="tower">
  <svg viewBox="0 0 200 500" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="70" y="120" width="60" height="360" fill="white" rx="4"/>
    <rect x="60" y="80" width="80" height="20" fill="white" rx="2"/>
    <rect x="55" y="60" width="90" height="20" fill="white" rx="2"/>
    <rect x="50" y="40" width="100" height="20" fill="white" rx="2"/>
    <rect x="45" y="20" width="110" height="20" fill="white" rx="2"/>
    <rect x="40" y="5" width="120" height="18" fill="white" rx="3"/>
    <rect x="78" y="150" width="10" height="40" fill="#080C10" rx="2"/>
    <rect x="95" y="150" width="10" height="40" fill="#080C10" rx="2"/>
    <rect x="112" y="150" width="10" height="40" fill="#080C10" rx="2"/>
    <rect x="78" y="210" width="10" height="40" fill="#080C10" rx="2"/>
    <rect x="95" y="210" width="10" height="40" fill="#080C10" rx="2"/>
    <rect x="112" y="210" width="10" height="40" fill="#080C10" rx="2"/>
    <rect x="60" y="470" width="80" height="20" fill="white" rx="2"/>
    <path d="M10 100 Q40 120 70 140" stroke="white" stroke-width="2" opacity="0.4" fill="none"/>
    <path d="M190 100 Q160 120 130 140" stroke="white" stroke-width="2" opacity="0.4" fill="none"/>
  </svg>
</div>
<div class="eyebrow">Ancient wisdom · Modern intelligence</div>
<div class="title">
  <span class="t1">Barjeel</span>
  <span class="t2">AI</span>
</div>
<p class="subtitle">
  For 1,000 years, the <em>Barjeel</em> wind tower cooled Emirati homes without electricity —
  reading wind direction, tracking heat, directing airflow. We've rebuilt that same adaptive
  intelligence for every modern building in Dubai.
</p>
<div class="scroll"><div class="scroll-line"></div>SCROLL TO EXPLORE</div>
</body></html>
""", height=540, scrolling=False)

    # ── The Problem ──────────────────────────────────────
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">The Challenge</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dubai\'s cooling problem, in numbers</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, cls, desc) in zip([c1,c2,c3,c4],[
        ("70%", "amber", "of commercial building energy goes to cooling"),
        ("30%+", "red",  "of that cooling is wasted on unoccupied zones"),
        ("45°C+", "amber","peak summer temperature driving HVAC into overdrive"),
        ("2050",  "teal", "UAE Net Zero deadline — building efficiency is central"),
    ]):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-value {cls}">{val}</div>
                <div class="card-body">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── What Barjeel AI does ─────────────────────────────
    st.markdown('<div class="section-eyebrow">Our Approach</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Three layers of intelligence</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    for col, (num, title, desc, sub) in zip([s1,s2,s3],[
        ("01", "Predictive Forecasting",
         "A Gradient Boosting model trained on UAE climate patterns — outdoor temperature, solar radiation, occupancy, humidity, and weekday behaviour — forecasts zone-level cooling demand before peaks arrive.",
         f"R² {r2:.3f} · MAPE {mape:.1f}%"),
        ("02", "Zone-Level Dispatch",
         "Cool only where heat is generated. Barjeel AI maps 9 building zones across 3 floors, directing airflow proportionally to real-time occupancy density — not fixed schedules.",
         "Up to 48% reduction in standby mode"),
        ("03", "Digital Twin Engine",
         "A live 3D building model simulates every HVAC decision before it executes. Stress-test heat waves, occupancy surges, and humidity spikes safely — before they reach the real system.",
         "Scenario planning + live monitoring"),
    ]):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">{num}</div>
                <div class="card-title">{title}</div>
                <div class="card-body">{desc}</div>
                <div style="margin-top:14px;padding-top:10px;border-top:1px solid var(--border);
                            font-family:'Space Mono',monospace;font-size:0.62rem;color:var(--teal);">
                    {sub}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# ██  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Dashboard":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:28px;">
        <div class="section-eyebrow">Live Monitoring</div>
        <div class="section-title">Building Intelligence Dashboard</div>
        <div class="section-desc">Real-time AI snapshot — cooling load, energy savings, and carbon impact at a glance.</div>
    </div>
    """, unsafe_allow_html=True)

    latest = data.iloc[-1]
    rad_l  = max(0, 1050 * np.sin(np.pi * max(float(latest.hour) - 6, 0) / 12)) if 6 < int(latest.hour) < 18 else 0
    pred   = predict(int(latest.hour), float(latest.outdoor_temp), int(latest.occupancy), float(latest.humidity), rad_l, int(latest.is_weekday))
    ai     = get_ai_decision(int(latest.hour), float(latest.outdoor_temp), int(latest.occupancy), float(latest.humidity), pred)
    rate   = ai["rate"]
    opt    = pred * (1 - rate)
    saved  = pred - opt
    co2_kg = saved * 0.45   # UAE grid: ~0.45 kg CO₂ per kWh

    # KPI strip
    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-item">
            <div class="kpi-num">{pred:.1f}</div>
            <div class="kpi-lbl">Predicted Load (kWh)</div>
            <div class="kpi-sub"><span class="pulse-dot"></span>AI FORECAST</div>
        </div>
        <div class="kpi-divider"></div>
        <div class="kpi-item">
            <div class="kpi-num" style="color:var(--cyan);">{opt:.1f}</div>
            <div class="kpi-lbl">Optimised Load (kWh)</div>
            <div class="kpi-sub">{rate*100:.0f}% REDUCTION</div>
        </div>
        <div class="kpi-divider"></div>
        <div class="kpi-item">
            <div class="kpi-num" style="color:var(--sand);">{saved:.1f}</div>
            <div class="kpi-lbl">Energy Saved (kWh)</div>
            <div class="kpi-sub">VS UNOPTIMISED</div>
        </div>
        <div class="kpi-divider"></div>
        <div class="kpi-item">
            <div class="kpi-num" style="color:#7FFFD4;">{co2_kg:.1f}</div>
            <div class="kpi-lbl">CO₂ Avoided (kg)</div>
            <div class="kpi-sub">THIS HOUR</div>
        </div>
        <div class="kpi-divider"></div>
        <div class="kpi-item">
            <div class="kpi-num" style="color:var(--sand);">{latest['outdoor_temp']:.1f}°C</div>
            <div class="kpi-lbl">Outdoor Temp</div>
            <div class="kpi-sub">LIVE SENSOR</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([5, 3])

    with left:
        # 48h load chart
        recent = data.tail(48).copy()
        preds_s = [predict(int(r.hour), float(r.outdoor_temp), int(r.occupancy), float(r.humidity), float(r.solar_radiation), int(r.is_weekday)) for _, r in recent.iterrows()]
        opt_s   = [p * (1 - get_ai_decision(int(r.hour), float(r.outdoor_temp), int(r.occupancy), float(r.humidity), p)["rate"]) for p, (_, r) in zip(preds_s, recent.iterrows())]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(48)), y=recent["cooling_load"].tolist(), name="Baseline HVAC",
            fill="tozeroy", line=dict(color=EMBER, width=1.5), fillcolor="rgba(184,92,28,0.12)"))
        fig.add_trace(go.Scatter(x=list(range(48)), y=opt_s, name="Barjeel AI",
            fill="tozeroy", line=dict(color=TEAL, width=2), fillcolor="rgba(0,206,209,0.10)"))
        fig.update_layout(height=320, **PLOT)
        fig.update_xaxes(title="Hour", tickvals=list(range(0,48,6)), ticktext=[f"-{48-i}h" for i in range(0,48,6)])
        fig.update_yaxes(title="kWh")
        st.plotly_chart(fig, use_container_width=True)

        # Hourly profile
        hourly = data.groupby("hour")[["cooling_load","occupancy"]].mean().reset_index()
        fig2   = go.Figure()
        fig2.add_trace(go.Bar(x=hourly["hour"], y=hourly["cooling_load"], name="Avg Load",
            marker_color="rgba(0,206,209,0.55)", marker_line_width=0))
        fig2.add_trace(go.Scatter(x=hourly["hour"], y=hourly["occupancy"]/10, name="Occupancy ÷10",
            yaxis="y2", line=dict(color=SAND, width=2, dash="dot")))
        fig2.update_layout(height=240, yaxis2=dict(overlaying="y", side="right", showgrid=False, gridcolor="rgba(0,0,0,0)"), **PLOT)
        fig2.update_xaxes(title="Hour of Day")
        fig2.update_yaxes(title="kWh")
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        # AI mode
        st.markdown(f"""
        <div class="ai-panel">
            <div class="ai-mode-badge {ai['badge']}">{ai['icon']} &nbsp; {ai['mode']}</div>
            <div class="ai-decision">{ai['decision']}</div>
            <div class="ai-why"><strong>Why:</strong> {ai['why']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Zone snapshot
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:var(--dim);letter-spacing:2px;margin-bottom:10px;">ZONE COOLING SNAPSHOT</div>""", unsafe_allow_html=True)
        mat, opt_mat = zone_matrix(pred, rate, seed=int(pred*10))
        zone_names   = [["F3-A","F3-B","F3-C"],["F2-A","F2-B","F2-C"],["F1-A","F1-B","F1-C"]]
        for fi in range(3):
            cols = st.columns(3)
            for zi in range(3):
                v = opt_mat[fi][zi]
                cls = "zone-cool" if v < 55 else "zone-warm" if v < 80 else "zone-hot"
                with cols[zi]:
                    st.markdown(f"""<div class="zone-cell {cls}"><div class="zone-name">{zone_names[fi][zi]}</div><div class="zone-val">{v:.0f}</div><div class="zone-unit">kW</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Environmental snapshot
        for label, val, color in [
            ("Outdoor Temp",  f"{latest['outdoor_temp']:.1f} °C",  SAND),
            ("Humidity",      f"{latest['humidity']:.0f}%",         TEAL),
            ("Occupancy",     f"{int(latest['occupancy'])} ppl",    CYAN),
            ("Solar Rad.",    f"{latest['solar_radiation']:.0f} W/m²", SAND),
            ("Hour",          f"{int(latest['hour']):02d}:00",      TEAL),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.84rem;">
                <span style="color:var(--dim);">{label}</span>
                <span style="color:{color};font-weight:600;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ██  DIGITAL TWIN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Digital Twin":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div class="section-eyebrow">Interactive Simulation</div>
        <div class="section-title">Building Digital Twin</div>
        <div class="section-desc">Tune conditions and watch the AI recalculate in real time. Stress-test heat waves before they happen.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: sim_h   = st.slider("Hour", 0, 23, 14)
    with c2: sim_t   = st.slider("Outdoor Temp (°C)", 22, 50, 42)
    with c3: sim_occ = st.slider("Occupancy", 0, 1400, 800, 50)
    with c4: sim_hum = st.slider("Humidity (%)", 25, 90, 58)
    with c5: sim_rad = st.slider("Solar (W/m²)", 0, 1100, 700, 50)

    sim_pred = predict(sim_h, sim_t, sim_occ, sim_hum, sim_rad)
    sim_ai   = get_ai_decision(sim_h, sim_t, sim_occ, sim_hum, sim_pred)
    sim_rate = sim_ai["rate"]
    sim_opt  = sim_pred * (1 - sim_rate)
    sim_saved= sim_pred - sim_opt
    sim_co2  = sim_saved * 0.45
    mat, opt_mat = zone_matrix(sim_pred, sim_rate, seed=sim_occ)

    # Summary strip
    st.markdown(f"""
    <div class="ai-panel" style="margin-bottom:24px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">
            <div style="flex:1;min-width:260px;">
                <div class="ai-mode-badge {sim_ai['badge']}">{sim_ai['icon']} &nbsp; {sim_ai['mode']}</div>
                <div class="ai-decision" style="margin-top:8px;">{sim_ai['decision']}</div>
                <div class="ai-why" style="margin-top:8px;"><strong>Why:</strong> {sim_ai['why']}</div>
            </div>
            <div style="display:flex;gap:20px;flex-wrap:wrap;">
                <div style="text-align:center;min-width:80px;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:var(--dim);letter-spacing:1.5px;">PREDICTED</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;color:var(--teal);">{sim_pred:.1f}</div>
                    <div style="font-size:0.7rem;color:var(--dim);">kWh</div>
                </div>
                <div style="text-align:center;min-width:80px;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:var(--dim);letter-spacing:1.5px;">OPTIMISED</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;color:var(--cyan);">{sim_opt:.1f}</div>
                    <div style="font-size:0.7rem;color:var(--dim);">kWh</div>
                </div>
                <div style="text-align:center;min-width:80px;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:var(--dim);letter-spacing:1.5px;">SAVING</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;color:var(--sand);">{sim_rate*100:.0f}%</div>
                    <div style="font-size:0.7rem;color:var(--dim);">{sim_saved:.1f} kWh</div>
                </div>
                <div style="text-align:center;min-width:80px;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:var(--dim);letter-spacing:1.5px;">CO₂ AVOIDED</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;color:#7FFFD4;">{sim_co2:.1f}</div>
                    <div style="font-size:0.7rem;color:var(--dim);">kg</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        fig_heat = go.Figure(go.Heatmap(
            z=opt_mat,
            x=["Zone A","Zone B","Zone C"],
            y=["Floor 3","Floor 2","Floor 1"],
            colorscale=[[0,"#0a3040"],[0.4,"#00CED1"],[0.7,"#E8C97A"],[1.0,"#B85C1C"]],
            text=opt_mat.round(1), texttemplate="%{text} kW",
            textfont=dict(size=13, color="white"),
            colorbar=dict(title=dict(text="kW", font=dict(color="#6A8090")), tickfont=dict(color="#6A8090")),
        ))
        fig_heat.update_layout(
            title=dict(text="Zone Cooling Distribution (kW)", font=dict(color=TEAL, size=13, family="Outfit")),
            height=380, **PLOT,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with t2:
        fig_3d = go.Figure()
        zone_labels  = ["Zone A","Zone B","Zone C"]
        floor_labels = ["Floor 1","Floor 2","Floor 3"]
        cmap = {"cool":"#00CED1","warm":"#E8C97A","hot":"#B85C1C"}
        for fi in range(3):
            for zi in range(3):
                v = opt_mat[fi][zi]
                col = cmap["cool"] if v < 55 else cmap["warm"] if v < 80 else cmap["hot"]
                h   = v / 115
                fig_3d.add_trace(go.Mesh3d(
                    x=[zi,zi+.85,zi+.85,zi,  zi,zi+.85,zi+.85,zi],
                    y=[fi,fi,fi+.55,fi+.55,   fi,fi,fi+.55,fi+.55],
                    z=[0,0,0,0,h,h,h,h],
                    color=col, opacity=0.80, showlegend=False,
                    name=f"{floor_labels[fi]} {zone_labels[zi]} — {v:.1f} kW"
                ))
        fig_3d.update_layout(
            title=dict(text="3D Thermal Load Model", font=dict(color=TEAL, size=13, family="Outfit")),
            scene=dict(
                xaxis=dict(title="Zone", tickvals=[0.4,1.4,2.4], ticktext=zone_labels, gridcolor="#112a45"),
                yaxis=dict(title="Floor", tickvals=[0.3,1.3,2.3], ticktext=floor_labels, gridcolor="#112a45"),
                zaxis=dict(title="Intensity", gridcolor="#112a45"),
                bgcolor="rgba(8,12,16,0)",
                camera=dict(eye=dict(x=1.8,y=1.8,z=1.3)),
            ),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#6A8090",
            height=400, margin=dict(t=30,b=10),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # Scenario buttons
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:0.6rem;color:var(--dim);letter-spacing:2px;margin-bottom:14px;">QUICK SCENARIOS</div>', unsafe_allow_html=True)
    sb1, sb2, sb3 = st.columns(3)
    with sb1:
        if st.button("🌡️ +5°C Heat Wave"):
            hp  = predict(sim_h, min(sim_t+5,50), sim_occ, sim_hum, sim_rad)
            hai = get_ai_decision(sim_h, min(sim_t+5,50), sim_occ, sim_hum, hp)
            ho  = hp * (1-hai["rate"])
            st.warning(f"Load: {hp:.1f} → {ho:.1f} kWh · Mode: {hai['mode']}")
            st.info(hai['decision'])
    with sb2:
        if st.button("👥 Full Occupancy"):
            hp  = predict(sim_h, sim_t, 1400, sim_hum, sim_rad)
            hai = get_ai_decision(sim_h, sim_t, 1400, sim_hum, hp)
            ho  = hp * (1-hai["rate"])
            st.warning(f"Load: {hp:.1f} → {ho:.1f} kWh · Mode: {hai['mode']}")
    with sb3:
        if st.button("🌙 3AM Standby"):
            hp  = predict(3, max(sim_t-5,22), 0, sim_hum, 0, 1)
            hai = get_ai_decision(3, max(sim_t-5,22), 0, sim_hum, hp)
            ho  = hp * (1-hai["rate"])
            st.success(f"Standby: {hp:.1f} → {ho:.1f} kWh · Saved: {(hp-ho):.1f} kWh overnight")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ██  SOLUTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Solution":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:28px;">
        <div class="section-eyebrow">How It Works</div>
        <div class="section-title">Problem → Solution → Architecture</div>
        <div class="section-desc">How Barjeel AI's precision cooling loop works — from sensor reading to airflow dispatch.</div>
    </div>
    """, unsafe_allow_html=True)

    # Problem vs Solution
    st.markdown("""
    <div class="vs-table">
        <div class="vs-col-bad">
            <div class="vs-label bad">⚠ Traditional HVAC</div>
            <div class="vs-item"><span style="color:var(--amber);">✗</span>&nbsp; Fixed hourly schedules — ignores real-time occupancy</div>
            <div class="vs-item"><span style="color:var(--amber);">✗</span>&nbsp; All zones cooled equally regardless of who is where</div>
            <div class="vs-item"><span style="color:var(--amber);">✗</span>&nbsp; Reactive — responds only after comfort has failed</div>
            <div class="vs-item"><span style="color:var(--amber);">✗</span>&nbsp; No integration with solar radiation or live humidity</div>
            <div class="vs-item"><span style="color:var(--amber);">✗</span>&nbsp; 30%+ energy wasted on empty corridors and offices</div>
        </div>
        <div class="vs-mid">→</div>
        <div class="vs-col-good">
            <div class="vs-label good">✦ Barjeel AI</div>
            <div class="vs-item"><span style="color:var(--teal);">✦</span>&nbsp; Predicts demand ahead — pre-cools before peaks arrive</div>
            <div class="vs-item"><span style="color:var(--teal);">✦</span>&nbsp; 9-zone precision dispatch — airflow matched to density</div>
            <div class="vs-item"><span style="color:var(--teal);">✦</span>&nbsp; 6 adaptive AI modes — from Standby to Extreme Heat</div>
            <div class="vs-item"><span style="color:var(--teal);">✦</span>&nbsp; 6-feature model including solar radiation & weekday patterns</div>
            <div class="vs-item"><span style="color:var(--teal);">✦</span>&nbsp; Up to 48% reduction in low-occupancy periods</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">The Precision Cooling Loop</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:0;">
        <div class="arch-row">
            <div class="arch-box">
                <div class="arch-box-label">INPUT 1</div>
                <div class="arch-box-title">🌡️ Climate Sensors</div>
                <div class="arch-box-desc">Outdoor temp, humidity, solar radiation (W/m²), hour of day, weekday flag</div>
            </div>
            <div class="arch-box">
                <div class="arch-box-label">INPUT 2</div>
                <div class="arch-box-title">👥 Occupancy Layer</div>
                <div class="arch-box-desc">PIR sensors, CO₂ ppm, badge access, zone-level density mapping</div>
            </div>
            <div class="arch-box">
                <div class="arch-box-label">INPUT 3</div>
                <div class="arch-box-title">🏗️ BMS Feed</div>
                <div class="arch-box-desc">Smart meters, HVAC actuators, zone damper positions, chiller COP</div>
            </div>
        </div>
        <div class="arch-arrow">↓</div>
        <div class="arch-center">
            <div class="arch-center-title">🧠 Gradient Boosting Thermal Core</div>
            <div class="arch-center-desc">300-tree ensemble · 6 features · UAE climate physics · R² {r2:.3f} · MAPE {mape:.1f}%</div>
        </div>
        <div class="arch-arrow">↓</div>
        <div class="arch-row">
            <div class="arch-box">
                <div class="arch-box-label">OUTPUT 1</div>
                <div class="arch-box-title">🎯 Zone Dispatch</div>
                <div class="arch-box-desc">9-zone airflow allocation weighted by occupancy × thermal load</div>
            </div>
            <div class="arch-box">
                <div class="arch-box-label">OUTPUT 2</div>
                <div class="arch-box-title">🌐 Digital Twin</div>
                <div class="arch-box-desc">3D live building model · what-if scenarios · heat wave simulation</div>
            </div>
            <div class="arch-box">
                <div class="arch-box-label">OUTPUT 3</div>
                <div class="arch-box-title">📉 Energy & Carbon Report</div>
                <div class="arch-box-desc">Real-time savings · CO₂ avoided per hour · monthly trend tracking</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Model Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What drives the AI?</div>', unsafe_allow_html=True)

    feat_names = {"hour":"Hour of Day","outdoor_temp":"Outdoor Temp","occupancy":"Occupancy",
                  "humidity":"Humidity","solar_radiation":"Solar Radiation","is_weekday":"Weekday"}
    imp_df = pd.DataFrame({"Feature":[feat_names.get(f,f) for f in features], "Importance":model.feature_importances_}).sort_values("Importance", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
        marker=dict(color=imp_df["Importance"], colorscale=[[0,"#0a3040"],[0.5,TEAL],[1.0,CYAN]], line_width=0),
        text=[f"{v:.3f}" for v in imp_df["Importance"]], textposition="outside",
        textfont=dict(color="#6A8090", size=11),
    ))
    fig_imp.update_layout(height=260, **PLOT)
    fig_imp.update_xaxes(title="Importance Score")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ██  ROI CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ROI":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:28px;">
        <div class="section-eyebrow">Business Case</div>
        <div class="section-title">ROI & Carbon Calculator</div>
        <div class="section-desc">Model the financial and environmental return of deploying Barjeel AI in your building.</div>
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1: ann_kwh  = st.number_input("Annual HVAC Energy (kWh)", 10_000, 10_000_000, 600_000, 10_000)
    with rc2: tariff   = st.number_input("Electricity Tariff (AED/kWh)", 0.10, 2.00, 0.38, 0.01)
    with rc3: save_pct = st.slider("AI Saving Rate (%)", 5, 40, 20)
    with rc4: building = st.selectbox("Building Type", ["Commercial Office","Hotel","Mall / Retail","Hospital","Mixed Use"])

    capex_map = {"Commercial Office":280_000,"Hotel":350_000,"Mall / Retail":420_000,"Hospital":310_000,"Mixed Use":295_000}
    capex = capex_map.get(building, 300_000)

    saved_kwh = ann_kwh * (save_pct / 100)
    saved_aed = saved_kwh * tariff
    co2_t     = saved_kwh * 0.45 / 1000        # tonnes CO₂
    ann_cost  = ann_kwh * tariff
    payback_m = (capex / saved_aed) * 12 if saved_aed > 0 else 0
    trees_eq  = int(co2_t * 45)                # 1 tree ≈ absorbs ~22 kg CO₂/yr → 45 trees/tonne

    # Carbon banner — prominent
    st.markdown(f"""
    <div class="carbon-banner">
        <div style="flex:1;min-width:180px;">
            <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#7FFFD4;letter-spacing:2px;margin-bottom:8px;">ANNUAL CARBON IMPACT</div>
            <div style="font-family:'Playfair Display',serif;font-size:0.9rem;color:var(--dim);line-height:1.6;">
                Eliminating <strong style="color:#7FFFD4;">{co2_t:.1f} tonnes</strong> of CO₂ per year is equivalent to
                planting <strong style="color:#7FFFD4;">{trees_eq:,} trees</strong> — or taking
                <strong style="color:#7FFFD4;">{int(co2_t/2.3)}</strong> cars off the road.
            </div>
        </div>
        <div class="carbon-stat"><div class="carbon-num">{co2_t:.1f}t</div><div class="carbon-lbl">CO₂ / year avoided</div></div>
        <div class="carbon-stat"><div class="carbon-num">{trees_eq:,}</div><div class="carbon-lbl">Tree equivalent</div></div>
        <div class="carbon-stat"><div class="carbon-num">{co2_t*1000/8760:.1f}kg</div><div class="carbon-lbl">CO₂ avoided per hour</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Impact row
    i1, i2, i3, i4 = st.columns(4)
    for col, (val, label) in zip([i1,i2,i3,i4],[
        (f"{saved_kwh:,.0f} kWh", "Annual Energy Saved"),
        (f"AED {saved_aed:,.0f}", "Annual Cost Savings"),
        (f"AED {capex:,}", "System Investment"),
        (f"{payback_m:.0f} months", "Estimated Payback"),
    ]):
        with col:
            st.markdown(f"""
            <div class="impact-card">
                <div class="impact-num" style="font-size:2rem;">{val}</div>
                <div class="impact-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        yrs = np.arange(0, 8, 0.1)
        cum = saved_aed * yrs - capex
        fig_roi = go.Figure()
        fig_roi.add_hline(y=0, line_color=SAND, line_dash="dot", line_width=1)
        fig_roi.add_trace(go.Scatter(
            x=yrs, y=cum, fill="tozeroy",
            line=dict(color=TEAL, width=2.5),
            fillcolor="rgba(0,206,209,0.07)",
            name="Cumulative Net Benefit (AED)",
        ))
        if payback_m > 0:
            fig_roi.add_vline(x=payback_m/12, line_color=SAND, line_dash="dash",
                annotation_text=f"  Breakeven: {payback_m:.0f} months",
                annotation_font=dict(color=SAND, size=11))
        fig_roi.update_layout(height=280, **PLOT)
        fig_roi.update_xaxes(title="Years")
        fig_roi.update_yaxes(title="Net Benefit (AED)")
        st.plotly_chart(fig_roi, use_container_width=True)

        # Carbon trajectory chart
        cum_co2 = co2_t * yrs
        fig_co2 = go.Figure()
        fig_co2.add_trace(go.Scatter(
            x=yrs, y=cum_co2, fill="tozeroy",
            line=dict(color=CYAN, width=2),
            fillcolor="rgba(127,255,212,0.07)",
            name="Cumulative CO₂ Avoided (tonnes)",
        ))
        fig_co2.update_layout(height=200, **PLOT, title=dict(text="Cumulative Carbon Avoided", font=dict(color=TEAL, size=12)))
        fig_co2.update_xaxes(title="Years")
        fig_co2.update_yaxes(title="Tonnes CO₂")
        st.plotly_chart(fig_co2, use_container_width=True)

    with right:
        st.markdown(f"""
        <div class="roi-summary">
            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:var(--teal);letter-spacing:2px;margin-bottom:14px;">SUMMARY</div>
            <div class="roi-row"><span class="roi-key">Building type</span><span class="roi-val">{building}</span></div>
            <div class="roi-row"><span class="roi-key">Current annual HVAC cost</span><span class="roi-val sand">AED {ann_cost:,.0f}</span></div>
            <div class="roi-row"><span class="roi-key">Post-Barjeel AI cost</span><span class="roi-val teal">AED {ann_cost-saved_aed:,.0f}</span></div>
            <div class="roi-row"><span class="roi-key">Annual savings</span><span class="roi-val cyan">AED {saved_aed:,.0f}</span></div>
            <div class="roi-row"><span class="roi-key">System investment</span><span class="roi-val">AED {capex:,}</span></div>
            <div class="roi-row"><span class="roi-key">Payback period</span><span class="roi-val teal">{payback_m:.0f} mo · {payback_m/12:.1f} yrs</span></div>
            <div class="roi-row"><span class="roi-key">CO₂ avoided / year</span><span class="roi-val cyan">{co2_t:.1f} tonnes</span></div>
            <div class="roi-row"><span class="roi-key">Tree equivalent / year</span><span class="roi-val cyan">{trees_eq:,} trees</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ██  DATA
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Data":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:28px;">
        <div class="section-eyebrow">Transparency</div>
        <div class="section-title">Data & Model Reports</div>
        <div class="section-desc">90-day UAE climate simulation dataset and full model validation metrics.</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, (val, label) in zip([m1,m2,m3,m4],[
        (f"{r2:.4f}", "R² Score"),
        (f"{mape:.2f}%", "MAPE"),
        (f"{mae:.2f} kWh", "MAE"),
        (f"{rmse:.2f} kWh", "RMSE"),
    ]):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:22px 16px;">
                <div class="card-value">{val}</div>
                <div class="card-label" style="margin-top:8px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    feat_names = {"hour":"Hour","outdoor_temp":"Outdoor Temp","occupancy":"Occupancy",
                  "humidity":"Humidity","solar_radiation":"Solar Radiation","is_weekday":"Weekday"}
    imp2 = pd.DataFrame({"Feature":[feat_names.get(f,f) for f in features], "Importance":model.feature_importances_}).sort_values("Importance", ascending=False)

    fi1, fi2 = st.columns(2)
    with fi1:
        fig_f = px.bar(imp2, x="Feature", y="Importance", color="Importance",
                       color_continuous_scale=[[0,"#0a3040"],[0.5,TEAL],[1.0,CYAN]])
        fig_f.update_layout(height=280, coloraxis_showscale=False, **PLOT)
        fig_f.update_xaxes(title="")
        fig_f.update_yaxes(title="Importance")
        st.plotly_chart(fig_f, use_container_width=True)

    with fi2:
        feat_sample = data.sample(500, random_state=42)
        y_true = feat_sample["cooling_load"].values
        y_pred = model.predict(feat_sample[features])
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred, y=y_true - y_pred, mode="markers",
            marker=dict(color=TEAL, size=5, opacity=0.45),
        ))
        fig_res.add_hline(y=0, line_color=SAND, line_dash="dot")
        fig_res.update_layout(height=280, **PLOT, title=dict(text="Residual Plot", font=dict(color=TEAL, size=12)))
        fig_res.update_xaxes(title="Predicted kWh")
        fig_res.update_yaxes(title="Residual")
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:var(--dim);letter-spacing:2px;margin-bottom:10px;">RECENT DATA SAMPLE</div>""", unsafe_allow_html=True)
    st.dataframe(
        data.tail(40).rename(columns=feat_names).style.background_gradient(subset=["cooling_load"], cmap="Blues"),
        use_container_width=True, hide_index=True,
    )
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇  Download Full Dataset (.csv)", csv, "barjeelai_dataset.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
    <div class="footer-left">
        <strong style="color:var(--sand);">Barjeel AI</strong> &nbsp;·&nbsp;
        Sameeha Siddiqui &amp; Shahed Marashdeh &nbsp;·&nbsp; Dubai, UAE
    </div>
    <div class="footer-right">R²: {r2:.4f} · MAPE: {mape:.2f}% · MAE: {mae:.2f} kWh · 6 FEATURES</div>
</div>
""", unsafe_allow_html=True)
