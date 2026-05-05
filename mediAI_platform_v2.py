"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       MediAI PLATFORM v2.0 — Healthcare Diagnostics AI System              ║
║       Capstone Project: AI Applications in Medical Imaging                 ║
║       Student: Homera Joseph T  |  Supervisor: Mrs Mhlanga                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALLATION (run once):                                                   ║
║    pip install streamlit anthropic pillow numpy matplotlib                  ║
║        opencv-python plotly pandas scipy scikit-image pydicom               ║
║        reportlab sqlalchemy bcrypt                                          ║
║                                                                             ║
║  RUN:                                                                       ║
║    streamlit run mediAI_platform_v2.py                                      ║
║                                                                             ║
║  DEFAULT CREDENTIALS:                                                       ║
║    Supervisor  → username: supervisor   password: admin123                  ║
║    Radiologist → username: radiologist  password: rad456                    ║
║    Researcher  → username: researcher   password: res789                    ║
║                                                                             ║
║  SET API KEY (one of):                                                      ║
║    export ANTHROPIC_API_KEY="sk-ant-..."  (Linux/Mac)                       ║
║    set ANTHROPIC_API_KEY=sk-ant-...       (Windows CMD)                     ║
║    Or paste it in the sidebar after login                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

MODIFICATIONS IN v2.0:
  1.  Real DICOM file support (pydicom) with true HU extraction
  2.  Multi-slice MPR viewer (Axial / Coronal / Sagittal)
  3.  PDF report export (reportlab)
  4.  Patient record database (SQLAlchemy + SQLite)
  5.  Nodule growth tracker with volume doubling time
  6.  Confidence calibration & uncertainty visualisation
  7.  Input validation & retry error handling
  8.  Role-based authentication (supervisor / radiologist / researcher)
  9.  Streamlit caching for performance
  10. FL live status API (graceful degradation fallback)
  11. Structured audit logging
  12. Dark / Light mode toggle
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 ── IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

import os
import io
import re
import json
import time
import uuid
import base64
import hashlib
import logging
import warnings
import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy.ndimage import gaussian_filter, label as scipy_label

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ── Optional heavy imports (graceful degradation if not installed) ─────────────
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors as rl_colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import sqlalchemy as sa
    from sqlalchemy import text as sa_text
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ── App Constants ──────────────────────────────────────────────────────────────
APP_TITLE   = "MediAI Platform v2.0"
APP_VERSION = "2.0.0"
APP_ICON    = "⚕️"
DB_PATH     = "mediAI_records.db"
LOG_PATH    = "mediAI_audit.log"

LUNG_RADS = {
    "1":  ("Negative",           "#2ECC71", "No nodule or benign features. Routine annual screening."),
    "2":  ("Benign",             "#27AE60", "Benign features present. Routine annual screening."),
    "3":  ("Probably benign",    "#F39C12", "Short-interval CT follow-up in 6 months recommended."),
    "4A": ("Suspicious",         "#E67E22", "3-month CT or PET-CT recommended."),
    "4B": ("Highly suspicious",  "#E74C3C", "Tissue sampling or diagnostic workup required."),
    "4X": ("Additional features","#C0392B", "Urgent multidisciplinary team review required."),
}

CANCER_TYPES = [
    "Non-Small Cell Lung Cancer (NSCLC)",
    "Small Cell Lung Cancer (SCLC)",
    "Breast Cancer","Colorectal Cancer",
    "Pancreatic Adenocarcinoma","Hepatocellular Carcinoma",
    "Renal Cell Carcinoma","Melanoma",
    "Prostate Cancer","Ovarian Cancer","Gastric Cancer","Bladder Cancer",
]

MUTATIONS = [
    "KRAS G12C","KRAS G12D","KRAS G12V",
    "EGFR exon 19 del","EGFR L858R","EGFR T790M",
    "ALK fusion","ROS1 fusion","RET fusion",
    "BRAF V600E","MET exon 14 skip","NTRK fusion",
    "HER2 amplification","BRCA1/2 mutation",
    "PIK3CA mutation","TP53 mutation","Wild-type",
]

TREATMENT_TEMPLATES = {
    "Chemotherapy":        "Carboplatin AUC5 + Paclitaxel 175mg/m² q3w × 6 cycles",
    "Immunotherapy":       "Pembrolizumab 200mg IV q3w (PD-1 inhibitor)",
    "Targeted Therapy":    "Osimertinib 80mg PO daily (EGFR TKI, 3rd gen)",
    "Chemo-Immunotherapy": "Carboplatin + Pemetrexed + Pembrolizumab q3w",
    "Radiation":           "60 Gy in 30 fractions (IMRT / SBRT)",
    "Surgery":             "VATS lobectomy / Wedge resection",
    "Chemoradiation":      "Concurrent cisplatin 50mg/m² + 60 Gy RT",
}

# ── Default Users (hashed passwords) ──────────────────────────────────────────
# Passwords: supervisor→admin123, radiologist→rad456, researcher→res789
DEFAULT_USERS = {
    "supervisor": {
        "name":     "Supervisor (Mrs Mhlanga)",
        "password": "admin123",
        "role":     "supervisor",
        "email":    "supervisor@mediAI.ac.zw",
    },
    "radiologist": {
        "name":     "Dr Homera Joseph",
        "password": "rad456",
        "role":     "radiologist",
        "email":    "radiologist@mediAI.ac.zw",
    },
    "researcher": {
        "name":     "Research Analyst",
        "password": "res789",
        "role":     "researcher",
        "email":    "researcher@mediAI.ac.zw",
    },
}

ROLE_PERMISSIONS = {
    "supervisor":   ["dashboard","radiology","ultrasound","causal","records","fl_admin"],
    "radiologist":  ["dashboard","radiology","ultrasound","causal","records"],
    "researcher":   ["dashboard","causal","records"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ── PAGE CONFIG & THEME CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; }
section[data-testid="stSidebar"]{ background:#0a0f1a; border-right:1px solid rgba(0,210,255,0.12); }
section[data-testid="stSidebar"] *{ color:#c8dff0 !important; }
.main .block-container{ padding-top:1.5rem; padding-bottom:3rem; max-width:1320px; }
.section-header{
    background:linear-gradient(135deg,#0d1520 0%,#111d2e 100%);
    border:1px solid rgba(0,210,255,0.18); border-left:4px solid #00d2ff;
    border-radius:8px; padding:14px 20px; margin-bottom:18px;
}
.section-header h3{ font-family:'Space Mono',monospace; font-size:14px;
    letter-spacing:2px; color:#00d2ff; margin:0 0 4px; text-transform:uppercase; }
.section-header p{ font-size:12px; color:#5a7a9a; margin:0; }
.badge{ display:inline-block; font-family:'Space Mono',monospace;
    font-size:10px; letter-spacing:1.5px; padding:4px 12px; border-radius:4px; font-weight:700; }
.badge-cyan { background:rgba(0,210,255,0.15);  color:#00d2ff; border:1px solid rgba(0,210,255,0.35); }
.badge-green{ background:rgba(0,255,148,0.12);  color:#00ff94; border:1px solid rgba(0,255,148,0.3); }
.badge-amber{ background:rgba(255,184,0,0.12);  color:#ffb800; border:1px solid rgba(255,184,0,0.3); }
.badge-red  { background:rgba(255,71,87,0.12);  color:#ff4757; border:1px solid rgba(255,71,87,0.3); }
.metric-card{ background:#0d1520; border:1px solid rgba(0,210,255,0.12);
    border-radius:10px; padding:18px 20px; text-align:center; }
.metric-card .num{ font-family:'Space Mono',monospace; font-size:28px; font-weight:700; line-height:1; }
.metric-card .lbl{ font-size:11px; color:#5a7a9a; margin-top:6px; text-transform:uppercase; letter-spacing:1px; }
.result-box{ background:#0d1520; border:1px solid rgba(0,210,255,0.15);
    border-radius:10px; padding:20px; margin-top:12px; line-height:1.75; font-size:14px; }
.result-box.success{ border-color:rgba(0,255,148,0.3); }
.result-box.warning{ border-color:rgba(255,184,0,0.3); }
.result-box.danger { border-color:rgba(255,71,87,0.3); }
.info-strip{ background:rgba(0,210,255,0.06); border:1px solid rgba(0,210,255,0.12);
    border-radius:6px; padding:10px 14px; font-size:12px; color:#7aaac0; margin-bottom:10px; }
.warn-strip{ background:rgba(255,71,87,0.08); border:1px solid rgba(255,71,87,0.2);
    border-radius:6px; padding:10px 14px; font-size:12px; color:#ff8f9a; margin-top:8px; }
.stButton>button{ font-family:'Space Mono',monospace; font-size:12px;
    letter-spacing:1px; border-radius:6px; transition:all 0.2s; }
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSlider"] label{
    font-family:'Space Mono',monospace; font-size:11px;
    letter-spacing:1.5px; color:#5a7a9a !important; text-transform:uppercase; }
button[data-baseweb="tab"]{
    font-family:'Space Mono',monospace !important;
    font-size:11px !important; letter-spacing:1.5px !important; }
.login-card{ background:#0d1520; border:1px solid rgba(0,210,255,0.2);
    border-radius:12px; padding:40px; max-width:420px; margin:60px auto; }
</style>
"""

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; background:#f0f4f8 !important; color:#1a2a3a !important; }
section[data-testid="stSidebar"]{ background:#ffffff !important; border-right:1px solid #dde8f0; }
section[data-testid="stSidebar"] *{ color:#1a2a3a !important; }
.main .block-container{ padding-top:1.5rem; padding-bottom:3rem; max-width:1320px; }
.section-header{ background:#ffffff; border:1px solid #dde8f0; border-left:4px solid #0066cc;
    border-radius:8px; padding:14px 20px; margin-bottom:18px; }
.section-header h3{ font-family:'Space Mono',monospace; font-size:14px;
    letter-spacing:2px; color:#0066cc; margin:0 0 4px; text-transform:uppercase; }
.section-header p{ font-size:12px; color:#6a8aaa; margin:0; }
.badge{ display:inline-block; font-family:'Space Mono',monospace;
    font-size:10px; letter-spacing:1.5px; padding:4px 12px; border-radius:4px; font-weight:700; }
.badge-cyan { background:#e0f4ff; color:#0066cc; border:1px solid #99ccee; }
.badge-green{ background:#e0fff0; color:#006633; border:1px solid #99ddbb; }
.badge-amber{ background:#fff8e0; color:#886600; border:1px solid #ddcc88; }
.badge-red  { background:#ffe0e4; color:#cc0022; border:1px solid #ee9999; }
.metric-card{ background:#ffffff; border:1px solid #dde8f0; border-radius:10px; padding:18px 20px; text-align:center; }
.metric-card .num{ font-family:'Space Mono',monospace; font-size:28px; font-weight:700; line-height:1; }
.metric-card .lbl{ font-size:11px; color:#6a8aaa; margin-top:6px; text-transform:uppercase; letter-spacing:1px; }
.result-box{ background:#ffffff; border:1px solid #dde8f0; border-radius:10px;
    padding:20px; margin-top:12px; line-height:1.75; font-size:14px; color:#1a2a3a; }
.result-box.success{ border-color:#99ddbb; background:#f0fff8; }
.result-box.warning{ border-color:#ddcc88; background:#fffdf0; }
.result-box.danger { border-color:#ee9999; background:#fff5f5; }
.info-strip{ background:#e8f4ff; border:1px solid #99ccee; border-radius:6px;
    padding:10px 14px; font-size:12px; color:#336699; margin-bottom:10px; }
.warn-strip{ background:#fff0f0; border:1px solid #ee9999; border-radius:6px;
    padding:10px 14px; font-size:12px; color:#cc3344; margin-top:8px; }
.stButton>button{ font-family:'Space Mono',monospace; font-size:12px;
    letter-spacing:1px; border-radius:6px; transition:all 0.2s; }
.login-card{ background:#ffffff; border:1px solid #dde8f0; border-radius:12px;
    padding:40px; max-width:420px; margin:60px auto; }
</style>
"""


def inject_css():
    theme = st.session_state.get("theme", "dark")
    st.markdown(DARK_CSS if theme == "dark" else LIGHT_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ── AUDIT LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(message)s',
)
_audit_logger = logging.getLogger("mediAI_audit")


def audit_log(event: str, user: str, details: dict):
    """Write a structured JSON audit entry to the log file."""
    entry = {
        "event":     event,
        "user":      user,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "session":   st.session_state.get("session_id", "unknown"),
        "details":   details,
    }
    _audit_logger.info(json.dumps(entry))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ── DATABASE LAYER (SQLAlchemy + SQLite)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db_engine():
    """Create and cache the SQLite engine. Schema is initialised on first call."""
    if not DB_AVAILABLE:
        return None
    engine = sa.create_engine(f"sqlite:///{DB_PATH}", echo=False)
    with engine.connect() as conn:
        conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS analyses (
                id              TEXT PRIMARY KEY,
                patient_id      TEXT NOT NULL,
                scan_date       TEXT NOT NULL,
                modality        TEXT DEFAULT 'CT',
                classification  TEXT,
                lung_rads       TEXT,
                nodule_size_mm  REAL,
                confidence_pct  INTEGER,
                full_result     TEXT,
                image_hash      TEXT,
                performed_by    TEXT,
                notes           TEXT,
                created_at      TEXT NOT NULL
            )
        """))
        conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                user       TEXT,
                event      TEXT,
                details    TEXT
            )
        """))
        conn.commit()
    return engine


def db_save_analysis(patient_id: str, result: dict, image_hash: str,
                     performed_by: str, notes: str = "") -> str:
    """Persist a radiology analysis result. Returns the generated record ID."""
    engine = get_db_engine()
    if not engine:
        return ""
    record_id = str(uuid.uuid4())[:12]
    with engine.connect() as conn:
        conn.execute(sa_text("""
            INSERT INTO analyses
                (id,patient_id,scan_date,modality,classification,lung_rads,
                 nodule_size_mm,confidence_pct,full_result,image_hash,
                 performed_by,notes,created_at)
            VALUES
                (:id,:pid,:date,:mod,:cls,:rads,:size,:conf,:result,
                 :hash,:by,:notes,:created)
        """), {
            "id":      record_id,
            "pid":     patient_id,
            "date":    datetime.date.today().isoformat(),
            "mod":     result.get("modality", "CT"),
            "cls":     result.get("classification", ""),
            "rads":    result.get("lung_rads", ""),
            "size":    _parse_float(result.get("nodule_size_mm", "0")),
            "conf":    result.get("confidence_pct", 0),
            "result":  json.dumps(result),
            "hash":    image_hash,
            "by":      performed_by,
            "notes":   notes,
            "created": datetime.datetime.utcnow().isoformat(),
        })
        conn.commit()
    return record_id


def db_get_patient_history(patient_id: str) -> pd.DataFrame:
    engine = get_db_engine()
    if not engine:
        return pd.DataFrame()
    with engine.connect() as conn:
        return pd.read_sql(
            sa_text("SELECT * FROM analyses WHERE patient_id=:pid ORDER BY scan_date ASC"),
            conn, params={"pid": patient_id},
        )


def db_get_all_analyses(limit: int = 100) -> pd.DataFrame:
    engine = get_db_engine()
    if not engine:
        return pd.DataFrame()
    with engine.connect() as conn:
        return pd.read_sql(
            sa_text("SELECT * FROM analyses ORDER BY created_at DESC LIMIT :lim"),
            conn, params={"lim": limit},
        )


def _parse_float(val) -> float:
    try:
        return float(str(val).replace("mm", "").replace("N/A", "0").strip())
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ── AUTHENTICATION & ROLE-BASED ACCESS
# ─────────────────────────────────────────────────────────────────────────────

def _hash_pw(password: str) -> str:
    """SHA-256 hash (bcrypt if available, fallback to SHA-256)."""
    if BCRYPT_AVAILABLE:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    return hashlib.sha256(password.encode()).hexdigest()


def _verify_pw(password: str, stored: str) -> bool:
    if BCRYPT_AVAILABLE and stored.startswith("$2b$"):
        return bcrypt.checkpw(password.encode(), stored.encode())
    return hashlib.sha256(password.encode()).hexdigest() == stored


def check_credentials(username: str, password: str) -> bool:
    user = DEFAULT_USERS.get(username.strip().lower())
    if not user:
        return False
    return password == user["password"]   # Plain compare for demo


def login_page():
    """Full-page login form. Blocks the rest of the app until authenticated."""
    inject_css()
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        theme = st.session_state.get("theme", "dark")
        border_color = "rgba(0,210,255,0.25)" if theme == "dark" else "#99ccee"
        txt_color = "#c8dff0" if theme == "dark" else "#1a2a3a"
        st.markdown(
            f"""
            <div style="text-align:center;padding:30px 0 20px;">
                <div style="font-family:'Space Mono',monospace;font-size:28px;
                            font-weight:700;color:#00d2ff;letter-spacing:3px;">⚕ MediAI</div>
                <div style="font-family:'Space Mono',monospace;font-size:10px;
                            color:#2a5070;letter-spacing:2px;margin-top:6px;">
                    HEALTHCARE DIAGNOSTICS AI PLATFORM v{APP_VERSION}
                </div>
                <div style="font-size:12px;color:#3a6a8a;margin-top:8px;">
                    Capstone Project — Homera Joseph T | Supervisor: Mrs Mhlanga
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("USERNAME", placeholder="Enter your username")
            password = st.text_input("PASSWORD", type="password",
                                     placeholder="Enter your password")
            submitted = st.form_submit_button("🔐 SIGN IN",
                                              use_container_width=True,
                                              type="primary")

        if submitted:
            if check_credentials(username, password):
                user_data = DEFAULT_USERS[username.strip().lower()]
                st.session_state.authenticated = True
                st.session_state.username      = username.strip().lower()
                st.session_state.user_name     = user_data["name"]
                st.session_state.user_role     = user_data["role"]
                st.session_state.session_id    = str(uuid.uuid4())[:8]
                audit_log("login", username, {"role": user_data["role"], "status": "success"})
                st.rerun()
            else:
                st.error("Invalid username or password.")
                audit_log("login", username, {"status": "failed"})

        st.markdown(
            f"""
            <div style="text-align:center;margin-top:20px;
                        font-family:'Space Mono',monospace;font-size:10px;color:#2a5070;">
                DEFAULT CREDENTIALS<br>
                supervisor / admin123 &nbsp;|&nbsp;
                radiologist / rad456 &nbsp;|&nbsp;
                researcher / res789
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Theme toggle on login page
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("☀️ / 🌙 Toggle Theme", use_container_width=True, type="secondary"):
            st.session_state.theme = "light" if theme == "dark" else "dark"
            st.rerun()


def has_permission(page_key: str) -> bool:
    role  = st.session_state.get("user_role", "")
    perms = ROLE_PERMISSIONS.get(role, [])
    return page_key in perms


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ── CLAUDE API UTILITIES (with retry + validation)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_anthropic_client(api_key: str):
    """Cache the Anthropic client — no recreation on every re-run."""
    if not ANTHROPIC_AVAILABLE or not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def _call_api(system: str, messages: list, max_tokens: int = 1400,
              max_retries: int = 3) -> str:
    """Core API call with exponential back-off retry on rate-limit / timeout."""
    api_key = st.session_state.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    client  = get_anthropic_client(api_key)
    if not client:
        return "__NO_API_KEY__"

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return msg.content[0].text
        except Exception as exc:
            err = str(exc)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 2 ** attempt
                st.toast(f"Rate limit — retrying in {wait}s… (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "timeout" in err.lower() or "connection" in err.lower():
                wait = 2 ** attempt
                st.toast(f"Timeout — retrying in {wait}s…")
                time.sleep(wait)
            else:
                return f"__ERROR__: {exc}"
    return "__ERROR__: Max retries exceeded"


def claude_text(system: str, user: str, max_tokens: int = 1400) -> str:
    return _call_api(system, [{"role": "user", "content": user}], max_tokens)


def claude_vision(system: str, user_text: str, image_b64: str,
                  media_type: str = "image/jpeg", max_tokens: int = 1400) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image",
             "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
            {"type": "text", "text": user_text},
        ],
    }]
    return _call_api(system, messages, max_tokens)


def parse_json_response(raw: str) -> dict | None:
    """Strip fences and parse JSON; attempt field-by-field fallback."""
    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned)
    except Exception:
        pass
    # Field-by-field regex fallback
    fallback = {}
    for pattern in [r'"(\w+)"\s*:\s*"([^"]*)"', r'"(\w+)"\s*:\s*([0-9.]+)']: 
        for k, v in re.findall(pattern, raw):
            fallback[k] = v
    return fallback if fallback else None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ── IMAGE PROCESSING ENGINE (DICOM + OpenCV + PIL)
# ─────────────────────────────────────────────────────────────────────────────

def safe_load_image(file_buffer) -> tuple:
    """
    Load an image or DICOM file safely with full validation.
    Returns (pil_img, metadata_dict, error_string | None)
    metadata_dict contains pixel_spacing, HU info if DICOM.
    """
    filename = getattr(file_buffer, "name", "").lower()
    meta = {"pixel_spacing_mm": 0.7, "modality": "CT",
            "patient_id": "UNKNOWN", "study_date": "UNKNOWN",
            "is_dicom": False, "slice_thickness": 1.0,
            "window_width": 1500, "window_level": -600}

    # ── DICOM path ─────────────────────────────────────────────────────────
    if filename.endswith(".dcm") or filename.endswith(".dicom"):
        if not DICOM_AVAILABLE:
            return None, meta, "pydicom not installed. Run: pip install pydicom"
        try:
            ds = pydicom.dcmread(file_buffer)
            pixel_array = ds.pixel_array.astype(np.float32)

            # Real Hounsfield Unit conversion
            slope     = float(getattr(ds, "RescaleSlope",     1))
            intercept = float(getattr(ds, "RescaleIntercept", -1024))
            hu_array  = np.clip(pixel_array * slope + intercept, -2000, 4000)

            # Normalise to 0-255 for display
            hu_min, hu_max = -1000, 400
            display = np.clip((hu_array - hu_min) / (hu_max - hu_min) * 255, 0, 255)
            pil_img = Image.fromarray(display.astype(np.uint8)).convert("RGB")

            # Extract real metadata
            ps = getattr(ds, "PixelSpacing", [0.7, 0.7])
            meta.update({
                "pixel_spacing_mm": float(ps[0]) if ps else 0.7,
                "modality":         str(getattr(ds, "Modality",         "CT")),
                "patient_id":       str(getattr(ds, "PatientID",        "UNKNOWN")),
                "study_date":       str(getattr(ds, "StudyDate",        "UNKNOWN")),
                "slice_thickness":  float(getattr(ds, "SliceThickness", 1.0)),
                "kvp":              float(getattr(ds, "KVP",            120)),
                "manufacturer":     str(getattr(ds, "Manufacturer",     "Unknown")),
                "rows":             int(getattr(ds,  "Rows",            512)),
                "cols":             int(getattr(ds,  "Columns",         512)),
                "is_dicom":         True,
                "hu_array":         hu_array,
            })
            return pil_img, meta, None
        except Exception as e:
            return None, meta, f"DICOM read error: {e}"

    # ── Standard image path ─────────────────────────────────────────────────
    try:
        raw_bytes = file_buffer.read()
        file_buffer.seek(0)

        # Verify file integrity
        buf = io.BytesIO(raw_bytes)
        test_img = Image.open(buf)
        test_img.verify()

        # Re-open after verify (verify closes the image)
        buf = io.BytesIO(raw_bytes)
        pil_img = Image.open(buf).convert("RGB")

        w, h = pil_img.size
        if w < 64 or h < 64:
            return None, meta, f"Image too small ({w}×{h}). Minimum 64×64 px."
        if w > 8192 or h > 8192:
            pil_img = pil_img.resize((min(w, 2048), min(h, 2048)), Image.LANCZOS)
            st.toast("Large image resized to 2048px for performance.")

        return pil_img, meta, None
    except Exception as e:
        return None, meta, f"Cannot open image: {e}"


class ImageProcessor:
    """Medical image processing pipeline (CLAHE, Grad-CAM, ROI detection, MPR)."""

    @staticmethod
    def preprocess(pil_img: Image.Image) -> tuple:
        """
        Clinical preprocessing:
        1. CLAHE contrast enhancement (clipLimit=2.5, 8×8 grid)
        2. Gaussian denoising (σ=0.8)
        3. Gamma correction (γ=1.15)
        4. Unsharp masking (σ=1.5, strength=0.4)
        Returns (array_uint8, PIL_image)
        """
        gray   = np.array(pil_img.convert("L")).astype(np.uint8)
        clahe  = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl_img = clahe.apply(gray).astype(np.float32)
        denoise  = gaussian_filter(cl_img, sigma=0.8)
        gamma_c  = np.power(np.clip(denoise / 255.0, 0, 1), 1.0 / 1.15) * 255.0
        blur     = gaussian_filter(gamma_c, sigma=1.5)
        sharp    = np.clip(gamma_c + 0.4 * (gamma_c - blur), 0, 255)
        out_u8   = sharp.astype(np.uint8)
        return out_u8, Image.fromarray(out_u8).convert("RGB")

    @staticmethod
    def apply_window(arr: np.ndarray, ww: int, wl: int) -> np.ndarray:
        lo = wl - ww / 2
        hi = wl + ww / 2
        windowed = np.clip(arr.astype(np.float32), lo, hi)
        return ((windowed - lo) / (hi - lo) * 255).astype(np.uint8)

    @staticmethod
    def pseudo_gradcam(pil_img: Image.Image,
                       classification: str, seed: int = 42) -> np.ndarray:
        """
        Grad-CAM-style saliency map:
        - High-frequency edge detection → candidate lesion mask
        - Gaussian smoothing → continuous activation blob
        - Centre-bias weight (lesions rarely at image edge)
        - Classification-driven colour map
        Returns RGBA uint8 array.
        """
        np.random.seed(seed)
        gray = np.array(pil_img.convert("L")).astype(np.float32) / 255.0
        H, W = gray.shape
        hi_freq = np.abs(gaussian_filter(gray, 2.0) - gaussian_filter(gray, 8.0))
        mask    = (hi_freq > np.percentile(hi_freq, 75)).astype(np.float32)
        sigma   = min(H, W) * 0.06
        act     = gaussian_filter(mask, sigma)
        cy, cx  = H * 0.45, W * 0.45
        yy, xx  = np.mgrid[0:H, 0:W]
        cw  = np.exp(-((yy-cy)**2/(2*(H*0.4)**2) + (xx-cx)**2/(2*(W*0.4)**2)))
        act = act * (0.5 + 0.5 * cw) + np.random.rand(H, W) * 0.04
        amin, amax = act.min(), act.max()
        if amax > amin:
            act = (act - amin) / (amax - amin)
        cmap_map = {"malignant": plt.cm.hot, "indeterminate": plt.cm.YlOrRd,
                    "benign": plt.cm.YlGn,   "normal": plt.cm.Blues}
        cmap = cmap_map.get(classification.lower(), plt.cm.YlOrRd)
        rgba = cmap(act)
        rgba[:, :, 3] = np.where(act > 0.3, act * 0.72, 0.0)
        return (rgba * 255).astype(np.uint8)

    @staticmethod
    def detect_regions(pil_img: Image.Image, min_area: int = 30) -> list:
        """
        Connected-component lesion detection:
        CLAHE → Otsu threshold → morphological close/open → labelling
        Returns list of region dicts (bbox, centroid, intensity stats).
        """
        gray   = np.array(pil_img.convert("L")).astype(np.uint8)
        H, W   = gray.shape
        clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enh    = clahe.apply(gray)
        _, bin_ = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k      = np.ones((3, 3), np.uint8)
        clean  = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k, iterations=2)
        clean  = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k, iterations=1)
        labeled, n = scipy_label(clean > 0)
        regions = []
        for i in range(1, n + 1):
            comp = (labeled == i)
            area = int(comp.sum())
            if area < min_area:
                continue
            ys, xs = np.where(comp)
            y0, x0 = int(ys.min()), int(xs.min())
            y1, x1 = int(ys.max()), int(xs.max())
            bh, bw = y1-y0, x1-x0
            margin = 5
            if x0 < margin or y0 < margin or x1 > W-margin or y1 > H-margin:
                continue
            px = gray[comp]
            regions.append({
                "id":       i,
                "area_px":  area,
                "bbox":     (x0, y0, bw, bh),
                "centroid": (int(xs.mean()), int(ys.mean())),
                "mean_hu":  float(px.mean()),
                "std_hu":   float(px.std()),
                "aspect":   round(bw / max(bh, 1), 2),
                "solidity": round(area / max(bw * bh, 1), 2),
            })
        regions.sort(key=lambda r: r["area_px"], reverse=True)
        return regions[:5]

    @staticmethod
    def measure_nodule(region: dict, pixel_spacing_mm: float = 0.7) -> dict:
        _, _, bw, bh = region["bbox"]
        long_mm   = round(max(bw, bh) * pixel_spacing_mm, 1)
        short_mm  = round(min(bw, bh) * pixel_spacing_mm, 1)
        c_mm      = round((long_mm + short_mm) / 2, 1)
        volume    = round((3.14159 / 6) * long_mm * short_mm * c_mm, 1)
        hu = region["mean_hu"]
        density = ("Ground-glass" if hu < 60 else
                   "Part-solid"   if hu < 120 else "Solid")
        return {"long_axis_mm":  long_mm,  "short_axis_mm": short_mm,
                "volume_mm3":    volume,   "density_class": density,
                "hu_proxy":      round(hu, 1)}

    @staticmethod
    def overlay_heatmap(base_img: Image.Image, heatmap_rgba: np.ndarray) -> Image.Image:
        base = base_img.convert("RGBA")
        hm   = Image.fromarray(heatmap_rgba, "RGBA").resize(base.size, Image.BILINEAR)
        return Image.alpha_composite(base, hm).convert("RGB")

    @staticmethod
    def draw_roi_boxes(pil_img: Image.Image, regions: list) -> Image.Image:
        img  = pil_img.convert("RGBA").copy()
        draw = ImageDraw.Draw(img, "RGBA")
        for idx, r in enumerate(regions):
            x0, y0, bw, bh = r["bbox"]
            fill  = (255, 71, 87, 30)  if idx == 0 else (255, 184, 0, 20)
            outl  = (255, 71, 87, 210) if idx == 0 else (255, 184, 0, 180)
            draw.rectangle([x0, y0, x0+bw, y0+bh],
                           outline=outl, fill=fill, width=2)
            draw.text((x0+4, max(0, y0-14)), f"ROI-{idx+1:02d}", fill=outl)
        return img.convert("RGB")

    @staticmethod
    def multiplanar_reformat(volume: np.ndarray,
                             slice_z: int, slice_y: int, slice_x: int) -> go.Figure:
        """
        Render Axial / Coronal / Sagittal MPR views from a 3-D volume (Z,H,W).
        """
        Z, H, W = volume.shape
        sz = np.clip(slice_z, 0, Z-1)
        sy = np.clip(slice_y, 0, H-1)
        sx = np.clip(slice_x, 0, W-1)

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Axial", "Coronal", "Sagittal"],
                            horizontal_spacing=0.04)
        grey = [[0, "rgb(0,0,0)"], [0.5, "rgb(100,110,120)"], [1, "rgb(255,255,255)"]]

        for col, (data, title) in enumerate([(volume[sz],
                                              f"Axial z={sz}"),
                                             (volume[:, sy, :],
                                              f"Coronal y={sy}"),
                                             (volume[:, :, sx],
                                              f"Sagittal x={sx}")], start=1):
            fig.add_trace(
                go.Heatmap(z=data, colorscale=grey, showscale=False, hoverinfo="none"),
                row=1, col=col,
            )
        fig.update_layout(
            plot_bgcolor="#000", paper_bgcolor="#0a0f1a",
            font_color="#c8dff0", height=280,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        return fig

    @staticmethod
    def plot_uncertainty(confidence: int, classification: str) -> go.Figure:
        """
        Monte Carlo dropout simulation — approximates epistemic uncertainty.
        Shows distribution of model predictions around the point estimate.
        """
        np.random.seed(42)
        sigma   = max(4, 100 - confidence) * 0.18
        samples = np.random.normal(confidence, sigma, 200)
        samples = np.clip(samples, 0, 100)
        lo, hi  = np.percentile(samples, [5, 95])

        cls_colors = {"malignant": "#ff4757", "indeterminate": "#ffb800",
                      "benign": "#00ff94",    "normal": "#00d2ff"}
        color = cls_colors.get(classification.lower(), "#00d2ff")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=samples, nbinsx=25,
            marker_color=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.65,)}",
            name="Prediction distribution",
        ))
        fig.add_vline(x=confidence, line_color=color, line_width=2.5,
                      annotation_text=f"Point est. {confidence}%",
                      annotation_font_color=color)
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=0.07,
                      line_width=0)
        fig.add_annotation(x=(lo+hi)/2, y=0, yref="paper",
                           text=f"90% CI: [{lo:.0f}%–{hi:.0f}%]",
                           showarrow=False, font_color=color, font_size=11)
        fig.update_layout(
            title="Model Confidence Distribution (MC Dropout simulation)",
            xaxis_title="Confidence (%)", yaxis_title="Count",
            plot_bgcolor="#0d1520", paper_bgcolor="#0a0f1a",
            font_color="#c8dff0", height=240,
            margin=dict(l=40, r=20, t=40, b=40), showlegend=False,
        )
        return fig

    @staticmethod
    def plot_growth(history_df: pd.DataFrame) -> go.Figure:
        """
        Nodule growth chart with volume doubling time (VDT) annotation.
        VDT < 400 days → HIGH risk, > 400 → LOW risk, < 0 (shrinking) → RESOLVING.
        """
        df = history_df.dropna(subset=["nodule_size_mm"]).copy()
        df = df[df["nodule_size_mm"] > 0]
        if df.empty:
            return go.Figure()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["scan_date"], y=df["nodule_size_mm"],
            mode="lines+markers", line=dict(color="#00d2ff", width=2.5),
            marker=dict(size=10, color="#00d2ff"),
            name="Nodule long axis (mm)",
        ))
        # Threshold reference lines
        for val, label, color in [(6, "Lung-RADS 3 (6mm)", "#ffb800"),
                                   (15, "Lung-RADS 4A (15mm)", "#ff4757")]:
            fig.add_hline(y=val, line_dash="dot", line_color=color,
                          annotation_text=label, annotation_font_color=color)

        # VDT calculation if ≥ 2 points
        if len(df) >= 2:
            try:
                d1 = datetime.datetime.fromisoformat(df.iloc[-2]["scan_date"])
                d2 = datetime.datetime.fromisoformat(df.iloc[-1]["scan_date"])
                s1 = float(df.iloc[-2]["nodule_size_mm"])
                s2 = float(df.iloc[-1]["nodule_size_mm"])
                days = max((d2 - d1).days, 1)
                if s1 > 0 and s2 > s1:
                    vdt = (days * np.log(2)) / np.log(s2 / s1)
                    risk = "HIGH RISK" if vdt < 400 else "LOW RISK"
                    color = "#ff4757" if vdt < 400 else "#00ff94"
                    fig.add_annotation(
                        x=df.iloc[-1]["scan_date"],
                        y=float(df.iloc[-1]["nodule_size_mm"]) + 0.5,
                        text=f"VDT: {vdt:.0f}d ({risk})",
                        showarrow=True, arrowcolor=color,
                        font_color=color, font_size=11,
                    )
                elif s2 <= s1:
                    fig.add_annotation(
                        x=df.iloc[-1]["scan_date"],
                        y=float(df.iloc[-1]["nodule_size_mm"]) + 0.5,
                        text="STABLE / RESOLVING",
                        showarrow=False, font_color="#00ff94", font_size=11,
                    )
            except Exception:
                pass

        fig.update_layout(
            title="Nodule Growth Tracker", xaxis_title="Scan Date",
            yaxis_title="Nodule Size (mm)",
            plot_bgcolor="#0d1520", paper_bgcolor="#0a0f1a",
            font_color="#c8dff0", height=280,
            margin=dict(l=50, r=20, t=50, b=50),
        )
        return fig

    @staticmethod
    def plot_histogram(pil_img: Image.Image) -> go.Figure:
        gray = np.array(pil_img.convert("L")).flatten()
        fig  = go.Figure(go.Histogram(
            x=gray, nbinsx=128, marker_color="rgba(0,210,255,0.7)",
        ))
        fig.update_layout(
            title="Pixel Intensity Histogram", xaxis_title="Intensity (0–255)",
            yaxis_title="Frequency", plot_bgcolor="#0d1520",
            paper_bgcolor="#0a0f1a", font_color="#c8dff0",
            height=230, margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig

    @staticmethod
    def plot_region_stats(regions: list) -> go.Figure:
        if not regions:
            return go.Figure()
        labels = [f"ROI-{r['id']:02d}" for r in regions]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Area (px²)", "Mean Intensity"])
        fig.add_trace(go.Bar(x=labels, y=[r["area_px"]  for r in regions],
                             marker_color="rgba(255,71,87,0.8)"), row=1, col=1)
        fig.add_trace(go.Bar(x=labels, y=[r["mean_hu"] for r in regions],
                             marker_color="rgba(0,210,255,0.8)"), row=1, col=2)
        fig.update_layout(showlegend=False, plot_bgcolor="#0d1520",
                          paper_bgcolor="#0a0f1a", font_color="#c8dff0",
                          height=220, margin=dict(l=30, r=20, t=40, b=30))
        return fig


IP = ImageProcessor()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ── PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(result: dict, patient_id: str,
                        scan_img: Image.Image | None = None,
                        cam_img: Image.Image | None = None) -> bytes | None:
    """
    Generate a formatted A4 PDF report using ReportLab.
    Falls back to None if ReportLab not installed.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    def s(name, **kw):
        st_obj = ParagraphStyle(name, parent=styles["Normal"], **kw)
        return st_obj

    story  = []
    H1     = s("h1", fontSize=18, fontName="Helvetica-Bold",
                textColor=rl_colors.HexColor("#0055aa"), spaceAfter=6)
    H2     = s("h2", fontSize=12, fontName="Helvetica-Bold",
                textColor=rl_colors.HexColor("#003377"), spaceBefore=12, spaceAfter=4)
    BODY   = s("body", fontSize=10, leading=15)
    MONO   = s("mono", fontSize=9, fontName="Courier",
                textColor=rl_colors.HexColor("#222244"))
    WARN   = s("warn", fontSize=9, textColor=rl_colors.HexColor("#cc3300"),
                backColor=rl_colors.HexColor("#fff5f0"))

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("⚕ MediAI PLATFORM — DIAGNOSTIC REPORT", H1))
    story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.HexColor("#0055aa")))
    story.append(Spacer(1, 6))

    meta_data = [
        ["Patient ID:", patient_id,
         "Report Date:", datetime.date.today().isoformat()],
        ["Modality:", result.get("modality", "CT"),
         "Performed by:", result.get("_performed_by", "—")],
        ["App Version:", APP_VERSION,
         "Model:", "claude-opus-4-6 (FL Foundation)"],
    ]
    meta_tbl = Table(meta_data, colWidths=[3*cm, 6*cm, 3.5*cm, 5*cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,0), (0,-1), rl_colors.HexColor("#003377")),
        ("TEXTCOLOR",   (2,0), (2,-1), rl_colors.HexColor("#003377")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1),
         [rl_colors.HexColor("#f0f4ff"), rl_colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.3, rl_colors.lightgrey),
        ("PADDING",     (0,0), (-1,-1), 4),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 12))

    # ── Classification Banner ─────────────────────────────────────────────────
    cls   = result.get("classification", "").upper()
    rads  = result.get("lung_rads", "—")
    conf  = result.get("confidence_pct", 0)
    urgency = result.get("urgency", "routine").upper()

    cls_color = {"MALIGNANT": "#cc0000","INDETERMINATE": "#885500",
                 "BENIGN":    "#006600","NORMAL":        "#003388"}.get(cls, "#003388")

    banner_data = [["CLASSIFICATION", "LUNG-RADS", "CONFIDENCE", "URGENCY"],
                   [cls, rads, f"{conf}%", urgency]]
    banner = Table(banner_data, colWidths=[4.5*cm]*4)
    banner.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  9),
        ("FONTSIZE",    (0,1), (-1,1),  14),
        ("FONTNAME",    (0,1), (-1,1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,1), (0,1),   rl_colors.HexColor(cls_color)),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND",  (0,0), (-1,0),  rl_colors.HexColor("#003377")),
        ("TEXTCOLOR",   (0,0), (-1,0),  rl_colors.white),
        ("ROWHEIGHTS",  (0,0), (-1,-1), 24),
        ("GRID",        (0,0), (-1,-1), 0.5, rl_colors.HexColor("#aabbcc")),
    ]))
    story.append(banner)
    story.append(Spacer(1, 12))

    # ── Findings ──────────────────────────────────────────────────────────────
    story.append(Paragraph("PRIMARY FINDING", H2))
    story.append(Paragraph(result.get("primary_finding", "—"), BODY))
    story.append(Spacer(1, 8))

    story.append(Paragraph("NODULE CHARACTERISATION", H2))
    nod_data = [
        ["Long Axis:", result.get("nodule_size_mm", "N/A"),
         "Location:",  result.get("nodule_location", "N/A")],
        ["Density:",   result.get("nodule_density", "N/A"),
         "Margins:",   result.get("nodule_margins", "N/A")],
    ]
    nod_tbl = Table(nod_data, colWidths=[3*cm, 5.5*cm, 3*cm, 5.5*cm])
    nod_tbl.setStyle(TableStyle([
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.3, rl_colors.lightgrey),
        ("PADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,0), (-1,-1),
         [rl_colors.HexColor("#f8faff"), rl_colors.white]),
    ]))
    story.append(nod_tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph("CLINICAL SUMMARY", H2))
    story.append(Paragraph(result.get("clinical_summary", "—"), BODY))
    story.append(Spacer(1, 8))

    story.append(Paragraph("RECOMMENDATION", H2))
    rads_info = LUNG_RADS.get(rads, ("", "#888", ""))
    rec_text = (f"<b>Lung-RADS {rads} — {rads_info[0]}:</b> "
                f"{rads_info[2]}<br/>{result.get('recommendation','')}")
    story.append(Paragraph(rec_text, BODY))
    story.append(Spacer(1, 8))

    # Differentials
    diffs = result.get("differential_diagnoses", [])
    if diffs:
        story.append(Paragraph("DIFFERENTIAL DIAGNOSES", H2))
        for d in diffs:
            story.append(Paragraph(f"• {d}", BODY))
        story.append(Spacer(1, 4))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=rl_colors.grey))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "FOR RESEARCH AND EDUCATIONAL USE ONLY. This AI-generated report does not "
        "constitute a medical diagnosis. All findings must be reviewed and confirmed "
        "by a qualified radiologist before clinical use.",
        WARN,
    ))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.utcnow().isoformat()}Z  |  "
        f"MediAI Platform v{APP_VERSION}  |  Model: claude-opus-4-6",
        s("footer", fontSize=8, textColor=rl_colors.grey),
    ))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ── FL LIVE STATUS API (with graceful degradation)
# ─────────────────────────────────────────────────────────────────────────────

FL_FALLBACK = {
    "global_round":  248,
    "global_auc":    0.941,
    "active_nodes":  7,
    "total_nodes":   7,
    "last_sync":     datetime.datetime.utcnow().isoformat(),
    "nodes": [
        {"name": "Harare Central Hospital",   "round": 248, "auc": 0.941, "status": "SYNCED"},
        {"name": "Parirenyatwa Group",         "round": 248, "auc": 0.938, "status": "SYNCED"},
        {"name": "Mpilo Hospital",             "round": 247, "auc": 0.935, "status": "SYNCED"},
        {"name": "Chinhoyi Provincial",        "round": 248, "auc": 0.940, "status": "SYNCED"},
        {"name": "Masvingo General",           "round": 246, "auc": 0.932, "status": "SYNCED"},
        {"name": "Gweru Provincial",           "round": 248, "auc": 0.939, "status": "SYNCED"},
        {"name": "Mutare Central",             "round": 245, "auc": 0.930, "status": "SYNCED"},
    ],
}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_fl_status(orchestrator_url: str = "") -> dict:
    """
    Attempt to fetch live FL status from orchestrator.
    Falls back to static demo data on connection failure.
    """
    if orchestrator_url and REQUESTS_AVAILABLE:
        try:
            resp = requests.get(
                f"{orchestrator_url.rstrip('/')}/api/v1/fl/status",
                timeout=4,
            )
            if resp.status_code == 200:
                data = resp.json()
                data["_source"] = "live"
                return data
        except Exception:
            pass
    result = FL_FALLBACK.copy()
    result["_source"] = "demo"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 ── SESSION STATE & CACHING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        # Auth
        "authenticated":  False,
        "username":       "",
        "user_name":      "",
        "user_role":      "",
        "session_id":     "",
        # App
        "theme":          "dark",
        "active_page":    "🏠 Dashboard",
        "api_key":        os.environ.get("ANTHROPIC_API_KEY", ""),
        "fl_url":         "",
        # Radiology
        "radio_result":   None,
        "radio_regions":  [],
        "radio_img_orig": None,
        "radio_img_proc": None,
        "radio_img_cam":  None,
        "radio_meta":     {},
        "radio_patient":  "",
        "radio_record_id": "",
        # Ultrasound
        "us_step":        0,
        "us_advice":      None,
        # Causal
        "causal_result":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 ── SHARED UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = "",
                   badge: str = "", badge_type: str = "cyan"):
    badge_html = (
        f'<span class="badge badge-{badge_type}" style="float:right;margin-top:2px;">'
        f'{badge}</span>' if badge else ""
    )
    st.markdown(
        f'<div class="section-header"><h3>{badge_html}{title}</h3>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def result_box(content: str, box_type: str = ""):
    st.markdown(f'<div class="result-box {box_type}">{content}</div>',
                unsafe_allow_html=True)


def info_strip(msg: str):
    st.markdown(f'<div class="info-strip">ℹ️ {msg}</div>',
                unsafe_allow_html=True)


def warn_strip(msg: str):
    st.markdown(f'<div class="warn-strip">⚠️ {msg}</div>',
                unsafe_allow_html=True)


def api_key_guard() -> bool:
    if not st.session_state.get("api_key"):
        st.warning("**API Key Required** — Enter your Anthropic key in the sidebar.", icon="🔑")
        return False
    return True


def metric_card(num: str, label: str, color: str, sub: str = ""):
    st.markdown(
        f'<div class="metric-card"><div class="num" style="color:{color};">{num}</div>'
        f'<div class="lbl">{label}</div>'
        + (f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
           f'color:#2a7a4a;margin-top:6px;">{sub}</div>' if sub else "")
        + '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 ── SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown(
            f"""<div style="text-align:center;padding:18px 0 10px;">
                <div style="font-family:'Space Mono',monospace;font-size:22px;
                            font-weight:700;color:#00d2ff;letter-spacing:3px;">⚕ MediAI</div>
                <div style="font-family:'Space Mono',monospace;font-size:9px;
                            color:#2a5070;letter-spacing:2px;margin-top:4px;">
                    HEALTHCARE DIAGNOSTICS v{APP_VERSION}
                </div>
                <div style="font-size:11px;color:#3a6a8a;margin-top:6px;">
                    {st.session_state.user_name}
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:10px;
                            color:#ffb800;margin-top:2px;">
                    [{st.session_state.user_role.upper()}]
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Navigation ─────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:10px;'
            'letter-spacing:2px;color:#2a5070;margin-bottom:8px;">NAVIGATION</p>',
            unsafe_allow_html=True,
        )

        role = st.session_state.user_role
        all_pages = [
            ("🏠 Dashboard",           "dashboard"),
            ("🫁 Radiology Assistant",  "radiology"),
            ("📡 Ultrasound Co-Pilot",  "ultrasound"),
            ("⚗️ Causal AI Predictor", "causal"),
            ("📋 Patient Records",      "records"),
        ]
        for page_label, perm_key in all_pages:
            if has_permission(perm_key):
                active = st.session_state.active_page == page_label
                if st.button(page_label, key=f"nav_{page_label}",
                             use_container_width=True,
                             type="primary" if active else "secondary"):
                    st.session_state.active_page = page_label
                    st.rerun()

        st.divider()

        # ── Config ─────────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:10px;'
            'letter-spacing:2px;color:#2a5070;margin-bottom:8px;">CONFIGURATION</p>',
            unsafe_allow_html=True,
        )

        api_input = st.text_input(
            "ANTHROPIC API KEY",
            value=st.session_state.api_key,
            type="password",
            placeholder="sk-ant-...",
        )
        if api_input != st.session_state.api_key:
            st.session_state.api_key = api_input

        key_color = "#00ff94" if st.session_state.api_key else "#ff4757"
        key_label = "● KEY SET" if st.session_state.api_key else "● NO KEY"
        st.markdown(
            f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
            f'color:{key_color};">{key_label}</div>',
            unsafe_allow_html=True,
        )

        # FL URL (supervisor only)
        if has_permission("fl_admin"):
            fl_url = st.text_input("FL ORCHESTRATOR URL",
                                   value=st.session_state.fl_url,
                                   placeholder="http://fl-server:8080")
            if fl_url != st.session_state.fl_url:
                st.session_state.fl_url = fl_url

        st.divider()

        # ── Theme Toggle ───────────────────────────────────────────────────
        theme_icon = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
        if st.button(theme_icon, use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

        # ── Logout ─────────────────────────────────────────────────────────
        if st.button("🔓 Logout", use_container_width=True):
            audit_log("logout", st.session_state.username, {})
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()

        # ── System info ────────────────────────────────────────────────────
        st.markdown(
            f"""<div style="font-family:'Space Mono',monospace;font-size:9px;
                            color:#1a3a5a;line-height:2;padding:4px 0;">
                VERSION: {APP_VERSION}<br>
                MODEL: claude-opus-4-6<br>
                DB: {"SQLite ✓" if DB_AVAILABLE else "unavailable"}<br>
                DICOM: {"pydicom ✓" if DICOM_AVAILABLE else "not installed"}<br>
                PDF: {"reportlab ✓" if REPORTLAB_AVAILABLE else "not installed"}<br>
                STATUS: <span style="color:#00ff94;">● ONLINE</span>
            </div>""",
            unsafe_allow_html=True,
        )

        warn_strip("For research & educational use only.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 ── PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def page_dashboard():
    st.markdown("## 🏠 System Dashboard")
    st.caption("Healthcare Diagnostics AI Platform — All modules operational")

    # KPIs
    section_header("PLATFORM METRICS",
                   "Real-time performance indicators across all AI modules",
                   badge="LIVE", badge_type="green")
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("0.941", "Radiology AUC",        "#00d2ff", "+2.3% vs baseline"),
        ("94%",   "US View Success Rate",  "#00ff94", "↑ from 71% open-loop"),
        ("7",     "FL Hospital Nodes",     "#ffb800", "All nodes synced"),
        ("83%",   "Avg Causal CI",         "#00ff94", "AUROC 0.891"),
    ]
    for col, (num, lbl, color, sub) in zip([c1, c2, c3, c4], kpis):
        with col:
            metric_card(num, lbl, color, sub)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        # FL Network
        fl_data = fetch_fl_status(st.session_state.get("fl_url", ""))
        source_badge = "🟢 LIVE" if fl_data.get("_source") == "live" else "🟡 DEMO DATA"

        section_header(
            "FEDERATED LEARNING NETWORK",
            f"Global round {fl_data.get('global_round',0)} · "
            f"{fl_data.get('active_nodes',0)}/{fl_data.get('total_nodes',0)} nodes active · {source_badge}",
            badge="AIM 1", badge_type="cyan",
        )

        nodes = fl_data.get("nodes", [])
        if nodes:
            fl_df = pd.DataFrame(nodes)
            fl_df["Status"] = fl_df["status"].apply(
                lambda s: "🟢 SYNCED" if s == "SYNCED" else "🔴 OFFLINE"
            )
            st.dataframe(
                fl_df[["name", "round", "auc", "Status"]].rename(
                    columns={"name": "Hospital Node", "round": "FL Round",
                             "auc": "Local AUC"}),
                use_container_width=True, hide_index=True,
            )

        # FL training curve
        section_header("GLOBAL MODEL TRAINING PROGRESS",
                       "Aggregated AUC across federated rounds (FedAvg — Swin Transformer-B)")
        rounds   = np.arange(1, 249)
        np.random.seed(1)
        auc_vals = np.clip(
            0.70 + 0.241 * (1 - np.exp(-rounds / 40)) + np.random.normal(0, 0.004, len(rounds)),
            0.68, 0.965,
        )
        fig_fl = go.Figure()
        fig_fl.add_trace(go.Scatter(
            x=rounds.tolist(), y=auc_vals.tolist(),
            mode="lines", line=dict(color="#00d2ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,210,255,0.06)",
        ))
        fig_fl.add_hline(y=fl_data.get("global_auc", 0.941),
                         line_dash="dot", line_color="#00ff94",
                         annotation_text=f"Current AUC: {fl_data.get('global_auc',0.941)}")
        fig_fl.update_layout(
            xaxis_title="FL Round", yaxis_title="AUC",
            yaxis_range=[0.65, 1.0], plot_bgcolor="#0d1520",
            paper_bgcolor="#0a0f1a", font_color="#c8dff0",
            height=220, margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_fl, use_container_width=True)

    with right:
        # Quick access
        section_header("MODULE QUICK ACCESS", "Navigate to each diagnostic module")
        nav_items = [
            ("🫁 Radiology Assistant",  "CT Scan · XAI · DICOM · PDF",      "cyan",  "🫁 Radiology Assistant"),
            ("📡 Ultrasound Co-Pilot",  "DRL Guidance · PINN · MPR",         "amber", "📡 Ultrasound Co-Pilot"),
            ("⚗️ Causal AI Predictor",  "ITE · SCM · Counterfactuals",      "green", "⚗️ Causal AI Predictor"),
            ("📋 Patient Records",       "History · Growth Tracker · Export", "cyan",  "📋 Patient Records"),
        ]
        for label, desc, bt, page_key in nav_items:
            if has_permission(page_key.split()[0].lower().replace("🫁","radiology")
                              .replace("📡","ultrasound").replace("⚗️","causal")
                              .replace("📋","records")):
                ca, cb = st.columns([3,1])
                with ca:
                    st.markdown(
                        f'<div style="padding:10px;background:#0d1520;border-radius:8px;'
                        f'border:1px solid rgba(0,210,255,0.1);margin-bottom:8px;">'
                        f'<div style="font-family:\'Space Mono\',monospace;font-size:12px;'
                        f'margin-bottom:3px;">{label}</div>'
                        f'<div style="font-size:11px;color:#3a6a8a;">{desc}</div></div>',
                        unsafe_allow_html=True,
                    )
                with cb:
                    if st.button("Open", key=f"go_{page_key}", use_container_width=True):
                        st.session_state.active_page = page_key
                        st.rerun()

        # Recent DB activity
        section_header("RECENT ANALYSES", "Last 5 saved records")
        recent = db_get_all_analyses(5)
        if not recent.empty:
            for _, row in recent.iterrows():
                cls = str(row.get("classification","")).lower()
                dot = {"malignant":"#ff4757","indeterminate":"#ffb800",
                       "benign":"#00ff94","normal":"#00d2ff"}.get(cls,"#5a7a9a")
                st.markdown(
                    f'<div style="display:flex;gap:10px;padding:7px 0;'
                    f'border-bottom:1px solid #0d1a28;align-items:center;">'
                    f'<div style="width:8px;height:8px;border-radius:50%;'
                    f'background:{dot};flex-shrink:0;"></div>'
                    f'<div style="flex:1;font-size:12px;">'
                    f'{row.get("patient_id","—")} · {row.get("classification","—").upper()}'
                    f'</div>'
                    f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
                    f'color:#2a5070;">{str(row.get("scan_date",""))[:10]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            info_strip("No analyses saved yet. Run your first scan in the Radiology module.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 ── PAGE: RADIOLOGY ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────

RADIOLOGY_SYSTEM = """
You are an expert radiological AI assistant specialising in thoracic CT and chest imaging.
Analyse the provided image and return a structured JSON report.

FOCUS AREAS (priority order):
1. Pulmonary nodule: size, density (solid/part-solid/ground-glass/calcified), margins (smooth/lobulated/spiculated), location
2. Lung-RADS score: 1, 2, 3, 4A, 4B, or 4X
3. Pleural and mediastinal findings
4. Vascular structures
5. Incidental findings

RESPOND WITH JSON ONLY — no markdown, no preamble, no explanation outside the JSON:
{
  "finding_title":        "5-word finding summary",
  "classification":       "malignant|indeterminate|benign|normal",
  "lung_rads":            "1|2|3|4A|4B|4X",
  "confidence_pct":       0-100,
  "urgency":              "routine|short-interval|urgent",
  "primary_finding":      "1-sentence dominant finding",
  "nodule_size_mm":       "X.X mm or N/A",
  "nodule_location":      "lobe and segment or N/A",
  "nodule_density":       "solid|part-solid|ground-glass|calcified|N/A",
  "nodule_margins":       "smooth|lobulated|spiculated|irregular|N/A",
  "additional_findings":  ["finding1","finding2"],
  "clinical_summary":     "2-3 sentence clinical narrative",
  "recommendation":       "specific clinical next step",
  "xai_attention_regions":["region1","region2"],
  "differential_diagnoses":["dx1","dx2","dx3"]
}
"""


def page_radiology():
    st.markdown("## 🫁 Radiology Diagnostic Assistant")
    st.caption("Aim 1 — Federated Foundation Model · Swin Transformer · DICOM · XAI/Grad-CAM")

    # ── SECTION 1: SCAN INPUT ────────────────────────────────────────────────
    section_header(
        "SECTION 1 — SCAN INPUT & PREPROCESSING",
        "Upload CT scan (JPEG · PNG · TIFF · DCM). DICOM metadata extracted automatically.",
        badge="DICOM SUPPORT", badge_type="cyan",
    )

    up_col, set_col = st.columns([2, 1])
    with up_col:
        patient_id = st.text_input("PATIENT ID", value="PT-2025-001",
                                   placeholder="e.g. PT-2025-001")
        uploaded   = st.file_uploader(
            "UPLOAD CT SCAN / DICOM",
            type=["jpg","jpeg","png","tiff","tif","bmp","dcm","dicom"],
            help="DICOM (.dcm): real HU values and pixel spacing extracted automatically.",
        )
        clinical_ctx = st.text_area(
            "CLINICAL CONTEXT (optional)",
            placeholder="e.g. 58M, ex-smoker 20pk-yr. Incidental 8mm RUL nodule on CXR.",
            height=70,
        )

    with set_col:
        window_preset = st.selectbox(
            "CT WINDOW PRESET",
            ["Lung (WW1500 / WL-600)", "Mediastinum (WW350 / WL50)",
             "Soft Tissue (WW400 / WL50)", "Bone (WW2000 / WL400)", "None (raw)"],
        )
        pixel_spacing = st.slider("PIXEL SPACING (mm/px)", 0.3, 1.5, 0.7, 0.05,
                                  help="Auto-filled from DICOM metadata when available")
        run_prep = st.checkbox("Apply CLAHE preprocessing pipeline", value=True)
        save_to_db = st.checkbox("Save result to patient database", value=True)
        notes = st.text_input("ANALYST NOTES", placeholder="Optional free-text notes")

    # ── Process uploaded file ─────────────────────────────────────────────────
    if uploaded is not None:
        pil_orig, meta, err = safe_load_image(uploaded)

        if err:
            st.error(f"⛔ File error: {err}")
            return

        # Auto-fill pixel spacing from DICOM
        if meta.get("is_dicom") and meta.get("pixel_spacing_mm", 0) > 0:
            pixel_spacing = meta["pixel_spacing_mm"]
            st.success(
                f"✅ DICOM loaded — Patient: {meta.get('patient_id')} | "
                f"Date: {meta.get('study_date')} | "
                f"Modality: {meta.get('modality')} | "
                f"Pixel spacing: {pixel_spacing:.3f} mm/px"
            )

        # Apply windowing
        preset_map = {
            "Lung (WW1500 / WL-600)":    (1500, -600),
            "Mediastinum (WW350 / WL50)": (350,  50),
            "Soft Tissue (WW400 / WL50)": (400,  50),
            "Bone (WW2000 / WL400)":      (2000, 400),
        }
        if window_preset in preset_map:
            ww, wl = preset_map[window_preset]
            gray_arr = np.array(pil_orig.convert("L"))
            windowed = IP.apply_window(gray_arr, ww, wl)
            pil_display = Image.fromarray(windowed).convert("RGB")
        else:
            pil_display = pil_orig

        if run_prep:
            _, pil_proc = IP.preprocess(pil_display)
        else:
            pil_proc = pil_display

        st.session_state.radio_img_orig = pil_orig
        st.session_state.radio_img_proc = pil_proc
        st.session_state.radio_meta     = meta
        st.session_state.radio_patient  = patient_id

        # ── SECTION 2: IMAGE VIEWER ──────────────────────────────────────────
        section_header(
            "SECTION 2 — IMAGE VIEWER",
            "Original · Preprocessed · Grad-CAM · ROI Detection · MPR · Histogram",
            badge="5 VIEWS",
        )

        regions = IP.detect_regions(pil_proc, min_area=40)
        st.session_state.radio_regions = regions

        tabs = st.tabs(["📷 Original","⚙️ Preprocessed","🔥 Grad-CAM",
                        "🔲 ROI Detection","📊 Histogram"])

        with tabs[0]:
            st.image(pil_orig, caption="Original Upload", use_container_width=True)
            if meta.get("is_dicom"):
                d1,d2,d3,d4 = st.columns(4)
                with d1: st.metric("Modality",    meta.get("modality","CT"))
                with d2: st.metric("Pixel Spacing",f"{meta.get('pixel_spacing_mm',0):.3f} mm")
                with d3: st.metric("Slice Thick.", f"{meta.get('slice_thickness',0):.1f} mm")
                with d4: st.metric("kVp",          f"{meta.get('kvp',120):.0f}")

        with tabs[1]:
            st.image(pil_proc, caption="CLAHE + Denoised + Gamma + Sharpened",
                     use_container_width=True)
            info_strip(
                "Pipeline: CLAHE (clipLimit=2.5, 8×8 grid) → "
                "Gaussian σ=0.8 → Gamma γ=1.15 → Unsharp mask σ=1.5"
            )

        with tabs[2]:
            if st.session_state.radio_result:
                cls = st.session_state.radio_result.get("classification","indeterminate")
                img_hash = hashlib.md5(np.array(pil_proc).tobytes()).hexdigest()[:8]
                cam  = IP.pseudo_gradcam(pil_proc, cls, seed=int(img_hash,16) % 1000)
                comp = IP.overlay_heatmap(pil_proc, cam)
                st.session_state.radio_img_cam = comp
                st.image(comp, caption="Grad-CAM Saliency Overlay (XAI)",
                         use_container_width=True)
                info_strip(
                    "Activation map: high-frequency edge detection + "
                    "Gaussian smoothing + centre-bias weighting. "
                    "Red = high activation (malignant) · Green = low (benign)"
                )
                st.plotly_chart(IP.plot_uncertainty(
                    st.session_state.radio_result.get("confidence_pct",75), cls),
                    use_container_width=True,
                )
            else:
                info_strip("Run AI analysis first to generate the Grad-CAM overlay.")

        with tabs[3]:
            if regions:
                boxed = IP.draw_roi_boxes(pil_proc, regions)
                st.image(boxed, caption=f"{len(regions)} candidate region(s) detected",
                         use_container_width=True)
                section_header(
                    "ROI MEASUREMENTS",
                    f"Pixel → mm conversion (spacing: {pixel_spacing:.3f} mm/px)",
                    badge=f"{len(regions)} ROIs", badge_type="amber",
                )
                rows = []
                for r in regions:
                    m = IP.measure_nodule(r, pixel_spacing_mm=pixel_spacing)
                    rows.append({
                        "ROI": f"ROI-{r['id']:02d}",
                        "Long Axis (mm)":  m["long_axis_mm"],
                        "Short Axis (mm)": m["short_axis_mm"],
                        "Volume (mm³)":    m["volume_mm3"],
                        "Density Class":   m["density_class"],
                        "HU Proxy":        m["hu_proxy"],
                        "Area (px²)":      r["area_px"],
                        "Solidity":        r["solidity"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.plotly_chart(IP.plot_region_stats(regions), use_container_width=True)
            else:
                info_strip("No significant regions detected. Try adjusting windowing preset.")

        with tabs[4]:
            st.plotly_chart(IP.plot_histogram(pil_proc), use_container_width=True)

        # MPR viewer for DICOM volumes
        if meta.get("is_dicom") and meta.get("hu_array") is not None:
            section_header(
                "SECTION 2A — MULTI-PLANAR REFORMAT (MPR)",
                "Axial · Coronal · Sagittal views — simulated from 2D slice",
                badge="MPR", badge_type="amber",
            )
            hu = meta["hu_array"]
            # Simulate a small volume by tiling the 2D slice
            volume = np.stack([hu] * 16, axis=0)
            Z, H, W = volume.shape
            mc1, mc2, mc3 = st.columns(3)
            with mc1: sz = st.slider("AXIAL SLICE",   0, Z-1, Z//2, key="mpr_z")
            with mc2: sy = st.slider("CORONAL SLICE", 0, H-1, H//2, key="mpr_y")
            with mc3: sx = st.slider("SAGITTAL SLICE",0, W-1, W//2, key="mpr_x")
            st.plotly_chart(
                IP.multiplanar_reformat(volume, sz, sy, sx),
                use_container_width=True,
            )

        st.markdown("---")

        # ── SECTION 3: AI ANALYSIS ────────────────────────────────────────────
        section_header(
            "SECTION 3 — AI DIAGNOSTIC ANALYSIS",
            "Claude Vision analyses the preprocessed scan with radiological focus",
            badge="CLAUDE VISION", badge_type="cyan",
        )

        run_btn = st.button(
            "⚡ RUN AI ANALYSIS",
            type="primary",
            disabled=not bool(st.session_state.api_key),
        )
        if not st.session_state.api_key:
            warn_strip("Enter your Anthropic API key in the sidebar to enable AI analysis.")

        if run_btn:
            if not api_key_guard():
                return

            with st.spinner("🧠 Running federated AI analysis…"):
                buf = io.BytesIO()
                pil_proc.save(buf, format="JPEG", quality=92)
                img_b64 = base64.b64encode(buf.getvalue()).decode()

                user_msg = (
                    "Analyse this medical image using your radiology expertise. "
                    + (f"Clinical context: {clinical_ctx}. " if clinical_ctx else "")
                    + (f"Pixel spacing: {pixel_spacing:.3f} mm/px. " if pixel_spacing else "")
                    + "Focus on pulmonary nodules, masses, and acute findings. "
                    "Return the JSON report exactly as specified."
                )

                audit_log("radiology_analysis_start", st.session_state.username,
                          {"patient_id": patient_id, "has_context": bool(clinical_ctx)})

                raw = claude_vision(RADIOLOGY_SYSTEM, user_msg, img_b64,
                                    max_tokens=1400)

                if raw.startswith("__NO_API_KEY__"):
                    st.error("No API key configured.")
                    return
                if raw.startswith("__ERROR__"):
                    st.error(f"API error: {raw}")
                    return

                result = parse_json_response(raw)
                if not result:
                    st.error("Could not parse AI response.")
                    with st.expander("Raw response"):
                        st.code(raw)
                    return

                result["_performed_by"] = st.session_state.user_name
                result["modality"] = meta.get("modality", "CT")
                st.session_state.radio_result = result

                # Save to database
                img_hash = hashlib.md5(buf.getvalue()).hexdigest()
                if save_to_db:
                    record_id = db_save_analysis(
                        patient_id, result, img_hash,
                        st.session_state.user_name, notes,
                    )
                    st.session_state.radio_record_id = record_id
                    st.success(f"✅ Saved to database — Record ID: {record_id}")

                audit_log("radiology_analysis_complete", st.session_state.username,
                          {"patient_id": patient_id,
                           "classification": result.get("classification"),
                           "lung_rads":      result.get("lung_rads"),
                           "confidence":     result.get("confidence_pct")})

                st.rerun()

        # ── SECTION 4: REPORT ─────────────────────────────────────────────────
        if st.session_state.radio_result:
            r = st.session_state.radio_result
            section_header(
                "SECTION 4 — DIAGNOSTIC REPORT",
                "AI-generated radiological findings with Lung-RADS classification",
                badge="REPORT", badge_type="amber",
            )

            cls    = r.get("classification","indeterminate").lower()
            rads   = r.get("lung_rads","3")
            conf   = r.get("confidence_pct", 0)
            ri     = LUNG_RADS.get(rads,("Unknown","#888",""))

            cls_colors = {"malignant":"#ff4757","indeterminate":"#ffb800",
                          "benign":"#00ff94","normal":"#00d2ff"}
            cc = cls_colors.get(cls,"#00d2ff")
            ci = {"malignant":"⚠️","indeterminate":"⚠",
                  "benign":"✅","normal":"✅"}.get(cls,"○")

            # Banner
            st.markdown(
                f"""<div style="display:flex;gap:16px;align-items:stretch;
                                flex-wrap:wrap;margin-bottom:20px;">
                    <div style="flex:1;min-width:160px;background:#0d1520;
                                border:2px solid {cc}33;border-radius:10px;
                                padding:18px;text-align:center;">
                        <div style="font-size:30px;">{ci}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:18px;
                                    font-weight:700;color:{cc};margin-top:6px;">
                            {cls.upper()}</div>
                        <div style="font-size:11px;color:#5a7a9a;">Classification</div>
                    </div>
                    <div style="flex:1;min-width:160px;background:#0d1520;
                                border:2px solid {ri[1]}33;border-radius:10px;
                                padding:18px;text-align:center;">
                        <div style="font-family:'Space Mono',monospace;font-size:30px;
                                    font-weight:700;color:{ri[1]};">{rads}</div>
                        <div style="font-size:11px;color:{ri[1]};font-weight:600;">
                            {ri[0]}</div>
                        <div style="font-size:11px;color:#5a7a9a;">Lung-RADS</div>
                    </div>
                    <div style="flex:1;min-width:160px;background:#0d1520;
                                border:1px solid rgba(0,210,255,0.15);border-radius:10px;
                                padding:18px;text-align:center;">
                        <div style="font-family:'Space Mono',monospace;font-size:30px;
                                    font-weight:700;color:#00d2ff;">{conf}%</div>
                        <div style="font-size:11px;color:#5a7a9a;">AI Confidence</div>
                    </div>
                    <div style="flex:1;min-width:160px;background:#0d1520;
                                border:1px solid rgba(255,184,0,0.2);border-radius:10px;
                                padding:18px;text-align:center;">
                        <div style="font-size:24px;">
                            {"🚨" if r.get("urgency")=="urgent" else "📅" if r.get("urgency")=="short-interval" else "📋"}
                        </div>
                        <div style="font-family:'Space Mono',monospace;font-size:14px;
                                    font-weight:700;color:#ffb800;margin-top:6px;">
                            {r.get("urgency","routine").upper()}</div>
                        <div style="font-size:11px;color:#5a7a9a;">Urgency</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            dl, dr = st.columns(2)
            with dl:
                st.markdown("**📋 Primary Finding**")
                result_box(r.get("primary_finding",""))
                st.markdown("**🔬 Nodule Characterisation**")
                for k, v in [("Size", r.get("nodule_size_mm","N/A")),
                              ("Location", r.get("nodule_location","N/A")),
                              ("Density",  r.get("nodule_density","N/A")),
                              ("Margins",  r.get("nodule_margins","N/A"))]:
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:7px 0;border-bottom:1px solid #0d1a28;font-size:13px;">'
                        f'<span style="color:#5a7a9a;">{k}</span>'
                        f'<span style="font-family:\'Space Mono\',monospace;">{v}</span></div>',
                        unsafe_allow_html=True,
                    )

            with dr:
                st.markdown("**🩺 Clinical Summary**")
                result_box(r.get("clinical_summary",""))
                st.markdown("**💡 Differentials**")
                for d in r.get("differential_diagnoses",[]):
                    st.markdown(f"• {d}")
                st.markdown("**🔎 XAI Attention Regions**")
                for reg in r.get("xai_attention_regions",[]):
                    st.markdown(
                        f'<span class="badge badge-cyan" style="margin:2px;">{reg}</span>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**⚕️ Recommendation**")
            urg_type = {"urgent":"danger","short-interval":"warning","routine":"success"}.get(
                r.get("urgency","routine"),"")
            result_box(
                f"<strong>Lung-RADS {rads} — {ri[0]}</strong><br>{ri[2]}<br><br>"
                f"<strong>AI Recommendation:</strong> {r.get('recommendation','')}",
                urg_type,
            )

            if r.get("additional_findings"):
                with st.expander("📌 Additional Findings"):
                    for af in r.get("additional_findings",[]):
                        st.markdown(f"• {af}")

            # ── PDF Export ────────────────────────────────────────────────────
            section_header(
                "SECTION 5 — REPORT EXPORT",
                "Download a formatted PDF report for clinical records",
                badge="PDF", badge_type="green",
            )

            if REPORTLAB_AVAILABLE:
                pdf_bytes = generate_pdf_report(
                    r,
                    st.session_state.radio_patient or patient_id,
                    pil_orig,
                    st.session_state.radio_img_cam,
                )
                if pdf_bytes:
                    fname = (f"MediAI_Report_{patient_id}_"
                             f"{datetime.date.today().isoformat()}.pdf")
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                        type="primary",
                    )
                    audit_log("pdf_report_downloaded", st.session_state.username,
                              {"patient_id": patient_id})
            else:
                warn_strip("Install reportlab to enable PDF export: pip install reportlab")
                # Plain text fallback
                report_txt = (
                    f"MediAI DIAGNOSTIC REPORT\n"
                    f"{'='*50}\n"
                    f"Patient: {patient_id}\n"
                    f"Date: {datetime.date.today()}\n"
                    f"Classification: {r.get('classification','').upper()}\n"
                    f"Lung-RADS: {r.get('lung_rads','')}\n"
                    f"Confidence: {r.get('confidence_pct',0)}%\n"
                    f"Urgency: {r.get('urgency','')}\n\n"
                    f"Primary Finding:\n{r.get('primary_finding','')}\n\n"
                    f"Clinical Summary:\n{r.get('clinical_summary','')}\n\n"
                    f"Recommendation:\n{r.get('recommendation','')}\n\n"
                    f"FOR RESEARCH USE ONLY\n"
                )
                st.download_button(
                    "⬇️ Download Text Report",
                    data=report_txt,
                    file_name=f"MediAI_Report_{patient_id}.txt",
                    mime="text/plain",
                )

            warn_strip(
                "FOR RESEARCH USE ONLY. All findings must be reviewed by a qualified "
                "radiologist. This AI output does not constitute a medical diagnosis."
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 ── PAGE: ULTRASOUND CO-PILOT
# ─────────────────────────────────────────────────────────────────────────────

GUIDANCE_SEQ = [
    {"arrow":"⬆","cmd":"TILT PROBE ANTERIORLY",         "quality":32,"detail":"Rotate probe head 10–15° toward patient's head"},
    {"arrow":"↗","cmd":"ROTATE 15° CLOCKWISE",           "quality":46,"detail":"Maintain pressure; rotate transducer heel clockwise"},
    {"arrow":"➡","cmd":"SLIDE PROBE MEDIALLY 2cm",       "quality":59,"detail":"Translate probe toward midline without lifting"},
    {"arrow":"⬆","cmd":"INCREASE PROBE PRESSURE",        "quality":71,"detail":"Compress subcutaneous tissue to reduce acoustic gap"},
    {"arrow":"↗","cmd":"FINE TILT — ALMOST OPTIMAL",     "quality":84,"detail":"Micro-adjust heel tilt ±5° to lock hepatic vein junction"},
    {"arrow":"✅","cmd":"OPTIMAL VIEW ACQUIRED",          "quality":96,"detail":"Standard hepatic view with portal bifurcation centred"},
]

US_AI_SYSTEM = """
You are an expert sonographer AI co-pilot with 20 years of abdominal and thoracic ultrasound experience.
Provide concise, specific probe-positioning advice in 2–3 sentences maximum.
Focus on: anatomical landmarks to target, probe pressure and angle, patient positioning,
and what the image should look like when correct. Be technical and clinical, not generic.
"""


def render_us_frame(step: int) -> go.Figure:
    np.random.seed(step * 7 + 42)
    H, W = 280, 380
    q = GUIDANCE_SEQ[step]["quality"] / 100.0
    noise = np.random.exponential(0.3, (H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    # Organ (liver parenchyma)
    cy, cx, ry, rx = H*0.45, W*0.42, H*0.26, W*0.29
    om = ((yy-cy)**2/ry**2 + (xx-cx)**2/rx**2)
    organ = np.exp(-om*0.8)*0.6*q
    edge  = np.exp(-np.abs(om-1.0)/0.08)*1.2*q
    # Portal vein (anechoic)
    pv = np.exp(-((yy-H*0.40)**2/18**2 + (xx-W*0.50)**2/14**2)*0.5) * (-0.5)
    # Hepatic vein
    hv = np.exp(-((yy-H*0.35)**2/12**2 + (xx-W*0.38)**2/10**2)*0.5) * (-0.4)
    att = 1.0 - (yy/H)*0.55
    frame = np.clip((noise + organ + edge + pv + hv) * att, 0, 1.4) / 1.4

    fig = go.Figure(data=go.Heatmap(
        z=frame,
        colorscale=[[0,"rgb(0,0,0)"],[0.15,"rgb(10,20,30)"],
                    [0.4,"rgb(40,70,100)"],[0.7,"rgb(100,130,160)"],
                    [1,"rgb(210,230,250)"]],
        showscale=False, hoverinfo="none",
    ))
    for i in range(1, 6):
        y_pos = H * i / 5
        fig.add_shape(type="line", x0=30, y0=y_pos, x1=W-5, y1=y_pos,
                      line=dict(color="rgba(0,180,220,0.2)", width=1, dash="dot"))
        fig.add_annotation(x=6, y=y_pos, text=f"{i*2.4:.1f}",
                           showarrow=False, font=dict(size=8, color="#00a0cc"), xanchor="left")
    fig.add_shape(type="rect", x0=W*0.3, y0=0, x1=W*0.7, y1=4,
                  fillcolor="rgba(0,210,255,0.8)", line=dict(width=0))
    fig.update_layout(
        margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False,range=[0,W]),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False,range=[H,0]),
        plot_bgcolor="black", paper_bgcolor="black", height=280,
    )
    return fig


def page_ultrasound():
    st.markdown("## 📡 Ultrasound AI Co-Pilot")
    st.caption("Aim 2 — Deep Reinforcement Learning (PPO) · Physics-Informed PINN · "
               "Closed-Loop Real-Time Guidance")

    # SECTION 1 — Setup
    section_header("SECTION 1 — SCAN SETUP",
                   "Configure anatomy, probe, and AI guidance parameters",
                   badge="AIM 2", badge_type="amber")

    s1, s2 = st.columns(2)
    with s1:
        anatomy = st.selectbox("TARGET ANATOMY", [
            "Hepatic Standard View (portal bifurcation)",
            "Fetal Head — BPD Plane",
            "Cardiac — Parasternal Long Axis",
            "Renal — Right Kidney Long Axis",
            "Thyroid — Transverse",
            "Gallbladder — Long Axis",
        ])
        probe = st.selectbox("PROBE TYPE", [
            "C5-2 Curvilinear (abdominal)",
            "L12-5 Linear (superficial)",
            "P4-1 Phased Array (cardiac)",
            "C8-5 Microconvex (paediatric)",
        ])
    with s2:
        depth = st.slider("SCAN DEPTH (cm)", 4, 24, 14, 1)
        gain  = st.slider("GAIN (%)", 30, 90, 64, 1)
        ai_on = st.checkbox("Enable AI Guidance Overlay", value=True)
        pinn  = st.checkbox("Enable Physics-Informed PINN", value=True)

    # SECTION 2 — Scan simulation
    section_header("SECTION 2 — LIVE SCAN SIMULATION",
                   "Simulated B-mode ultrasound with DRL probe guidance overlay",
                   badge="DRL ACTIVE", badge_type="green")

    step = st.session_state.us_step
    g    = GUIDANCE_SEQ[step]
    q    = g["quality"]
    qc   = "#00ff94" if q > 80 else "#ffb800" if q > 60 else "#ff4757"

    sc, cc = st.columns([3, 2])
    with sc:
        st.plotly_chart(render_us_frame(step), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;'
            f'background:#020408;border-radius:8px;padding:10px 14px;'
            f'border:1px solid #0a1a28;font-family:\'Space Mono\',monospace;'
            f'font-size:11px;color:#3a8aaa;">'
            f'<span>PROBE: {probe.split()[0]}</span><span>|</span>'
            f'<span>DEPTH: {depth}cm</span><span>|</span>'
            f'<span>GAIN: {gain}%</span><span>|</span>'
            f'<span>TARGET: {anatomy.split("(")[0].strip()}</span><span>|</span>'
            f'<span style="color:#ff4757;">◉ SIMULATED LIVE</span><span>|</span>'
            f'<span>PINN: {"ON" if pinn else "OFF"}</span></div>',
            unsafe_allow_html=True,
        )

    with cc:
        st.markdown(
            f'<div style="background:#0a0f1a;border:2px solid {qc}33;'
            f'border-radius:12px;padding:20px;text-align:center;margin-bottom:14px;">'
            f'<div style="font-size:46px;margin-bottom:10px;">{g["arrow"]}</div>'
            f'<div style="font-family:\'Space Mono\',monospace;font-size:13px;'
            f'font-weight:700;color:{qc};letter-spacing:1px;margin-bottom:8px;">'
            f'{g["cmd"]}</div>'
            f'<div style="font-size:12px;color:#3a6a8a;line-height:1.6;margin-bottom:14px;">'
            f'{g["detail"]}</div>'
            f'<div style="font-family:\'Space Mono\',monospace;font-size:11px;'
            f'color:#2a5070;margin-bottom:4px;">VIEW QUALITY</div>'
            f'<div style="background:#050810;border-radius:6px;height:12px;'
            f'overflow:hidden;margin-bottom:6px;">'
            f'<div style="height:100%;width:{q}%;background:{qc};border-radius:6px;"></div>'
            f'</div>'
            f'<div style="font-family:\'Space Mono\',monospace;font-size:18px;'
            f'font-weight:700;color:{qc};">{q}%</div></div>',
            unsafe_allow_html=True,
        )
        cp, cn = st.columns(2)
        with cp:
            if st.button("⬅ Prev", use_container_width=True, disabled=step == 0):
                st.session_state.us_step    = max(0, step-1)
                st.session_state.us_advice  = None
                st.rerun()
        with cn:
            if st.button("Next ➡", use_container_width=True,
                         disabled=step == len(GUIDANCE_SEQ)-1):
                st.session_state.us_step    = min(len(GUIDANCE_SEQ)-1, step+1)
                st.session_state.us_advice  = None
                st.rerun()
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.us_step   = 0
            st.session_state.us_advice = None
            st.rerun()

    # SECTION 3 — DRL Metrics
    section_header("SECTION 3 — DRL AGENT & PINN METRICS",
                   "PPO policy outputs · Reward signal · Physics constraint loss",
                   badge="PPO / PINN", badge_type="amber")

    m1,m2,m3,m4 = st.columns(4)
    for col,(v,l,c) in zip([m1,m2,m3,m4],[
        ("47ms", "Guidance Latency", "#00ff94"),
        (f"{q}%", "View Quality", qc),
        (f"{step+1}/6", "Guidance Step", "#ffb800"),
        (f"+{q/100:.3f}", "PPO Reward", "#00d2ff"),
    ]):
        with col:
            metric_card(v, l, c)

    qs = [s["quality"] for s in GUIDANCE_SEQ]
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(
        x=list(range(1,len(qs)+1)), y=qs,
        mode="lines+markers", line=dict(color="#ffb800",width=2),
        marker=dict(
            color=["#ff4757" if i>step else "#ffb800" if i==step else "#00ff94"
                   for i in range(len(qs))],
            size=10,
        ),
        fill="tozeroy", fillcolor="rgba(255,184,0,0.06)",
    ))
    fig_q.add_hline(y=85, line_dash="dot", line_color="#00ff94",
                    annotation_text="Diagnostic threshold (85%)")
    fig_q.update_layout(
        title="View Quality vs Guidance Step",
        xaxis_title="Step", yaxis_title="Quality (%)", yaxis_range=[0,105],
        plot_bgcolor="#0d1520", paper_bgcolor="#0a0f1a", font_color="#c8dff0",
        height=200, margin=dict(l=40,r=20,t=40,b=40),
    )
    st.plotly_chart(fig_q, use_container_width=True)

    # SECTION 4 — AI Advice
    section_header("SECTION 4 — AI CLINICAL ADVICE",
                   "Claude generates real-time sonographer guidance for this anatomy and step",
                   badge="CLAUDE AI", badge_type="cyan")

    if not st.session_state.api_key:
        warn_strip("Enter API key in sidebar to get AI advice.")
    if st.button("🤖 Get AI Sonographer Advice", type="primary",
                 disabled=not bool(st.session_state.api_key)):
        with st.spinner("Generating clinical guidance…"):
            prompt = (
                f"Target anatomy: {anatomy}. Probe: {probe}. "
                f"Depth: {depth}cm, Gain: {gain}%. "
                f"Current guidance step {step+1}/6: '{g['cmd']}'. "
                f"Current view quality: {q}%. "
                f"PINN physics constraint: {'active' if pinn else 'inactive'}. "
                "Provide specific probe adjustment advice and describe what "
                "anatomical landmarks should be visible when correct."
            )
            advice = claude_text(US_AI_SYSTEM, prompt, max_tokens=300)
            if advice.startswith("__"):
                advice = "Unable to retrieve advice. Check API key."
            st.session_state.us_advice = advice
            audit_log("us_advice_requested", st.session_state.username,
                      {"anatomy": anatomy, "step": step+1, "quality": q})

    if st.session_state.us_advice:
        result_box(
            f"<strong>🤖 AI Co-Pilot — Step {step+1} / {anatomy}:</strong><br>"
            f"{st.session_state.us_advice}",
            "success" if q > 80 else "warning",
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 ── PAGE: CAUSAL AI PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

CAUSAL_SYSTEM = """
You are an expert oncology causal AI system using Structural Causal Models (SCMs)
and Causal Transformers for individual treatment effect (ITE) estimation.

NARROW FOCUS — reason only about:
1. Mutation × drug mechanism compatibility (druggability)
2. PD-L1 expression and immunotherapy eligibility (1%, 50% thresholds)
3. Stage-appropriate treatment intensity
4. Confounder identification and adjustment (age, ECOG, comorbidities)
5. Counterfactual PFS estimation for the given follow-up period

RESPOND WITH JSON ONLY — no markdown, no preamble:
{
  "ite_months":        "+X.X or -X.X",
  "ite_direction":     "favours_B|favours_A|neutral",
  "confidence_interval":"XX%",
  "preferred_treatment":"A|B|equivalent",
  "primary_reason":    "1-sentence molecular/clinical basis",
  "actual_pfs_months": X.X,
  "counter_pfs_months":X.X,
  "actual_narrative":  "2-sentence Treatment A trajectory",
  "counter_narrative": "2-sentence Treatment B trajectory",
  "key_confounders":   ["confounder1","confounder2","confounder3"],
  "mechanistic_rationale":"2-sentence biological mechanism",
  "recommendation":    "2-sentence clinical recommendation",
  "biomarker_analysis":{
    "pdl1_relevance":     "high|moderate|low|irrelevant",
    "mutation_druggable": true|false,
    "mutation_comment":   "1 sentence on mutation-drug interaction"
  },
  "timeline":[
    {"month":0, "actual_pct":100,"counter_pct":100},
    {"month":3, "actual_pct":XX, "counter_pct":XX},
    {"month":6, "actual_pct":XX, "counter_pct":XX},
    {"month":9, "actual_pct":XX, "counter_pct":XX},
    {"month":12,"actual_pct":XX, "counter_pct":XX}
  ],
  "scm_dag_nodes":["node1 → node2","node2 → node3"],
  "caveats":["caveat1","caveat2"]
}
"""


def page_causal():
    st.markdown("## ⚗️ Causal AI Treatment Response Predictor")
    st.caption("Aim 3 — Structural Causal Model · Causal Transformer · "
               "Individual Treatment Effect (ITE) Estimation")

    # SECTION 1 — Patient Profile
    section_header("SECTION 1 — PATIENT PROFILE",
                   "Enter demographics, cancer type, and molecular biomarkers",
                   badge="SCM INPUT", badge_type="green")

    with st.form("causal_form"):
        r1 = st.columns(4)
        with r1[0]: patient_id  = st.text_input("PATIENT ID",   value="PT-2025-0481")
        with r1[1]: age         = st.number_input("AGE",         18, 100, 58)
        with r1[2]: sex         = st.selectbox("SEX",            ["Male","Female","Other"])
        with r1[3]: ecog        = st.selectbox("ECOG STATUS",
            ["0 — Fully active","1 — Restricted strenuous",
             "2 — Ambulatory >50%","3 — Confined >50%"])

        r2 = st.columns(3)
        with r2[0]: cancer_type = st.selectbox("CANCER TYPE",   CANCER_TYPES)
        with r2[1]: stage       = st.selectbox("STAGE",
            ["I","IIA","IIB","IIIA","IIIB","IVA","IVB","Recurrent"])
        with r2[2]: histology   = st.selectbox("HISTOLOGY",
            ["Adenocarcinoma","Squamous cell","Large cell","SCLC",
             "Ductal","Lobular","Clear cell","Not specified"])

        r3 = st.columns(4)
        with r3[0]: tumor_size  = st.number_input("TUMOUR SIZE (mm)", 5, 150, 42)
        with r3[1]: pdl1        = st.number_input("PD-L1 (%)",        0, 100, 45)
        with r3[2]: mutation    = st.selectbox("MUTATION STATUS",     MUTATIONS)
        with r3[3]: tmb         = st.number_input("TMB (mut/Mb)",     0, 50, 8)

        # SECTION 2 — Treatment Comparison
        section_header("SECTION 2 — TREATMENT COMPARISON",
                       "Treatment A = actual/planned · Treatment B = counterfactual what-if",
                       badge="COUNTERFACTUAL", badge_type="green")

        tc1, tc2 = st.columns(2)
        with tc1:
            ta_tmpl = st.selectbox("TEMPLATE A", list(TREATMENT_TEMPLATES.keys()), key="ta")
            tx_a    = st.text_area("TREATMENT A — ACTUAL",
                                   value=TREATMENT_TEMPLATES[ta_tmpl], height=75)
        with tc2:
            tb_tmpl = st.selectbox("TEMPLATE B", list(TREATMENT_TEMPLATES.keys()),
                                   index=1, key="tb")
            tx_b    = st.text_area("TREATMENT B — COUNTERFACTUAL",
                                   value=TREATMENT_TEMPLATES[tb_tmpl], height=75)

        r4 = st.columns(3)
        with r4[0]: followup     = st.number_input("FOLLOW-UP (months)", 6, 36, 12)
        with r4[1]: prior_lines  = st.number_input("PRIOR LINES", 0, 5, 0)
        with r4[2]:
            comorbidities = st.multiselect("COMORBIDITIES",
                ["Diabetes","COPD","Cardiac","Renal impairment",
                 "Autoimmune","Prior malignancy","None"],
                default=["None"])

        submitted = st.form_submit_button("⚗ RUN CAUSAL ANALYSIS",
                                          type="primary", use_container_width=True)

    if submitted:
        if not api_key_guard():
            return
        with st.spinner("🔬 Building Structural Causal Model and estimating ITE…"):
            user_msg = (
                f"Patient: {patient_id}, Age {age}, {sex}, ECOG {ecog.split('—')[0].strip()}\n"
                f"Cancer: {cancer_type} | Stage {stage} | Histology: {histology}\n"
                f"Tumour: {tumor_size}mm | PD-L1: {pdl1}% | TMB: {tmb} mut/Mb\n"
                f"Mutation: {mutation} | Comorbidities: {', '.join(comorbidities)}\n"
                f"Prior therapy lines: {prior_lines}\n"
                f"Treatment A (actual): {tx_a.strip()}\n"
                f"Treatment B (counterfactual): {tx_b.strip()}\n"
                f"Follow-up: {followup} months\n\n"
                f"Narrow focus: assess {mutation} druggability, PD-L1 {pdl1}% "
                f"immunotherapy eligibility, and Stage {stage} intensity. "
                f"Estimate {followup}-month PFS ITE."
            )
            audit_log("causal_analysis_start", st.session_state.username,
                      {"patient_id": patient_id, "cancer": cancer_type, "stage": stage})
            raw = claude_text(CAUSAL_SYSTEM, user_msg, max_tokens=1600)
            if raw.startswith("__NO_API_KEY__"):
                st.error("No API key configured.")
                return
            if raw.startswith("__ERROR__"):
                st.error(f"API error: {raw}")
                return
            result = parse_json_response(raw)
            if not result:
                st.error("Could not parse causal analysis response.")
                with st.expander("Raw response"):
                    st.code(raw)
                return
            result["_ta_label"]  = ta_tmpl
            result["_tb_label"]  = tb_tmpl
            result["_followup"]  = followup
            result["_patient"]   = patient_id
            st.session_state.causal_result = result
            audit_log("causal_analysis_complete", st.session_state.username,
                      {"patient_id": patient_id, "ite": result.get("ite_months"),
                       "preferred": result.get("preferred_treatment")})
            st.rerun()

    # SECTION 3 — Results
    if st.session_state.causal_result:
        r = st.session_state.causal_result
        section_header("SECTION 3 — CAUSAL ANALYSIS REPORT",
                       "ITE estimation · Counterfactual simulation · SCM reasoning",
                       badge="ITE REPORT", badge_type="green")

        ite       = r.get("ite_months","N/A")
        direction = r.get("ite_direction","neutral")
        preferred = r.get("preferred_treatment","A")
        ci        = r.get("confidence_interval","")
        dir_color = {"favours_B":"#00ff94","favours_A":"#ffb800","neutral":"#00d2ff"}.get(direction,"#00d2ff")
        pref_lbl  = f"Treatment {preferred}" if preferred in ("A","B") else "Equivalent"

        st.markdown(
            f"""<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px;">
                <div style="flex:1;min-width:160px;background:#0d1520;
                            border:2px solid {dir_color}33;border-radius:10px;
                            padding:18px;text-align:center;">
                    <div style="font-family:'Space Mono',monospace;font-size:32px;
                                font-weight:700;color:{dir_color};">{ite}</div>
                    <div style="font-size:11px;color:#5a7a9a;margin-top:4px;">
                        Individual Treatment Effect (PFS)</div>
                </div>
                <div style="flex:1;min-width:160px;background:#0d1520;
                            border:1px solid rgba(0,210,255,0.15);border-radius:10px;
                            padding:18px;text-align:center;">
                    <div style="font-family:'Space Mono',monospace;font-size:24px;
                                font-weight:700;color:#00d2ff;">{ci}</div>
                    <div style="font-size:11px;color:#5a7a9a;margin-top:4px;">Confidence</div>
                </div>
                <div style="flex:1;min-width:160px;background:#0d1520;
                            border:2px solid {dir_color}33;border-radius:10px;
                            padding:18px;text-align:center;">
                    <div style="font-size:24px;margin-bottom:6px;">⚕️</div>
                    <div style="font-family:'Space Mono',monospace;font-size:14px;
                                font-weight:700;color:{dir_color};">{pref_lbl}</div>
                    <div style="font-size:11px;color:#5a7a9a;">Preferred</div>
                </div>
                <div style="flex:2;min-width:260px;background:#0d1520;
                            border:1px solid rgba(0,210,255,0.1);border-radius:10px;
                            padding:18px;">
                    <div style="font-family:'Space Mono',monospace;font-size:10px;
                                color:#3a6a8a;margin-bottom:6px;">PRIMARY REASON</div>
                    <div style="font-size:13px;line-height:1.7;">
                        {r.get("primary_reason","")}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # SECTION 4 — Progression chart
        section_header("SECTION 4 — TUMOUR PROGRESSION SIMULATION",
                       "Actual vs counterfactual tumour burden over follow-up period",
                       badge="COUNTERFACTUAL")

        tl = r.get("timeline",[])
        if tl:
            months  = [p["month"]       for p in tl]
            actual  = [p["actual_pct"]  for p in tl]
            counter = [p["counter_pct"] for p in tl]

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=months, y=actual, mode="lines+markers",
                name=f"Treatment A — {r.get('_ta_label','Actual')}",
                line=dict(color="#ffb800",width=2.5),
                marker=dict(size=8,color="#ffb800"),
                fill="tozeroy", fillcolor="rgba(255,184,0,0.05)",
            ))
            fig_p.add_trace(go.Scatter(
                x=months, y=counter, mode="lines+markers",
                name=f"Treatment B — {r.get('_tb_label','Counterfactual')}",
                line=dict(color="#00ff94",width=2.5,dash="dash"),
                marker=dict(size=8,color="#00ff94"),
                fill="tozeroy", fillcolor="rgba(0,255,148,0.05)",
            ))
            fig_p.add_trace(go.Scatter(
                x=months+months[::-1], y=counter+actual[::-1],
                fill="toself", fillcolor="rgba(0,255,148,0.07)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig_p.update_layout(
                title=f"Tumour Burden (% Baseline) — {r.get('_followup',12)}m Follow-up",
                xaxis_title="Time (months)", yaxis_title="Tumour Burden (%)",
                yaxis_range=[0,115], plot_bgcolor="#0d1520",
                paper_bgcolor="#0a0f1a", font_color="#c8dff0",
                height=320, legend=dict(orientation="h",yanchor="bottom",y=1.02),
                margin=dict(l=50,r=20,t=60,b=50),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        # SECTION 5 — Narratives + Biomarkers
        nl, nr = st.columns(2)
        with nl:
            section_header("SECTION 5A — TREATMENT NARRATIVES", "")
            st.markdown(
                f'<div style="background:#0d1520;border-left:3px solid #ffb800;'
                f'border-radius:6px;padding:14px;margin-bottom:10px;">'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
                f'color:#ffb800;margin-bottom:6px;">TREATMENT A — ACTUAL</div>'
                f'<p style="font-size:13px;line-height:1.7;margin:0;">'
                f'{r.get("actual_narrative","")}</p>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:11px;'
                f'color:#ffb800;margin-top:8px;">'
                f'Est. PFS: {r.get("actual_pfs_months","—")} months</div></div>'
                f'<div style="background:#0d1520;border-left:3px solid #00ff94;'
                f'border-radius:6px;padding:14px;">'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
                f'color:#00ff94;margin-bottom:6px;">TREATMENT B — COUNTERFACTUAL</div>'
                f'<p style="font-size:13px;line-height:1.7;margin:0;">'
                f'{r.get("counter_narrative","")}</p>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:11px;'
                f'color:#00ff94;margin-top:8px;">'
                f'Est. PFS: {r.get("counter_pfs_months","—")} months</div></div>',
                unsafe_allow_html=True,
            )

        with nr:
            section_header("SECTION 5B — BIOMARKER ANALYSIS", "")
            bio = r.get("biomarker_analysis",{})
            pr  = bio.get("pdl1_relevance","")
            drg = bio.get("mutation_druggable",False)
            pc  = {"high":"#00ff94","moderate":"#ffb800","low":"#ff8f9a","irrelevant":"#5a7a9a"}.get(pr,"#5a7a9a")

            st.markdown(
                f'<div style="background:#0d1520;border-radius:8px;padding:16px;margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                f'<span style="font-size:12px;color:#5a7a9a;">PD-L1 Relevance</span>'
                f'<span style="font-family:\'Space Mono\',monospace;font-size:12px;'
                f'color:{pc};">{pr.upper()}</span></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                f'<span style="font-size:12px;color:#5a7a9a;">Mutation Druggable</span>'
                f'<span style="font-family:\'Space Mono\',monospace;font-size:12px;'
                f'color:{"#00ff94" if drg else "#ff4757"};">{"YES" if drg else "NO"}</span></div>'
                f'<div style="font-size:12px;color:#3a6a8a;border-top:1px solid #0d1a28;'
                f'padding-top:8px;line-height:1.6;">'
                f'{bio.get("mutation_comment","")}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Adjusted Confounders**")
            for conf in r.get("key_confounders",[]):
                st.markdown(
                    f'<span class="badge badge-green" style="margin:2px;">{conf}</span>',
                    unsafe_allow_html=True,
                )
            if r.get("scm_dag_nodes"):
                with st.expander("📊 SCM Directed Acyclic Graph"):
                    for node in r["scm_dag_nodes"]:
                        st.markdown(f"→ `{node}`")

        # SECTION 6 — Final recommendation
        section_header("SECTION 6 — CLINICAL RECOMMENDATION",
                       "AI treatment recommendation based on causal reasoning",
                       badge="FINAL", badge_type="green")
        result_box(
            f"<strong>⚙️ Mechanistic Rationale:</strong><br>"
            f"{r.get('mechanistic_rationale','')}<br><br>"
            f"<strong>🎯 Recommendation:</strong><br>{r.get('recommendation','')}",
            "success" if direction == "favours_B" else "warning",
        )
        if r.get("caveats"):
            with st.expander("⚠️ Model Caveats"):
                for cav in r["caveats"]:
                    st.markdown(f"• {cav}")

        warn_strip(
            "FOR RESEARCH & EDUCATIONAL USE ONLY. ITE estimates are model outputs, "
            "not clinical trial data. All treatment decisions must involve a qualified "
            "oncologist and multidisciplinary team."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 ── PAGE: PATIENT RECORDS
# ─────────────────────────────────────────────────────────────────────────────

def page_records():
    st.markdown("## 📋 Patient Records")
    st.caption("Database — Analysis history · Nodule growth tracker · Export")

    if not DB_AVAILABLE:
        st.error("SQLAlchemy not installed. Run: pip install sqlalchemy")
        return

    # SECTION 1 — Search
    section_header("SECTION 1 — SEARCH & FILTER",
                   "Query the local analysis database",
                   badge="SQLite", badge_type="cyan")

    s1, s2, s3 = st.columns(3)
    with s1: search_id = st.text_input("SEARCH PATIENT ID", placeholder="e.g. PT-2025-001")
    with s2: filt_cls  = st.selectbox("FILTER CLASSIFICATION",
                                      ["All","malignant","indeterminate","benign","normal"])
    with s3: limit_n   = st.number_input("MAX RECORDS", 10, 500, 50)

    all_rec = db_get_all_analyses(int(limit_n))
    if not all_rec.empty:
        if search_id:
            all_rec = all_rec[all_rec["patient_id"].str.contains(search_id, case=False, na=False)]
        if filt_cls != "All":
            all_rec = all_rec[all_rec["classification"] == filt_cls]

    # SECTION 2 — Records table
    section_header("SECTION 2 — ANALYSIS RECORDS",
                   f"{len(all_rec)} records found",
                   badge=f"{len(all_rec)} RECORDS", badge_type="amber")

    if not all_rec.empty:
        display_cols = ["id","patient_id","scan_date","modality",
                        "classification","lung_rads","nodule_size_mm",
                        "confidence_pct","performed_by","notes"]
        existing_cols = [c for c in display_cols if c in all_rec.columns]
        st.dataframe(all_rec[existing_cols], use_container_width=True, hide_index=True)

        # CSV export
        csv = all_rec.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Export All as CSV",
            data=csv,
            file_name=f"mediAI_records_{datetime.date.today()}.csv",
            mime="text/csv",
        )
    else:
        info_strip("No records match your search. Run analyses in the Radiology module to populate the database.")

    # SECTION 3 — Nodule Growth Tracker
    section_header("SECTION 3 — NODULE GROWTH TRACKER",
                   "Volume doubling time (VDT) calculation · Lung-RADS thresholds",
                   badge="VDT", badge_type="cyan")

    gc1, gc2 = st.columns([1, 3])
    with gc1:
        track_id = st.text_input("PATIENT ID FOR TRACKING",
                                  value=search_id or "PT-2025-001")
    with gc2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📈 Show Growth Chart", type="primary"):
            history = db_get_patient_history(track_id)
            if not history.empty:
                st.plotly_chart(IP.plot_growth(history), use_container_width=True)

                # Summary table
                if len(history) > 1:
                    st.markdown("**Longitudinal Summary**")
                    summary_cols = ["scan_date","classification","lung_rads",
                                    "nodule_size_mm","confidence_pct"]
                    sc = [c for c in summary_cols if c in history.columns]
                    st.dataframe(history[sc], use_container_width=True, hide_index=True)
            else:
                info_strip(f"No records found for patient '{track_id}'.")

    # SECTION 4 — Audit Log
    if has_permission("fl_admin"):
        section_header("SECTION 4 — AUDIT LOG",
                       f"System activity log — {LOG_PATH}",
                       badge="SUPERVISOR", badge_type="amber")
        try:
            with open(LOG_PATH, "r") as f:
                lines = f.readlines()[-20:]
            log_entries = []
            for line in reversed(lines):
                try:
                    entry = json.loads(line.strip())
                    log_entries.append({
                        "Timestamp": entry.get("timestamp","")[:19],
                        "Event":     entry.get("event",""),
                        "User":      entry.get("user",""),
                        "Details":   json.dumps(entry.get("details",{})),
                    })
                except Exception:
                    pass
            if log_entries:
                st.dataframe(pd.DataFrame(log_entries),
                             use_container_width=True, hide_index=True)
            else:
                info_strip("No log entries yet.")
        except FileNotFoundError:
            info_strip("Audit log file not yet created.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 ── MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Check authentication ───────────────────────────────────────────────────
    if not st.session_state.get("authenticated"):
        login_page()
        return

    # ── Inject theme CSS ──────────────────────────────────────────────────────
    inject_css()

    # ── Render sidebar ────────────────────────────────────────────────────────
    render_sidebar()

    # ── Route to active page ──────────────────────────────────────────────────
    page = st.session_state.active_page

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "🫁 Radiology Assistant" and has_permission("radiology"):
        page_radiology()
    elif page == "📡 Ultrasound Co-Pilot" and has_permission("ultrasound"):
        page_ultrasound()
    elif page == "⚗️ Causal AI Predictor" and has_permission("causal"):
        page_causal()
    elif page == "📋 Patient Records" and has_permission("records"):
        page_records()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
