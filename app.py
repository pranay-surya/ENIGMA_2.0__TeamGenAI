import os
import warnings
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
import joblib
from scipy.integrate import simpson

warnings.filterwarnings("ignore")

try:
    import mne
    mne.set_log_level("ERROR")
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroScan — EEG Schizophrenia Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — clinical white & blue
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@300;400;500&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet">

<style>
/* ── Palette ─────────────────────────────────────────────── */
:root {
    --white:      #ffffff;
    --bg:         #f0f4f9;
    --bg2:        #e8eef7;
    --surface:    #ffffff;
    --surface2:   #f7faff;
    --blue-900:   #1a2f6e;
    --blue-700:   #1d4ed8;
    --blue-600:   #2563eb;
    --blue-500:   #3b82f6;
    --blue-400:   #60a5fa;
    --blue-100:   #dbeafe;
    --blue-50:    #eff6ff;
    --border:     #cddcf5;
    --border2:    #bfcfe8;
    --text:       #0f172a;
    --text2:      #1e3a5f;
    --muted:      #64748b;
    --muted2:     #94a3b8;
    --green:      #059669;
    --green-bg:   #ecfdf5;
    --amber:      #d97706;
    --amber-bg:   #fffbeb;
    --red:        #dc2626;
    --red-bg:     #fef2f2;
    --shadow:     0 1px 3px rgba(30,64,175,0.08), 0 1px 2px rgba(30,64,175,0.06);
    --shadow-md:  0 4px 16px rgba(30,64,175,0.10), 0 2px 6px rgba(30,64,175,0.07);
    --radius:     10px;
    --font-body:  'IBM Plex Sans', sans-serif;
    --font-mono:  'IBM Plex Mono', monospace;
    --font-head:  'Playfair Display', serif;
}

/* ── Global reset ────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }


/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--blue-900) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #e2eaf8 !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #c3d4f0 !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] input[type="text"] {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stToggle label {
    color: #c3d4f0 !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: var(--blue-600) !important;
    border: none !important;
    color: white !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.30) !important;
}
.stButton > button:hover {
    background: var(--blue-700) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.40) !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ───────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--blue-50) !important;
    border: 2px dashed var(--blue-400) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--white) !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 0 !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 24px !important;
    background: transparent !important;
    margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--blue-600) !important;
    border-bottom-color: var(--blue-600) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    padding: 24px !important;
}

/* ── Inputs ──────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--white) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* ── Dataframe ───────────────────────────────────────────── */
.dataframe { font-size: 12px !important; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--blue-400); border-radius: 3px; }

/* ── Progress bar ────────────────────────────────────────── */
.stProgress > div > div { background: var(--blue-600) !important; }

/* ── HR ──────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
TARGET_CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4","T6","O1","O2"
]
BANDS = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
BAND_COLORS = {
    "delta": "#6366f1",
    "theta": "#2563eb",
    "alpha": "#0891b2",
    "beta":  "#0369a1",
    "gamma": "#1e40af",
}
TOPO_POS = {
    "Fp1":(-0.18,0.87),"Fp2":(0.18,0.87),
    "F7":(-0.55,0.50),"F3":(-0.30,0.55),"Fz":(0.00,0.60),"F4":(0.30,0.55),"F8":(0.55,0.50),
    "T3":(-0.75,0.00),"C3":(-0.38,0.00),"Cz":(0.00,0.00),"C4":(0.38,0.00),"T4":(0.75,0.00),
    "T5":(-0.55,-0.50),"P3":(-0.30,-0.55),"Pz":(0.00,-0.55),"P4":(0.30,-0.55),
    "T6":(0.55,-0.50),"O1":(-0.18,-0.87),"O2":(0.18,-0.87),
}
FRONTAL_CH   = ["Fp1","Fp2","F3","Fz","F4","F7","F8"]
TEMPORAL_CH  = ["T3","T4","T5","T6"]
OCCIPITAL_CH = ["O1","O2"]
PARIETAL_CH  = ["P3","Pz","P4"]
CENTRAL_CH   = ["C3","Cz","C4"]

# ── Design tokens ─────────────────────────────────────────────────────────────
C_BG      = "#f0f4f9"
C_WHITE   = "#ffffff"
C_BLUE900 = "#1a2f6e"
C_BLUE700 = "#1d4ed8"
C_BLUE600 = "#2563eb"
C_BLUE500 = "#3b82f6"
C_BLUE100 = "#dbeafe"
C_BLUE50  = "#eff6ff"
C_BORDER  = "#cddcf5"
C_TEXT    = "#0f172a"
C_TEXT2   = "#1e3a5f"
C_MUTED   = "#64748b"
C_MUTED2  = "#94a3b8"
C_GREEN   = "#059669"
C_AMBER   = "#d97706"
C_RED     = "#dc2626"


def hex_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def risk_color(pct: float) -> str:
    if pct < 35:   return C_GREEN
    elif pct < 65: return C_AMBER
    return C_RED

def risk_label(pct: float) -> str:
    if pct < 35:   return "LOW RISK"
    elif pct < 65: return "MODERATE RISK"
    return "HIGH RISK"

def risk_bg(pct: float) -> str:
    if pct < 35:   return "#ecfdf5"
    elif pct < 65: return "#fffbeb"
    return "#fef2f2"

# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def page_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C_BLUE900} 0%,#1e40af 100%);
                border-radius:12px;padding:28px 32px;margin-bottom:28px;
                box-shadow:0 4px 20px rgba(26,47,110,0.25);">
        <div style="font-family:'Playfair Display',serif;font-size:26px;font-weight:700;
                    color:#ffffff;letter-spacing:-0.01em;">{title}</div>
        {"" if not subtitle else f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:13px;color:#93c5fd;margin-top:6px;font-weight:400;">{subtitle}</div>'}
    </div>""", unsafe_allow_html=True)


def card_open(title: str = "", accent: bool = False):
    border = f"2px solid {C_BLUE600}" if accent else f"1px solid {C_BORDER}"
    st.markdown(f"""
    <div style="background:{C_WHITE};border:{border};border-radius:10px;
                padding:20px 24px;margin-bottom:16px;
                box-shadow:0 1px 4px rgba(30,64,175,0.07);">
        {"" if not title else f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{C_BLUE600};margin-bottom:14px;">{title}</div>'}
    </div>""", unsafe_allow_html=True)


def kv_row(label: str, value: str, color: str = C_TEXT):
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:9px 0;border-bottom:1px solid {C_BORDER};">
        <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                     color:{C_MUTED};font-weight:500;">{label}</span>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                     font-weight:500;color:{color};">{value}</span>
    </div>""", unsafe_allow_html=True)


def section_divider(label: str):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:24px 0 18px 0;">
        <div style="width:4px;height:18px;background:{C_BLUE600};border-radius:2px;"></div>
        <span style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;font-weight:600;
                     color:{C_TEXT2};letter-spacing:0.02em;">{label}</span>
        <div style="flex:1;height:1px;background:{C_BORDER};"></div>
    </div>""", unsafe_allow_html=True)


def plotly_medical(fig, title="", height=360):
    fig.update_layout(
        title=dict(text=title, font=dict(family="IBM Plex Sans", size=13, color=C_TEXT2)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C_WHITE,
        font=dict(family="IBM Plex Sans", color=C_TEXT, size=11),
        xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, zerolinecolor=C_BORDER),
        yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, zerolinecolor=C_BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER),
        margin=dict(l=40, r=20, t=36 if title else 16, b=40),
        height=height,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None, "Model file not found."
    try:
        payload = joblib.load(path)
        if isinstance(payload, dict):
            return payload, None
        return {"model": payload, "scaler": None, "classes": payload.classes_}, None
    except Exception as e:
        return None, str(e)

# ══════════════════════════════════════════════════════════════════════════════
#  EEG PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def standardize_channels(raw):
    raw.rename_channels(lambda x: x.strip())
    raw.pick_types(eeg=True)
    missing = [ch for ch in TARGET_CHANNELS if ch not in raw.ch_names]
    if missing:
        info = mne.create_info(missing, raw.info["sfreq"], ch_types="eeg")
        zeros = np.zeros((len(missing), raw.n_times))
        raw.add_channels([mne.io.RawArray(zeros, info)], force_update_info=True)
    raw.pick_channels(TARGET_CHANNELS)
    raw.set_montage("standard_1020", on_missing="ignore")
    return raw


def load_eeg(filepath: str):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw = standardize_channels(raw)
    raw.resample(250)
    raw.notch_filter(50, method="iir")
    raw.filter(0.5, 45, method="iir")
    raw.set_eeg_reference("average")
    return raw


def create_epochs(raw, duration=1.0, overlap=0.5):
    events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    return mne.Epochs(raw, events, tmin=0, tmax=duration,
                      baseline=None, preload=True, verbose=False)


def extract_features(epochs):
    psd   = epochs.compute_psd(method="welch", fmin=0.5, fmax=45, verbose=False)
    psds  = psd.get_data()
    freqs = psd.freqs
    n_epochs, n_channels, _ = psds.shape
    features = []
    ch_band_accum = {ch: {b: [] for b in BANDS} for ch in TARGET_CHANNELS}

    for ep in range(n_epochs):
        ep_feat = []
        for ch_idx, ch in enumerate(TARGET_CHANNELS):
            signal      = psds[ep, ch_idx]
            total_power = simpson(signal, x=freqs)
            for band, (fmin, fmax) in BANDS.items():
                mask       = (freqs >= fmin) & (freqs <= fmax)
                band_power = simpson(signal[mask], x=freqs[mask])
                rel_power  = band_power / (total_power + 1e-10)
                ep_feat.extend([band_power, rel_power])
                ch_band_accum[ch][band].append(band_power)
        features.append(ep_feat)

    per_channel_band = {
        ch: {b: float(np.mean(ch_band_accum[ch][b])) for b in BANDS}
        for ch in TARGET_CHANNELS
    }
    return np.array(features).mean(axis=0), per_channel_band, psds, freqs

# ══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED CLINICAL SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def _ch_avg(pcb: dict, channels: list, band: str) -> float:
    vals = [pcb[ch][band] for ch in channels if ch in pcb]
    return float(np.mean(vals)) if vals else 0.0

def _rel(pcb: dict, channels: list, band: str) -> float:
    total = sum(_ch_avg(pcb, channels, b) for b in BANDS) + 1e-10
    return _ch_avg(pcb, channels, band) / total


def rule_based_score(per_ch_band: dict) -> dict:
    rules = {}

    # Rule 1: Frontal alpha suppression (hypofrontality)
    fa_rel = _rel(per_ch_band, FRONTAL_CH, "alpha")
    rules["Alpha Suppression (Frontal)"] = {
        "score":        float(np.clip(1.0 - fa_rel / 0.30, 0.0, 1.0)),
        "value":        f"{fa_rel:.3f}",
        "normal_range": "> 0.30",
        "finding":      "Hypofrontality — reduced frontal alpha power",
        "weight":       0.20,
    }

    # Rule 2: Theta/Alpha Ratio
    gt  = _ch_avg(per_ch_band, TARGET_CHANNELS, "theta")
    ga  = _ch_avg(per_ch_band, TARGET_CHANNELS, "alpha") + 1e-10
    tar = gt / ga
    rules["Theta / Alpha Ratio (TAR)"] = {
        "score":        float(np.clip((tar - 0.5) / 1.5, 0.0, 1.0)),
        "value":        f"{tar:.3f}",
        "normal_range": "0.40 – 0.70",
        "finding":      "Elevated TAR — cognitive slowing biomarker",
        "weight":       0.20,
    }

    # Rule 3: Frontal delta excess
    fd_rel = _rel(per_ch_band, FRONTAL_CH, "delta")
    rules["Frontal Delta Excess"] = {
        "score":        float(np.clip((fd_rel - 0.08) / 0.22, 0.0, 1.0)),
        "value":        f"{fd_rel:.3f}",
        "normal_range": "< 0.12",
        "finding":      "Excess frontal delta — cognitive disorganization",
        "weight":       0.15,
    }

    # Rule 4: Global slow-wave dominance
    slow = (_ch_avg(per_ch_band, TARGET_CHANNELS, "delta") +
            _ch_avg(per_ch_band, TARGET_CHANNELS, "theta"))
    fast = (_ch_avg(per_ch_band, TARGET_CHANNELS, "alpha") +
            _ch_avg(per_ch_band, TARGET_CHANNELS, "beta") + 1e-10)
    swd  = slow / fast
    rules["Slow-Wave Dominance Index"] = {
        "score":        float(np.clip((swd - 0.5) / 1.5, 0.0, 1.0)),
        "value":        f"{swd:.3f}",
        "normal_range": "0.40 – 0.60",
        "finding":      "Slow wave excess over fast oscillations",
        "weight":       0.15,
    }

    # Rule 5: Posterior alpha loss
    post_ch        = OCCIPITAL_CH + PARIETAL_CH
    post_alpha_rel = _rel(per_ch_band, post_ch, "alpha")
    rules["Posterior Alpha Loss"] = {
        "score":        float(np.clip(1.0 - post_alpha_rel / 0.40, 0.0, 1.0)),
        "value":        f"{post_alpha_rel:.3f}",
        "normal_range": "> 0.40",
        "finding":      "Reduced posterior alpha — impaired sensory gating",
        "weight":       0.15,
    }

    # Rule 6: Temporal hemispheric asymmetry
    t3_ta = (per_ch_band.get("T3",{}).get("theta",0) /
             (per_ch_band.get("T3",{}).get("alpha",1e-10)))
    t4_ta = (per_ch_band.get("T4",{}).get("theta",0) /
             (per_ch_band.get("T4",{}).get("alpha",1e-10)))
    asym  = abs(t3_ta - t4_ta) / (max(t3_ta, t4_ta) + 1e-10)
    rules["Temporal Asymmetry (T3 / T4)"] = {
        "score":        float(np.clip(asym / 0.5, 0.0, 1.0)),
        "value":        f"{asym:.3f}",
        "normal_range": "< 0.15",
        "finding":      "Left-right temporal theta/alpha asymmetry",
        "weight":       0.08,
    }

    # Rule 7: Gamma disruption
    gg_rel = _rel(per_ch_band, TARGET_CHANNELS, "gamma")
    rules["Gamma Disruption"] = {
        "score":        float(np.clip(1.0 - gg_rel / 0.04, 0.0, 1.0)),
        "value":        f"{gg_rel:.4f}",
        "normal_range": "> 0.040",
        "finding":      "Reduced gamma — impaired sensory binding",
        "weight":       0.04,
    }

    # Rule 8: Frontal beta anomaly
    fb_rel    = _rel(per_ch_band, FRONTAL_CH, "beta")
    beta_anom = (max(0.0, fb_rel - 0.30) / 0.20 +
                 max(0.0, 0.10 - fb_rel) / 0.10)
    rules["Beta Anomaly (Frontal)"] = {
        "score":        float(np.clip(beta_anom, 0.0, 1.0)),
        "value":        f"{fb_rel:.3f}",
        "normal_range": "0.12 – 0.30",
        "finding":      "Abnormal frontal beta — arousal dysregulation",
        "weight":       0.03,
    }

    total_w    = sum(r["weight"] for r in rules.values())
    rule_score = sum(r["score"] * r["weight"] for r in rules.values()) / total_w
    rule_pct   = float(np.clip(rule_score * 100.0, 0.0, 100.0))

    return {
        "rules":             rules,
        "rule_pct":          rule_pct,
        "tar":               tar,
        "swd":               swd,
        "frontal_alpha_rel": fa_rel,
        "post_alpha_rel":    post_alpha_rel,
    }


def ensemble_score(ml_pct: float, rule_pct: float, single_class: bool):
    if single_class:
        final  = 0.80 * rule_pct + 0.20 * ml_pct
        method = "Rule-Based (80%) + Heuristic (20%)"
    else:
        final  = 0.50 * ml_pct + 0.50 * rule_pct
        method = "ML Model (50%) + Rule-Based (50%)"
    return float(np.clip(final, 0.0, 100.0)), method

# ══════════════════════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(filepath: str, payload: dict):
    raw    = load_eeg(filepath)
    epochs = create_epochs(raw)
    feat_vec, per_ch_band, psds, freqs = extract_features(epochs)

    model  = payload["model"]
    scaler = payload.get("scaler")

    feat_input = feat_vec.reshape(1, -1)

    expected_dims = None
    if hasattr(model, "n_features_in_"):
        expected_dims = model.n_features_in_
    elif hasattr(model, "feature_importances_"):
        expected_dims = len(model.feature_importances_)

    actual_dims  = feat_input.shape[1]
    dim_mismatch = expected_dims is not None and actual_dims != expected_dims
    if dim_mismatch:
        if actual_dims < expected_dims:
            feat_input = np.pad(feat_input, ((0,0),(0, expected_dims - actual_dims)))
        else:
            feat_input = feat_input[:, :expected_dims]

    if scaler is not None:
        try:
            feat_input = scaler.transform(feat_input)
        except Exception:
            pass

    classes            = model.classes_
    single_class_model = len(classes) == 1
    pred               = model.predict(feat_input)[0]

    rb = rule_based_score(per_ch_band)

    if single_class_model:
        n_ch_feat = len(BANDS) * 2
        def avg_rel(bidx):
            idxs = [ci * n_ch_feat + bidx * 2 + 1 for ci in range(len(TARGET_CHANNELS))]
            vals = [feat_vec[i] for i in idxs if i < len(feat_vec)]
            return float(np.mean(vals)) if vals else 0.0
        sd     = (avg_rel(0) + avg_rel(1)) - avg_rel(2)
        ml_pct = float(100.0 / (1.0 + np.exp(-12.0 * (sd - 0.15))))
        heuristic_used = True
    elif hasattr(model, "predict_proba"):
        prob       = model.predict_proba(feat_input)[0]
        class_list = list(classes)
        idx1       = class_list.index(1) if 1 in class_list else len(prob) - 1
        ml_pct     = float(prob[idx1]) * 100.0
        heuristic_used = False
    else:
        ml_pct         = 100.0 if pred == 1 else 0.0
        heuristic_used = False

    risk_pct, ens_method = ensemble_score(ml_pct, rb["rule_pct"], single_class_model)
    final_pred = 1 if risk_pct >= 50 else 0

    return {
        "prediction":      final_pred,
        "risk_pct":        risk_pct,
        "ml_pct":          ml_pct,
        "rule_pct":        rb["rule_pct"],
        "ensemble_method": ens_method,
        "rules":           rb["rules"],
        "rb_metrics": {
            "tar":               rb["tar"],
            "swd":               rb["swd"],
            "frontal_alpha_rel": rb["frontal_alpha_rel"],
            "post_alpha_rel":    rb["post_alpha_rel"],
        },
        "feat_vec":        feat_vec,
        "per_ch_band":     per_ch_band,
        "psds":            psds,
        "freqs":           freqs,
        "n_channels":      len(TARGET_CHANNELS),
        "n_epochs":        len(epochs),
        "sfreq":           raw.info["sfreq"],
        "duration_s":      raw.times[-1],
        "classes":         classes,
        "feat_dims":       actual_dims,
        "model_dims":      expected_dims,
        "dim_mismatch":    dim_mismatch,
        "single_class":    single_class_model,
        "heuristic_used":  heuristic_used,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  DEMO DATA
# ══════════════════════════════════════════════════════════════════════════════
def generate_demo_result(schiz: bool = True):
    rng = np.random.default_rng(42 if schiz else 7)
    per_ch_band = {}
    for ch in TARGET_CHANNELS:
        frontal  = ch in FRONTAL_CH
        occipital = ch in (OCCIPITAL_CH + PARIETAL_CH)
        if schiz:
            per_ch_band[ch] = {
                "delta": rng.uniform(18, 28) if frontal else rng.uniform(10, 18),
                "theta": rng.uniform(14, 22),
                "alpha": rng.uniform(2, 6) if frontal else rng.uniform(5, 10),
                "beta":  rng.uniform(5, 10),
                "gamma": rng.uniform(1, 3),
            }
        else:
            per_ch_band[ch] = {
                "delta": rng.uniform(1.5, 4),
                "theta": rng.uniform(3, 6),
                "alpha": rng.uniform(22, 35) if occipital else rng.uniform(10, 18),
                "beta":  rng.uniform(3, 6),
                "gamma": rng.uniform(0.3, 1),
            }

    rb       = rule_based_score(per_ch_band)
    sd       = (rng.uniform(0.35,0.45) if schiz else rng.uniform(0.06,0.10)) - \
               (rng.uniform(0.09,0.15) if schiz else rng.uniform(0.38,0.50))
    ml_pct   = float(100.0 / (1.0 + np.exp(-12.0 * (sd - 0.15))))
    rp, meth = ensemble_score(ml_pct, rb["rule_pct"], True)

    freqs = np.linspace(0.5, 45, 200)
    return {
        "prediction":      1 if rp >= 50 else 0,
        "risk_pct":        rp,
        "ml_pct":          ml_pct,
        "rule_pct":        rb["rule_pct"],
        "ensemble_method": meth,
        "rules":           rb["rules"],
        "rb_metrics": {
            "tar":               rb["tar"],
            "swd":               rb["swd"],
            "frontal_alpha_rel": rb["frontal_alpha_rel"],
            "post_alpha_rel":    rb["post_alpha_rel"],
        },
        "per_ch_band":     per_ch_band,
        "freqs":           freqs,
        "n_channels":      19,
        "n_epochs":        48,
        "sfreq":           250.0,
        "duration_s":      120.0,
        "classes":         np.array([0, 1]),
        "psds":            None,
        "feat_vec":        rng.uniform(0, 1, 190),
        "single_class":    True,
        "heuristic_used":  True,
        "dim_mismatch":    False,
        "demo":            True,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def chart_risk_gauge(risk_pct: float) -> go.Figure:
    clr = risk_color(risk_pct)
    lbl = risk_label(risk_pct)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number=dict(suffix="%", font=dict(family="Playfair Display", size=38, color=clr)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color=C_MUTED, family="IBM Plex Sans", size=10),
                      tickcolor=C_BORDER, nticks=6),
            bar=dict(color=clr, thickness=0.24),
            bgcolor=C_WHITE,
            bordercolor=C_BORDER,
            borderwidth=1,
            steps=[
                dict(range=[0,  35], color="rgba(5,150,105,0.08)"),
                dict(range=[35, 65], color="rgba(217,119,6,0.08)"),
                dict(range=[65,100], color="rgba(220,38,38,0.08)"),
            ],
            threshold=dict(line=dict(color=clr, width=3), thickness=0.8, value=risk_pct),
        ),
    ))
    fig.add_annotation(text=lbl, x=0.5, y=0.18, showarrow=False,
                       font=dict(family="IBM Plex Sans", size=12, color=clr))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color=C_TEXT),
        margin=dict(l=20, r=20, t=10, b=10),
        height=240,
    )
    return fig


def chart_band_radar(per_ch_band: dict) -> go.Figure:
    avg   = {b: float(np.mean([per_ch_band[ch][b] for ch in TARGET_CHANNELS])) for b in BANDS}
    vals  = list(avg.values())
    norm  = [v / (max(vals) + 1e-10) for v in vals]
    names = list(avg.keys())
    norm.append(norm[0]); names.append(names[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm, theta=names, fill="toself",
        fillcolor=hex_alpha(C_BLUE500, 0.12),
        line=dict(color=C_BLUE600, width=2),
        name="Band Power",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=C_WHITE,
            radialaxis=dict(visible=True, showticklabels=False,
                            gridcolor=C_BORDER, linecolor=C_BORDER),
            angularaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER,
                             tickfont=dict(family="IBM Plex Sans", color=C_TEXT, size=12)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=50, r=50, t=20, b=20),
        height=280,
    )
    return fig


def chart_band_bars(per_ch_band: dict) -> go.Figure:
    avg = {b: float(np.mean([per_ch_band[ch][b] for ch in TARGET_CHANNELS])) for b in BANDS}
    colors = [C_BLUE900, C_BLUE700, C_BLUE600, C_BLUE500, "#60a5fa"]
    fig = go.Figure(go.Bar(
        x=[b.upper() for b in avg],
        y=list(avg.values()),
        marker_color=colors,
        marker_line=dict(color="rgba(0,0,0,0)"),
        text=[f"{v:.2f}" for v in avg.values()],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10, color=C_MUTED),
    ))
    plotly_medical(fig, "Average Band Power — All Channels", 300)
    fig.update_layout(showlegend=False)
    return fig


def chart_heatmap(per_ch_band: dict) -> go.Figure:
    bands = list(BANDS.keys())
    z = [[per_ch_band[ch][b] for b in bands] for ch in TARGET_CHANNELS]
    z_arr  = np.array(z)
    z_norm = (z_arr - z_arr.min(axis=0)) / (z_arr.max(axis=0) - z_arr.min(axis=0) + 1e-10)

    colorscale = [
        [0.0,  "#eff6ff"],
        [0.25, "#bfdbfe"],
        [0.5,  "#3b82f6"],
        [0.75, "#1d4ed8"],
        [1.0,  "#1a2f6e"],
    ]
    fig = go.Figure(go.Heatmap(
        z=z_norm,
        x=[b.upper() for b in bands],
        y=TARGET_CHANNELS,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(thickness=12, len=0.8,
                      tickfont=dict(family="IBM Plex Mono", color=C_MUTED, size=10),
                      outlinecolor=C_BORDER, outlinewidth=1),
        hovertemplate="Channel: %{y}<br>Band: %{x}<br>Power: %{z:.3f}<extra></extra>",
    ))
    plotly_medical(fig, height=460)
    fig.update_layout(plot_bgcolor=C_WHITE)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return fig


def chart_topomap(per_ch_band: dict, band: str) -> plt.Figure:
    values  = np.array([per_ch_band[ch][band] for ch in TARGET_CHANNELS])
    vn      = (values - values.min()) / ((values.max() - values.min()) + 1e-10)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_facecolor(C_WHITE)

    head = plt.Circle((0,0), 1.0, fill=False, color=C_BORDER, linewidth=2)
    ax.add_patch(head)
    ax.plot([0,0],   [1.0, 1.12], color=C_BORDER, linewidth=2)
    ax.plot([-1.0,-1.08],[0.05,0.05], color=C_BORDER, linewidth=2)
    ax.plot([ 1.0, 1.08],[0.05,0.05], color=C_BORDER, linewidth=2)

    cmap = LinearSegmentedColormap.from_list(
        "med", ["#eff6ff","#bfdbfe","#3b82f6","#1d4ed8","#1a2f6e"]
    )
    norm = plt.Normalize(0, 1)

    for ch, v in zip(TARGET_CHANNELS, vn):
        x, y = TOPO_POS[ch]
        for radius, alpha_val in [(0.11, 0.07), (0.07, 0.14), (0.045, 0.55)]:
            c   = cmap(v)
            glow = plt.Circle((x,y), radius, color=(*c[:3], alpha_val), zorder=3)
            ax.add_patch(glow)
        dot = plt.Circle((x,y), 0.038, color=cmap(v), zorder=5)
        ax.add_patch(dot)
        ax.text(x, y-0.11, ch, ha="center", va="top",
                fontsize=6.2, color=C_MUTED, fontfamily="monospace", zorder=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors=C_MUTED)
    cbar.outline.set_edgecolor(C_BORDER)

    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{band.upper()} Band", color=C_TEXT2,
                 fontsize=10, fontfamily="sans-serif", pad=8)
    fig.patch.set_edgecolor(C_BORDER)
    fig.tight_layout()
    return fig


def chart_psd(freqs: np.ndarray, psds: np.ndarray) -> go.Figure:
    avg_psd = psds.mean(axis=(0,1))
    colors  = [C_BLUE900, C_BLUE700, C_BLUE600, C_BLUE500, "#60a5fa"]
    fig = go.Figure()
    for i, (band, (fmin, fmax)) in enumerate(BANDS.items()):
        mask = (freqs >= fmin) & (freqs <= fmax)
        fig.add_trace(go.Scatter(
            x=freqs[mask], y=avg_psd[mask],
            name=band.upper(),
            line=dict(color=colors[i], width=2),
            fill="tozeroy",
            fillcolor=hex_alpha(colors[i], 0.10),
        ))
    plotly_medical(fig, "Power Spectral Density", 320)
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Power (uV²/Hz)")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 24px 0;border-bottom:1px solid rgba(255,255,255,0.12);margin-bottom:20px;">
        <div style="font-family:'Playfair Display',serif;font-size:20px;font-weight:700;
                    color:#ffffff;letter-spacing:-0.01em;">NeuroScan</div>
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;
                    color:#93c5fd;margin-top:4px;font-weight:400;letter-spacing:0.04em;">
            EEG SCHIZOPHRENIA DETECTION
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#93c5fd;margin-bottom:10px;">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Analysis", "Visualizations", "Research Pipeline", "Model Info"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12);margin:18px 0;'>", unsafe_allow_html=True)

    st.markdown(f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#93c5fd;margin-bottom:8px;">Model Path</div>', unsafe_allow_html=True)
    model_path = st.text_input("", value="./data/processed_aszed/eeg_model.pkl",
                               label_visibility="collapsed")

    payload, model_err = load_model(model_path)
    if payload:
        classes = payload.get("classes", [])
        is_single = len(classes) == 1
        dot_color = "#f59e0b" if is_single else "#34d399"
        status_txt = "Single-class" if is_single else "Ready"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
            <div style="width:8px;height:8px;background:{dot_color};border-radius:50%;"></div>
            <span style="font-family:IBM Plex Sans,sans-serif;font-size:12px;color:#c3d4f0;">{status_txt} · classes: {list(classes)}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:12px;color:#fca5a5;margin-top:4px;">Not found — demo mode active</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12);margin:18px 0;'>", unsafe_allow_html=True)

    st.markdown(f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#93c5fd;margin-bottom:8px;">Demo Mode</div>', unsafe_allow_html=True)
    use_demo = st.toggle("Use demo data", value=(payload is None))
    if use_demo:
        demo_schiz = st.toggle("Simulate schizophrenia", value=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12);margin:18px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;color:#93c5fd;line-height:1.8;">
        ASZED Dataset · 1,932 recordings<br>
        19 channels · 10-20 system<br>
        Random Forest · 300 trees<br>
        8 clinical biomarker rules<br>
        Ensemble scoring
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "result" not in st.session_state:
    st.session_state.result = None

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if page == "Analysis":
    page_header("EEG Analysis Interface",
                "Upload a patient EDF recording to generate a schizophrenia risk assessment report")

    # ── Upload + pipeline info ────────────────────────────────────────────────
    col_up, col_pipe = st.columns([2, 1], gap="large")

    with col_up:
        st.markdown(f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:12px;font-weight:600;color:{C_TEXT2};margin-bottom:8px;">Patient EDF File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["edf"], label_visibility="collapsed")
        run_btn  = st.button("Run Analysis", use_container_width=True)

    with col_pipe:
        st.markdown(f"""
        <div style="background:{C_BLUE50};border:1px solid {C_BLUE100};border-radius:10px;
                    padding:18px 20px;height:100%;">
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;font-weight:600;
                        letter-spacing:0.07em;text-transform:uppercase;color:{C_BLUE600};margin-bottom:12px;">
                Processing Pipeline
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{C_TEXT2};line-height:2.0;">
                01&nbsp; Load EDF / standardize channels<br>
                02&nbsp; Resample 250 Hz<br>
                03&nbsp; IIR notch 50 Hz + bandpass 0.5-45 Hz<br>
                04&nbsp; Average reference<br>
                05&nbsp; Epoch 1s / 50% overlap<br>
                06&nbsp; Welch PSD extraction<br>
                07&nbsp; 5-band power features<br>
                08&nbsp; Random Forest inference<br>
                09&nbsp; 8-rule clinical scoring<br>
                10&nbsp; Ensemble risk calculation
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    result = None

    if use_demo and (run_btn or st.session_state.result is None):
        with st.spinner("Generating demo data..."):
            result = generate_demo_result(demo_schiz)
            st.session_state.result = result

    elif run_btn and uploaded is not None:
        if not MNE_AVAILABLE:
            st.error("MNE-Python not installed. Run: pip install mne")
        elif payload is None:
            st.error("No model loaded. Check the model path in the sidebar.")
        else:
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Processing EEG recording..."):
                    prog = st.progress(0, text="Loading and filtering...")
                    result = run_pipeline(tmp_path, payload)
                    prog.progress(100, text="Analysis complete")
                    st.session_state.result = result
            except Exception as e:
                st.error(f"Processing failed: {e}")
            finally:
                os.unlink(tmp_path)

    elif run_btn and uploaded is None and not use_demo:
        st.warning("Please upload an EDF file or enable demo data in the sidebar.")

    result = st.session_state.result

    # ── Results ───────────────────────────────────────────────────────────────
    if result:
        demo_tag = " &nbsp;·&nbsp; DEMO DATA" if result.get("demo") else ""
        st.markdown(f"""
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;font-weight:600;
                    letter-spacing:0.07em;text-transform:uppercase;color:{C_BLUE600};
                    margin-bottom:18px;">Analysis Report{demo_tag}</div>""",
                    unsafe_allow_html=True)

        # ── Diagnostic warnings ───────────────────────────────────────────────
        if not result.get("demo"):
            if result.get("single_class"):
                st.markdown(f"""
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-left:4px solid {C_AMBER};
                            border-radius:8px;padding:12px 16px;margin-bottom:16px;
                            font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:{C_TEXT};">
                    <strong style="color:{C_AMBER};">Single-class model detected</strong> —
                    The model was trained on only one class {list(result["classes"])}.
                    Risk score uses the rule-based engine (80%) with heuristic fallback (20%).
                    To fix: correct <code>get_label()</code> and retrain the model.
                </div>""", unsafe_allow_html=True)
            if result.get("dim_mismatch"):
                st.markdown(f"""
                <div style="background:#fef2f2;border:1px solid #fca5a5;border-left:4px solid {C_RED};
                            border-radius:8px;padding:12px 16px;margin-bottom:16px;
                            font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:{C_TEXT};">
                    <strong style="color:{C_RED};">Feature dimension mismatch</strong> —
                    EDF produced {result.get("feat_dims")} features but model expects {result.get("model_dims")}.
                    Array was auto-padded/truncated. Results may be inaccurate.
                </div>""", unsafe_allow_html=True)

        # ── Three score cards ─────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3, gap="large")
        for col, lbl, pct, sub in [
            (c1, "Ensemble Risk Score", result["risk_pct"],
             result.get("ensemble_method","Rule + ML")),
            (c2, "ML Model Score", result.get("ml_pct", result["risk_pct"]),
             "Random Forest" if not result.get("heuristic_used") else "Heuristic fallback"),
            (c3, "Rule-Based Score", result.get("rule_pct", result["risk_pct"]),
             "8 clinical biomarkers"),
        ]:
            clr = risk_color(pct)
            bg  = risk_bg(pct)
            col.markdown(f"""
            <div style="background:{bg};border:1px solid {clr}44;border-top:4px solid {clr};
                        border-radius:10px;padding:20px 22px;text-align:center;
                        box-shadow:0 2px 8px {clr}18;">
                <div style="font-family:'IBM Plex Sans',sans-serif;font-size:10px;font-weight:600;
                            letter-spacing:0.08em;text-transform:uppercase;color:{clr};margin-bottom:8px;">{lbl}</div>
                <div style="font-family:'Playfair Display',serif;font-size:40px;font-weight:700;
                            color:{clr};line-height:1;">{pct:.1f}%</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{C_MUTED};
                            margin-top:6px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Gauge + Recording info + Interpretation ────────────────────────────
        col_g, col_s, col_i = st.columns([1.1, 1, 1.4], gap="large")

        with col_g:
            st.markdown(f"""
            <div style="background:{C_WHITE};border:1px solid {C_BORDER};border-radius:10px;
                        padding:16px 18px;box-shadow:0 1px 4px rgba(30,64,175,0.07);">
                <div style="font-family:'IBM Plex Sans',sans-serif;font-size:10px;font-weight:600;
                            letter-spacing:0.08em;text-transform:uppercase;color:{C_BLUE600};margin-bottom:4px;">
                    Risk Gauge
                </div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(chart_risk_gauge(result["risk_pct"]),
                            use_container_width=True, config={"displayModeBar": False})
            clr = risk_color(result["risk_pct"])
            bg  = risk_bg(result["risk_pct"])
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {clr}44;border-radius:6px;
                        padding:8px 14px;text-align:center;">
                <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                             font-weight:700;color:{clr};letter-spacing:0.06em;">
                    {risk_label(result["risk_pct"])}
                </span>
            </div>""", unsafe_allow_html=True)

        with col_s:
            st.markdown(f"""
            <div style="background:{C_WHITE};border:1px solid {C_BORDER};border-radius:10px;
                        padding:20px 22px;box-shadow:0 1px 4px rgba(30,64,175,0.07);">
                <div style="font-family:'IBM Plex Sans',sans-serif;font-size:10px;font-weight:600;
                            letter-spacing:0.08em;text-transform:uppercase;color:{C_BLUE600};margin-bottom:12px;">
                    Recording Details
                </div>
            </div>""", unsafe_allow_html=True)
            kv_row("Channels",    str(result["n_channels"]))
            kv_row("Epochs",      str(result["n_epochs"]))
            kv_row("Sample Rate", f'{result["sfreq"]:.0f} Hz')
            kv_row("Duration",    f'{result["duration_s"]:.1f} s')
            pred_lbl  = "Schizophrenia" if result["prediction"] == 1 else "Control"
            pred_clr  = C_RED if result["prediction"] == 1 else C_GREEN
            kv_row("Prediction", pred_lbl, pred_clr)

            rb = result.get("rb_metrics", {})
            if rb:
                st.markdown(f'<div style="height:10px"></div>', unsafe_allow_html=True)
                tar = rb.get("tar", 0)
                swd = rb.get("swd", 0)
                fa  = rb.get("frontal_alpha_rel", 0)
                pa  = rb.get("post_alpha_rel", 0)
                kv_row("TAR (Theta/Alpha)",     f"{tar:.3f}",
                       C_RED if tar > 1.0 else C_GREEN)
                kv_row("Slow-Wave Dominance",   f"{swd:.3f}",
                       C_RED if swd > 1.0 else C_GREEN)
                kv_row("Frontal Alpha (rel)",   f"{fa:.3f}",
                       C_RED if fa < 0.18 else C_GREEN)
                kv_row("Posterior Alpha (rel)", f"{pa:.3f}",
                       C_RED if pa < 0.25 else C_GREEN)

        with col_i:
            risk = result["risk_pct"]
            if risk < 35:
                ititle = "Low Schizophrenia Risk"
                icolor = C_GREEN; ibg = "#ecfdf5"
                ibody  = "EEG patterns fall within the expected range for a healthy individual. Alpha dominance is preserved and slow-wave activity is not elevated. No significant biomarkers associated with schizophrenia were detected. Routine follow-up is recommended."
            elif risk < 65:
                ititle = "Moderate Schizophrenia Risk"
                icolor = C_AMBER; ibg = "#fffbeb"
                ibody  = "EEG patterns show atypical features that warrant clinical attention. Partial frontal alpha suppression and elevated theta activity have been detected. A comprehensive neuropsychiatric evaluation is recommended."
            else:
                ititle = "High Schizophrenia Risk"
                icolor = C_RED; ibg = "#fef2f2"
                ibody  = "EEG patterns exhibit strong biomarkers associated with schizophrenia: elevated delta/theta, suppressed frontal and posterior alpha, abnormal TAR, and temporal asymmetry. Urgent clinical referral and comprehensive neuropsychiatric assessment are advised."

            st.markdown(f"""
            <div style="background:{ibg};border:1px solid {icolor}44;
                        border-left:4px solid {icolor};border-radius:10px;
                        padding:20px 22px;">
                <div style="font-family:'IBM Plex Sans',sans-serif;font-size:10px;font-weight:600;
                            letter-spacing:0.07em;text-transform:uppercase;color:{icolor};margin-bottom:8px;">
                    Clinical Interpretation
                </div>
                <div style="font-family:'Playfair Display',serif;font-size:16px;font-weight:600;
                            color:{icolor};margin-bottom:12px;">{ititle}</div>
                <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;
                            color:{C_TEXT};line-height:1.75;">{ibody}</div>
            </div>""", unsafe_allow_html=True)

            # Rule-by-rule breakdown
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            section_divider("Biomarker Breakdown")

            for rname, rd in result.get("rules", {}).items():
                sc  = rd["score"]
                clr = risk_color(sc * 100)
                bw  = int(sc * 100)
                st.markdown(f"""
                <div style="margin-bottom:11px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;
                                     font-weight:500;color:{C_TEXT};">{rname}</span>
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{C_MUTED};">
                            {rd['value']} &nbsp;|&nbsp; ref: {rd['normal_range']} &nbsp;|&nbsp; w:{int(rd['weight']*100)}%
                        </span>
                    </div>
                    <div style="background:{C_BORDER};border-radius:3px;height:6px;overflow:hidden;">
                        <div style="width:{bw}%;background:{clr};height:6px;border-radius:3px;"></div>
                    </div>
                    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:10px;
                                color:{C_MUTED};margin-top:2px;">{rd['finding']}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:10px;color:{C_MUTED2};
                        border-top:1px solid {C_BORDER};padding-top:10px;margin-top:6px;">
                For research and screening purposes only. Not a clinical diagnosis.
                Consult a licensed neurologist for interpretation.
            </div>""", unsafe_allow_html=True)

        # ── Band charts ───────────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        section_divider("Frequency Band Analysis")
        col_r, col_b = st.columns([1, 1.6], gap="large")
        with col_r:
            st.markdown(f'<div style="background:{C_WHITE};border:1px solid {C_BORDER};border-radius:10px;padding:16px 18px;">', unsafe_allow_html=True)
            st.plotly_chart(chart_band_radar(result["per_ch_band"]),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div style="background:{C_WHITE};border:1px solid {C_BORDER};border-radius:10px;padding:16px 18px;">', unsafe_allow_html=True)
            st.plotly_chart(chart_band_bars(result["per_ch_band"]),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Visualizations":
    page_header("Brain Activity Visualizations",
                "Explainable spatial and spectral decomposition of EEG signals")

    result = st.session_state.result
    if result is None:
        if use_demo:
            result = generate_demo_result(True)
            st.session_state.result = result
        else:
            st.info("Run an analysis first on the Analysis page, or enable demo data in the sidebar.")
            st.stop()

    tab_topo, tab_heat, tab_psd = st.tabs([
        "Topographic Maps", "Channel Heatmap", "PSD Spectrum"
    ])

    # ── Topographic maps ──────────────────────────────────────────────────────
    with tab_topo:
        st.markdown(f"""
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:{C_MUTED};
                    margin-bottom:20px;line-height:1.7;">
            Topographic maps show normalised band power projected across the 19 scalp electrodes.
            Darker blue indicates elevated activity at that scalp location.
        </div>""", unsafe_allow_html=True)

        cols = st.columns(5, gap="small")
        for i, band in enumerate(BANDS):
            with cols[i]:
                fig = chart_topomap(result["per_ch_band"], band)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        st.markdown("<hr>", unsafe_allow_html=True)
        section_divider("Key Electrodes in Schizophrenia Research")
        eoi = pd.DataFrame({
            "Electrode": ["Fz","Cz","T3","T4","Fp1","Fp2"],
            "Region":    ["Frontal Midline","Central Midline","Left Temporal",
                          "Right Temporal","Left Prefrontal","Right Prefrontal"],
            "Clinical Relevance": [
                "Impaired executive function, reduced frontal alpha",
                "Sensorimotor gating deficits, mismatch negativity",
                "Auditory processing — hallucination correlates",
                "Auditory processing — hemispheric asymmetry",
                "Working memory deficits, hypofrontality",
                "Working memory deficits, hypofrontality",
            ],
        })
        st.dataframe(eoi, use_container_width=True, hide_index=True)

    # ── Channel heatmap ───────────────────────────────────────────────────────
    with tab_heat:
        st.markdown(f"""
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:{C_MUTED};
                    margin-bottom:20px;line-height:1.7;">
            Each cell represents the normalised band power for a channel–band pair.
            Darker blue indicates higher relative power.
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(chart_heatmap(result["per_ch_band"]),
                        use_container_width=True, config={"displayModeBar": False})

    # ── PSD spectrum ──────────────────────────────────────────────────────────
    with tab_psd:
        if result.get("psds") is not None:
            st.plotly_chart(chart_psd(result["freqs"], result["psds"]),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            freqs = result["freqs"]
            rng   = np.random.default_rng(42)
            psd_d = (2.5 / (freqs + 1)**1.2 +
                     rng.uniform(0, 0.04, len(freqs)) +
                     0.3 * np.exp(-0.5 * ((freqs - 10) / 2)**2))
            colors = [C_BLUE900, C_BLUE700, C_BLUE600, C_BLUE500, "#60a5fa"]
            fig = go.Figure()
            for i, (band, (fmin, fmax)) in enumerate(BANDS.items()):
                mask = (freqs >= fmin) & (freqs <= fmax)
                fig.add_trace(go.Scatter(
                    x=freqs[mask], y=psd_d[mask],
                    name=band.upper(),
                    line=dict(color=colors[i], width=2),
                    fill="tozeroy",
                    fillcolor=hex_alpha(colors[i], 0.10),
                ))
            plotly_medical(fig, "Power Spectral Density (Demo)", 320)
            fig.update_xaxes(title_text="Frequency (Hz)")
            fig.update_yaxes(title_text="Power (uV²/Hz)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<hr>", unsafe_allow_html=True)
        section_divider("Frequency Band Clinical Reference")
        band_ref = pd.DataFrame({
            "Band":       ["Delta","Theta","Alpha","Beta","Gamma"],
            "Range (Hz)": ["0.5 – 4","4 – 8","8 – 13","13 – 30","30 – 45"],
            "In Schizophrenia": [
                "Often elevated — cognitive disorganization",
                "Often elevated — positive symptom correlate",
                "Reduced — hypofrontality marker",
                "Variable — arousal dysregulation",
                "Disrupted — impaired sensory binding",
            ],
        })
        st.dataframe(band_ref, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RESEARCH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Research Pipeline":
    page_header("Research Dataset Pipeline",
                "Batch process ASZED EDF recordings and export results")

    col_cfg, col_info = st.columns([1.5, 1], gap="large")

    with col_cfg:
        section_divider("Batch Configuration")
        data_root     = st.text_input("Dataset root directory",
                                      value=r"C:\Users\Dell\Desktop\EEG_detection\data\ASZED")
        output_dir    = st.text_input("Output directory", value="./data/processed_aszed")
        label_index   = st.number_input("Label folder index (from filename)",
                                        min_value=-5, max_value=-1, value=-2, step=1)
        label_map_str = st.text_input("Label mapping (folder:class, comma separated)",
                                      value="1:1, 2:0")
        c_dry, c_csv = st.columns(2)
        dry_run  = c_dry.toggle("Dry run", value=True)
        save_csv = c_csv.toggle("Save CSV", value=True)
        run_batch = st.button("Start Batch Processing", use_container_width=True)

    with col_info:
        st.markdown(f"""
        <div style="background:{C_BLUE50};border:1px solid {C_BLUE100};border-radius:10px;
                    padding:20px 22px;">
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:10px;font-weight:600;
                        letter-spacing:0.07em;text-transform:uppercase;color:{C_BLUE600};margin-bottom:12px;">
                ASZED Dataset
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{C_TEXT2};line-height:2.0;">
                Files: 1,932 EDF recordings<br>
                Structure: node / subset / subject / session<br>
                Format: European Data Format (.edf)<br>
                Channels: 19-ch 10-20 system<br>
                Labels: encoded in folder hierarchy
            </div>
            <div style="margin-top:14px;font-family:'IBM Plex Sans',sans-serif;font-size:10px;
                        font-weight:600;letter-spacing:0.07em;text-transform:uppercase;
                        color:{C_BLUE600};margin-bottom:8px;">Path Structure</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{C_MUTED};
                        background:{C_WHITE};border-radius:6px;padding:10px;line-height:1.9;">
                ASZED/version_1.1/<br>
                &nbsp;&nbsp;node_1/subset_1/<br>
                &nbsp;&nbsp;&nbsp;&nbsp;subject_10/<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1/ ← label folder<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Phase 1.edf
            </div>
        </div>""", unsafe_allow_html=True)

    if run_batch:
        try:
            label_map = {k.strip(): int(v.strip())
                         for pair in label_map_str.split(",")
                         for k, v in [pair.split(":")]}
        except Exception:
            st.error("Invalid label mapping. Use format: 1:1, 2:0")
            st.stop()

        if not os.path.exists(data_root):
            st.info("Directory not found — showing simulated results for demonstration.")
            rng = np.random.default_rng(0)
            records = []
            for i in range(20):
                lbl  = int(rng.integers(0, 2))
                risk = float(rng.uniform(65,90) if lbl==1 else rng.uniform(8,35))
                records.append({"file": f"subject_{10+i}/Phase {rng.integers(1,5)}.edf",
                                 "label": lbl, "risk_pct": round(risk,2),
                                 "prediction": int(risk>50),
                                 "correct": int(lbl==int(risk>50))})
            df = pd.DataFrame(records)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Files", len(df))
            c2.metric("Schizophrenia", int(df["label"].sum()))
            c3.metric("Control", int((df["label"]==0).sum()))
            c4.metric("Accuracy", f"{df['correct'].mean():.1%}")
            st.dataframe(df, use_container_width=True, hide_index=True)
            fig_d = px.histogram(df, x="risk_pct",
                                 color=df["label"].map({0:"Control",1:"Schizophrenia"}),
                                 nbins=10, barmode="overlay",
                                 color_discrete_map={"Control":C_BLUE500,"Schizophrenia":C_RED})
            plotly_medical(fig_d, "Risk Score Distribution", 300)
            st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar":False})
            if save_csv:
                st.download_button("Download Results CSV", df.to_csv(index=False),
                                   "batch_results.csv", "text/csv")
        else:
            edf_files = [os.path.join(r, f)
                         for r, _, fs in os.walk(data_root)
                         for f in fs if f.lower().endswith(".edf")]
            st.info(f"Found {len(edf_files)} EDF files.")
            os.makedirs(output_dir, exist_ok=True)
            records, prog = [], st.progress(0, text="Processing...")
            status = st.empty()
            for idx, fp in enumerate(edf_files):
                parts  = fp.replace("\\","/").split("/")
                folder = parts[label_index] if abs(label_index) <= len(parts) else None
                label  = label_map.get(folder, -1)
                if label == -1:
                    continue
                rec = {"file": fp, "label": label}
                if not dry_run and payload:
                    try:
                        r = run_pipeline(fp, payload)
                        rec.update({"risk_pct": round(r["risk_pct"],2),
                                    "prediction": int(r["prediction"]),
                                    "correct": int(label == r["prediction"])})
                    except Exception as e:
                        rec["error"] = str(e)
                records.append(rec)
                prog.progress((idx+1)/len(edf_files), text=f"{idx+1}/{len(edf_files)}")
                status.text(os.path.basename(fp))
            prog.empty(); status.empty()
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True, hide_index=True)
            if save_csv:
                path = os.path.join(output_dir, "batch_results.csv")
                df.to_csv(path, index=False)
                st.success(f"Saved to {path}")
                st.download_button("Download Results CSV", df.to_csv(index=False),
                                   "batch_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Info":
    page_header("Model Information",
                "Architecture, clinical context, and system configuration")

    col_l, col_r = st.columns([1.1, 1], gap="large")

    with col_l:
        section_divider("Model Card")
        kv_row("Architecture",   "Random Forest Classifier")
        kv_row("Estimators",     "300 trees")
        kv_row("Class weights",  "balanced")
        kv_row("Random seed",    "42")
        kv_row("Input features", "190 (19 ch × 5 bands × 2 stats)")
        kv_row("Feature types",  "Absolute + relative band power")
        kv_row("Epoch length",   "1 second")
        kv_row("Epoch overlap",  "50%")
        kv_row("Sample rate",    "250 Hz")
        kv_row("PSD method",     "Welch")
        kv_row("Reference",      "Average")
        kv_row("Notch filter",   "50 Hz IIR")
        kv_row("Bandpass",       "0.5 – 45 Hz IIR")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        section_divider("Rule Engine — 8 Clinical Biomarkers")
        rules_info = [
            ("Alpha Suppression (Frontal)", "20%", "Frontal alpha rel. power < 0.30"),
            ("Theta/Alpha Ratio (TAR)",     "20%", "TAR outside 0.40–0.70"),
            ("Frontal Delta Excess",        "15%", "Frontal delta rel. power > 0.12"),
            ("Slow-Wave Dominance Index",   "15%", "(Delta+Theta)/(Alpha+Beta) > 0.60"),
            ("Posterior Alpha Loss",        "15%", "Posterior alpha rel. power < 0.40"),
            ("Temporal Asymmetry T3/T4",    "8%",  "Left-right TAR asymmetry > 0.15"),
            ("Gamma Disruption",            "4%",  "Global gamma rel. power < 0.04"),
            ("Beta Anomaly (Frontal)",      "3%",  "Frontal beta outside 0.12–0.30"),
        ]
        for name, wt, desc in rules_info:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:flex-start;
                        padding:9px 0;border-bottom:1px solid {C_BORDER};">
                <div>
                    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                                font-weight:500;color:{C_TEXT};">{name}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                                color:{C_MUTED};margin-top:2px;">{desc}</div>
                </div>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                             font-weight:600;color:{C_BLUE600};white-space:nowrap;margin-left:12px;">{wt}</span>
            </div>""", unsafe_allow_html=True)

        if payload:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            section_divider("Loaded Model Status")
            model_obj = payload.get("model")
            is_single = len(payload.get("classes",[])) == 1
            kv_row("Status",      "Loaded", C_GREEN)
            kv_row("Classes",     str(list(payload.get("classes",[]))),
                   C_AMBER if is_single else C_GREEN)
            kv_row("Has scaler",  "Yes" if payload.get("scaler") else "No")
            if hasattr(model_obj, "n_estimators"):
                kv_row("n_estimators", str(model_obj.n_estimators))
            if hasattr(model_obj, "feature_importances_"):
                kv_row("Feature dims", str(len(model_obj.feature_importances_)))

    with col_r:
        st.markdown(f"""
        <div style="background:{C_BLUE50};border:1px solid {C_BLUE100};
                    border-left:4px solid {C_BLUE600};border-radius:10px;
                    padding:22px 24px;margin-bottom:16px;">
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:10px;font-weight:600;
                        letter-spacing:0.07em;text-transform:uppercase;color:{C_BLUE600};margin-bottom:12px;">
                Clinical Context
            </div>
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;
                        color:{C_TEXT};line-height:1.85;">
                Schizophrenia affects approximately <strong>1% of the global population</strong>
                and typically has a diagnostic delay of 1–2 years due to reliance on
                behavioural observation alone.<br><br>
                EEG biomarkers offer a low-cost, non-invasive alternative for early screening.
                Key markers include:<br><br>
                <strong>Elevated delta and theta power</strong> — cognitive disorganization<br>
                <strong>Reduced alpha coherence</strong> — hypofrontality marker<br>
                <strong>Disrupted gamma oscillations</strong> — impaired sensory binding<br>
                <strong>Mismatch negativity deficits</strong> — auditory processing impairment<br>
                <strong>Temporal hemispheric asymmetry</strong> — hallucination correlate
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#fef2f2;border:1px solid #fca5a5;
                    border-left:4px solid {C_RED};border-radius:10px;
                    padding:22px 24px;margin-bottom:16px;">
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:10px;font-weight:600;
                        letter-spacing:0.07em;text-transform:uppercase;color:{C_RED};margin-bottom:12px;">
                Important Limitations
            </div>
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                        color:{C_TEXT};line-height:1.9;">
                This system is a <strong>research prototype</strong>, not a certified medical device.<br>
                — Label quality depends on correct ASZED path labeling<br>
                — Zero-padded missing channels may affect accuracy<br>
                — Cross-dataset generalisation has not been validated<br>
                — Single-class model falls back to rule-based scoring<br>
                — All outputs require clinical interpretation by a specialist
            </div>
        </div>""", unsafe_allow_html=True)

        section_divider("Dependencies")
        deps = [("streamlit","UI framework"),("mne","EEG processing"),
                ("scikit-learn","Random Forest, metrics"),("scipy","PSD, filtering"),
                ("numpy","Numerical arrays"),("pandas","Tabular data"),
                ("plotly","Interactive charts"),("matplotlib","Topographic maps"),
                ("joblib","Model serialization")]
        for lib, purpose in deps:
            kv_row(lib, purpose)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        reqs = "\n".join(["streamlit>=1.35.0","mne>=1.7.0","scikit-learn>=1.4.0",
                          "scipy>=1.13.0","numpy>=1.26.0","pandas>=2.2.0",
                          "plotly>=5.20.0","matplotlib>=3.9.0","joblib>=1.4.0"])
        st.download_button("Download requirements.txt", data=reqs,
                           file_name="requirements.txt", mime="text/plain")