"""Predictive maintenance dashboard.

Streamlit app that scores machine sensor readings with a trained Random Forest,
asks an LLM (Groq / Llama 3.1) for a short diagnosis and forwards the alert to an
n8n webhook so the operator gets notified.

Configuration is read from environment variables or `.streamlit/secrets.toml`:
    GROQ_API_KEY      - Groq API key (required for the AI diagnosis)
    N8N_WEBHOOK_URL   - n8n webhook URL (optional; notifications are skipped if unset)
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "failure_model.pkl"


def get_setting(name: str) -> str | None:
    """Read a setting from env vars first, then from Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets.toml present
        return None


GROQ_API_KEY = get_setting("GROQ_API_KEY")
N8N_WEBHOOK_URL = get_setting("N8N_WEBHOOK_URL")

st.set_page_config(page_title="Control Predictiu | Cartonajes Ribas", layout="wide")

st.markdown("""
    <style>
    /* App background */
    .stApp { background-color: #f8fafc; }

    /* Sliders: light grey track, green fill and handle */
    div[data-baseweb="slider"] { gap: 0; }
    div[data-baseweb="slider"] > div:first-child { background-color: #e2e8f0 !important; }
    div[data-baseweb="slider"] > div:first-child > div { background-color: #2D5A27 !important; }

    /* Corporate green button */
    .stButton>button {
        background-color: #2D5A27 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-weight: bold !important;
    }
    .stButton>button:hover { background-color: #1e3d1a !important; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


model = load_model()

# Simulated sensor history so the telemetry chart has some context
if "hist_torque" not in st.session_state:
    st.session_state.hist_torque = np.random.normal(40, 1, 50).tolist()
    st.session_state.hist_temp = np.random.normal(38, 0.5, 50).tolist()

# --- HEADER ---
col_logo, col_titol = st.columns([1, 6])
with col_logo:
    st.markdown("<div style='font-size: 52px; line-height: 1;'>🏭</div>", unsafe_allow_html=True)
with col_titol:
    st.markdown("<h2 style='margin-bottom: 0px; padding-bottom: 0px;'>Panell de Control Predictiu</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #2D5A27; font-weight: 600;'>Cartonajes Ribas</p>", unsafe_allow_html=True)

st.markdown("---")

# --- SIDEBAR: sensor inputs ---
with st.sidebar:
    st.title("Panell de Control")
    st.markdown("---")
    st.subheader("🎛️ Sensors Màquina Plegadora")

    tipus = st.selectbox("Tipus de Màquina", ["L", "M", "H"])
    temp_proces = st.slider("Temp. Procés [°C]", 30.0, 70.0, 38.0)
    rpm = st.slider("Velocitat Rotació [rpm]", 1000, 3000, 2800)
    torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0)

    with st.expander("Sensors Secundaris"):
        temp_aire = st.slider("Temp. Aire [°C]", 20.0, 40.0, 25.0)
        desgast = st.slider("Desgast Eina [min]", 0, 300, 130)

# --- PREDICTION ---
input_df = pd.DataFrame({
    "Type": [tipus],
    "Air temperature [°C]": [temp_aire],
    "Process temperature [°C]": [temp_proces],
    "Rotational speed [rpm]": [rpm],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [desgast],
})
prob = model.predict_proba(input_df)[0][1]

# --- MAIN LAYOUT ---
c1, c2 = st.columns([1, 1.2])

with c1:
    with st.container(border=True):
        st.subheader("Estat Operatiu")

        if prob < 0.3:
            st.success(f"✅ **SISTEMA ESTABLE** (Risc: {prob:.1%})")
        elif prob < 0.7:
            st.warning(f"⚠️ **REVISIÓ RECOMANADA** (Risc: {prob:.1%})")
        else:
            st.error(f"🚨 **ALERTA CRÍTICA** (Risc: {prob:.1%})")

        st.write("")

        if st.button("Generar Diagnòstic i Notificar", use_container_width=True):
            client = get_groq_client()
            if client is None:
                st.error("Falta configurar GROQ_API_KEY (vegeu el README).")
            else:
                diag = None
                with st.spinner("Analitzant amb Groq..."):
                    try:
                        prompt = (
                            f"La màquina Plegadora té Torque {torque}Nm i Temp {temp_proces}C. "
                            f"Risc de fallada: {prob*100:.1f}%. Fes un diagnòstic de 2 frases en català."
                        )
                        res = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="llama-3.1-8b-instant",
                        )
                        diag = res.choices[0].message.content
                        st.info(f"**Assistent IA:** {diag}")
                    except Exception as e:
                        st.error(f"Error de l'API de Groq: {e}")

                # Send the alert to n8n (optional)
                if diag and N8N_WEBHOOK_URL:
                    try:
                        requests.post(
                            N8N_WEBHOOK_URL,
                            json={
                                "maquina_id": "PLEGADORA",
                                "missatge": diag,
                                "probabilitat": f"{prob:.1%}",
                                "alarma": "CRÍTICA" if prob > 0.7 else "INFO",
                            },
                            timeout=10,
                        ).raise_for_status()
                        st.toast("Notificació enviada a l'operari!", icon="📩")
                    except requests.RequestException as e:
                        st.warning(f"No s'ha pogut enviar la notificació a n8n: {e}")
                elif diag:
                    st.caption("N8N_WEBHOOK_URL no configurat: notificació no enviada.")

with c2:
    with st.container(border=True):
        st.subheader("Telemetria en Temps Real")

        t_data = st.session_state.hist_torque.copy()
        p_data = st.session_state.hist_temp.copy()
        t_data.append(torque)
        p_data.append(temp_proces)

        chart_df = pd.DataFrame({
            "Torque (Nm)": t_data[-50:],
            "Temp (°C)": p_data[-50:],
        })
        st.line_chart(chart_df, height=250)
