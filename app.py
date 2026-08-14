import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Styles ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&family=Nunito:wght@400;500;600;700&display=swap');

    :root {
        --pink: #E91E63;
        --pink-deep: #C2185B;
        --pink-deeper: #AD1457;
        --pink-soft: #F8BBD0;
        --pink-mist: #FCE4EC;
        --ink: #4A2C34;
        --muted: #8A6A73;
        --green: #43A047;
        --green-deep: #2E7D32;
        --red: #E53935;
    }

    * { font-family: 'Nunito', sans-serif; }
    h1, h2, h3, .display { font-family: 'Quicksand', sans-serif; }

    .stApp {
        background: linear-gradient(160deg, #FFF5F8 0%, #FCE4EC 100%);
    }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1000px;
    }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #E91E63 0%, #AD1457 100%);
        padding: 2.2rem 2rem 2rem;
        border-radius: 22px;
        text-align: center;
        color: #fff;
        box-shadow: 0 14px 40px rgba(173, 20, 87, 0.28);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero p {
        margin: 0.55rem 0 0;
        font-size: 1.05rem;
        font-weight: 500;
        opacity: 0.94;
    }

    /* Illustration band */
    .illo {
        display: flex;
        justify-content: center;
        margin: 0.4rem 0 0.2rem;
    }
    .illo img { width: 100%; max-width: 520px; height: auto; }

    /* Section heading */
    .section-title {
        font-family: 'Quicksand', sans-serif;
        color: var(--pink-deep);
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0.2rem 0 0.2rem;
    }
    .section-sub {
        color: var(--muted);
        font-size: 0.95rem;
        margin: 0 0 0.4rem;
    }

    /* Inputs */
    .stNumberInput label {
        color: var(--ink) !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    .stNumberInput input {
        border: 1.5px solid var(--pink-soft) !important;
        border-radius: 10px !important;
        color: var(--ink) !important;
        background: #fff !important;
    }
    .stNumberInput input:focus {
        border-color: var(--pink) !important;
        box-shadow: 0 0 0 3px rgba(233, 30, 99, 0.12) !important;
    }
    .stNumberInput button {
        background: var(--pink-mist) !important;
        border-color: var(--pink-soft) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #E91E63 0%, #C2185B 100%) !important;
        color: #fff !important;
        font-family: 'Quicksand', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        padding: 0.75rem 3rem !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 8px 24px rgba(233, 30, 99, 0.32) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        width: 100% !important;
        margin-top: 0.6rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(233, 30, 99, 0.42) !important;
    }

    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 18px;
        text-align: center;
        margin-top: 1.6rem;
        box-shadow: 0 10px 34px rgba(0, 0, 0, 0.08);
        animation: slideUp 0.45s ease-out;
    }
    .benign-card { background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); border: 2px solid var(--green); }
    .malignant-card { background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%); border: 2px solid var(--red); }

    .result-badge {
        display: inline-block;
        font-family: 'Quicksand', sans-serif;
        padding: 0.7rem 2rem;
        border-radius: 40px;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
    }
    .benign-badge { background: var(--green); color: #fff; }
    .malignant-badge { background: var(--red); color: #fff; }

    .confidence-text {
        font-size: 1.15rem;
        color: var(--ink);
        margin: 0.6rem 0;
        font-weight: 700;
    }

    .progress-container {
        background: rgba(255,255,255,0.7);
        border-radius: 20px;
        height: 26px;
        margin: 1.1rem auto;
        overflow: hidden;
        max-width: 380px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.08);
    }
    .progress-bar {
        height: 100%;
        border-radius: 20px;
        display: flex; align-items: center; justify-content: center;
        color: #fff; font-weight: 700; font-size: 0.9rem;
    }

    .recommendation {
        background: rgba(255,255,255,0.85);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-top: 1rem;
        font-size: 0.98rem;
        color: var(--ink);
        text-align: left;
        line-height: 1.5;
        border-left: 4px solid var(--pink);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.6rem;
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 2.4rem;
        background: rgba(255,255,255,0.7);
        border-radius: 16px;
        line-height: 1.55;
    }
    .footer .cause { color: var(--pink-deep); font-weight: 700; margin-top: 0.7rem; }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.55rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    """Load the pickled model and scaler."""
    try:
        with open("model.h5", "rb") as f:
            saved_data = pickle.load(f)
            model = saved_data["model"]
            scaler = saved_data.get("scaler")
        return model, scaler
    except FileNotFoundError:
        st.error("Model file 'model.h5' was not found. Add it to the app directory and reload.")
        st.stop()
    except Exception as e:
        st.error(f"The model could not be loaded: {str(e)}")
        st.stop()

model, scaler = load_model()

# ---------------- Header + illustration ----------------
st.markdown("""
<div class="hero">
    <h1>🎗️ Breast Cancer Prediction System</h1>
    <p>Early detection saves lives — enter cell-nucleus measurements to estimate the diagnosis</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="illo"><img src="data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgNjQwIDI2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icmliIiB4MT0iMCIgeTE9IjAiIHgyPSIxIiB5Mj0iMSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iI0YwNjI5MiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiNDMjE4NUIiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgoKICA8Y2lyY2xlIGN4PSIxNjAiIGN5PSIxMzIiIHI9Ijk4IiBmaWxsPSIjRjhCQkQwIiBvcGFjaXR5PSIwLjQ1Ii8+CiAgPGNpcmNsZSBjeD0iNDc1IiBjeT0iMTIwIiByPSI4MCIgZmlsbD0iI0Y0OEZCMSIgb3BhY2l0eT0iMC4yOCIvPgoKICA8cG9seWxpbmUgcG9pbnRzPSI0MCwxOTIgMTUwLDE5MiAxODIsMTkyIDIwMiwxNTIgMjI3LDIyNiAyNTcsMTIwIDI4NywxOTIgMzYwLDE5MiA0MjAsMTkyIDQ1MCwxNjIgNDgwLDIwNiA1MTIsMTkyIDYwMCwxOTIiCiAgICAgICAgICAgIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0FEMTQ1NyIgc3Ryb2tlLXdpZHRoPSI0IiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2UtbGluZWNhcD0icm91bmQiIG9wYWNpdHk9IjAuNiIvPgoKICA8IS0tIFJJQkJPTjogc2luZ2xlIHRvcCBsb29wLCB0YWlscyBjcm9zcyBsb3cgLS0+CiAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTAwLDIyKSI+CiAgICA8ZyBmaWxsPSJub25lIiBzdHJva2U9InVybCgjcmliKSIgc3Ryb2tlLXdpZHRoPSIxOCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIj4KICAgICAgPHBhdGggZD0iTSA0MCwxNzggQyA1MiwxNTAgNTgsMTIwIDYyLDEwMCBDIDY2LDgwIDQ0LDc0IDQ0LDU0IEMgNDQsMzQgNTgsMjYgNjgsMzQgQyA3OCw0MiA3OCw2NiA3NCw5MiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAsMCkgc2NhbGUoLTEsMSkiIGQ9Ik0gNDAsMTc4IEMgNTIsMTUwIDU4LDEyMCA2MiwxMDAgQyA2Niw4MCA0NCw3NCA0NCw1NCBDIDQ0LDM0IDU4LDI2IDY4LDM0IEMgNzgsNDIgNzgsNjYgNzQsOTIiLz4KICAgIDwvZz4KICA8L2c+CgogIDxwYXRoIGQ9Ik0gNTA1LDY2IGMgLTYsLTExIC0yNCwtOCAtMjQsNiBjIDAsMTEgMTQsMTggMjQsMjggYyAxMCwtMTAgMjQsLTE3IDI0LC0yOCBjIDAsLTE0IC0xOCwtMTcgLTI0LC02IHoiIGZpbGw9IiNFQzQwN0EiLz4KCiAgPHBhdGggZD0iTSA5MiwyMDggQyA3NywxOTkgNzUsMTgxIDkzLDE3NSBDIDk3LDE5MyAxMDUsMjAxIDkyLDIwOCBaIiBmaWxsPSIjNjZCQjZBIiBvcGFjaXR5PSIwLjg1Ii8+CiAgPHBhdGggZD0iTSAxMDYsMjExIEMgMTE4LDE5OSAxMzYsMjAxIDEzNiwyMTcgQyAxMjAsMjE5IDExMCwyMTkgMTA2LDIxMSBaIiBmaWxsPSIjODFDNzg0IiBvcGFjaXR5PSIwLjg1Ii8+CgogIDxnIGZpbGw9IiNGMDYyOTIiPgogICAgPGNpcmNsZSBjeD0iMzYwIiBjeT0iNjgiIHI9IjQiLz4KICAgIDxjaXJjbGUgY3g9IjQwOCIgY3k9IjE1MCIgcj0iMyIvPgogICAgPGNpcmNsZSBjeD0iMzAwIiBjeT0iMjEyIiByPSIzIi8+CiAgICA8Y2lyY2xlIGN4PSI1NDgiIGN5PSIxNzgiIHI9IjQiLz4KICA8L2c+Cjwvc3ZnPgo=" alt="Breast cancer awareness illustration"></div>', unsafe_allow_html=True)

# ---------------- Inputs ----------------
st.markdown('<div class="section-title">Patient measurements</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Typical healthy ranges are shown in each field’s tooltip.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    radius = st.number_input("Mean Radius", min_value=0.0, max_value=50.0, value=0.0, step=0.01,
                             help="Average distance from center to perimeter (6.0 - 28.0)")
    perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=250.0, value=0.0, step=0.1,
                                help="Average perimeter of cell nucleus (40.0 - 190.0)")
    smoothness = st.number_input("Mean Smoothness", min_value=0.0, max_value=0.3, value=0.0, step=0.001,
                                 format="%.4f", help="Local variation in radius lengths (0.05 - 0.16)")
    concavity = st.number_input("Mean Concavity", min_value=0.0, max_value=0.5, value=0.0, step=0.001,
                                format="%.4f", help="Severity of concave portions (0.0 - 0.43)")
    symmetry = st.number_input("Mean Symmetry", min_value=0.0, max_value=0.5, value=0.0, step=0.001,
                               format="%.4f", help="Symmetry of the cell (0.10 - 0.30)")

with col2:
    texture = st.number_input("Mean Texture", min_value=0.0, max_value=50.0, value=0.0, step=0.01,
                              help="Standard deviation of gray-scale values (9.0 - 40.0)")
    area = st.number_input("Mean Area", min_value=0.0, max_value=3000.0, value=0.0, step=1.0,
                           help="Average area of cell nucleus (140.0 - 2500.0)")
    compactness = st.number_input("Mean Compactness", min_value=0.0, max_value=0.5, value=0.0, step=0.001,
                                  format="%.4f", help="(perimeter squared / area - 1.0) (0.02 - 0.35)")
    concave_points = st.number_input("Mean Concave Points", min_value=0.0, max_value=0.3, value=0.0, step=0.001,
                                     format="%.4f", help="Number of concave portions (0.0 - 0.20)")
    fractal = st.number_input("Mean Fractal Dimension", min_value=0.0, max_value=0.2, value=0.0, step=0.0001,
                              format="%.5f", help="Coastline approximation - 1 (0.05 - 0.10)")

# ---------------- Predict ----------------
if st.button("Predict Diagnosis"):
    if radius == 0 or texture == 0 or perimeter == 0 or area == 0:
        st.error("Please fill in Mean Radius, Texture, Perimeter and Area with non-zero values.")
    else:
        try:
            features = np.array([[
                radius, texture, perimeter, area, smoothness,
                compactness, concavity, concave_points, symmetry, fractal
            ]])

            if scaler is not None:
                features = scaler.transform(features)

            # NOTE: in scikit-learn's dataset, class 0 = malignant, class 1 = benign
            proba = model.predict_proba(features)[0]   # [P(malignant), P(benign)]
            malignant_prob = proba[0]

            malignant = malignant_prob > 0.5
            confidence = malignant_prob if malignant else (1 - malignant_prob)
            confidence_percent = confidence * 100

            if malignant:
                st.markdown(f"""
                <div class="result-card malignant-card">
                    <div class="result-badge malignant-badge">MALIGNANT</div>
                    <div class="confidence-text">Confidence: {confidence_percent:.1f}%</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {confidence_percent}%; background: linear-gradient(90deg, #E53935 0%, #C62828 100%);">
                            {confidence_percent:.1f}%
                        </div>
                    </div>
                    <div class="recommendation">
                        <strong>Recommendation:</strong> This result suggests potential malignancy.
                        Please consult a healthcare professional promptly for further diagnostic testing.
                        Early detection is crucial for effective treatment.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card benign-card">
                    <div class="result-badge benign-badge">BENIGN</div>
                    <div class="confidence-text">Confidence: {confidence_percent:.1f}%</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {confidence_percent}%; background: linear-gradient(90deg, #43A047 0%, #2E7D32 100%);">
                            {confidence_percent:.1f}%
                        </div>
                    </div>
                    <div class="recommendation">
                        <strong>Recommendation:</strong> This result suggests the tumor is benign.
                        Continue with regular check-ups and follow your doctor’s advice for ongoing monitoring.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"The prediction could not be completed: {str(e)}")

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
    <strong>Medical disclaimer:</strong> This tool is for educational and informational purposes only.
    It is not a substitute for professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider with any medical questions.
    <div class="cause">🎗️ Supporting breast cancer awareness — early detection saves lives</div>
</div>
""", unsafe_allow_html=True)