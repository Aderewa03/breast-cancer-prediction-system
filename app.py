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

# Custom CSS for Pink Theme
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #E91E63 0%, #C2185B 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(233, 30, 99, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Input container */
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(233, 30, 99, 0.1);
        border: 2px solid #F8BBD0;
        margin-bottom: 2rem;
    }
    
    /* Section headers */
    .section-header {
        color: #C2185B;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #E91E63;
    }
    
    /* Warning box */
    .warning-box {
        background: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: #E65100;
        font-size: 0.9rem;
    }
    
    /* Input labels */
    .stNumberInput label {
        color: #C2185B !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Number inputs */
    .stNumberInput input {
        border: 2px solid #F8BBD0 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput input:focus {
        border-color: #E91E63 !important;
        box-shadow: 0 0 0 2px rgba(233, 30, 99, 0.1) !important;
    }
    
    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #E91E63 0%, #C2185B 100%) !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 3rem !important;
        border-radius: 25px !important;
        border: none !important;
        box-shadow: 0 6px 25px rgba(233, 30, 99, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 1rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(233, 30, 99, 0.5) !important;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        animation: slideUp 0.5s ease-out;
    }
    
    .benign-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border: 3px solid #4CAF50;
    }
    
    .malignant-card {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border: 3px solid #F44336;
    }
    
    .result-badge {
        display: inline-block;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .benign-badge {
        background: #4CAF50;
        color: white;
    }
    
    .malignant-badge {
        background: #F44336;
        color: white;
    }
    
    .confidence-text {
        font-size: 1.3rem;
        color: #424242;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .recommendation {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-size: 1rem;
        color: #424242;
        border-left: 4px solid #E91E63;
    }
    
    /* Progress bar container */
    .progress-container {
        background: #f0f0f0;
        border-radius: 20px;
        height: 30px;
        margin: 1.5rem auto;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        max-width: 400px;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #E91E63 0%, #C2185B 100%);
        border-radius: 20px;
        transition: width 1s ease-out;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #757575;
        font-size: 0.9rem;
        margin-top: 3rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(233, 30, 99, 0.1);
    }
    
    /* Animation */
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    """Load the pickled model and scaler"""
    try:
        with open("model.h5", "rb") as f:
            saved_data = pickle.load(f)
            model = saved_data["model"]
            scaler = saved_data.get("scaler")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model file 'model.h5' not found. Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load model
model, scaler = load_model()

# Header
st.markdown("""
<div class="main-header">
    <h1>🎗️ Breast Cancer Prediction System</h1>
    <p>AI-Powered Early Detection • Early Detection Saves Lives</p>
</div>
""", unsafe_allow_html=True)

# Warning about auto-fill
st.markdown("""
<div class="warning-box">
    <strong>ℹ️ Note:</strong> This simplified version uses only 10 mean features. 
    The remaining 20 features (standard error and worst values) are auto-filled with zeros. 
    This may slightly reduce prediction accuracy compared to the full 30-feature model.
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📋 Patient Features Input</div>', unsafe_allow_html=True)

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    radius = st.number_input(
        "Mean Radius",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.01,
        help="Average distance from center to perimeter (6.0 - 28.0)"
    )
    
    perimeter = st.number_input(
        "Mean Perimeter",
        min_value=0.0,
        max_value=250.0,
        value=0.0,
        step=0.1,
        help="Average perimeter of cell nucleus (40.0 - 190.0)"
    )
    
    smoothness = st.number_input(
        "Mean Smoothness",
        min_value=0.0,
        max_value=0.3,
        value=0.0,
        step=0.001,
        format="%.4f",
        help="Local variation in radius lengths (0.05 - 0.16)"
    )
    
    concavity = st.number_input(
        "Mean Concavity",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.001,
        format="%.4f",
        help="Severity of concave portions (0.0 - 0.43)"
    )
    
    symmetry = st.number_input(
        "Mean Symmetry",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.001,
        format="%.4f",
        help="Symmetry of the cell (0.10 - 0.30)"
    )

with col2:
    texture = st.number_input(
        "Mean Texture",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.01,
        help="Standard deviation of gray-scale values (9.0 - 40.0)"
    )
    
    area = st.number_input(
        "Mean Area",
        min_value=0.0,
        max_value=3000.0,
        value=0.0,
        step=1.0,
        help="Average area of cell nucleus (140.0 - 2500.0)"
    )
    
    compactness = st.number_input(
        "Mean Compactness",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.001,
        format="%.4f",
        help="(perimeter² / area - 1.0) (0.02 - 0.35)"
    )
    
    concave_points = st.number_input(
        "Mean Concave Points",
        min_value=0.0,
        max_value=0.3,
        value=0.0,
        step=0.001,
        format="%.4f",
        help="Number of concave portions (0.0 - 0.20)"
    )
    
    fractal = st.number_input(
        "Mean Fractal Dimension",
        min_value=0.0,
        max_value=0.2,
        value=0.0,
        step=0.0001,
        format="%.5f",
        help="Coastline approximation - 1 (0.05 - 0.10)"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button("🔬 PREDICT DIAGNOSIS"):
    # Validate inputs
    if radius == 0 or texture == 0 or perimeter == 0 or area == 0:
        st.error("⚠️ Please fill in all required fields with non-zero values.")
    else:
        try:
            # Prepare features array with 30 features
            # First 10: Mean values (user input)
            mean_features = [
                radius, texture, perimeter, area, smoothness,
                compactness, concavity, concave_points, symmetry, fractal
            ]
            
            # Next 10: Standard Error values (auto-filled with zeros)
            se_features = [0.0] * 10
            
            # Last 10: Worst values (auto-filled with zeros)
            worst_features = [0.0] * 10
            
            # Combine all 30 features
            all_features = mean_features + se_features + worst_features
            
            # Convert to numpy array
            features = np.array([all_features])
            
            # Scale features if scaler exists
            if scaler is not None:
                features = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features)
            
            # Extract probability
            probability = float(prediction[0]) if prediction.ndim == 1 else float(prediction[0][0])
            
            # Determine result
            malignant = probability > 0.5
            confidence = probability if malignant else (1 - probability)
            confidence_percent = confidence * 100
            
            # Display result
            if malignant:
                st.markdown(f"""
                <div class="result-card malignant-card">
                    <div class="result-badge malignant-badge">
                        ⚠️ MALIGNANT
                    </div>
                    <div class="confidence-text">
                        Confidence: {confidence_percent:.1f}%
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {confidence_percent}%;">
                            {confidence_percent:.1f}%
                        </div>
                    </div>
                    <div class="recommendation">
                        <strong>⚕️ Recommendation:</strong> This result indicates potential malignancy. 
                        Please consult with a healthcare professional immediately for further diagnostic 
                        testing and evaluation. Early detection is crucial for effective treatment.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card benign-card">
                    <div class="result-badge benign-badge">
                        ✓ BENIGN
                    </div>
                    <div class="confidence-text">
                        Confidence: {confidence_percent:.1f}%
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {confidence_percent}%; background: linear-gradient(90deg, #4CAF50 0%, #388E3C 100%);">
                            {confidence_percent:.1f}%
                        </div>
                    </div>
                    <div class="recommendation">
                        <strong>✅ Recommendation:</strong> This result suggests the tumor is benign. 
                        However, continue with regular check-ups and follow your doctor's advice for 
                        ongoing monitoring and health maintenance.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("💡 Debug info: If you see a feature mismatch error, please check your model.h5 file.")

# Footer
st.markdown("""
<div class="footer">
    <p><strong>⚠️ Medical Disclaimer:</strong> This system is for educational and informational purposes only. 
    It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.</p>
    <p style="margin-top: 1rem; color: #E91E63; font-weight: 600;">
    🎗️ Supporting Breast Cancer Awareness • Early Detection Saves Lives
    </p>
</div>
""", unsafe_allow_html=True)
