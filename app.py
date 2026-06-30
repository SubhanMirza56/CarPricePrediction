import streamlit as st
import numpy as np
import joblib
import time

# ──────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Load model and scaler
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("LinerRegressionModel.pkl")
    scaler = joblib.load("scalerFeature.joblib")
    return model, scaler

model, scaler = load_artifacts()

# ──────────────────────────────────────────────
# Encoding dictionaries
# ──────────────────────────────────────────────
brand_map = {"Ford": 0, "Audi": 1, "Honda": 2, "Toyota": 3, "BMW": 4}
model_map = {"Model A": 0, "Model B": 1, "Model C": 2, "Model D": 3, "Model E": 4}
fuel_map = {"Diesel": 0, "Petrol": 1, "Electric": 2}
trans_map = {"Automatic": 0, "Manual": 1}

# ──────────────────────────────────────────────
# Custom CSS styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Overall app background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    /* Main content card */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }

    /* Title */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }

    /* Form card */
    div[data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 2rem 2rem 1.2rem 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(8px);
    }

    /* Input labels */
    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    /* Selectbox / number input boxes */
    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        color: #f1f5f9 !important;
    }

    /* Submit button */
    .stFormSubmitButton button {
        width: 100%;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 0;
        margin-top: 0.8rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stFormSubmitButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4);
    }

    /* Result card */
    .result-card {
        margin-top: 1.6rem;
        padding: 1.6rem;
        border-radius: 18px;
        text-align: center;
        background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(129,140,248,0.15));
        border: 1px solid rgba(129, 140, 248, 0.35);
    }
    .result-label {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .result-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #34d399;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(148,163,184,0.1);
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 About")
    st.write(
        "This tool estimates a car's market price using a trained "
        "regression model based on brand, model, engine size, mileage, "
        "fuel type, transmission, and age."
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Fill in the car details\n"
        "2. Click **Predict Price**\n"
        "3. Get an instant estimate"
    )
    st.markdown("---")
    st.caption("Built with Streamlit • Linear Regression")

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">🚗 Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Get an instant, data-driven estimate of your car\'s value</div>',
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────
# Form
# ──────────────────────────────────────────────
with st.form("prediction_form"):

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("🏷️ Car Brand", list(brand_map.keys()))
        engine_size = st.number_input(
            "⚙️ Engine Size (L)",
            min_value=0.0,
            max_value=10.0,
            value=None,
            step=0.1,
            placeholder="Enter the engine size"
        )
        fuel = st.selectbox("⛽ Fuel Type", list(fuel_map.keys()))

    with col2:
        model_name = st.selectbox("🚘 Model", list(model_map.keys()))
        mileage = st.number_input(
            "🛣️ Mileage (km)",
            min_value=0,
            max_value=500000,
            value=None,
            step=1000,
            placeholder="Enter the mileage"
        )
        transmission = st.selectbox("🔧 Transmission", list(trans_map.keys()))

    car_age = st.slider("📅How Old your Car is  (years)", min_value=0, max_value=30, value=5)

    predict = st.form_submit_button("🔮 Predict Price")

# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
if predict:
    if engine_size is None or mileage is None:
        st.warning("⚠️ Please enter both Engine Size and Mileage before predicting.")
        st.stop()

    with st.spinner("Crunching the numbers..."):
        time.sleep(0.6)  # small delay purely for UX feel

        brand_encoded = brand_map[brand]
        model_encoded = model_map[model_name]
        fuel_encoded = fuel_map[fuel]
        transmission_encoded = trans_map[transmission]

        scaled = scaler.transform([[engine_size, mileage]])
        engine_scaled, mileage_scaled = scaled[0][0], scaled[0][1]

        features = np.array([[
            brand_encoded,
            model_encoded,
            engine_scaled,
            mileage_scaled,
            fuel_encoded,
            transmission_encoded,
            car_age
        ]])

        prediction = model.predict(features)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated Market Price</div>
            <div class="result-value">${prediction[0]:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.balloons()

    with st.expander("📋 See the details used in this prediction"):
        st.write(f"**Brand:** {brand}")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Engine Size:** {engine_size} L")
        st.write(f"**Mileage:** {mileage:,} km")
        st.write(f"**Fuel Type:** {fuel}")
        st.write(f"**Transmission:** {transmission}")
        st.write(f"**Car Age:** {car_age} years")