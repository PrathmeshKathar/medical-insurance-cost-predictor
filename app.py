import streamlit as st
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class AppConfig:
    """Application configuration constants"""
    MODEL_PATH: str = "optimized_insurance_model.pkl"
    PAGE_TITLE: str = "Medical Insurance Cost Predictor"
    PAGE_ICON: str = "🏥"
    MIN_AGE: int = 18
    MAX_AGE: int = 100
    MIN_BMI: float = 10.0
    MAX_BMI: float = 50.0
    MAX_CHILDREN: int = 5

@dataclass
class UserInput:
    """User input data structure"""
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# ============================================================================
# STYLING & UI COMPONENTS
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling to the app"""
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    .section-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .input-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .result-card {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .prediction-amount {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .factor-item {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #007bff;
        border: 1px solid #e0e0e0;
    }
    
    .stButton > button {
        background: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render the main application header"""
    st.markdown(
        f'<h1 class="main-header">{AppConfig.PAGE_ICON} {AppConfig.PAGE_TITLE}</h1>', 
        unsafe_allow_html=True
    )

# ============================================================================
# DATA PROCESSING & BUSINESS LOGIC
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained machine learning model"""
    try:
        return joblib.load(AppConfig.MODEL_PATH)
    except FileNotFoundError:
        st.error(f"⚠️ Model file '{AppConfig.MODEL_PATH}' not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def get_bmi_category(bmi: float) -> Tuple[str, str]:
    """Get BMI category and corresponding emoji"""
    if bmi < 18.5:
        return "Underweight", "🔵"
    elif bmi < 25:
        return "Normal", "🟢"
    elif bmi < 30:
        return "Overweight", "🟡"
    else:
        return "Obese", "🔴"

def get_cost_level(prediction: float) -> Tuple[str, str]:
    """Determine cost level based on prediction"""
    if prediction < 5000:
        return "Low Cost", "💚"
    elif prediction < 15000:
        return "Moderate Cost", "💛"
    else:
        return "High Cost", "🔴"

def analyze_risk_factors(user_input: UserInput) -> List[Tuple[str, str]]:
    """Analyze risk factors based on user input"""
    factors = []
    
    if user_input.smoker == "yes":
        factors.append((
            "🚬 Smoking", 
            "Major cost driver - significantly increases premiums"
        ))
    
    if user_input.bmi > 30:
        factors.append((
            "⚖️ High BMI", 
            "Obesity increases health risks and costs"
        ))
    
    if user_input.age > 50:
        factors.append((
            "👤 Age Factor", 
            "Higher age correlates with increased costs"
        ))
    
    if user_input.children > 2:
        factors.append((
            "👶 Family Size", 
            "Multiple dependents increase coverage costs"
        ))
    
    return factors

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_input_form() -> UserInput:
    """Render the input form and return user input"""
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📋 Personal Information</h2>', unsafe_allow_html=True)
    
    # Basic Details Section
    st.markdown("#### 👤 Basic Details")
    col_a, col_b = st.columns(2)
    
    with col_a:
        age = st.slider("Age", AppConfig.MIN_AGE, AppConfig.MAX_AGE, 25)
        sex = st.selectbox("Gender", ["male", "female"])
    
    with col_b:
        bmi = st.number_input(
            "BMI", 
            min_value=AppConfig.MIN_BMI, 
            max_value=AppConfig.MAX_BMI, 
            value=20.0, 
            step=0.1
        )
        children = st.slider("Children", 0, AppConfig.MAX_CHILDREN, 0)
    
    # Lifestyle & Location Section
    st.markdown("---")
    st.markdown("#### 🏠 Lifestyle & Location")
    col_c, col_d = st.columns(2)
    
    with col_c:
        smoker = st.selectbox("Smoking Status", ["no", "yes"])
    
    with col_d:
        region = st.selectbox(
            "Region", 
            ["northeast", "northwest", "southeast", "southwest"]
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return UserInput(age, sex, bmi, children, smoker, region)

def render_user_profile(user_input: UserInput):
    """Render user profile summary"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 Your Profile</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown(f"**👤 Age:** {user_input.age} years")
        st.markdown(f"**⚧ Gender:** {user_input.sex.title()}")
        st.markdown(f"**⚖️ BMI:** {user_input.bmi}")
    
    with col_info2:
        st.markdown(f"**👶 Children:** {user_input.children}")
        st.markdown(f"**🚬 Smoker:** {user_input.smoker.title()}")
        st.markdown(f"**🗺️ Region:** {user_input.region.title()}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # BMI Category
    bmi_category, bmi_color = get_bmi_category(user_input.bmi)
    st.markdown(f"**BMI Category:** {bmi_color} {bmi_category}")
    st.markdown('</div>', unsafe_allow_html=True)

def render_prediction_results(prediction: float, user_input: UserInput):
    """Render prediction results and analysis"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    # Main prediction display
    st.markdown(
        f'<div class="prediction-amount">${prediction:,.2f}</div>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem;">Estimated Annual Premium</p>', 
        unsafe_allow_html=True
    )
    
    # Cost level indicator
    cost_level, cost_emoji = get_cost_level(prediction)
    st.markdown(
        f'<div style="text-align: center; font-size: 1.3rem; font-weight: 500;">'
        f'{cost_emoji} {cost_level}</div>', 
        unsafe_allow_html=True
    )
    
    # Risk factors analysis
    st.markdown("#### 📈 Cost Factors Analysis")
    factors = analyze_risk_factors(user_input)
    
    if factors:
        for factor, description in factors:
            st.markdown(
                f'<div class="factor-item">'
                f'<strong>{factor}</strong><br>'
                f'<small>{description}</small>'
                f'</div>', 
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="factor-item">'
            '<strong>💚 Low Risk Profile</strong><br>'
            '<small>You have relatively few high-risk factors</small>'
            '</div>', 
            unsafe_allow_html=True
        )
    
    # Payment breakdown
    st.markdown("---")
    st.markdown("#### 💰 Payment Breakdown")
    col1, col2, col3 = st.columns(3)
    
    monthly = prediction / 12
    quarterly = prediction / 4
    
    col1.markdown(f"**Monthly:** ${monthly:.2f}")
    col2.markdown(f"**Quarterly:** ${quarterly:.2f}")
    col3.markdown(f"**Annual:** ${prediction:,.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def make_prediction(model, user_input: UserInput) -> Optional[float]:
    """Make prediction using the loaded model"""
    try:
        # Convert user input to DataFrame
        input_data = pd.DataFrame([{
            'age': user_input.age,
            'sex': user_input.sex,
            'bmi': user_input.bmi,
            'children': user_input.children,
            'smoker': user_input.smoker,
            'region': user_input.region
        }])
        
        # Make prediction
        prediction = model.predict(input_data)
        return prediction[0]
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None

def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; opacity: 0.7; padding: 2rem 0;">'
        f'<small>{AppConfig.PAGE_ICON} {AppConfig.PAGE_TITLE}</small>'
        '</div>', 
        unsafe_allow_html=True
    )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    # Configure page
    st.set_page_config(
        page_title=AppConfig.PAGE_TITLE,
        page_icon=AppConfig.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply styling
    apply_custom_css()
    
    # Render header
    render_header()
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error(f"❌ Cannot load model. Please check if '{AppConfig.MODEL_PATH}' exists.")
        return
    
    # Main layout
    col1, col2 = st.columns([3, 2], gap="large")
    
    # Input form
    with col1:
        user_input = render_input_form()
    
    # User profile
    with col2:
        render_user_profile(user_input)
    
    # Prediction section
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🔮 Calculate Insurance Cost"):
            prediction = make_prediction(model, user_input)
            
            if prediction is not None:
                render_prediction_results(prediction, user_input)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    render_footer()

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()