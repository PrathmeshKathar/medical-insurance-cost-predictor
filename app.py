import streamlit as st
import joblib
import numpy as np

# Configure page settings
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

# Theme toggle in the top-right corner
col_header1, col_header2, col_header3 = st.columns([6, 1, 1])
with col_header3:
    if st.button("🌓 Theme", key="theme_toggle"):
        st.session_state.theme = 'Dark' if st.session_state.theme == 'Light' else 'Light'
        st.rerun()

# Apply theme-specific CSS
if st.session_state.theme == 'Dark':
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .input-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem 0;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem 0;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    
    .section-header {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: #ffffff;
    }
    
    .stNumberInput > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: #ffffff;
    }
    
    .stSlider > div > div {
        color: #ffffff;
    }
    
    .prediction-amount {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0;
    }
    
    .factor-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: 'Inter', sans-serif;
        color: #1e293b;
        min-height: 100vh;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #1e293b !important;
    }
    
    .stText, div[data-testid="stText"] {
        color: #1e293b !important;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(135deg, #22543d 0%, #2d5a3d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(34, 84, 61, 0.1);
    }
    
    .card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(20px);
    }
    
    .input-card {
        background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .input-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #22543d 0%, #2f855a 100%);
    }
    
    .result-card {
        background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(79, 70, 229, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #1a5838 0%, #22543d 100%);
    }
    
    .section-header {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(135deg, #22543d 0%, #2f855a 100%);
        border-radius: 2px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #22543d 0%, #2f855a 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 8px 20px rgba(34, 84, 61, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1a5838 0%, #276749 100%);
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(34, 84, 61, 0.4);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #cbd5e1;
        color: #1e293b;
        box-shadow: 0 4px 6px rgba(15, 23, 42, 0.05);
    }
    
    .metric-container p, .metric-container div {
        color: #1e293b !important;
    }
    
    .stSelectbox > div > div, .stSelectbox label {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        color: #1e293b !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #22543d;
        box-shadow: 0 0 0 3px rgba(34, 84, 61, 0.1);
    }
    
    .stNumberInput > div > div, .stNumberInput label {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        color: #1e293b !important;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div:focus-within {
        border-color: #22543d;
        box-shadow: 0 0 0 3px rgba(34, 84, 61, 0.1);
    }
    
    .stSlider > div > div, .stSlider label {
        color: #1e293b !important;
    }
    
    .stSlider .stMarkdown {
        color: #1e293b !important;
    }
    
    label, .stMarkdown label {
        color: #334155 !important;
        font-weight: 500 !important;
    }
    
    .prediction-amount {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #1a5838 0%, #22543d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0;
        text-shadow: 0 2px 4px rgba(26, 88, 56, 0.1);
    }
    
    .factor-item {
        background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-left: 4px solid #22543d;
        color: #1e293b;
        box-shadow: 0 4px 6px rgba(34, 84, 61, 0.1);
        transition: all 0.3s ease;
    }
    
    .factor-item:hover {
        transform: translateX(4px);
        box-shadow: 0 6px 12px rgba(34, 84, 61, 0.15);
    }
    
    .factor-item p, .factor-item div, .factor-item strong, .factor-item small {
        color: #1e293b !important;
    }
    
    /* Enhanced text visibility */
    * {
        color: #1e293b !important;
    }
    
    .stApp * {
        color: #1e293b !important;
    }
    
    /* Add subtle animations */
    .input-card, .result-card, .card {
        transition: all 0.3s ease;
    }
    
    .input-card:hover, .result-card:hover, .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 25px 50px rgba(15, 23, 42, 0.15);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #22543d 0%, #2f855a 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1a5838 0%, #276749 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("insurance_model.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model file 'insurance_model.pkl' not found. Please ensure the model file is in the same directory.")
        return None

model = load_model()

# Main header
st.markdown('<h1 class="main-header">🏥 Medical Insurance Cost Predictor</h1>', unsafe_allow_html=True)

if model is not None:
    # Create main layout
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">📋 Personal Information</h2>', unsafe_allow_html=True)
        
        # Personal details section
        st.markdown("#### 👤 Basic Details")
        col_a, col_b = st.columns(2)
        
        with col_a:
            age = st.slider("Age", 18, 100, 25, help="Your current age in years")
            sex = st.selectbox("Gender", ["male", "female"])
        
        with col_b:
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=20.0, step=0.1,
                                 help="Body Mass Index (weight in kg / height in m²)")
            children = st.slider("Children", 0, 5, 0, help="Number of dependents covered")
        
        st.markdown("---")
        
        # Lifestyle and location section
        st.markdown("#### 🏠 Lifestyle & Location")
        col_c, col_d = st.columns(2)
        
        with col_c:
            smoker = st.selectbox("Smoking Status", ["no", "yes"], 
                                help="Are you a smoker?")
        
        with col_d:
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"],
                                help="Your geographical region")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">📊 Your Profile</h2>', unsafe_allow_html=True)
        
        # Display current selection in a nice format
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown(f"**👤 Age:** {age} years")
            st.markdown(f"**⚧ Gender:** {sex.title()}")
            st.markdown(f"**⚖️ BMI:** {bmi}")
        
        with col_info2:
            st.markdown(f"**👶 Children:** {children}")
            st.markdown(f"**🚬 Smoker:** {smoker.title()}")
            st.markdown(f"**🗺️ Region:** {region.title()}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # BMI Category
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "🔵"
        elif bmi < 25:
            bmi_category = "Normal"
            bmi_color = "🟢"
        elif bmi < 30:
            bmi_category = "Overweight"
            bmi_color = "🟡"
        else:
            bmi_category = "Obese"
            bmi_color = "🔴"
        
        st.markdown(f"**BMI Category:** {bmi_color} {bmi_category}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction section
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    # Convert inputs to model format
    input_data = np.array([
        age,
        1 if sex == "male" else 0,
        bmi,
        children,
        1 if smoker == "yes" else 0,
        1 if region == "northeast" else
        2 if region == "northwest" else
        3 if region == "southeast" else
        4
    ]).reshape(1, -1)
    
    # Centered predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🔮 Calculate Insurance Cost"):
            try:
                prediction = model.predict(input_data)
                
                # Display result
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-amount">${prediction[0]:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('<p style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">Estimated Annual Premium</p>', unsafe_allow_html=True)
                
                # Cost analysis
                if prediction[0] < 5000:
                    cost_level = "Low Cost"
                    cost_emoji = "💚"
                    cost_color = "#10b981"
                elif prediction[0] < 15000:
                    cost_level = "Moderate Cost" 
                    cost_emoji = "💛"
                    cost_color = "#f59e0b"
                else:
                    cost_level = "High Cost"
                    cost_emoji = "🔴"
                    cost_color = "#ef4444"
                
                st.markdown(f'<div style="text-align: center; font-size: 1.3rem; color: {cost_color}; font-weight: 600; margin-bottom: 2rem;">{cost_emoji} {cost_level}</div>', unsafe_allow_html=True)
                
                # Risk factors analysis
                st.markdown("#### 📈 Cost Factors Analysis")
                
                factors = []
                if smoker == "yes":
                    factors.append(("🚬 Smoking", "Major cost driver - significantly increases premiums"))
                if bmi > 30:
                    factors.append(("⚖️ High BMI", "Obesity increases health risks and costs"))
                if age > 50:
                    factors.append(("👤 Age Factor", "Higher age correlates with increased costs"))
                if children > 2:
                    factors.append(("👶 Family Size", "Multiple dependents increase coverage costs"))
                
                if factors:
                    for factor, description in factors:
                        st.markdown(f'<div class="factor-item"><strong>{factor}</strong><br><small>{description}</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="factor-item"><strong>💚 Low Risk Profile</strong><br><small>You have relatively few high-risk factors</small></div>', unsafe_allow_html=True)
                
                # Monthly breakdown
                monthly_cost = prediction[0] / 12
                st.markdown("---")
                st.markdown("#### 💰 Payment Breakdown")
                
                col_pay1, col_pay2, col_pay3 = st.columns(3)
                with col_pay1:
                    st.markdown(f"**Monthly:** ${monthly_cost:.2f}")
                with col_pay2:
                    st.markdown(f"**Quarterly:** ${prediction[0]/4:.2f}")
                with col_pay3:
                    st.markdown(f"**Annual:** ${prediction[0]:,.2f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.error("❌ Cannot load the machine learning model. Please check if 'insurance_model.pkl' exists in the current directory.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    f'<div style="text-align: center; opacity: 0.7; padding: 2rem 0;"><small>🏥 Medical Insurance Cost Predictor | Current Theme: {st.session_state.theme}</small></div>', 
    unsafe_allow_html=True
)