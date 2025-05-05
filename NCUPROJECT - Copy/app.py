import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import base64
from datetime import datetime
import warnings
# import joblib  # Comment out to avoid ModuleNotFoundError
import sys
from streamlit_option_menu import option_menu
# Import PDF and Excel libraries conditionally
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    import xlsxwriter
    HAS_REPORT_LIBS = True
except ImportError:
    HAS_REPORT_LIBS = False

# Import utility modules
from utils.feature_names import (
    get_display_name, 
    get_feature_name,
    get_feature_display_names,
    DEFAULT_FEATURES
)
from utils.report_generator import generate_enhanced_excel_report, generate_enhanced_pdf_report
from utils.xai_utils import (
    create_shap_plots, 
    create_lime_explanation, 
    create_pdp_plots
)
from utils.recommendation_engine import get_recommendations
from about import show_about_page
from retrofit_analysis import display_retrofit_analysis_page


# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Building Energy Analysis Dashboard",
    page_icon="https://img.icons8.com/fluency/96/000000/energy-meter.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set custom theme with beautiful design
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* BASE STYLES AND VARIABLES */
    :root {
        --primary-color: #0f766e;        /* Teal 700 */
        --primary-light: #14b8a6;        /* Teal 500 */
        --secondary-color: #10b981;      /* Emerald 500 */
        --accent-color: #3b82f6;         /* Blue 500 */
        --background-color: #f8fafc;     /* Slate 50 */
        --surface-color: #ffffff;        /* White */
        --text-color: #1e293b;           /* Slate 800 */
        --text-muted: #64748b;           /* Slate 500 */
        --success-color: #10b981;        /* Emerald 500 */
        --warning-color: #f59e0b;        /* Amber 500 */
        --error-color: #ef4444;          /* Red 500 */
        --border-radius: 12px;
        --card-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        --transition-normal: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    /* CORE APP STYLING */
    body {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp, header[data-testid="stHeader"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
        background: linear-gradient(135deg, var(--background-color) 0%, #eef9f2 100%) !important;
    }
    
    /* TYPOGRAPHY */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        color: var(--primary-color) !important;
        font-weight: 600;
    }
    
    h1 {
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem;
        position: relative;
    }
    
    h1::after {
        content: "";
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 2px;
    }
    
    h2 {
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        color: var(--primary-color) !important;
    }
    
    h3 {
        font-size: 1.4rem !important;
        margin-top: 1.5rem !important;
        border-left: 4px solid var(--primary-color);
        padding-left: 0.8rem;
        color: var(--primary-color) !important;
    }
    
    /* BODY TEXT */
    .stMarkdown p, .css-1d391kg, .css-1v3fvcr, .stTextInput label {
        color: var(--text-color) !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    /* GLASS CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--card-shadow);
        transition: var(--transition-normal);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* BUTTONS */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        padding: 0.6rem 1.5rem !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(15, 118, 110, 0.2) !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        text-transform: none !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 15px rgba(15, 118, 110, 0.3) !important;
    }
    
    .stButton button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 5px rgba(15, 118, 110, 0.2) !important;
    }
    
    /* METRICS */
    div[data-testid="stMetricValue"] {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* BEAUTIFUL CARDS */
    .metric-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(255, 255, 255, 0.6);
        transition: var(--transition-normal);
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #e7f6fd 0%, #c8f5e2 100%);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
        box-shadow: var(--card-shadow);
        transition: var(--transition-normal);
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff9c4, #ffecb3);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--warning-color);
        box-shadow: var(--card-shadow);
    }
    
    /* TABLE STYLING */
    div.stDataFrame {
        border-radius: var(--border-radius);
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: var(--card-shadow);
    }
    
    div.stDataFrame th {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        font-weight: 600;
        padding: 12px;
        border: none;
        text-align: left;
        font-size: 0.9rem;
    }
    
    div.stDataFrame td {
        padding: 10px 12px;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.95rem;
        color: var(--text-color);
    }
    
    div.stDataFrame tr:nth-child(even) {
        background-color: rgba(248, 250, 252, 0.5);
    }
    
    div.stDataFrame tr:hover {
        background-color: rgba(16, 185, 129, 0.05);
    }
    
    /* TABS */
    button[role="tab"] {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-color) !important;
        font-weight: 500 !important;
        padding: 0.8rem 1.2rem !important;
        background: rgba(255, 255, 255, 0.5) !important;
        border-radius: 8px 8px 0 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-bottom: none !important;
        margin-right: 5px !important;
        transition: all 0.2s ease !important;
    }
    
    button[role="tab"][aria-selected="true"] {
        color: var(--primary-color) !important;
        background: white !important;
        font-weight: 600 !important;
        border-bottom-color: transparent !important;
        border-top: 3px solid var(--primary-color) !important;
    }
    
    button[role="tab"]:hover {
        background: rgba(255, 255, 255, 0.8) !important;
        color: var(--primary-color) !important;
    }
    
    /* SLIDERS */
    div.stSlider div[data-baseweb="thumb"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
        border: 2px solid white !important;
        width: 22px !important;
        height: 22px !important;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.2) !important;
    }
    
    div.stSlider div[data-baseweb="track"] {
        background: rgba(20, 184, 166, 0.2) !important;
        height: 6px !important;
        border-radius: 3px !important;
    }
    
    div.stSlider div[data-baseweb="track"] div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
    }
    
    /* SELECT BOXES */
    div[data-baseweb="select"] > div:first-child {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(15, 118, 110, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-baseweb="select"] > div:first-child:hover {
        border-color: var(--primary-color) !important;
    }
    
    /* EXPANDERS */
    details[data-stale="false"] summary {
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.05) 0%, rgba(16, 185, 129, 0.05) 100%) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border: 1px solid rgba(15, 118, 110, 0.1) !important;
        transition: all 0.2s ease !important;
    }
    
    details[data-stale="false"] summary:hover {
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%) !important;
    }
    
    details[open] summary {
        border-bottom-left-radius: 0 !important;
        border-bottom-right-radius: 0 !important;
        border-bottom: 1px solid rgba(15, 118, 110, 0.1) !important;
    }
    
    details[open] > div:first-child {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(15, 118, 110, 0.1) !important;
        border-top: none !important;
        border-bottom-left-radius: 8px !important;
        border-bottom-right-radius: 8px !important;
        padding: 1.5rem !important;
    }
    
    /* ANIMATIONS */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(15, 118, 110, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(15, 118, 110, 0); }
        100% { box-shadow: 0 0 0 0 rgba(15, 118, 110, 0); }
    }
    
    .animated-fade {
        animation: fadeIn 0.8s ease-out;
    }
    
    .animated-slide {
        animation: slideIn 0.6s ease-out;
    }
    
    .animated-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Page specific styles */
    .eui-gauge-container {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: var(--card-shadow);
        transition: var(--transition-normal);
    }
    
    .eui-gauge-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    .recommendation-item {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
        box-shadow: var(--card-shadow);
        transition: var(--transition-normal);
    }
    
    .recommendation-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    
    .metric-unit {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }
    
    /* Dashboard rating cards */
    .rating-card {
        border-radius: var(--border-radius);
        padding: 1.5rem;
        color: white;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: var(--card-shadow);
        min-height: 200px;
        transition: var(--transition-normal);
    }
    
    .rating-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
    }
    
    .rating-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
    }
    
    .rating-desc {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Beautiful comparison chart container */
    .chart-container {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: var(--card-shadow);
        margin-top: 2rem;
        transition: var(--transition-normal);
    }
    
    .chart-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f766e 0%, #047857 100%) !important;
        border-radius: 0 20px 20px 0;
    }
    
    section[data-testid="stSidebar"] .css-6qob1r {
        background: transparent !important;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
        -webkit-text-fill-color: white !important;
    }
    
    /* SIDEBAR NAVIGATION */
    .css-8hkptd {
        margin-top: 2rem !important;
    }
    
    /* STEP PROGRESS INDICATOR */
    .step-progress {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .step-progress::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 3px;
        background: rgba(15, 118, 110, 0.2);
        transform: translateY(-50%);
        z-index: 0;
    }
    
    .step-item {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        border: 2px solid var(--primary-color);
        color: var(--primary-color);
        position: relative;
        z-index: 1;
    }
    
    .step-item.active {
        background: var(--primary-color);
        color: white;
        box-shadow: 0 0 0 5px rgba(15, 118, 110, 0.2);
    }
    
    .step-item.completed {
        background: var(--success-color);
        border-color: var(--success-color);
        color: white;
    }
    
    /* Better visualization containers */
    .viz-container {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: var(--card-shadow);
        margin-bottom: 2rem;
        transition: var(--transition-normal);
    }
    
    .viz-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid rgba(15, 118, 110, 0.1);
        color: var(--text-muted);
    }
</style>
""", unsafe_allow_html=True)



# Suppress warnings
warnings.filterwarnings('ignore')

# Feature flags for available libraries
HAS_VIZ = True
HAS_ML = True
HAS_XAI = False  # Set to False since we don't have SHAP and LIME
HAS_PDP = True

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    HAS_VIZ = False

# Try to import ML libraries
try:
    from sklearn.inspection import partial_dependence
except ImportError:
    HAS_ML = False
    HAS_PDP = False

# Create directories for assets if they don't exist
os.makedirs('attached_assets', exist_ok=True)
os.makedirs('utils', exist_ok=True)

# Function to load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    """Load the model and preprocessor"""
    try:
        # Try to load from both current directory and attached_assets
        model_paths = ['lgbm_optimized_eui.pkl', 'attached_assets\lgbm_optimized_eui.pkl']
        preprocessor_paths = ['lgbm_optimized_eui.pkl', 'attached_assets/preprocessor_eui_ordinal.pkl']

        # Try each possible path
        model = None
        preprocessor = None

        try:
            # First try to import LightGBM to ensure it's available
            import lightgbm as lgb

            for model_path in model_paths:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    break

            for preprocessor_path in preprocessor_paths:
                if os.path.exists(preprocessor_path):
                    with open(preprocessor_path, 'rb') as f:
                        preprocessor = pickle.load(f)
                    break
        except (ImportError, ModuleNotFoundError, OSError) as lgbm_error:
            # If LightGBM has system library issues, log the error
            st.error(f"⚠️ USING DEMO MODEL: The pre-trained LightGBM model could not be loaded due to system dependencies. Predictions will be approximate.")
            model = None
            preprocessor = None

        if model is not None and preprocessor is not None:
            return model, preprocessor
        else:
            # Create simple model and preprocessor if files don't exist or couldn't be loaded
            st.warning("""
            ## DEMO MODE ACTIVATED

            Using a simplified model with sample data. Predictions will be approximate and for demonstration purposes only.

            For accurate results:
            1. This app requires the LightGBM model and preprocessor files
            2. System dependencies for LightGBM need to be installed
            3. Contact your system administrator for assistance
            """)
            return create_simple_model_and_preprocessor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return create_simple_model_and_preprocessor()

def create_simple_model_and_preprocessor():
    """Create a simple model and preprocessor for demonstration"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    # Create a simple random forest model
    model = RandomForestRegressor(n_estimators=50, random_state=42)

    # Define categorical variables for preprocessing
    categorical_features = ['Weather_File', 'Building_Type', 'Renewable_Energy_Usage']

    # Define categories for ordinal encoding based on the data provided in aad.txt
    weather_file_categories = ['Historical', '2030']  
    building_type_categories = ['Detached', 'Bungalow', 'Semi Detached', 'Terraced']
    renewable_energy_categories = ['Yes', 'No']

    # Create preprocessing pipeline with proper ordinal encoding for categorical features
    # Using basic OrdinalEncoder configuration without additional parameters
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(categories=[weather_file_categories, building_type_categories, renewable_energy_categories]), categorical_features)
        ],
        remainder='passthrough'
    )

    # Generate sample training data
    features = [
        'HVAC_Efficiency', 'Domestic_Hot_Water_Usage',
        'Lighting_Density', 'Occupancy_Level', 'Equipment_Density',
        'Heating_Setpoint_Temperature', 'Heating_Setback_Temperature',
        'Window_to_Wall_Ratio', 'Total_Building_Area',
        'Floor_Insulation_U-Value_Capped', 'Door_Insulation_U-Value_Capped',
        'Roof_Insulation_U-Value_Capped', 'Window_Insulation_U-Value_Capped',
        'Wall_Insulation_U-Value_Capped', 'Air_Change_Rate_Capped',
        'Weather_File', 'Building_Type', 'Renewable_Energy_Usage'
    ]

    np.random.seed(42)  # For reproducibility
    n_samples = 500

    data = {
        'HVAC_Efficiency': np.random.uniform(0.6, 0.95, n_samples),
        'Domestic_Hot_Water_Usage': np.random.uniform(20, 100, n_samples),
        'Lighting_Density': np.random.uniform(5, 20, n_samples),
        'Occupancy_Level': np.random.uniform(0.01, 0.1, n_samples),
        'Equipment_Density': np.random.uniform(5, 30, n_samples),
        'Heating_Setpoint_Temperature': np.random.uniform(18, 23, n_samples),
        'Heating_Setback_Temperature': np.random.uniform(15, 18, n_samples),
        'Window_to_Wall_Ratio': np.random.uniform(0.1, 0.6, n_samples),
        'Total_Building_Area': np.random.uniform(100, 10000, n_samples),
        'Floor_Insulation_U-Value_Capped': np.random.uniform(0.1, 1.5, n_samples),
        'Door_Insulation_U-Value_Capped': np.random.uniform(0.1, 2.0, n_samples),
        'Roof_Insulation_U-Value_Capped': np.random.uniform(0.1, 1.0, n_samples),
        'Window_Insulation_U-Value_Capped': np.random.uniform(0.5, 3.0, n_samples),
        'Wall_Insulation_U-Value_Capped': np.random.uniform(0.1, 1.5, n_samples),
        'Air_Change_Rate_Capped': np.random.uniform(0.5, 5.0, n_samples),
        'Weather_File': np.random.choice(weather_file_categories, n_samples),
        'Building_Type': np.random.choice(building_type_categories, n_samples),
        'Renewable_Energy_Usage': np.random.choice(renewable_energy_categories, n_samples)
    }

    X = pd.DataFrame(data)

    # Create target variable (EUI) that depends on the features
    y = (
        0.5 * X['Floor_Insulation_U-Value_Capped'] +
        0.3 * X['Window_Insulation_U-Value_Capped'] +
        0.2 * X['Wall_Insulation_U-Value_Capped'] +
        0.1 * X['Air_Change_Rate_Capped'] -
        0.4 * X['HVAC_Efficiency'] +
        0.2 * X['Domestic_Hot_Water_Usage'] +
        0.1 * X['Lighting_Density'] +
        0.1 * X['Equipment_Density'] +
        0.1 * (X['Building_Type'] == 'Detached').astype(int) +
        0.05 * (X['Weather_File'] == '2030').astype(int) -
        0.2 * (X['Renewable_Energy_Usage'] == 'Yes').astype(int) +
        0.5 * X['Heating_Setpoint_Temperature'] -
        0.3 * X['Heating_Setback_Temperature'] +
        0.3 * X['Window_to_Wall_Ratio'] +
        0.05 * np.sqrt(X['Total_Building_Area']) +
        np.random.normal(0, 10, n_samples)
    ) * 50 + 150  # Scale to typical EUI range

    # Fit preprocessor and model
    X_transformed = preprocessor.fit_transform(X)
    model.fit(X_transformed, y)

    return model, preprocessor

# Function to create default inputs from ranges
def get_default_inputs():
    return {
        'Building_Orientation': 0,
        'Floor_Insulation_U-Value_Capped': 0.3,
        'Door_Insulation_U-Value_Capped': 2.2,
        'Roof_Insulation_U-Value_Capped': 0.7,
        'Window_Insulation_U-Value_Capped': 2.0,
        'Wall_Insulation_U-Value_Capped': 0.9,
        'Air_Change_Rate_Capped': 1.5,
        'HVAC_Efficiency': 3.0,
        'Domestic_Hot_Water_Usage': 1.5,
        'Lighting_Density': 5.0,
        'Occupancy_Level': 4.0,
        'Equipment_Density': 15.0,
        'Heating_Setpoint_Temperature': 21.0,
        'Heating_Setback_Temperature': 16.0,
        'Window_to_Wall_Ratio': 0.3,
        'Total_Building_Area': 2000.0,
        'Weather_File': 'Historical',
        'Building_Type': 'Detached',
        'Renewable_Energy_Usage': 'No'
    }

# Function to extract feature ranges from data
def get_feature_ranges():
    ranges = {
        'Floor_Insulation_U-Value_Capped': (0.15, 1.6),
        'Door_Insulation_U-Value_Capped': (0.81, 5.7),
        'Roof_Insulation_U-Value_Capped': (0.07, 2.28),
        'Window_Insulation_U-Value_Capped': (0.73, 5.75),
        'Wall_Insulation_U-Value_Capped': (0.1, 2.4),
        'Air_Change_Rate_Capped': (0.5, 5.0),
        'HVAC_Efficiency': (0.3, 4.5),
        'Domestic_Hot_Water_Usage': (0.5, 3.5),
        'Lighting_Density': (1.0, 9.0),
        'Occupancy_Level': (1.0, 6.0),
        'Equipment_Density': (5.0, 30.0),
        'Heating_Setpoint_Temperature': (18.0, 23.0),
        'Heating_Setback_Temperature': (15.0, 18.0),
        'Window_to_Wall_Ratio': (0.1, 0.6),
        'Total_Building_Area': (100.0, 10000.0)
    }
    return ranges

# Function to predict energy use intensity
def get_efficiency_rating(eui):
    """Return energy efficiency rating based on EUI value"""
    if eui < 50:
        return "A+", "#10b981"  # Emerald 500
    elif eui < 100:
        return "A", "#34d399"   # Emerald 400
    elif eui < 150:
        return "B", "#6ee7b7"   # Emerald 300
    elif eui < 200:
        return "C", "#fbbf24"   # Amber 400
    elif eui < 250:
        return "D", "#f59e0b"   # Amber 500
    elif eui < 300:
        return "E", "#f97316"   # Orange 500
    elif eui < 350:
        return "F", "#ef4444"   # Red 500
    else:
        return "G", "#b91c1c"   # Red 700

def create_gauge_chart(eui_value, key_suffix=""):
    """Create a beautiful gauge chart for the EUI value"""
    if eui_value is None:
        eui_value = 0

    rating, color = get_efficiency_rating(eui_value)

    # Create a more visually appealing gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=eui_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Energy Use Intensity", 'font': {'size': 18, 'family': "Inter, sans-serif", 'color': "#1e293b"}},
        number={'font': {'size': 28, 'family': "Poppins, sans-serif", 'color': color}, 'suffix': " kWh/m²/yr"},
        gauge={
            'axis': {'range': [0, 400], 'tickwidth': 1, 'tickcolor': "#64748b", 'tickfont': {'size': 12}},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 50], 'color': '#10b981'},
                {'range': [50, 100], 'color': '#34d399'},
                {'range': [100, 150], 'color': '#6ee7b7'},
                {'range': [150, 200], 'color': '#fcd34d'},
                {'range': [200, 250], 'color': '#f59e0b'},
                {'range': [250, 300], 'color': '#f97316'},
                {'range': [300, 350], 'color': '#ef4444'},
                {'range': [350, 400], 'color': '#b91c1c'},
            ],
            'threshold': {
                'line': {'color': "#1e293b", 'width': 4},
                'thickness': 0.8,
                'value': eui_value
            }
        }
    ))
    
    # Add rating annotation
    fig.add_annotation(
        x=0.5,
        y=0.25,
        text=f"Rating: {rating}",
        font=dict(size=24, family="Poppins, sans-serif", color=color),
        showarrow=False,
        xanchor='center'
    )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig

def predict_eui(inputs, model, preprocessor):
    """Predict EUI based on inputs"""
    try:
        # Validate inputs
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary")

        required_features = [
            'HVAC_Efficiency', 'Domestic_Hot_Water_Usage',
            'Building_Orientation', 'Lighting_Density', 'Occupancy_Level', 
            'Equipment_Density', 'Heating_Setpoint_Temperature', 
            'Heating_Setback_Temperature', 'Window_to_Wall_Ratio', 
            'Total_Building_Area', 'Floor_Insulation_U-Value_Capped', 
            'Door_Insulation_U-Value_Capped', 'Roof_Insulation_U-Value_Capped', 
            'Window_Insulation_U-Value_Capped', 'Wall_Insulation_U-Value_Capped', 
            'Air_Change_Rate_Capped', 'Weather_File', 'Building_Type', 
            'Renewable_Energy_Usage'
        ]

        # Check for missing features
        missing_features = [f for f in required_features if f not in inputs]
        if missing_features:
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")

        # Create DataFrame from inputs with correct column order
        input_df = pd.DataFrame([inputs])[required_features]

        # Validate numeric values and special features
        numeric_features = [f for f in required_features if f not in ['Weather_File', 'Building_Type', 'Renewable_Energy_Usage']]
        for feature in numeric_features:
            if not isinstance(inputs[feature], (int, float)):
                raise ValueError(f"{feature} must be a numeric value")
            if inputs[feature] < 0:
                raise ValueError(f"{feature} cannot be negative")
            
            # Special validation for Building_Orientation
            if feature == 'Building_Orientation':
                valid_orientations = [0, 45, 90, 135, 180, 225, 270, 315]
                if inputs[feature] not in valid_orientations:
                    raise ValueError(f"Building_Orientation must be one of {valid_orientations}")


        # Transform inputs using the preprocessor
        try:
            # Extract categorical features
            categorical_features = ['Weather_File', 'Building_Type', 'Renewable_Energy_Usage']
            numeric_features = [f for f in required_features if f not in categorical_features]
            
            # Ensure input data is properly formatted
            input_df = input_df.copy()
            for feat in categorical_features:
                input_df[feat] = input_df[feat].astype(str)
            for feat in numeric_features:
                input_df[feat] = input_df[feat].astype(float)
            
            # Define categories for ordinal encoding
            weather_file_categories = ['Historical', '2030']
            building_type_categories = ['Detached', 'Bungalow', 'Semi Detached', 'Terraced']
            renewable_energy_categories = ['Yes', 'No']
            
            # Create a new preprocessor if the saved one fails
            try:
                input_transformed = preprocessor.transform(input_df)
            except (AttributeError, ValueError) as e:
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OrdinalEncoder
                
                # Create new preprocessor with basic ordinal encoding
                new_preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OrdinalEncoder(categories=[weather_file_categories, building_type_categories, renewable_energy_categories]), categorical_features)
                    ],
                    remainder='passthrough'
                )
                
                # Fit and transform with new preprocessor
                input_transformed = new_preprocessor.fit_transform(input_df)
            
            # Ensure input_transformed is 2D
            if len(input_transformed.shape) == 1:
                input_transformed = input_transformed.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(input_transformed)
            return float(prediction[0])
        except Exception as e:
            raise ValueError(f"Error in preprocessing/prediction: {str(e)}")


    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Function to convert EUI to cost and emissions
def calculate_cost_and_emissions(eui, area, electricity_cost=0.34, emissions_factor=0.233):
    """Calculate annual energy cost and CO2 emissions based on EUI and area"""
    if eui is None:
        return None, None, None

    # EUI is in kWh/m²/year
    total_energy = eui * area  # kWh/year

    # Calculate cost ($)
    annual_cost = total_energy * electricity_cost

    # Calculate emissions (kg CO2e)
    emissions = total_energy * emissions_factor

    # Calculate carbon equivalents for better visualization
    trees_equivalent = emissions / 21  # Average tree absorbs ~21 kg CO2 per year
    car_miles_equivalent = emissions * 2.5  # ~0.4 kg CO2 per mile driven
    flights_equivalent = emissions / 255  # ~255 kg CO2 per short-haul flight
    smartphone_charging = emissions / 0.005  # ~0.005 kg CO2 per smartphone charge

    # Create a dictionary of carbon equivalents
    carbon_equivalents = {
        'trees_planted': round(trees_equivalent, 1),
        'car_miles': round(car_miles_equivalent, 0),
        'flights': round(flights_equivalent, 1),
        'smartphone_charges': round(smartphone_charging, 0)
    }

    return annual_cost, emissions, carbon_equivalents

# Main layout of the app
def main():
    """Main function to display the dashboard"""

    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor()

    # Sidebar with modern design
    st.sidebar.markdown(
        """
        <div style="padding: 20px 0; text-align: center;">
            <img src="https://img.icons8.com/fluency/96/000000/energy-meter.png" width="80">
            <h1 style="color: white !important; font-size: 28px; margin-top: 10px; font-weight: 700; font-family: 'Poppins', sans-serif;">
                Building Energy Analysis
            </h1>
            <p style="color: rgba(255, 255, 255, 0.8) !important; font-size: 16px; margin-top: 5px;">
                Smart. Sustainable. Simple.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Move the navigation selector to the bottom of the sidebar
    st.sidebar.markdown("""
    <div style="height: 20px;"></div>
    <div style="background: rgba(255, 255, 255, 0.1); height: 1px; margin: 10px 0 30px 0;"></div>
    """, unsafe_allow_html=True)
    
    # Create a container at the bottom of the sidebar for the page selector
    with st.sidebar:
        app_mode = option_menu(
        None,  # No title
        ["About", "Prediction", "What-If Analysis", "Explainability", "Recommendations", "Retrofit Analysis", "Reports"],
        icons=['info-circle', 'bar-chart', 'sliders', 'lightbulb', 'list-check', 'tools', 'file-earmark-text'],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {
                "padding": "20px",
                "background-color": "#0f766e",
                "border-radius": "0px",
                "margin-bottom": "20px",
            },
            "icon": {
                "color": "white",
                "font-size": "18px",
            },
            "menu-icon": {  
                "color": "white",
            },
            "nav-link": {
                "font-size": "16px",
                "font-weight": "600",
                "color": "rgba(255, 255, 255, 0.8)",
                "background-color": "rgba(255, 255, 255, 0.05)",
                "padding": "15px",
                "border-radius": "12px",
                "margin-bottom": "10px",
                "text-align": "left",
            },
            "nav-link-selected": {
                "background-color": "rgba(255, 255, 255, 0.2)",
                "color": "white",
                "font-weight": "700",
                "border-left": "4px solid white",
            }
        }
    )
    

    # Initialize session state for tracking changes across interactions
    if 'inputs' not in st.session_state:
        st.session_state.inputs = get_default_inputs()

    if 'baseline_prediction' not in st.session_state:
        st.session_state.baseline_prediction = None
        st.session_state.baseline_cost = None
        st.session_state.baseline_emissions = None
        st.session_state.carbon_equivalents = None

    # Calculate baseline values if not already set
    if st.session_state.baseline_prediction is None:
        try:
            st.session_state.baseline_prediction = predict_eui(st.session_state.inputs, model, preprocessor)
            if st.session_state.baseline_prediction is not None:
                cost, emissions, carbon_equivalents = calculate_cost_and_emissions(
                    st.session_state.baseline_prediction,
                    st.session_state.inputs['Total_Building_Area']
                )
                st.session_state.baseline_cost = cost
                st.session_state.baseline_emissions = emissions
                st.session_state.carbon_equivalents = carbon_equivalents
        except Exception as e:
            st.error(f"Error calculating baseline values: {e}")

    if 'modified_inputs' not in st.session_state:
        st.session_state.modified_inputs = st.session_state.inputs.copy()

    if 'cumulative_changes' not in st.session_state:
        st.session_state.cumulative_changes = {}

    # Different app pages based on selected mode
    if app_mode == "About":
        show_about_page()

    elif app_mode == "Prediction":
        display_prediction_page(model, preprocessor)

    elif app_mode == "What-If Analysis":
        display_what_if_analysis(model, preprocessor)

    elif app_mode == "Explainability":
        display_explainability_page(model, preprocessor)

    elif app_mode == "Recommendations":
        display_recommendations_page(model, preprocessor)
        
    elif app_mode == "Retrofit Analysis":
        display_retrofit_analysis_page(model, preprocessor)
    elif app_mode == "Reports":
        display_report_generation_page(model, preprocessor)

def display_prediction_page(model, preprocessor):
    """Display the prediction page with beautiful UI elements"""

    # Create a container for the title with animation
    st.markdown("""
    <div class="animated-fade">
        <h1>Building Energy Prediction</h1>
        <p style="font-size: 1.2rem; margin-bottom: 2rem; color: #64748b;">
            Input your building characteristics and get instant energy performance predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How to use this page", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h3 style="margin-top: 0;">Quick Guide</h3>
            <p>This page allows you to input detailed building characteristics and get predictions for:</p>
            <ul>
                <li><strong>Energy Use Intensity (EUI):</strong> The annual energy consumption per square meter</li>
                <li><strong>Annual Energy Cost:</strong> Estimated cost based on current energy prices</li>
                <li><strong>CO₂ Emissions:</strong> Environmental impact of your building's energy use</li>
            </ul>
            <p>Adjust the sliders and dropdowns to match your building's specifications, then click 'Calculate Energy Performance'.</p>
        </div>
        """, unsafe_allow_html=True)

    # Progress indicator for form completion (visual guide)
    st.markdown("""
    <div class="step-progress animated-fade">
        <div class="step-item active">1</div>
        <div class="step-item">2</div>
        <div class="step-item">3</div>
        <div class="step-item">4</div>
    </div>
    """, unsafe_allow_html=True)

    # Create a beautiful glass card for the form
    st.markdown('<div class="glass-card animated-fade">', unsafe_allow_html=True)

    # Create columns for form layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
            <i style="margin-right: 10px;">🏢</i> Building Characteristics
        </h3>
        """, unsafe_allow_html=True)

        # Get feature ranges for sliders
        ranges = get_feature_ranges()

        # Building type with improved selection
        building_type = st.selectbox(
            get_display_name("Building_Type"),
            options=["Detached", "Bungalow", "Semi Detached", "Terraced"],
            index=["Detached", "Bungalow", "Semi Detached", "Terraced"].index(st.session_state.inputs["Building_Type"])
        )
        
        building_orientation = st.selectbox(
            get_display_name("Building_Orientation"),
            options=[0, 45, 90, 135, 180, 225, 270, 315],
            index=[0, 45, 90, 135, 180, 225, 270, 315].index(st.session_state.inputs.get("Building_Orientation", 0))
        )

        # Area and shape with better sliders
        total_area = st.slider(
            get_display_name("Total_Building_Area") + " (m²)",
            min_value=ranges["Total_Building_Area"][0],
            max_value=ranges["Total_Building_Area"][1],
            value=st.session_state.inputs["Total_Building_Area"],
            step=10.0,
            help="The total floor area of your building in square meters"
        )

        window_wall_ratio = st.slider(
            get_display_name("Window_to_Wall_Ratio"),
            min_value=ranges["Window_to_Wall_Ratio"][0],
            max_value=ranges["Window_to_Wall_Ratio"][1],
            value=st.session_state.inputs["Window_to_Wall_Ratio"],
            step=0.01,
            help="The ratio of window area to wall area. Higher values typically mean more natural light but may affect thermal performance."
        )

    with col2:
        st.markdown("""
        <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
            <i style="margin-right: 10px;">⚡</i> Energy Systems
        </h3>
        """, unsafe_allow_html=True)

        # HVAC and hot water with improved sliders
        hvac_efficiency = st.slider(
            get_display_name("HVAC_Efficiency") + " (COP)",
            min_value=ranges["HVAC_Efficiency"][0],
            max_value=ranges["HVAC_Efficiency"][1],
            value=st.session_state.inputs["HVAC_Efficiency"],
            step=0.1,
            help="Coefficient of Performance for your HVAC system. Higher values indicate better efficiency."
        )

        hot_water_usage = st.slider(
            get_display_name("Domestic_Hot_Water_Usage"),
            min_value=ranges["Domestic_Hot_Water_Usage"][0],
            max_value=ranges["Domestic_Hot_Water_Usage"][1],
            value=st.session_state.inputs["Domestic_Hot_Water_Usage"],
            step=0.1,
            help="Hot water consumption level. Higher values indicate greater usage."
        )

        # Heating temperatures with better explanation
        heating_setpoint = st.slider(
            get_display_name("Heating_Setpoint_Temperature") + " (°C)",
            min_value=ranges["Heating_Setpoint_Temperature"][0],
            max_value=ranges["Heating_Setpoint_Temperature"][1],
            value=st.session_state.inputs["Heating_Setpoint_Temperature"],
            step=0.5,
            help="Target temperature when the building is occupied. Lower values reduce energy consumption."
        )

        heating_setback = st.slider(
            get_display_name("Heating_Setback_Temperature") + " (°C)",
            min_value=ranges["Heating_Setback_Temperature"][0],
            max_value=ranges["Heating_Setback_Temperature"][1],
            value=st.session_state.inputs["Heating_Setback_Temperature"],
            step=0.5,
            help="Reduced temperature when the building is unoccupied. Lower values save more energy."
        )

        # Renewable energy with better selection
        renewable_energy = st.selectbox(
            get_display_name("Renewable_Energy_Usage"),
            options=["Yes", "No"],
            index=["Yes", "No"].index(st.session_state.inputs["Renewable_Energy_Usage"]),
            help="Indicates whether the building uses renewable energy sources (solar, wind, etc.)"
        )

        # Weather scenario with improved description
        weather_file = st.selectbox(
            get_display_name("Weather_File"),
            options=["Historical", "2030"],
            index=["Historical", "2030"].index(st.session_state.inputs["Weather_File"]),
            help="Historical uses current climate data, 2030 uses projected climate data for future conditions"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Create beautiful tabs for more inputs
    st.markdown('<div class="glass-card animated-fade" style="animation-delay: 0.2s;">', unsafe_allow_html=True)
    insulation_tab, occupancy_tab, other_tab = st.tabs(["Insulation & Envelope", "Occupancy & Usage", "Air & Lighting"])

    with insulation_tab:
        st.markdown("""
        <p style="color: var(--text-muted); margin-bottom: 1rem;">
            Building envelope characteristics determine how well your building retains heat. 
            Lower U-values indicate better insulation.
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            wall_insulation = st.slider(
                get_display_name("Wall_Insulation_U-Value_Capped") + " (W/m²K)",
                min_value=ranges["Wall_Insulation_U-Value_Capped"][0],
                max_value=ranges["Wall_Insulation_U-Value_Capped"][1],
                value=st.session_state.inputs["Wall_Insulation_U-Value_Capped"],
                step=0.01,
                help="Thermal transmittance of walls. Lower values indicate better insulation."
            )

            roof_insulation = st.slider(
                get_display_name("Roof_Insulation_U-Value_Capped") + " (W/m²K)",
                ranges["Roof_Insulation_U-Value_Capped"][0],
                ranges["Roof_Insulation_U-Value_Capped"][1],
                st.session_state.inputs["Roof_Insulation_U-Value_Capped"],
                0.01,
                help="Thermal transmittance of roof. Lower values indicate better insulation."
            )

            floor_insulation = st.slider(
                get_display_name("Floor_Insulation_U-Value_Capped") + " (W/m²K)",
                min_value=ranges["Floor_Insulation_U-Value_Capped"][0],
                max_value=ranges["Floor_Insulation_U-Value_Capped"][1],
                value=st.session_state.inputs["Floor_Insulation_U-Value_Capped"],
                step=0.01,
                help="Thermal transmittance of floor. Lower values indicate better insulation."
            )

        with col2:
            window_insulation = st.slider(
                get_display_name("Window_Insulation_U-Value_Capped") + " (W/m²K)",
                min_value=ranges["Window_Insulation_U-Value_Capped"][0],
                max_value=ranges["Window_Insulation_U-Value_Capped"][1],
                value=st.session_state.inputs["Window_Insulation_U-Value_Capped"],
                step=0.01,
                help="Thermal transmittance of windows. Lower values indicate better insulation."
            )

            door_insulation = st.slider(
                get_display_name("Door_Insulation_U-Value_Capped") + " (W/m²K)",
                min_value=ranges["Door_Insulation_U-Value_Capped"][0],
                max_value=ranges["Door_Insulation_U-Value_Capped"][1],
                value=st.session_state.inputs["Door_Insulation_U-Value_Capped"],
                step=0.01,
                help="Thermal transmittance of doors. Lower values indicate better insulation."
            )

            air_change_rate = st.slider(
                get_display_name("Air_Change_Rate_Capped") + " (ACH)",
                min_value=ranges["Air_Change_Rate_Capped"][0],
                max_value=ranges["Air_Change_Rate_Capped"][1],
                value=st.session_state.inputs["Air_Change_Rate_Capped"],
                step=0.1,
                help="Air changes per hour. Lower values indicate a more airtight building."
            )

    with occupancy_tab:
        st.markdown("""
        <p style="color: var(--text-muted); margin-bottom: 1rem;">
            Occupancy patterns and equipment usage significantly impact energy consumption.
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            occupancy_level = st.slider(
                get_display_name("Occupancy_Level") + " (persons/100m²)",
                min_value=ranges["Occupancy_Level"][0],
                max_value=ranges["Occupancy_Level"][1],
                value=st.session_state.inputs["Occupancy_Level"],
                step=0.1,
                help="Number of people per 100 square meters. Higher values may increase energy needs."
            )

        with col2:
            equipment_density = st.slider(
                get_display_name("Equipment_Density") + " (W/m²)",
                min_value=ranges["Equipment_Density"][0],
                max_value=ranges["Equipment_Density"][1],
                value=st.session_state.inputs["Equipment_Density"],
                step=0.5,
                help="Power density of equipment. Higher values increase internal heat gains and electricity use."
            )

    with other_tab:
        st.markdown("""
        <p style="color: var(--text-muted); margin-bottom: 1rem;">
            Lighting efficiency affects both electricity use and internal heat gains.
        </p>
        """, unsafe_allow_html=True)
        
        lighting_density = st.slider(
            get_display_name("Lighting_Density") + " (W/m²)",
            min_value=ranges["Lighting_Density"][0],
            max_value=ranges["Lighting_Density"][1],
            value=st.session_state.inputs["Lighting_Density"],
            step=0.1,
            help="Power density of lighting. Lower values indicate more efficient lighting systems."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Beautiful calculate button with animation
    st.markdown("""
    <div style="display: flex; justify-content: center; margin: 2rem 0; animation: fadeIn 0.8s 0.4s ease-out forwards; opacity: 0;">
    """, unsafe_allow_html=True)

    # Collect all inputs
    inputs = {
        'Building_Type': building_type,
        'Total_Building_Area': total_area,
        'Window_to_Wall_Ratio': window_wall_ratio,
        'HVAC_Efficiency': hvac_efficiency,
        'Domestic_Hot_Water_Usage': hot_water_usage,
        'Heating_Setpoint_Temperature': heating_setpoint,
        'Heating_Setback_Temperature': heating_setback,
        'Building_Orientation': building_orientation,
        'Renewable_Energy_Usage': renewable_energy,
        'Weather_File': weather_file,
        'Wall_Insulation_U-Value_Capped': wall_insulation,
        'Roof_Insulation_U-Value_Capped': roof_insulation,
        'Floor_Insulation_U-Value_Capped': floor_insulation,
        'Window_Insulation_U-Value_Capped': window_insulation,
        'Door_Insulation_U-Value_Capped': door_insulation,
        'Air_Change_Rate_Capped': air_change_rate,
        'Occupancy_Level': occupancy_level,
        'Equipment_Density': equipment_density,
        'Lighting_Density': lighting_density
    }

    # Update session state with current inputs
    if st.button("Calculate Energy Performance", key="calculate_button"):
        with st.spinner("Analyzing building performance..."):
            try:
                st.session_state.inputs = inputs
                prediction = predict_eui(inputs, model, preprocessor)
                
                if prediction is not None:
                    st.session_state.baseline_prediction = prediction
                    cost, emissions, carbon_equivalents = calculate_cost_and_emissions(
                        prediction,
                        inputs['Total_Building_Area']
                    )
                    st.session_state.baseline_cost = cost
                    st.session_state.baseline_emissions = emissions
                    st.session_state.carbon_equivalents = carbon_equivalents
                else:
                    st.error("Error: Could not generate prediction. Please check your inputs.")
                    return
            except Exception as e:
                st.error(f"Error calculating energy performance: {str(e)}")
                return

            # Reset the what-if analysis state
            st.session_state.modified_inputs = inputs.copy()
            st.session_state.cumulative_changes = {}

    st.markdown('</div>', unsafe_allow_html=True)

    # Display predictions with beautiful animated cards
    if st.session_state.baseline_prediction is not None:
        st.markdown("""
        <div class="animated-fade" style="animation-delay: 0.5s;">
            <h2>Energy Performance Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a row for the gauge chart and metrics
        gauge_col, metrics_col = st.columns([1, 2])

        with gauge_col:
            # Display the gauge chart in a beautiful container
            st.markdown('<div class="eui-gauge-container">', unsafe_allow_html=True)
            gauge_chart = create_gauge_chart(st.session_state.baseline_prediction, "main")
            st.plotly_chart(gauge_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with metrics_col:
            col1, col2, col3 = st.columns(3)

            with col1:
                rating, color = get_efficiency_rating(st.session_state.baseline_prediction)
                st.markdown(f"""
                <div class="metric-container animated-fade" style="animation-delay: 0.6s;">
                    <div class="metric-label">Energy Use Intensity</div>
                    <div class="metric-value" style="color:{color};">{st.session_state.baseline_prediction:.1f}</div>
                    <div class="metric-unit">kWh/m²/year</div>
                    <div style="margin-top:15px; font-weight:600; color:{color}; font-size: 1.2rem;">Rating: {rating}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-container animated-fade" style="animation-delay: 0.7s;">
                    <div class="metric-label">Annual Energy Cost</div>
                    <div class="metric-value" style="color:var(--primary-color);">${st.session_state.baseline_cost:.2f}</div>
                    <div class="metric-unit">per year</div>
                    <div style="margin-top:15px; color:var(--text-muted); font-size: 0.9rem;">Based on current energy prices</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-container animated-fade" style="animation-delay: 0.8s;">
                    <div class="metric-label">CO₂ Emissions</div>
                    <div class="metric-value" style="color:var(--secondary-color);">{st.session_state.baseline_emissions:.1f}</div>
                    <div class="metric-unit">kg CO₂e/year</div>
                    <div style="margin-top:15px; color:var(--text-muted); font-size: 0.9rem;">Environmental impact</div>
                </div>
                """, unsafe_allow_html=True)

        # Performance rating with beautiful card
        st.markdown('<div class="animated-fade" style="animation-delay: 0.9s;">', unsafe_allow_html=True)
        st.subheader("Energy Performance Rating")

        # Determine rating based on EUI
        if st.session_state.baseline_prediction < 100:
            rating = "A (Excellent)"
            color = "#10b981"  # Emerald 500
            bg_gradient = "linear-gradient(135deg, #10b981 0%, #34d399 100%)"
            description = "Excellent energy performance, minimal environmental impact."
        elif st.session_state.baseline_prediction < 150:
            rating = "B (Good)"
            color = "#34d399"  # Emerald 400
            bg_gradient = "linear-gradient(135deg, #34d399 0%, #6ee7b7 100%)"
            description = "Good energy performance, lower than average environmental impact."
        elif st.session_state.baseline_prediction < 200:
            rating = "C (Average)"
            color = "#fbbf24"  # Amber 400
            bg_gradient = "linear-gradient(135deg, #fbbf24 0%, #fcd34d 100%)"
            description = "Average energy performance, typical environmental impact."
        elif st.session_state.baseline_prediction < 250:
            rating = "D (Below Average)"
            color = "#f59e0b"  # Amber 500
            bg_gradient = "linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)"
            description = "Below average energy performance, higher than typical environmental impact."
        elif st.session_state.baseline_prediction < 300:
            rating = "E (Poor)"
            color = "#f97316"  # Orange 500
            bg_gradient = "linear-gradient(135deg, #f97316 0%, #fb923c 100%)"
            description = "Poor energy performance, considerable environmental impact."
        elif st.session_state.baseline_prediction < 350:
            rating = "F (Very Poor)"
            color = "#ef4444"  # Red 500
            bg_gradient = "linear-gradient(135deg, #ef4444 0%, #f87171 100%)"
            description = "Very poor energy performance, significant environmental impact."
        else:
            rating = "G (Very Poor)"
            color = "#ef4444"  # Red 500
            bg_gradient = "linear-gradient(135deg, #ef4444 0%, #f87171 100%)"
            description = "Very poor energy performance, significant environmental impact."

        # Display rating with beautiful styling
        st.markdown(f"""
        <div class="rating-card" style="background: {bg_gradient};">
            <div class="rating-title">Rating: {rating}</div>
            <div class="rating-desc">{description}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Carbon equivalences with beautiful icons
        if st.session_state.carbon_equivalents:
            st.markdown("""
            <div class="animated-fade" style="animation-delay: 1s;">
                <h3>Carbon Impact Visualization</h3>
                <p style="color: var(--text-muted); margin-bottom: 1rem;">
                    Your building's carbon emissions are equivalent to:
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create beautiful cards for carbon equivalents
            cols = st.columns(4)
            
            icons = ["🌳", "🚗", "✈️", "📱"]
            titles = ["Trees Needed", "Car Miles", "Flight Equivalents", "Phone Charges"]
            values = [
                st.session_state.carbon_equivalents["trees_planted"],
                st.session_state.carbon_equivalents["car_miles"],
                st.session_state.carbon_equivalents["flights"],
                st.session_state.carbon_equivalents["smartphone_charges"]
            ]
            descriptions = [
                "Trees needed per year to offset emissions",
                "Miles driven in an average car",
                "Short-haul flights",
                "Smartphone charges"
            ]
            
            for i, col in enumerate(cols):
                col.markdown(f"""
                <div class="metric-container animated-fade" style="animation-delay: {1.1 + i*0.1}s; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icons[i]}</div>
                    <div style="font-weight: 600; color: var(--text-color); font-size: 1.1rem;">{titles[i]}</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: var(--primary-color); margin: 0.5rem 0;">{values[i]:,}</div>
                    <div style="color: var(--text-muted); font-size: 0.9rem;">{descriptions[i]}</div>
                </div>
                """, unsafe_allow_html=True)

        # Generate report buttons with beautiful styling
        st.markdown("""
        <div class="animated-fade" style="animation-delay: 1.2s;">
            <h3>Generate Reports</h3>
            <p style="color: var(--text-muted); margin-bottom: 1rem;">
                Create detailed reports to share with stakeholders or for your records.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Generate PDF Report", key="pdf_button"):
                with st.spinner("Creating beautiful PDF report..."):
                    try:
                        pdf_data = generate_enhanced_pdf_report(
                            inputs=st.session_state.inputs,
                            eui=st.session_state.baseline_prediction,
                            cost=st.session_state.baseline_cost,
                            emissions=st.session_state.baseline_emissions,
                            rating=rating,
                            model=model,
                            preprocessor=preprocessor
                        )

                        # Create download button for PDF
                        b64_pdf = base64.b64encode(pdf_data).decode()
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 1rem;">
                            <a href="data:application/pdf;base64,{b64_pdf}" download="building_energy_report.pdf" 
                               style="display: inline-block; background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
                               color: white; padding: 0.6rem 1.5rem; border-radius: 50px; text-decoration: none; font-weight: 600;
                               box-shadow: 0 4px 10px rgba(15, 118, 110, 0.2); transition: all 0.3s ease;">
                                <i style="margin-right: 0.5rem;">📄</i> Download PDF Report
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating PDF report: {e}")

        with col2:
            if st.button("Generate Excel Report", key="excel_button"):
                with st.spinner("Creating detailed Excel report..."):
                    try:
                        excel_data = generate_enhanced_excel_report(
                            inputs=st.session_state.inputs,
                            eui=st.session_state.baseline_prediction,
                            cost=st.session_state.baseline_cost,
                            emissions=st.session_state.baseline_emissions,
                            rating=rating
                        )

                        # Create download button for Excel
                        b64_excel = base64.b64encode(excel_data).decode()
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 1rem;">
                            <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" 
                               download="building_energy_report.xlsx" 
                               style="display: inline-block; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
                               color: white; padding: 0.6rem 1.5rem; border-radius: 50px; text-decoration: none; font-weight: 600;
                               box-shadow: 0 4px 10px rgba(59, 130, 246, 0.2); transition: all 0.3s ease;">
                                <i style="margin-right: 0.5rem;">📊</i> Download Excel Report
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating Excel report: {e}")

    # Add footer
    st.markdown("""
    <div class="footer">
        <p>Building Energy Analysis Dashboard • Version 2.0</p>
        <p>For questions, feedback, or support, please contact the development team (The Green Guardians)</p>
    </div>
    """, unsafe_allow_html=True)

# Function to display the What-If Analysis page
def display_what_if_analysis(model, preprocessor):
    """Display the What-If Analysis page with enhanced interactive features and visualizations"""

    st.title("What-If Analysis")
    
    # Add a descriptive subtitle
    st.markdown("""
    <p style="font-size: 1.2rem; margin-bottom: 1.5rem; color: #64748b;">
        Explore how changes to your building would affect energy performance in real-time.
    </p>
    """, unsafe_allow_html=True)

    with st.expander("How to use this page", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h3 style="margin-top: 0;">Interactive Building Simulation</h3>
            <p>This tool allows you to experiment with different building parameters and instantly see their impact:</p>
            <ol>
                <li><strong>Select a parameter</strong> to modify from the dropdown menu</li>
                <li><strong>Adjust the value</strong> using the slider or selector</li>
                <li><strong>Apply the change</strong> to see its effect on energy performance</li>
                <li><strong>Make multiple changes</strong> to see their cumulative impact</li>
                <li><strong>Reset anytime</strong> to start over with your baseline building</li>
            </ol>
            <p>The dashboard shows real-time comparisons between your baseline building and the modified version.</p>
        </div>
        """, unsafe_allow_html=True)

    # Get feature ranges for sliders
    ranges = get_feature_ranges()

    # Create a two-column layout for the main content
    left_col, right_col = st.columns([3, 2])

    with left_col:
        # Display baseline prediction in a nice card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("Baseline Building Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            prediction_text = f"{st.session_state.baseline_prediction:.1f} kWh/m²/year" if st.session_state.baseline_prediction is not None else "N/A"
            rating, color = get_efficiency_rating(st.session_state.baseline_prediction) if st.session_state.baseline_prediction is not None else ("N/A", "#64748b")
            
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.2rem;">Energy Use Intensity</p>
                <p style="color: {color}; font-size: 1.8rem; font-weight: 700; margin: 0;">{prediction_text}</p>
                <p style="color: {color}; font-size: 1rem; margin-top: 0.2rem;">Rating: {rating}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            cost_text = f"${st.session_state.baseline_cost:.2f}" if st.session_state.baseline_cost is not None else "N/A"
            
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.2rem;">Annual Energy Cost</p>
                <p style="color: #0f766e; font-size: 1.8rem; font-weight: 700; margin: 0;">{cost_text}</p>
                <p style="color: #64748b; font-size: 1rem; margin-top: 0.2rem;">per year</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            emissions_text = f"{st.session_state.baseline_emissions:.1f} kg CO₂e/year" if st.session_state.baseline_emissions is not None else "N/A"
            
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.2rem;">CO₂ Emissions</p>
                <p style="color: #10b981; font-size: 1.8rem; font-weight: 700; margin: 0;">{emissions_text.split()[0]}</p>
                <p style="color: #64748b; font-size: 1rem; margin-top: 0.2rem;">kg CO₂e/year</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # What-if analysis interface
        st.markdown('<div class="glass-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.header("Modify Building Parameters")

        # Select parameter to modify with improved UI
        param_options = list(DEFAULT_FEATURES.keys())
        
        # Group parameters by category for better organization
        param_categories = {
            "Building Envelope": ["Wall Insulation", "Roof Insulation", "Floor Insulation", "Window Insulation", "Door Insulation", "Air Change Rate", "Window to Wall Ratio"],
            "Systems & Equipment": ["HVAC Efficiency", "Domestic Hot Water Usage", "Lighting Density", "Equipment Density"],
            "Operations": ["Heating Setpoint Temperature", "Heating Setback Temperature", "Occupancy Level", "Renewable Energy Usage"],
            "Building Properties": ["Building Type", "Total Building Area", "Building Orientation", "Weather File"]
        }
        
        # Create a more organized parameter selection
        st.subheader("Step 1: Select a parameter to modify")
        
        # Initialize or maintain the selected category and parameter in session state
        if 'selected_category' not in st.session_state:
            st.session_state.selected_category = "Building Envelope"
        if 'selected_param' not in st.session_state:
            st.session_state.selected_param = None
        
        # Create function to update the selected parameter when a button is clicked
        def set_param(param):
            st.session_state.selected_param = param
        
        # Function to change category
        def set_category(category):
            st.session_state.selected_category = category
        
        # Create a more user-friendly horizontal category selector
        category_cols = st.columns(len(param_categories))
        for i, (category, _) in enumerate(param_categories.items()):
            with category_cols[i]:
                # Highlight the selected category
                button_style = "primary" if category == st.session_state.selected_category else "secondary"
                if st.button(category, key=f"cat_{category}", use_container_width=True, type=button_style):
                    set_category(category)
        
        # Show parameters for the selected category
        category_params = [p for p in param_options if any(cp in p for cp in param_categories[st.session_state.selected_category])]
        
        if category_params:
            # Two-column layout for parameters
            col1, col2 = st.columns([3, 1])
            with col1:
                param_choice = st.selectbox(
                    f"Select a {st.session_state.selected_category.lower()} parameter",
                    category_params,
                    key=f"param_select_{st.session_state.selected_category}"
                )
            with col2:
                if st.button(f"Use parameter", key=f"use_param_{st.session_state.selected_category}", use_container_width=True):
                    set_param(param_choice)
        else:
            st.info(f"No {st.session_state.selected_category.lower()} parameters available.")
        
        # Display a message showing the currently selected parameter
        if st.session_state.selected_param:
            st.success(f"Currently modifying: **{st.session_state.selected_param}**")
        
        # Fallback if no parameter was selected yet
        if st.session_state.selected_param is None:
            selected_param = st.selectbox("Or select any parameter", param_options, key="fallback_param")
            if st.button("Use this parameter", key="use_fallback"):
                st.session_state.selected_param = selected_param
        
        # Use the selected parameter (either from tabs or fallback)
        selected_param = st.session_state.selected_param or param_options[0]

        # Get original feature name
        feature_name = get_feature_name(selected_param)

        # Get current value (from cumulative changes if exists, otherwise baseline)
        current_value = st.session_state.modified_inputs.get(feature_name, st.session_state.inputs[feature_name])

        # Display current value with better formatting
        st.markdown(f"""
        <div style="background-color: rgba(15, 118, 110, 0.1); padding: 0.8rem; border-radius: 0.5rem; margin: 1rem 0;">
            <p style="margin: 0; font-size: 0.9rem; color: #64748b;">Current value:</p>
            <p style="margin: 0; font-size: 1.2rem; font-weight: 600; color: #0f766e;">{current_value}</p>
        </div>
        """, unsafe_allow_html=True)

        # Step 2: Modify the parameter
        st.subheader("Step 2: Set a new value")

        # Create appropriate input widget based on parameter type
        new_value = None

        if feature_name in ['Building_Type', 'Weather_File', 'Renewable_Energy_Usage']:
            # Categorical parameters with improved UI
            if feature_name == 'Building_Type':
                options = ["Detached", "Bungalow", "Semi Detached", "Terraced"]
                icons = ["🏠", "🏡", "🏘️", "🏢"]
                
                # Create a more visual selector for building type
                st.write("Building Type:")
                cols = st.columns(len(options))
                for i, (col, option, icon) in enumerate(zip(cols, options, icons)):
                    with col:
                        selected = st.button(
                            f"{icon} {option}",
                            key=f"btn_{option}",
                            help=f"Select {option} building type",
                            use_container_width=True
                        )
                        if selected:
                            new_value = option
                
                # Fallback to regular selectbox if no button was clicked
                if new_value is None:
                    new_value = st.selectbox(
                        f"New value for {selected_param}",
                        options,
                        index=options.index(current_value)
                    )
                    
            elif feature_name == 'Weather_File':
                options = ["Historical", "2030"]
                
                # Create a more visual selector for weather file
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📜 Historical", use_container_width=True, help="Use historical weather data"):
                        new_value = "Historical"
                with col2:
                    if st.button("🔮 Future (2030)", use_container_width=True, help="Use projected 2030 weather data"):
                        new_value = "2030"
                
                # Fallback to regular selectbox if no button was clicked
                if new_value is None:
                    new_value = st.selectbox(
                        f"New value for {selected_param}",
                        options,
                        index=options.index(current_value)
                    )
                    
            elif feature_name == 'Renewable_Energy_Usage':
                options = ["Yes", "No"]
                
                # Create a more visual selector for renewable energy
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("♻️ Yes", use_container_width=True, help="Building uses renewable energy"):
                        new_value = "Yes"
                with col2:
                    if st.button("⚡ No", use_container_width=True, help="Building does not use renewable energy"):
                        new_value = "No"
                
                # Fallback to regular selectbox if no button was clicked
                if new_value is None:
                    new_value = st.selectbox(
                        f"New value for {selected_param}",
                        options,
                        index=options.index(current_value)
                    )
        else:
            # Numerical parameters - use slider with appropriate range and better UI
            if feature_name in ranges:
                min_val, max_val = ranges[feature_name]
                step = 0.01 if max_val - min_val < 1 else 0.1
                if feature_name == 'Total_Building_Area':
                    step = 10.0

                # Add units and help text based on parameter
                units = ""
                help_text = ""
                
                if "Temperature" in feature_name:
                    units = "°C"
                    help_text = "Higher values typically increase energy consumption"
                elif "U-Value" in feature_name:
                    units = "W/m²K"
                    help_text = "Lower values indicate better insulation and reduce energy consumption"
                elif feature_name == "HVAC_Efficiency":
                    units = "COP"
                    help_text = "Higher values indicate more efficient systems and reduce energy consumption"
                elif "Density" in feature_name:
                    units = "W/m²"
                    help_text = "Lower values typically reduce energy consumption"
                elif feature_name == "Air_Change_Rate_Capped":
                    units = "ACH"
                    help_text = "Lower values indicate a more airtight building and reduce energy consumption"
                elif feature_name == "Total_Building_Area":
                    units = "m²"
                    help_text = "Building floor area"
                elif feature_name == "Window_to_Wall_Ratio":
                    help_text = "Ratio of window area to wall area (0.1 = 10%)"
                
                # Create a more informative slider
                # Ensure consistent types to avoid errors
                min_val_typed = float(min_val) if isinstance(current_value, float) else int(min_val)
                max_val_typed = float(max_val) if isinstance(current_value, float) else int(max_val)
                step_typed = float(step) if isinstance(current_value, float) else int(step)
                current_value_typed = float(current_value) if isinstance(current_value, float) else int(current_value)
                
                new_value = st.slider(
                    f"New value for {selected_param} ({units})" if units else f"New value for {selected_param}",
                    min_value=min_val_typed,
                    max_value=max_val_typed,
                    value=current_value_typed,
                    step=step_typed,
                    help=help_text
                )
                
                # Add a visual indicator of change
                if new_value != current_value:
                    change_pct = ((new_value - current_value) / current_value) * 100 if current_value != 0 else 0
                    change_direction = "increase" if new_value > current_value else "decrease"
                    
                    st.markdown(f"""
                    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: {'#10b981' if 'U-Value' in feature_name and change_direction == 'decrease' or 'Efficiency' in feature_name and change_direction == 'increase' else '#ef4444'};">
                        {change_direction.title()} of {abs(change_pct):.1f}% from current value
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Fallback for parameters without defined ranges
                # Ensure consistent numeric types to avoid StreamlitMixedNumericTypesError
                if isinstance(current_value, float):
                    new_value = st.number_input(
                        f"New value for {selected_param}",
                        value=float(current_value),
                        step=0.1,
                        format="%.2f"
                    )
                else:
                    new_value = st.number_input(
                        f"New value for {selected_param}",
                        value=int(current_value),
                        step=1
                    )

        # Step 3: Apply changes or reset
        st.subheader("Step 3: Apply changes")
        
        # Buttons to apply change or reset with better styling
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Apply Change", type="primary", use_container_width=True):
                # Store the original value if this is the first change to this parameter
                if feature_name not in st.session_state.cumulative_changes:
                    st.session_state.cumulative_changes[feature_name] = st.session_state.inputs[feature_name]

                # Update the modified inputs
                st.session_state.modified_inputs[feature_name] = new_value

                # Re-run prediction with all cumulative changes
                new_prediction = predict_eui(st.session_state.modified_inputs, model, preprocessor)
                new_cost, new_emissions, carbon_equivalents = calculate_cost_and_emissions(
                    new_prediction,
                    st.session_state.modified_inputs.get('Total_Building_Area', st.session_state.inputs['Total_Building_Area'])
                )
                st.session_state.carbon_equivalents = carbon_equivalents

                # Store results to display
                st.session_state.new_prediction = new_prediction
                st.session_state.new_cost = new_cost
                st.session_state.new_emissions = new_emissions

                # Force a rerun to update the UI
                st.rerun()

        with col2:
            if st.button("Reset All Changes", type="secondary", use_container_width=True):
                # Reset all modifications
                st.session_state.modified_inputs = st.session_state.inputs.copy()
                st.session_state.cumulative_changes = {}

                # Reset prediction to baseline
                st.session_state.new_prediction = st.session_state.baseline_prediction
                st.session_state.new_cost = st.session_state.baseline_cost
                st.session_state.new_emissions = st.session_state.baseline_emissions

                # Force a rerun to update the UI
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Right column for results visualization
    with right_col:
        # Display results of the what-if analysis
        if st.session_state.cumulative_changes:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.header("Modified Building Performance")

            # Calculate new prediction if not already done
            if 'new_prediction' not in st.session_state:
                new_prediction = predict_eui(st.session_state.modified_inputs, model, preprocessor)
                new_cost, new_emissions, carbon_equivalents = calculate_cost_and_emissions(
                    new_prediction,
                    st.session_state.modified_inputs.get('Total_Building_Area', st.session_state.inputs['Total_Building_Area'])
                )
                st.session_state.carbon_equivalents = carbon_equivalents

                st.session_state.new_prediction = new_prediction
                st.session_state.new_cost = new_cost
                st.session_state.new_emissions = new_emissions

            # Calculate differences with None checks
            new_prediction = st.session_state.get('new_prediction', 0)
            baseline_prediction = st.session_state.get('baseline_prediction', 0)
            new_cost = st.session_state.get('new_cost', 0)
            baseline_cost = st.session_state.get('baseline_cost', 0)
            new_emissions = st.session_state.get('new_emissions', 0)
            baseline_emissions = st.session_state.get('baseline_emissions', 0)
            
            # Safety check to ensure we have valid values
            if new_prediction is None: new_prediction = 0
            if baseline_prediction is None: baseline_prediction = 0
            if new_cost is None: new_cost = 0
            if baseline_cost is None: baseline_cost = 0
            if new_emissions is None: new_emissions = 0
            if baseline_emissions is None: baseline_emissions = 0
            
            eui_diff = new_prediction - baseline_prediction
            eui_pct = (eui_diff / baseline_prediction) * 100 if baseline_prediction != 0 else 0

            cost_diff = new_cost - baseline_cost
            cost_pct = (cost_diff / baseline_cost) * 100 if baseline_cost != 0 else 0

            emissions_diff = new_emissions - baseline_emissions
            emissions_pct = (emissions_diff / baseline_emissions) * 100 if baseline_emissions != 0 else 0

            # Display metrics with deltas and improved styling
            col1, col2, col3 = st.columns(3)

            with col1:
                rating, color = get_efficiency_rating(st.session_state.new_prediction)
                delta_color = "normal" if eui_diff >= 0 else "inverse"
                
                st.markdown(f"""
                <div style="text-align: center;">
                    <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.2rem;">Energy Use Intensity</p>
                    <p style="color: {color}; font-size: 1.8rem; font-weight: 700; margin: 0;">{st.session_state.new_prediction:.1f}</p>
                    <p style="color: {color}; font-size: 1rem; margin-top: 0.2rem;">kWh/m²/year</p>
                    <p style="color: {'#ef4444' if eui_diff > 0 else '#10b981'}; font-size: 0.9rem; margin-top: 0.5rem;">
                        {eui_diff:.1f} kWh/m²/year ({eui_pct:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="text-align: center;">
                    <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.2rem;">Annual Energy Cost</p>
                    <p style="color: #0f766e; font-size: 1.8rem; font-weight: 700; margin: 0;">${st.session_state.new_cost:.2f}</p>
                    <p style="color: #64748b; font-size: 1rem; margin-top: 0.2rem;">per year</p>
                    <p style="color: {'#ef4444' if cost_diff > 0 else '#10b981'}; font-size: 0.9rem; margin-top: 0.5rem;">
                        ${cost_diff:.2f} ({cost_pct:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style="text-align: center;">
                    <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.2rem;">CO₂ Emissions</p>
                    <p style="color: #10b981; font-size: 1.8rem; font-weight: 700; margin: 0;">{st.session_state.new_emissions:.1f}</p>
                    <p style="color: #64748b; font-size: 1rem; margin-top: 0.2rem;">kg CO₂e/year</p>
                    <p style="color: {'#ef4444' if emissions_diff > 0 else '#10b981'}; font-size: 0.9rem; margin-top: 0.5rem;">
                        {emissions_diff:.1f} kg ({emissions_pct:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display gauge chart comparison
            st.markdown('<div class="glass-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            
            # Create gauge charts for baseline and modified
            baseline_gauge = create_gauge_chart(st.session_state.baseline_prediction, "baseline")
            modified_gauge = create_gauge_chart(st.session_state.new_prediction, "modified")
            
            # Create a figure with two subplots for the gauges
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}]],
                subplot_titles=("Baseline Building", "Modified Building")
            )
            
            # Add the gauge charts to the subplots
            baseline_rating, baseline_color = get_efficiency_rating(st.session_state.baseline_prediction)
            modified_rating, modified_color = get_efficiency_rating(st.session_state.new_prediction)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=st.session_state.baseline_prediction,
                    title={'text': f"Rating: {baseline_rating}", 'font': {'size': 16, 'color': baseline_color}},
                    number={'suffix': " kWh/m²/yr", 'font': {'size': 20, 'color': baseline_color}},
                    gauge={
                        'axis': {'range': [0, 400], 'tickwidth': 1, 'tickcolor': "#64748b"},
                        'bar': {'color': baseline_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#e2e8f0",
                        'steps': [
                            {'range': [0, 50], 'color': '#10b981'},
                            {'range': [50, 100], 'color': '#34d399'},
                            {'range': [100, 150], 'color': '#6ee7b7'},
                            {'range': [150, 200], 'color': '#fcd34d'},
                            {'range': [200, 250], 'color': '#f59e0b'},
                            {'range': [250, 300], 'color': '#f97316'},
                            {'range': [300, 350], 'color': '#ef4444'},
                            {'range': [350, 400], 'color': '#b91c1c'},
                        ],
                        'threshold': {
                            'line': {'color': "#1e293b", 'width': 4},
                            'thickness': 0.8,
                            'value': st.session_state.baseline_prediction
                        }
                    }
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=st.session_state.new_prediction,
                    title={'text': f"Rating: {modified_rating}", 'font': {'size': 16, 'color': modified_color}},
                    number={'suffix': " kWh/m²/yr", 'font': {'size': 20, 'color': modified_color}},
                    gauge={
                        'axis': {'range': [0, 400], 'tickwidth': 1, 'tickcolor': "#64748b"},
                        'bar': {'color': modified_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#e2e8f0",
                        'steps': [
                            {'range': [0, 50], 'color': '#10b981'},
                            {'range': [50, 100], 'color': '#34d399'},
                            {'range': [100, 150], 'color': '#6ee7b7'},
                            {'range': [150, 200], 'color': '#fcd34d'},
                            {'range': [200, 250], 'color': '#f59e0b'},
                            {'range': [250, 300], 'color': '#f97316'},
                            {'range': [300, 350], 'color': '#ef4444'},
                            {'range': [350, 400], 'color': '#b91c1c'},
                        ],
                        'threshold': {
                            'line': {'color': "#1e293b", 'width': 4},
                            'thickness': 0.8,
                            'value': st.session_state.new_prediction
                        }
                    }
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a comparison arrow between the gauges
            direction = "improved" if eui_diff < 0 else "worsened"
            arrow = "↓" if eui_diff < 0 else "↑"
            
            st.markdown(f"""
            <div style="text-align: center; margin: 0.5rem 0 1.5rem 0;">
                <p style="color: {'#10b981' if eui_diff < 0 else '#ef4444'}; font-size: 1.2rem; font-weight: 600;">
                    Energy performance {direction} by {abs(eui_pct):.1f}% {arrow}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display summary of applied changes
            st.markdown('<div class="glass-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.subheader("Applied Changes")

            changes_data = []
            for feature, original_value in st.session_state.cumulative_changes.items():
                display_name = get_display_name(feature)
                new_value = st.session_state.modified_inputs[feature]
                
                # Calculate percent change for numerical values
                pct_change = ""
                if isinstance(original_value, (int, float)) and isinstance(new_value, (int, float)):
                    if original_value != 0:
                        change = ((new_value - original_value) / original_value) * 100
                        pct_change = f" ({change:+.1f}%)"
                
                changes_data.append({
                    "Parameter": display_name,
                    "Original Value": original_value,
                    "New Value": f"{new_value}{pct_change}"
                })

            if changes_data:
                # Create a more visually appealing table
                for change in changes_data:
                    st.markdown(f"""
                    <div style="display: flex; margin-bottom: 0.8rem; padding: 0.5rem; background-color: rgba(255,255,255,0.5); border-radius: 0.5rem;">
                        <div style="flex: 2; font-weight: 600; color: #0f766e;">{change['Parameter']}</div>
                        <div style="flex: 1; text-align: center; color: #64748b;">{change['Original Value']}</div>
                        <div style="flex: 1; text-align: center; font-weight: 600; color: #0f766e;">{change['New Value']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Display recommendations based on changes
            st.markdown('<div class="glass-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            
            if eui_diff < 0:
                st.markdown(f"""
                <div style="background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; padding: 1rem; border-radius: 0.5rem;">
                    <h4 style="margin-top: 0; color: #10b981;">Positive Impact</h4>
                    <p>Your changes reduced energy use by {abs(eui_diff):.1f} kWh/m²/year ({abs(eui_pct):.1f}%), saving ${abs(cost_diff):.2f} and {abs(emissions_diff):.1f} kg CO₂e annually.</p>
                    <p style="margin-bottom: 0;">Consider applying these changes to your building to improve energy efficiency.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 1rem; border-radius: 0.5rem;">
                    <h4 style="margin-top: 0; color: #ef4444;">Negative Impact</h4>
                    <p>Your changes increased energy use by {eui_diff:.1f} kWh/m²/year ({eui_pct:.1f}%), costing an additional ${cost_diff:.2f} and producing {emissions_diff:.1f} kg CO₂e more annually.</p>
                    <p style="margin-bottom: 0;">Consider adjusting these parameters to find more energy-efficient options.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Display a prompt to make changes when no changes have been applied yet
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem;">
                <img src="https://img.icons8.com/fluency/96/000000/edit.png" width="64" style="margin-bottom: 1rem;">
                <h3 style="margin-top: 0; color: #0f766e;">Make Your First Change</h3>
                <p style="color: #64748b;">Select a parameter on the left and modify its value to see how it affects your building's energy performance.</p>
                <p style="color: #64748b; font-size: 0.9rem;">The results will appear here after you apply changes.</p>
            </div>
            """, unsafe_allow_html=True)

    # Display comparison visualization at the bottom
    if st.session_state.cumulative_changes:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Impact Visualization")

        # Create tabs for different visualizations
        viz_tabs = st.tabs(["Bar Chart", "Radar Chart", "Savings Breakdown"])
        
        with viz_tabs[0]:
            # Create data for comparison bar chart
            comparison_data = pd.DataFrame({
                'Scenario': ['Baseline', 'Modified'],
                'EUI (kWh/m²/year)': [st.session_state.baseline_prediction, st.session_state.new_prediction],
                'Annual Cost ($)': [st.session_state.baseline_cost, st.session_state.new_cost],
                'CO₂ Emissions (kg CO₂e/year)': [st.session_state.baseline_emissions, st.session_state.new_emissions]
            })

            # Melt data for easier plotting
            melted_data = pd.melt(comparison_data, id_vars=['Scenario'], 
                                var_name='Metric', value_name='Value')

            # Create bar chart with improved styling
            fig = px.bar(
                melted_data,
                x='Scenario',
                y='Value',
                color='Metric',
                barmode='group',
                title='Comparison of Energy Performance',
                color_discrete_sequence=['#0f766e', '#0284c7', '#10b981'],
                template="plotly_white"
            )
            
            # Improve layout
            fig.update_layout(
                legend_title="",
                xaxis_title="",
                yaxis_title="",
                font=dict(family="Inter, sans-serif", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:
            # Create radar chart for more visual comparison
            # Normalize the values for radar chart
            baseline_values = [
                st.session_state.baseline_prediction / 400 * 100,  # Normalize EUI to 0-100 scale
                st.session_state.baseline_cost / 5000 * 100,       # Normalize cost to 0-100 scale
                st.session_state.baseline_emissions / 5000 * 100   # Normalize emissions to 0-100 scale
            ]
            
            modified_values = [
                st.session_state.new_prediction / 400 * 100,
                st.session_state.new_cost / 5000 * 100,
                st.session_state.new_emissions / 5000 * 100
            ]
            
            # Create radar chart
            categories = ['Energy Use Intensity', 'Annual Cost', 'CO₂ Emissions']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=baseline_values,
                theta=categories,
                fill='toself',
                name='Baseline Building',
                line_color='#0f766e',
                fillcolor='rgba(15, 118, 110, 0.2)'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=modified_values,
                theta=categories,
                fill='toself',
                name='Modified Building',
                line_color='#0284c7',
                fillcolor='rgba(2, 132, 199, 0.2)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title="Performance Comparison (lower is better)",
                font=dict(family="Inter, sans-serif"),
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[2]:
            # Create a breakdown of savings/costs
            area = st.session_state.inputs.get('Total_Building_Area', 1000)
            
            # Calculate total energy savings
            energy_savings = (st.session_state.baseline_prediction - st.session_state.new_prediction) * area  # kWh/year
            cost_savings = st.session_state.baseline_cost - st.session_state.new_cost  # $/year
            emissions_savings = st.session_state.baseline_emissions - st.session_state.new_emissions  # kg CO2e/year
            
            # Create a more detailed breakdown
            st.markdown("""
            <h4 style="margin-bottom: 1rem;">Annual Savings Breakdown</h4>
            """, unsafe_allow_html=True)
            
            # Create columns for the breakdown
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background-color: rgba(15, 118, 110, 0.1); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h5 style="margin-top: 0; color: #0f766e;">Energy Savings</h5>
                    <p style="font-size: 1.8rem; font-weight: 700; margin: 0; color: {'#10b981' if energy_savings > 0 else '#ef4444'};">
                        {energy_savings:.1f} kWh
                    </p>
                    <p style="color: #64748b; font-size: 0.9rem;">per year</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: rgba(2, 132, 199, 0.1); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h5 style="margin-top: 0; color: #0284c7;">Cost Savings</h5>
                    <p style="font-size: 1.8rem; font-weight: 700; margin: 0; color: {'#10b981' if cost_savings > 0 else '#ef4444'};">
                        ${cost_savings:.2f}
                    </p>
                    <p style="color: #64748b; font-size: 0.9rem;">per year</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background-color: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h5 style="margin-top: 0; color: #10b981;">CO₂ Reduction</h5>
                    <p style="font-size: 1.8rem; font-weight: 700; margin: 0; color: {'#10b981' if emissions_savings > 0 else '#ef4444'};">
                        {emissions_savings:.1f} kg
                    </p>
                    <p style="color: #64748b; font-size: 0.9rem;">per year</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add long-term projections
            st.markdown("""
            <h4 style="margin: 1.5rem 0 1rem 0;">Long-term Projections</h4>
            """, unsafe_allow_html=True)
            
            # Create projection data
            years = list(range(1, 11))
            cumulative_cost_savings = [cost_savings * year for year in years]
            cumulative_emissions_savings = [emissions_savings * year for year in years]
            
            # Create projection chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=years,
                y=cumulative_cost_savings,
                mode='lines+markers',
                name='Cost Savings ($)',
                line=dict(color='#0284c7', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=years,
                y=cumulative_emissions_savings,
                mode='lines+markers',
                name='CO₂ Reduction (kg)',
                line=dict(color='#10b981', width=3),
                marker=dict(size=8),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='10-Year Cumulative Savings',
                xaxis=dict(
                    title='Years',
                    tickmode='linear',
                    tick0=1,
                    dtick=1
                ),
                yaxis=dict(
                    title='Cumulative Cost Savings ($)',
                    titlefont=dict(color='#0284c7'),
                    tickfont=dict(color='#0284c7')
                ),
                yaxis2=dict(
                    title='Cumulative CO₂ Reduction (kg)',
                    titlefont=dict(color='#10b981'),
                    tickfont=dict(color='#10b981'),
                    anchor='x',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a note about projections
            if cost_savings > 0:
                st.info(f"Over 10 years, these changes could save approximately ${cost_savings * 10:.2f} and reduce CO₂ emissions by {emissions_savings * 10:.1f} kg.")
            else:
                st.warning(f"Over 10 years, these changes could increase costs by approximately ${-cost_savings * 10:.2f} and increase CO₂ emissions by {-emissions_savings * 10:.1f} kg.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a section for next steps
        st.markdown('<div class="glass-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.subheader("Next Steps")
        
        # Use the values we already calculated above
        if new_prediction < baseline_prediction:
            st.markdown("""
            <p>Based on your modifications, here are some recommended next steps:</p>
            <ol>
                <li><strong>Explore more combinations</strong> - Try different parameter combinations to find the optimal configuration</li>
                <li><strong>View detailed recommendations</strong> - Check the Recommendations tab for specific improvement measures</li>
                <li><strong>Analyze cost-effectiveness</strong> - Compare the implementation costs with projected savings</li>
                <li><strong>Generate a report</strong> - Create a PDF or Excel report to share your findings</li>
            </ol>
            """, unsafe_allow_html=True)
            
            # Add buttons for quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Recommendations", use_container_width=True):
                    # This would ideally navigate to the recommendations page
                    st.session_state.app_mode = "Recommendations"
                    st.rerun()
            with col2:
                if st.button("Generate Report", use_container_width=True):
                    # This would ideally trigger report generation
                    st.info("Report generation functionality would be implemented here.")
        else:
            st.markdown("""
            <p>Your current changes increase energy use. Consider these next steps:</p>
            <ol>
                <li><strong>Try different parameters</strong> - Adjust other parameters to find energy-saving combinations</li>
                <li><strong>Check the Explainability tab</strong> - Understand which parameters have the biggest impact</li>
                <li><strong>Reset and start over</strong> - Clear your changes and try a different approach</li>
                <li><strong>View recommendations</strong> - See expert-suggested improvements in the Recommendations tab</li>
            </ol>
            """, unsafe_allow_html=True)
            
            # Add buttons for quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Explore Explainability", use_container_width=True):
                    # This would ideally navigate to the explainability page
                    st.session_state.app_mode = "Explainability"
                    st.rerun()
            with col2:
                if st.button("View Recommendations", use_container_width=True):
                    # This would ideally navigate to the recommendations page
                    st.session_state.app_mode = "Recommendations"
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Add footer
    st.markdown("""
    <div class="footer" style="margin-top: 2rem; text-align: center; color: #64748b; font-size: 0.9rem;">
        <p>Building Energy Analysis Dashboard • What-If Analysis Tool</p>
        <p>For questions or support, please contact the Green Guardians team</p>
    </div>
    """, unsafe_allow_html=True)

def display_explainability_page(model, preprocessor):
    """Display the explainability page with enhanced visualizations"""

    st.title("Model Explainability")
    
    # Add a descriptive subtitle
    st.markdown("""
    <p style="font-size: 1.2rem; margin-bottom: 1.5rem; color: #64748b;">
        Understand how different building parameters influence energy performance predictions.
    </p>
    """, unsafe_allow_html=True)

    with st.expander("How to use this page", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h3 style="margin-top: 0;">Understanding Model Insights</h3>
            <p>This page provides multiple ways to understand how the energy prediction model works:</p>
            <ul>
                <li><strong>Feature Importance</strong> - Discover which building parameters have the biggest impact on energy predictions</li>
                <li><strong>SHAP Analysis</strong> - See how each parameter contributes to increasing or decreasing energy use</li>
                <li><strong>Partial Dependence</strong> - Explore how changing specific parameters affects predictions</li>
                <li><strong>LIME Explanation</strong> - Get a personalized explanation for your specific building</li>
            </ul>
            <p>These visualizations help identify which building characteristics have the greatest potential for energy savings.</p>
        </div>
        """, unsafe_allow_html=True)

    # Tabs for different explainability methods with improved styling
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8fafc;
        border-radius: 4px;
        color: #64748b;
        font-size: 14px;
        font-weight: 500;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0f766e !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    shap_tab, pdp_tab, lime_tab = st.tabs([
        "🧩 SHAP Analysis", 
        "📈 Partial Dependence", 
        "🔍 LIME Explanation"
    ])

    with shap_tab:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("🔍 Model Explainability: SHAP Values")

        # Short intro
        st.markdown("""
        <p style="font-size: 1.1rem; color: #475569;">
            Understand how each feature impacts your building's predicted energy consumption using SHAP (SHapley Additive exPlanations).
        </p>
        """, unsafe_allow_html=True)

        # First card: What are SHAP values?
        st.markdown("""
        <div style="background-color: rgba(15, 118, 110, 0.08); padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1rem;">
            <h4 style="margin-top: 0; color: #0f766e;">What Are SHAP Values?</h4>
            <p>SHAP values break down the impact of each parameter:</p>
            <ul>
                <li><b>Red:</b> Features increasing predicted energy use.</li>
                <li><b>Blue:</b> Features reducing predicted energy use.</li>
            </ul>
            <p>The wider the bar, the stronger the effect.</p>
        </div>
        """, unsafe_allow_html=True)

        # Plot section
        with st.spinner("⚙️ Calculating SHAP values..."):
            try:
                shap_plots = create_shap_plots(model, preprocessor, st.session_state.inputs)
                if shap_plots:
                    if isinstance(shap_plots, list):
                        for i, fig in enumerate(shap_plots):
                            st.plotly_chart(fig, use_container_width=True, key=f"shap_plot_{i}")
                    else:
                        st.plotly_chart(shap_plots, use_container_width=True, key="shap_plot_single")

                    # Key Factors section
                    st.subheader("📈 Key Factors Impacting Energy Use")

                    st.markdown("""
                    <div style="background-color: rgba(239, 68, 68, 0.08); padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1rem; border-left: 5px solid #ef4444;">
                        <h4 style="margin-top: 0; color: #ef4444;">Top Drivers Increasing Energy Consumption 🚨</h4>
                        <ul>
                            <li>Higher heating setpoint temperatures</li>
                            <li>Older windows with poor U-values</li>
                            <li>Weak wall insulation</li>
                        </ul>
                    </div>

                    <div style="background-color: rgba(16, 185, 129, 0.08); padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1rem; border-left: 5px solid #10b981;">
                        <h4 style="margin-top: 0; color: #10b981;">Top Drivers Reducing Energy Consumption 🌱</h4>
                        <ul>
                            <li>Energy-efficient HVAC systems</li>
                            <li>LED lighting with lower density</li>
                            <li>Renewable energy systems (if applicable)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # Final Interpretation Card
                    st.markdown("""
                    <div style="background-color: #f1f5f9; padding: 1.25rem; border-radius: 0.75rem; margin-top: 1.5rem;">
                        <h4 style="margin-top: 0; color: #334155;">How to Read These Results:</h4>
                        <ul>
                            <li>Features at the top have the greatest influence on predictions.</li>
                            <li>Focus improvement efforts on reducing the top "red" factors.</li>
                            <li>Enhance the impact of top "blue" factors to lower energy use.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.error("Could not generate SHAP visualizations. Please try again.")
            except Exception as e:
                st.error(f"Error calculating SHAP values: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    with pdp_tab:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("Partial Dependence Plots")

        st.markdown("""
        <div style="background-color: rgba(15, 118, 110, 0.1); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h4 style="margin-top: 0; color: #0f766e;">What are Partial Dependence Plots?</h4>
            <p style="margin-bottom: 0;">These plots show how the predicted energy use changes when you vary a single parameter while holding all others constant. The line shows the average effect of changing that parameter across all buildings.</p>
        </div>
        """, unsafe_allow_html=True)

        # Select feature for partial dependence plot with improved UI
        features = list(st.session_state.inputs.keys())
        display_names = {get_display_name(f): f for f in features}

        # Default to showing HVAC Efficiency or Wall Insulation as they are usually important
        default_feature = "HVAC_Efficiency"
        if default_feature not in st.session_state.inputs:
            default_feature = "Wall_Insulation_U-Value_Capped"
        if default_feature not in st.session_state.inputs:
            default_feature = next(iter(st.session_state.inputs.keys()))

        default_display = get_display_name(default_feature)

        # Create a more visually appealing parameter selector
        st.markdown("""
        <h4 style="color: #0f766e; margin-bottom: 0.5rem;">Select a parameter to analyze</h4>
        """, unsafe_allow_html=True)
        
        selected_feature_display = st.selectbox(
            "Choose a building parameter to see how it affects energy use",
            list(display_names.keys()),
            index=list(display_names.keys()).index(default_display) if default_display in display_names else 0
        )
        selected_feature = display_names[selected_feature_display]

        # Automatically generate plot when parameter is selected
        with st.spinner(f"Analyzing how {selected_feature_display} affects energy use..."):
            try:
                # Create PDP plot
                pdp_fig = create_pdp_plots(
                    model, 
                    preprocessor, 
                    st.session_state.inputs,
                    selected_feature
                )

                if pdp_fig:
                    # Handle both Figure objects and direct data dictionaries
                    st.plotly_chart(pdp_fig, use_container_width=True, key=f"pdp_chart_{selected_feature}")

                    # Add an optimization insights box with improved styling
                    if 'U-Value' in selected_feature:
                        insight = "Lower values (better insulation) generally reduce energy consumption"
                        icon = "❄️"
                    elif selected_feature == 'HVAC_Efficiency':
                        insight = "Higher efficiency values reduce energy consumption substantially"
                        icon = "🔥"
                    elif 'Temperature' in selected_feature and 'Heating_Setpoint' in selected_feature:
                        insight = "Lower heating setpoints reduce energy consumption"
                        icon = "🌡️"
                    elif 'Temperature' in selected_feature and 'Setback' in selected_feature:
                        insight = "Setting back temperatures when spaces are unoccupied saves energy"
                        icon = "⏱️"
                    else:
                        insight = "Analyze the curve to find optimal values for energy savings"
                        icon = "📊"

                    st.markdown(f"""
                    <div style="background-color: rgba(2, 132, 199, 0.1); padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #0284c7;">
                        <h4 style="margin-top: 0; color: #0284c7;">{icon} Optimization Insight for {selected_feature_display}</h4>
                        <p style="margin-bottom: 0;">{insight}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Explanation text with improved styling
                    st.markdown(f"""
                    <div style="background-color: #f1f5f9; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                        <h4 style="margin-top: 0; color: #334155;">Interpretation for {selected_feature_display}</h4>
                        <p>The plot shows how changing {selected_feature_display} affects energy use predictions:</p>
                        <ul style="margin-bottom: 0;">
                            <li>The x-axis shows different values of {selected_feature_display}</li>
                            <li>The y-axis shows the predicted Energy Use Intensity (EUI)</li>
                            <li>The line shows the average effect of this parameter on energy use</li>
                            <li>Upward slopes indicate increasing energy use, downward slopes indicate decreasing energy use</li>
                            <li>Use this plot to identify optimal values for {selected_feature_display}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Could not generate Partial Dependence Plot. Please try again.")
            except Exception as e:
                st.error(f"Error generating Partial Dependence Plot: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with lime_tab:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("LIME (Local Interpretable Model-agnostic Explanations)")

        st.markdown("""
        <div style="background-color: rgba(15, 118, 110, 0.1); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h4 style="margin-top: 0; color: #0f766e;">What is LIME?</h4>
            <p style="margin-bottom: 0;">LIME provides an explanation for your specific building, showing how each parameter contributes to your prediction. Green bars show parameters that reduce energy use, and red bars show parameters that increase it.</p>
        </div>
        """, unsafe_allow_html=True)

        # Automatically generate LIME explanation
        with st.spinner("Generating personalized explanation for your building..."):
            try:
                # Create LIME explanation
                lime_fig = create_lime_explanation(
                    model, 
                    preprocessor, 
                    st.session_state.inputs
                )

                if lime_fig:
                    # Check if lime_fig is a Figure object or a data dictionary
                    if isinstance(lime_fig, dict):
                        st.plotly_chart(lime_fig, use_container_width=True, key="lime_chart_dict")
                    else:
                        # For backward compatibility with older code that might return a Figure object
                        st.plotly_chart(lime_fig, use_container_width=True, key="lime_chart_fig")

                    # Show specific recommendations based on LIME results with improved styling
                    st.subheader("Building-Specific Insights")

                    st.markdown(f"""
                    <div style="background-color: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #fbbf24;">
                        <h4 style="margin-top: 0; color: #d97706;">Tailored Recommendations</h4>
                        <p>Based on your building's specific characteristics, here are personalized recommendations:</p>
                        <ol style="margin-bottom: 0;">
                            <li><strong>Focus on insulation improvements</strong> - Prioritize wall, window and roof insulation upgrades as these have the largest impact on your building's energy use.</li>
                            <li><strong>Consider HVAC system upgrades</strong> - A more efficient heating/cooling system could significantly reduce your energy consumption.</li>
                            <li><strong>Adjust temperature setpoints</strong> - Small changes to heating and cooling setpoints can have a meaningful impact on energy use.</li>
                        </ol>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add a call-to-action button
                    if st.button("See Detailed Recommendations", use_container_width=True):
                        # This would ideally navigate to the recommendations page
                        st.session_state.app_mode = "Recommendations"
                        st.rerun()

                    # Explanation text with improved styling
                    st.markdown("""
                    <div style="background-color: #f1f5f9; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                        <h4 style="margin-top: 0; color: #334155;">Interpretation</h4>
                        <p>The LIME explanation shows how each parameter affects the energy use prediction for your specific building:</p>
                        <ul style="margin-bottom: 0;">
                            <li>Green bars represent parameters that reduce energy use</li>
                            <li>Red bars represent parameters that increase energy use</li>
                            <li>The length of each bar indicates the strength of the impact</li>
                            <li>Focus improvement efforts on reducing the impact of parameters with long red bars</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Could not generate LIME explanation. Please try again.")
            except Exception as e:
                st.error(f"Error generating LIME explanation: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Add footer
    st.markdown("""
    <div class="footer" style="margin-top: 2rem; text-align: center; color: #64748b; font-size: 0.9rem;">
        <p>Building Energy Analysis Dashboard • Model Explainability Tool</p>
        <p>For questions or support, please contact the Green Guardians team</p>
    </div>
    """, unsafe_allow_html=True)

def display_recommendation_item(rec, i, payback_text, highlight_color):
    """Display a single recommendation item with enhanced styling"""
    
    # Create a glass card for each recommendation
    st.markdown(f'<div class="glass-card" style="margin-bottom: 1.5rem; border-left: 5px solid {highlight_color};">', unsafe_allow_html=True)
    
    # Create a multicolumn layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <h3 style="color: #0f766e; margin-top: 0;">{i+1}. {rec['title']}</h3>
        <p style="color: #334155;">{rec['description']}</p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            f"""
            <div style="border-left: 3px solid {highlight_color}; padding-left: 15px; height: 100%; background-color: rgba(208, 240, 192, 0.3); border-radius: 0.5rem; padding: 1rem;">
                <p style="margin:0; font-size:14px; color:#64748b; font-weight:bold;">Energy Savings</p>
                <p style="margin:0; font-size:24px; font-weight:bold; color: {highlight_color};">{rec['potential_savings']}</p>
                <p style="margin:0; font-size:14px; color:#64748b;">kWh/m²/yr</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Create a more structured metrics display with better alignment
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background-color: rgba(208, 240, 192, 0.2); border-radius: 0.5rem;">
                <p style="margin:0; font-size:14px; color:#64748b; font-weight: bold;">Financial Benefit</p>
                <p style="margin:0; font-size:20px; color:#0f766e; font-weight: bold;">${rec.get('annual_savings', 0):,.2f}/yr</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with metric_cols[1]:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background-color: rgba(208, 240, 192, 0.2); border-radius: 0.5rem;">
                <p style="margin:0; font-size:14px; color:#64748b; font-weight:bold;">Implementation Cost</p>
                <p style="margin:0; font-size:20px; color:#0f766e; font-weight:bold;">${rec.get('estimated_cost', 0):,.2f}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with metric_cols[2]:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background-color: rgba(208, 240, 192, 0.2); border-radius: 0.5rem;">
                <p style="margin:0; font-size:14px; color:#64748b; font-weight:bold;">CO₂ Reduction</p>
                <p style="margin:0; font-size:20px; color:#0f766e; font-weight:bold;">{rec['co2_reduction']} kg/yr</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with metric_cols[3]:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background-color: rgba(208, 240, 192, 0.2); border-radius: 0.5rem;">
                <p style="margin:0; font-size:14px; color:#64748b; font-weight:bold;">Payback Period</p>
                <p style="margin:0; font-size:20px; color:#0f766e; font-weight:bold;">{payback_text}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Add implementation details
    st.markdown(f"""
    <div style="background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
        <h4 style="margin-top: 0; color: #334155;">Implementation Details</h4>
        <p style="color: #64748b; margin-bottom: 0;">
            {rec.get('implementation_details', 'Contact a qualified professional for detailed implementation guidance.')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_recommendations_page(model, preprocessor):
    """Display the recommendations page with enhanced UI and actionable suggestions"""

    st.title("Energy-Saving Recommendations")
    
    # Add a descriptive subtitle
    st.markdown("""
    <p style="font-size: 1.2rem; margin-bottom: 1.5rem; color: #64748b;">
        Discover tailored, actionable recommendations to improve your building's energy performance.
    </p>
    """, unsafe_allow_html=True)

    with st.expander("How to use this page", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h3 style="margin-top: 0;">Recommendation Guide</h3>
            <p>This page provides customized recommendations to improve your building's energy efficiency:</p>
            <ul>
                <li><strong>Quick Wins</strong> - Simple changes with minimal cost that can save energy immediately (payback < 2 years)</li>
                <li><strong>Medium-Term Improvements</strong> - Moderate investments with good energy-saving potential (payback 2-7 years)</li>
                <li><strong>Major Upgrades</strong> - Significant improvements for substantial energy savings (payback > 7 years)</li>
                <li><strong>Cost-Benefit Analysis</strong> - Detailed financial analysis including costs, savings, and payback periods</li>
            </ul>
            <p>All recommendations are customized to your building's specific characteristics and energy profile.</p>
        </div>
        """, unsafe_allow_html=True)

    # Get recommendations based on current inputs
    with st.spinner("Generating personalized recommendations for your building..."):
        recommendations = get_recommendations(
            st.session_state.inputs,
            st.session_state.baseline_prediction,
            model,
            preprocessor
        )

    if recommendations:
        # Display overall summary first to give context
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("Executive Summary")

        # Calculate cumulative savings
        total_savings = sum(rec['potential_savings'] for rec_list in recommendations.values() for rec in rec_list)
        total_co2 = sum(rec['co2_reduction'] for rec_list in recommendations.values() for rec in rec_list)

        if st.session_state.baseline_prediction > 0:
            pct_reduction = (total_savings / st.session_state.baseline_prediction) * 100
        else:
            pct_reduction = 0

        area = st.session_state.inputs.get('Total_Building_Area', 1000)
        annual_cost_savings = total_savings * area * 0.34

        # Count recommendations
        num_recommendations = sum(len(recs) for recs in recommendations.values())

        # Create summary cards with improved styling
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: rgba(15, 118, 110, 0.1); padding: 1.5rem; border-radius: 0.5rem; height: 100%;">
                <h3 style="margin-top: 0; color: #0f766e;">Potential Improvements</h3>
                <p>We've identified <strong>{num_recommendations} energy-saving measures</strong> for your building that could:</p>
                <ul>
                    <li>Reduce energy use by <strong>{total_savings:.1f} kWh/m²/year</strong> ({pct_reduction:.1f}% improvement)</li>
                    <li>Save approximately <strong>${annual_cost_savings:.2f} per year</strong> on energy costs</li>
                    <li>Reduce carbon emissions by <strong>{total_co2:.1f} kg CO₂e/year</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create a visual summary with icons
            st.markdown(f"""
            <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; height: 100%;">
                <h3 style="margin-top: 0; color: #334155;">Recommendation Breakdown</h3>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; color: #10b981; margin-bottom: 0.5rem;">🚀</div>
                        <div style="font-weight: bold; color: #10b981;">{len(recommendations['quick_wins'])}</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Quick Wins</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; color: #0284c7; margin-bottom: 0.5rem;">⚙️</div>
                        <div style="font-weight: bold; color: #0284c7;">{len(recommendations['medium_term'])}</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Medium-Term</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; color: #f59e0b; margin-bottom: 0.5rem;">🏗️</div>
                        <div style="font-weight: bold; color: #f59e0b;">{len(recommendations['major_upgrades'])}</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Major Upgrades</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Display tailored recommendations with improved styling
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8fafc;
            border-radius: 4px;
            color: #64748b;
            font-size: 14px;
            font-weight: 500;
            border: 1px solid #e2e8f0;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #0f766e !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        quick_wins_tab, medium_term_tab, major_upgrades_tab, cost_benefit_tab = st.tabs([
            "🚀 Quick Wins", 
            "⚙️ Medium-Term Improvements", 
            "🏗️ Major Upgrades",
            "💰 Cost-Benefit Analysis"
        ])

        with quick_wins_tab:
            st.header("Quick Wins")

            # Add context - Quick Wins with improved styling
            st.markdown("""
            <div style="background-color: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem; border-left: 4px solid #10b981;">
                <h3 style="margin-top: 0; color: #10b981;">About Quick Wins</h3>
                <p>These measures typically:</p>
                <ul style="margin-bottom: 0;">
                    <li>Have minimal or no upfront cost</li>
                    <li>Can be implemented immediately without specialized expertise</li>
                    <li>Provide fast payback periods (usually under 2 years)</li>
                    <li>Focus on operational changes and simple upgrades</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if recommendations['quick_wins']:
                for i, rec in enumerate(recommendations['quick_wins']):
                    payback_years = rec.get('estimated_cost', 0) / rec.get('annual_savings', 1) if rec.get('annual_savings', 0) > 0 else float('inf')
                    payback_text = f"{payback_years:.1f} years" if payback_years != float('inf') else "N/A"

                    # Determine color based on savings amount
                    if rec['potential_savings'] > 10:
                        highlight_color = "#10b981"  # Green for high savings
                    elif rec['potential_savings'] > 5:
                        highlight_color = "#34d399"  # Light green for medium savings
                    else:
                        highlight_color = "#6ee7b7"  # Very light green for lower savings

                    # Use the helper function to display recommendation consistently
                    display_recommendation_item(rec, i, payback_text, highlight_color)
            else:
                st.markdown("""
                <div style="background-color: #f1f5f9; padding: 1.5rem; border-radius: 0.5rem; text-align: center;">
                    <img src="https://img.icons8.com/fluency/96/000000/info-squared.png" width="48" style="margin-bottom: 1rem;">
                    <h4 style="margin-top: 0; color: #334155;">No quick wins identified</h4>
                    <p style="color: #64748b; margin-bottom: 0;">This could mean your building already has good operational efficiency.</p>
                </div>
                """, unsafe_allow_html=True)

        with medium_term_tab:
            st.header("Medium-Term Improvements")

            # Add context - Medium-Term Improvements with improved styling
            st.markdown("""
            <div style="background-color: rgba(2, 132, 199, 0.1); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem; border-left: 4px solid #0284c7;">
                <h3 style="margin-top: 0; color: #0284c7;">About Medium-Term Improvements</h3>
                <p>These measures typically:</p>
                <ul style="margin-bottom: 0;">
                    <li>Require moderate capital investment</li>
                    <li>May need professional installation or expertise</li>
                    <li>Provide reasonable payback periods (usually 2-7 years)</li>
                    <li>Focus on building system upgrades and component replacements</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if recommendations['medium_term']:
                for i, rec in enumerate(recommendations['medium_term']):
                    payback_years = rec.get('estimated_cost', 0) / rec.get('annual_savings', 1) if rec.get('annual_savings', 0) > 0 else float('inf')
                    payback_text = f"{payback_years:.1f} years" if payback_years != float('inf') else "N/A"

                    # Determine color based on savings amount
                    if rec['potential_savings'] > 20:
                        highlight_color = "#0284c7"  # Blue for high savings
                    elif rec['potential_savings'] > 10:
                        highlight_color = "#38bdf8"  # Medium blue for medium savings
                    else:
                        highlight_color = "#7dd3fc"  # Light blue for lower savings

                    # Use the helper function to display recommendation consistently
                    display_recommendation_item(rec, i, payback_text, highlight_color)
            else:
                st.markdown("""
                <div style="background-color: #f1f5f9; padding: 1.5rem; border-radius: 0.5rem; text-align: center;">
                    <img src="https://img.icons8.com/fluency/96/000000/info-squared.png" width="48" style="margin-bottom: 1rem;">
                    <h4 style="margin-top: 0; color: #334155;">No medium-term improvements identified</h4>
                    <p style="color: #64748b; margin-bottom: 0;">This could mean your building's systems are already reasonably efficient.</p>
                </div>
                """, unsafe_allow_html=True)

        with major_upgrades_tab:
            st.header("Major Upgrades")

            # Add context - Major Upgrades with improved styling
            st.markdown("""
            <div style="background-color: rgba(245, 158, 11, 0.1); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem; border-left: 4px solid #f59e0b;">
                <h3 style="margin-top: 0; color: #f59e0b;">About Major Upgrades</h3>
                <p>These measures typically:</p>
                <ul style="margin-bottom: 0;">
                    <li>Require significant capital investment</li>
                    <li>Need professional design, installation and commissioning</li>
                    <li>Have longer payback periods (usually 7+ years)</li>
                    <li>Provide transformational energy performance improvements</li>
                    <li>Often have additional non-energy benefits (comfort, value, resilience)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if recommendations['major_upgrades']:
                for i, rec in enumerate(recommendations['major_upgrades']):
                    payback_years = rec.get('estimated_cost', 0) / rec.get('annual_savings', 1) if rec.get('annual_savings', 0) > 0 else float('inf')
                    payback_text = f"{payback_years:.1f} years" if payback_years != float('inf') else "N/A"

                    # Determine color based on savings amount
                    if rec['potential_savings'] > 30:
                        highlight_color = "#f59e0b"  # Orange for high savings
                    elif rec['potential_savings'] > 15:
                        highlight_color = "#fbbf24"  # Medium orange for medium savings
                    else:
                        highlight_color = "#fcd34d"  # Light orange/amber for lower savings

                    # Use the helper function to display recommendation consistently
                    display_recommendation_item(rec, i, payback_text, highlight_color)
                    
                    # Add additional benefits info (only in major upgrades section)
                    st.markdown("""
                    <div style="background-color: rgba(245, 158, 11, 0.05); padding: 1rem; border-radius: 0.5rem; margin-top: -1rem; margin-bottom: 1.5rem;">
                        <p style="color: #92400e; margin: 0;"><strong>Additional benefits:</strong> Improved comfort, reduced maintenance, increased property value, enhanced resilience to climate change.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #f1f5f9; padding: 1.5rem; border-radius: 0.5rem; text-align: center;">
                    <img src="https://img.icons8.com/fluency/96/000000/info-squared.png" width="48" style="margin-bottom: 1rem;">
                    <h4 style="margin-top: 0; color: #334155;">No major upgrades identified</h4>
                    <p style="color: #64748b; margin-bottom: 0;">This could mean your building already has high-performance systems or the model can't determine major improvement opportunities based on the available data.</p>
                </div>
                """, unsafe_allow_html=True)

        with cost_benefit_tab:
            st.header("Cost-Benefit Analysis")

            # Add context - Financial Analysis with improved styling
            st.markdown("""
            <div style="background-color: rgba(20, 184, 166, 0.1); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem; border-left: 4px solid #14b8a6;">
                <h3 style="margin-top: 0; color: #14b8a6;">Financial Analysis</h3>
                <p>This analysis provides a detailed financial breakdown of all recommended measures:</p>
                <ul>
                    <li>Estimated upfront costs (material and installation)</li>
                    <li>Projected annual cost savings</li>
                    <li>Simple payback period (cost ÷ annual savings)</li>
                    <li>Environmental benefits (CO₂ reduction)</li>
                </ul>
                <p style="font-style: italic; color: #64748b; margin-bottom: 0;">Note: Actual costs and savings may vary. We recommend consulting with qualified professionals for detailed quotes.</p>
            </div>
            """, unsafe_allow_html=True)

            # Calculate payback period
            payback_data = []

            # Collect all recommendations
            all_recs = (
                recommendations['quick_wins'] +
                recommendations['medium_term'] +
                recommendations['major_upgrades']
            )

            # Create cost-benefit data
            for rec in all_recs:
                if 'estimated_cost' in rec and 'annual_savings' in rec:
                    payback_years = rec['estimated_cost'] / rec['annual_savings'] if rec['annual_savings'] > 0 else float('inf')

                    # Add implementation category
                    if rec in recommendations['quick_wins']:
                        category = "Quick Win"
                    elif rec in recommendations['medium_term']:
                        category = "Medium-Term"
                    else:
                        category = "Major Upgrade"

                    payback_data.append({
                        'Measure': rec['title'],
                        'Category': category,
                        'Estimated Cost ($)': rec['estimated_cost'],
                        'Annual Savings ($)': rec['annual_savings'],
                        'Payback Period (years)': round(payback_years, 1),
                        'CO₂ Reduction (kg)': rec['co2_reduction'],
                        'Energy Savings (kWh/m²/yr)': rec['potential_savings']
                    })

            if payback_data:
                # Create DataFrame and sort by payback period
                payback_df = pd.DataFrame(payback_data)
                payback_df = payback_df.sort_values('Payback Period (years)')

                # Display financial summary with improved styling
                total_investment = payback_df['Estimated Cost ($)'].sum()
                total_annual_savings = payback_df['Annual Savings ($)'].sum()
                avg_payback = total_investment / total_annual_savings if total_annual_savings > 0 else float('inf')

                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color: rgba(20, 184, 166, 0.1); padding: 1.5rem; border-radius: 0.5rem; text-align: center; height: 100%;">
                        <h4 style="margin-top: 0; color: #14b  text-align: center; height: 100%;">
                        <h4 style="margin-top: 0; color: #14b8a6;">Total Investment Required</h4>
                        <p style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0; color: #0f766e;">${total_investment:,.2f}</p>
                        <p style="color: #64748b; margin: 0;">One-time cost</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color: rgba(20, 184, 166, 0.1); padding: 1.5rem; border-radius: 0.5rem; text-align: center; height: 100%;">
                        <h4 style="margin-top: 0; color: #14b8a6;">Total Annual Savings</h4>
                        <p style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0; color: #0f766e;">${total_annual_savings:,.2f}</p>
                        <p style="color: #64748b; margin: 0;">Per year</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="background-color: rgba(20, 184, 166, 0.1); padding: 1.5rem; border-radius: 0.5rem; text-align: center; height: 100%;">
                        <h4 style="margin-top: 0; color: #14b8a6;">Overall Payback Period</h4>
                        <p style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0; color: #0f766e;">{avg_payback:.1f} years</p>
                        <p style="color: #64748b; margin: 0;">Return on investment</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display table with improved styling
                st.subheader("Detailed Financial Analysis")
                st.dataframe(
                    payback_df,
                    column_config={
                        "Measure": st.column_config.TextColumn("Measure"),
                        "Category": st.column_config.TextColumn("Category"),
                        "Estimated Cost ($)": st.column_config.NumberColumn(
                            "Estimated Cost ($)", 
                            format="$%.2f"
                        ),
                        "Annual Savings ($)": st.column_config.NumberColumn(
                            "Annual Savings ($)", 
                            format="$%.2f"
                        ),
                        "Payback Period (years)": st.column_config.NumberColumn(
                            "Payback Period (years)", 
                            format="%.1f years"
                        ),
                        "CO₂ Reduction (kg)": st.column_config.NumberColumn(
                            "CO₂ Reduction (kg)", 
                            format="%.1f kg"
                        ),
                        "Energy Savings (kWh/m²/yr)": st.column_config.NumberColumn(
                            "Energy Savings (kWh/m²/yr)", 
                            format="%.1f kWh/m²/yr"
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )

                # Create payback chart with improved styling
                st.subheader("Payback Period Analysis")

                fig = px.bar(
                    payback_df,
                    x='Measure',
                    y='Payback Period (years)',
                    color='Category',
                    color_discrete_map={
                        'Quick Win': '#10b981',
                        'Medium-Term': '#0284c7',
                        'Major Upgrade': '#f59e0b'
                    },
                    hover_data=['Estimated Cost ($)', 'Annual Savings ($)', 'CO₂ Reduction (kg)'],
                    labels={'Payback Period (years)': 'Years to Recover Investment'}
                )

                fig.update_layout(
                    xaxis_title="Recommended Measures",
                    yaxis_title="Payback Period (years)",
                    legend_title="Implementation Category",
                    plot_bgcolor='white',
                    font=dict(family="Inter, sans-serif", size=12),
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Add ROI analysis with improved styling
                st.subheader("Return on Investment (ROI) Analysis")

                # Calculate 10-year ROI for each measure
                year_range = 10
                payback_df['10-Year Return ($)'] = payback_df['Annual Savings ($)'] * year_range - payback_df['Estimated Cost ($)']
                payback_df['10-Year ROI (%)'] = (payback_df['10-Year Return ($)'] / payback_df['Estimated Cost ($)']) * 100

                # Create ROI chart with improved styling
                roi_fig = px.scatter(
                    payback_df,
                    x='Estimated Cost ($)',
                    y='Annual Savings ($)',
                    size='Energy Savings (kWh/m²/yr)',
                    color='Category',
                    color_discrete_map={
                        'Quick Win': '#10b981',
                        'Medium-Term': '#0284c7',
                        'Major Upgrade': '#f59e0b'
                    },
                    hover_data=['Measure', 'Payback Period (years)', '10-Year ROI (%)'],
                    text='Measure'
                )

                # Add payback period reference lines
                max_cost = payback_df['Estimated Cost ($)'].max() * 1.1

                for years in [2, 5, 10]:
                    roi_fig.add_shape(
                        type="line",
                        x0=0, y0=0,
                        x1=max_cost, y1=max_cost/years,
                        line=dict(color="rgba(100, 116, 139, 0.5)", width=1, dash="dot"),
                    )
                    # Add annotation
                    roi_fig.add_annotation(
                        x=max_cost*0.8,
                        y=max_cost*0.8/years,
                        text=f"{years} year payback",
                        showarrow=False,
                        font=dict(size=12, color="#64748b")
                    )

                roi_fig.update_layout(
                    title="Investment vs. Returns Analysis",
                    xaxis_title="Upfront Investment Cost ($)",
                    yaxis_title="Annual Cost Savings ($)",
                    plot_bgcolor='white',
                    font=dict(family="Inter, sans-serif", size=12),
                    height=500
                )

                roi_fig.update_traces(textposition='top center')

                st.plotly_chart(roi_fig, use_container_width=True)

                st.markdown("""
                <div style="background-color: rgba(2, 132, 199, 0.1); padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                    <p style="margin: 0; color: #0284c7;"><strong>Note:</strong> This analysis uses simple payback period and doesn't account for energy price inflation, maintenance savings, or other non-energy benefits. A professional energy audit or engineering assessment would provide more detailed financial projections.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style="background-color: #f1f5f9; padding: 1.5rem; border-radius: 0.5rem; text-align: center;">
                    <img src="https://img.icons8.com/fluency/96/000000/info-squared.png" width="48" style="margin-bottom: 1rem;">
                    <h4 style="margin-top: 0; color: #334155;">No cost-benefit data available</h4>
                    <p style="color: #64748b; margin-bottom: 0;">We couldn't generate financial analysis for your recommendations.</p>
                </div>
                """, unsafe_allow_html=True)

        # Add actionable next steps section with improved styling
        st.markdown('<div class="glass-card" style="margin-top: 2rem;">', unsafe_allow_html=True)
        st.header("Next Steps")
        
        with st.container():
            st.markdown("""
            <h3 style="color: #0f766e;">Implementing Your Energy Efficiency Roadmap</h3>
            <p>Here's a suggested approach to implementing these recommendations:</p>
            """, unsafe_allow_html=True)
            
            steps = [
                "**Start with the quick wins** - Implement low/no-cost operational changes immediately",
                "**Plan for medium-term improvements** - Budget for these moderate investments in your next fiscal cycle",
                "**Develop a long-term strategy** - Create a capital improvement plan for major upgrades",
                "**Seek professional assistance** - Consult with energy efficiency experts, contractors, or engineering firms for detailed assessments and implementation",
                "**Explore financial incentives** - Research utility rebates, tax incentives, and financing options for energy efficiency projects",
                "**Measure and verify** - Track energy usage before and after improvements to verify savings",
            ]
            
            # Create a more visually appealing steps list
            for i, step in enumerate(steps):
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 1rem; background-color: rgba(15, 118, 110, 0.05); padding: 1rem; border-radius: 0.5rem;">
                    <div style="background-color: #0f766e; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; flex-shrink: 0;">
                        {i+1}
                    </div>
                    <div>{step}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                <p style="margin: 0; color: #10b981;"><strong>Pro Tip:</strong> For best results, consider developing a comprehensive energy management plan that aligns with your budget cycles and building renovation schedules.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background-color: #f1f5f9; padding: 2rem; border-radius: 0.5rem; text-align: center; margin-top: 2rem;">
            <img src="https://img.icons8.com/fluency/96/000000/error.png" width="64" style="margin-bottom: 1rem;">
            <h3 style="margin-top: 0; color: #ef4444;">Could not generate recommendations</h3>
            <p style="color: #64748b; margin-bottom: 1rem;">Please try using the Prediction tab first to establish a baseline for your building.</p>
            <button style="background-color: #0f766e; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer;">Go to Prediction Tab</button>
        </div>
        """, unsafe_allow_html=True)
def display_report_generation_page(model, preprocessor):
    """Display enhanced report generation page with detailed PDF and Excel options"""
    st.markdown("<h1>Generate Enhanced Reports</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <p>Generate comprehensive, detailed reports with actionable insights and analysis of your building's energy performance. Download reports to share with stakeholders or for your records.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if baseline prediction exists
    if not hasattr(st.session_state, 'baseline_prediction') or st.session_state.baseline_prediction is None:
        st.warning("Please first go to the Prediction page to establish a baseline for your building.")
        return
    
    # Get the baseline and inputs
    baseline_eui = st.session_state.baseline_prediction
    inputs = st.session_state.inputs
    baseline_cost = st.session_state.baseline_cost
    baseline_emissions = st.session_state.baseline_emissions
    
    # Get the efficiency rating
    rating = get_efficiency_rating(baseline_eui)
    
    # Create tabs for PDF and Excel reports
    tab1, tab2 = st.tabs(["PDF Report", "Excel Report"])
    
    with tab1:
        st.markdown("<h3>Enhanced PDF Report</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>Report Content Options</h4>", unsafe_allow_html=True)
            include_recommendations = st.checkbox("Include Recommendations", value=True)
            include_cost_analysis = st.checkbox("Include Cost Analysis", value=True)
            include_carbon_footprint = st.checkbox("Include Carbon Footprint Analysis", value=True)
            include_benchmark = st.checkbox("Include Industry Benchmarks", value=True)
        
        with col2:
            st.markdown("<h4>Report Details</h4>", unsafe_allow_html=True)
            report_title = st.text_input("Report Title", value="Building Energy Performance Report")
            prepared_by = st.text_input("Prepared By", value="")
        
        # Generate PDF Report button
        if st.button("Generate Enhanced PDF Report"):
            with st.spinner("Generating PDF report..."):
                try:
                    # Generate enhanced PDF report
                    pdf_data = generate_enhanced_pdf_report(
                        inputs, 
                        baseline_eui, 
                        baseline_cost, 
                        baseline_emissions, 
                        rating,
                        {
                            'title': report_title,
                            'prepared_by': prepared_by,
                            'include_recommendations': include_recommendations,
                            'include_cost_analysis': include_cost_analysis,
                            'include_carbon_footprint': include_carbon_footprint,
                            'include_benchmark': include_benchmark
                        },
                        model, 
                        preprocessor
                    )
                    
                    # Create download button
                    b64_pdf = base64.b64encode(pdf_data).decode()
                    pdf_filename = f"enhanced_building_energy_report_{datetime.now().strftime('%Y%m%d')}.pdf"
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_filename}" class="contact-button">Download Enhanced PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF report: {e}")
    
    with tab2:
        st.markdown("<h3>Enhanced Excel Report</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>Report Content Options</h4>", unsafe_allow_html=True)
            include_data_tables = st.checkbox("Include Detailed Data Tables", value=True)
            include_charts = st.checkbox("Include Interactive Charts", value=True)
            include_roi_calculator = st.checkbox("Include ROI Calculator", value=True)
            include_year_projection = st.checkbox("Include 10-Year Projection", value=True)
        
        with col2:
            st.markdown("<h4>Analysis Parameters</h4>", unsafe_allow_html=True)
            energy_price_increase = st.slider("Annual Energy Price Increase (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.5) / 100
            projection_years = st.slider("Projection Period (Years)", min_value=5, max_value=20, value=10, step=1)
            discount_rate = st.slider("Discount Rate (%)", min_value=1.0, max_value=10.0, value=3.5, step=0.5) / 100
        
        # Generate Excel Report button
        if st.button("Generate Enhanced Excel Report"):
            with st.spinner("Generating Excel report..."):
                try:
                    # Generate enhanced Excel report
                    excel_data = generate_enhanced_excel_report(
                        inputs, 
                        baseline_eui, 
                        baseline_cost, 
                        baseline_emissions, 
                        rating,
                        {
                            'include_data_tables': include_data_tables,
                            'include_charts': include_charts,
                            'include_roi_calculator': include_roi_calculator,
                            'include_year_projection': include_year_projection,
                            'energy_price_increase': energy_price_increase,
                            'projection_years': projection_years,
                            'discount_rate': discount_rate
                        },
                        model, 
                        preprocessor
                    )
                    
                    # Create download button
                    b64_excel = base64.b64encode(excel_data).decode()
                    excel_filename = f"enhanced_building_energy_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="{excel_filename}" class="contact-button">Download Enhanced Excel Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("Excel report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating Excel report: {e}")
    # Add footer
    st.markdown("""
    <div class="footer" style="margin-top: 2rem; text-align: center; color: #64748b; font-size: 0.9rem;">
        <p>Building Energy Analysis Dashboard • Recommendations Tool</p>
        <p>For questions or support, please contact the Green Guardians team</p>
    </div>
    """, unsafe_allow_html=True)

# Run the main function if this file is executed directly
if __name__ == "__main__":
    main()
st.markdown("""
    <div style='text-align: center;'>
        <hr>
        <strong>Building Energy Analysis Dashboard <br>
        For questions, feedback, or support, please contact the development team (The Green Guardians (NCU))</strong> 
    </div>
""", unsafe_allow_html=True)