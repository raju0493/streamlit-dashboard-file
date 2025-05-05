import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import base64
from datetime import datetime
import warnings
import joblib

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Building Energy Analysis Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set custom theme
st.markdown("""
<style>
    /* Green theme inspired by CO2 emissions and sustainability */
    :root {
        --primary-color: #2e7d32;         /* Dark green */
        --secondary-color: #4caf50;       /* Medium green */
        --tertiary-color: #81c784;        /* Light green */
        --background-color: #f5f5f5;      /* Light gray background */
        --accent-color: #388e3c;          /* Accent green */
        --text-color: #263238;            /* Dark text */
        --light-text-color: #455a64;      /* Light text */
    }
    
    /* Apply theme colors */
    .stButton button {
        background-color: var(--primary-color);
        color: white;
        font-weight: 600;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: var(--accent-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Primary action button */
    .stButton.primary button {
        background-color: var(--primary-color);
        font-size: 1.1em;
        padding: 0.5em 1em;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    h1 {
        border-bottom: 2px solid var(--tertiary-color);
        padding-bottom: 0.3em;
        margin-bottom: 0.8em;
    }
    
    h3 {
        margin-top: 1.5em;
        border-left: 4px solid var(--primary-color);
        padding-left: 0.6em;
    }
    
    /* Metric containers */
    div[data-testid="stMetricValue"] {
        color: var(--primary-color);
        font-weight: bold;
        font-size: 1.1em !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.9em !important;
    }
    
    /* Card styling for containers */
    div.stBlock {
        border: 1px solid var(--tertiary-color);
        border-radius: 8px;
        padding: 15px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin-bottom: 1em;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--background-color);
        border-right: 1px solid #e0e0e0;
    }
    
    /* Progress bars */
    div[role="progressbar"] > div {
        background-color: var(--secondary-color);
        border-radius: 10px;
    }
    
    /* Tabs styling */
    button[role="tab"] {
        color: var(--text-color) !important;
        font-weight: 500;
    }
    
    button[role="tab"][aria-selected="true"] {
        color: var(--primary-color) !important;
        border-bottom-color: var(--primary-color) !important;
        font-weight: 600;
    }
    
    /* Table styling */
    div.stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
    }
    
    div.stDataFrame table {
        border-collapse: separate;
        border-spacing: 0;
    }
    
    div.stDataFrame th {
        background-color: var(--primary-color);
        color: white;
        font-weight: 600;
        padding: 10px;
        border: none;
        text-align: left;
        font-size: 0.9em;
    }
    
    div.stDataFrame td {
        padding: 8px 10px;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.9em;
    }
    
    div.stDataFrame tr:nth-child(even) {
        background-color: #f8f8f8;
    }
    
    div.stDataFrame tr:hover {
        background-color: #f0f7f0;
    }
    
    /* Slider styling */
    div.stSlider {
        padding-top: 1em;
        padding-bottom: 1em;
    }
    
    div.stSlider div[data-baseweb="slider"] {
        height: 6px;
    }
    
    div.stSlider div[data-baseweb="thumb"] {
        width: 18px;
        height: 18px;
        background-color: var(--primary-color);
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Select box styling */
    div.stSelectbox {
        margin-bottom: 0.8em;
    }
    
    div.stSelectbox > div[data-baseweb="select"] > div {
        border-radius: 4px;
        border-color: #dcdcdc;
    }
    
    div.stSelectbox > div[data-baseweb="select"] > div:hover {
        border-color: var(--secondary-color);
    }
    
    /* Number input styling */
    div.stNumberInput {
        margin-bottom: 0.8em;
    }
    
    div.stNumberInput > div > div > input {
        border-radius: 4px;
        border-color: #dcdcdc;
    }
    
    div.stNumberInput > div > div > input:focus {
        border-color: var(--secondary-color);
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: var(--text-color);
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Custom containers for metrics and cards */
    .metric-container {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin-bottom: 1em;
        border-left: 4px solid var(--primary-color);
    }
    
    .info-card {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 15px;
        margin: 1em 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .warning-card {
        background-color: #fff8e1;
        border-radius: 8px;
        padding: 15px;
        margin: 1em 0;
        border-left: 4px solid #ffb300;
    }
    
    /* Animation for loading/transitions */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Suppress warnings
warnings.filterwarnings('ignore')

# Feature flags for available libraries
HAS_VIZ = True
HAS_ML = True
HAS_XAI = True
HAS_PDP = True

try:
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    st.warning("Visualization libraries not available. Some charts will be disabled.")

try:
    from sklearn.inspection import permutation_importance
    from sklearn.inspection import partial_dependence
    HAS_ML = True
    HAS_PDP = True
except ImportError:
    HAS_ML = False
    HAS_PDP = False
    st.warning("Scikit-learn not available. Some machine learning features will be disabled.")

try:
    import shap
    import lime
    from lime import lime_tabular
    HAS_XAI = True
except ImportError:
    HAS_XAI = False
    st.warning("XAI libraries (SHAP, LIME) not available. Explainability features will be disabled.")

# Create directories for assets if they don't exist
os.makedirs('attached_assets', exist_ok=True)

# Function to load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    """Load the model and preprocessor"""
    try:
        # Try to load from both current directory and attached_assets
        model_paths = ['lgbm_optimized_eui.pkl', 'attached_assets/lgbm_optimized_eui.pkl']
        preprocessor_paths = ['preprocessor_eui_ordinal.pkl', 'attached_assets/preprocessor_eui_ordinal.pkl']
        
        # Try each possible path
        model = None
        preprocessor = None
        
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
        
        if model is not None and preprocessor is not None:
            return model, preprocessor
        else:
            # Create simple model and preprocessor if files don't exist
            st.info("Creating prediction model using sample data. For accurate results, place your model files in the current directory or 'attached_assets' folder.")
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
    categorical_features = ['Building_Orientation', 'Weather_File', 'Building_Type']
    
    # Define categories for ordinal encoding
    building_orientation_categories = ['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest']
    weather_file_categories = ['Cold', 'Temperate', 'Hot', 'Tropical', 'Arctic']
    building_type_categories = ['Residential', 'Office', 'Commercial', 'Industrial', 'Educational']
    
    # Create preprocessing pipeline with proper ordinal encoding for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(categories=[
                building_orientation_categories,
                weather_file_categories,
                building_type_categories
            ]), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Generate sample training data
    features = [
        'HVAC_Efficiency', 'Domestic_Hot_Water_Usage', 'Building_Orientation',
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
        'Building_Orientation': np.random.choice(building_orientation_categories, n_samples),
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
        'Renewable_Energy_Usage': np.random.uniform(0, 1, n_samples),
    }
    
    # Calculate energy use intensity (target variable)
    eui = 200 - (data['HVAC_Efficiency'] * 100) + (data['Lighting_Density'] * 2) + \
        (data['Equipment_Density'] * 1.5) - (data['Renewable_Energy_Usage'] * 100) + \
        (data['Window_Insulation_U-Value_Capped'] * 30) + \
        (data['Wall_Insulation_U-Value_Capped'] * 20) + \
        np.random.normal(0, 20, n_samples)  # Add some noise
    
    eui = np.clip(eui, 30, 400)  # Clip to reasonable range
    
    # Create dataframe
    df = pd.DataFrame(data)
    X = df[features]
    y = eui
    
    # Fit preprocessor and model
    X_processed = preprocessor.fit_transform(X)
    model.fit(X_processed, y)
    
    # Save model and preprocessor for future use
    with open('lgbm_optimized_eui.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('preprocessor_eui_ordinal.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    return model, preprocessor

# Function to make predictions
def predict_eui(params, model, preprocessor):
    """Predict Energy Use Intensity based on input parameters"""
    try:
        # Convert params to DataFrame
        input_df = pd.DataFrame([params])
        
        # Handle categorical features correctly
        categorical_cols = ['Building_Orientation', 'Weather_File', 'Building_Type']
        for col in categorical_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)
        
        # Apply preprocessing
        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Fallback calculation if prediction fails
        p = params
        fallback = 200 - (p['HVAC_Efficiency'] * 100) + (p['Lighting_Density'] * 2) + \
                  (p['Equipment_Density'] * 1.5) - (p['Renewable_Energy_Usage'] * 100) + \
                  (p['Window_Insulation_U-Value_Capped'] * 30) + \
                  (p['Wall_Insulation_U-Value_Capped'] * 20)
        return fallback

# Function to get energy efficiency rating
def get_efficiency_rating(eui):
    """Return energy efficiency rating based on EUI value"""
    if eui < 50:
        return "A+", "#1F9D55"  # Green
    elif eui < 100:
        return "A", "#2da44e"
    elif eui < 150:
        return "B", "#7cb305"
    elif eui < 200:
        return "C", "#d4b106"
    elif eui < 250:
        return "D", "#F59E0B"  # Orange
    elif eui < 300:
        return "E", "#fa8c16"
    elif eui < 350:
        return "F", "#f5222d"
    else:
        return "G", "#cf1322"  # Red

# Function to create gauge chart
def create_gauge_chart(eui_value, key_suffix=""):
    """Create a gauge chart for the EUI value"""
    rating, color = get_efficiency_rating(eui_value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=eui_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Energy Use Intensity (kWh/m²/year)", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 400], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#1F9D55'},
                {'range': [50, 100], 'color': '#2da44e'},
                {'range': [100, 150], 'color': '#7cb305'},
                {'range': [150, 200], 'color': '#d4b106'},
                {'range': [200, 250], 'color': '#F59E0B'},
                {'range': [250, 300], 'color': '#fa8c16'},
                {'range': [300, 350], 'color': '#f5222d'},
                {'range': [350, 400], 'color': '#cf1322'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': eui_value
            }
        }
    ))
    
    # Add efficiency rating as annotation
    fig.add_annotation(
        x=0.5,
        y=0.3,
        text=f"Rating: {rating}",
        font=dict(size=24, color=color, family="Arial Black"),
        showarrow=False
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Function to calculate energy cost
def calculate_energy_cost(eui, area, rate):
    """Calculate annual energy cost based on EUI, area, and energy rate"""
    annual_energy = eui * area  # kWh/year
    annual_cost = annual_energy * rate  # $/year
    return annual_energy, annual_cost

# Function to calculate CO2 emissions
def calculate_co2_emissions(annual_energy, emission_factor=0.5):
    """Calculate CO2 emissions based on annual energy consumption"""
    # emission_factor in kg CO2 per kWh
    co2_emissions = annual_energy * emission_factor  # kg CO2/year
    return co2_emissions

# Function to generate energy cost forecast
def generate_energy_forecast(eui, area, base_rate, annual_increase, years):
    """Generate energy cost forecast for multiple years"""
    forecast = []
    cumulative_cost = 0
    annual_energy = eui * area
    
    for year in range(1, years + 1):
        # Calculate energy rate with annual increase
        rate = base_rate * ((1 + annual_increase / 100) ** (year - 1))
        
        # Calculate cost
        cost = annual_energy * rate
        cumulative_cost += cost
        
        forecast.append({
            'Year': year,
            'Energy_Rate': rate,
            'Annual_Cost': cost,
            'Cumulative_Cost': cumulative_cost,
        })
    
    return pd.DataFrame(forecast)

# Function to generate carbon emission forecast
def generate_emissions_forecast(annual_energy, emission_factor, years):
    """Generate carbon emissions forecast for multiple years"""
    forecast = []
    annual_emissions = annual_energy * emission_factor
    cumulative_emissions = 0
    
    for year in range(1, years + 1):
        cumulative_emissions += annual_emissions
        
        forecast.append({
            'Year': year,
            'Annual_Emissions': annual_emissions,
            'Cumulative_Emissions': cumulative_emissions,
        })
    
    return pd.DataFrame(forecast)

# Function to generate recommendations
def generate_recommendations(params, eui, model, preprocessor):
    """Generate recommendations for improving energy efficiency"""
    try:
        recommendations = []
        feature_improvements = {
            'HVAC_Efficiency': ('increase', 0.1, 0.95),  # (direction, step, limit)
            'Lighting_Density': ('decrease', 2, 5),
            'Equipment_Density': ('decrease', 3, 5),
            'Window_Insulation_U-Value_Capped': ('decrease', 0.5, 0.8),
            'Wall_Insulation_U-Value_Capped': ('decrease', 0.3, 0.2),
            'Roof_Insulation_U-Value_Capped': ('decrease', 0.2, 0.15),
            'Floor_Insulation_U-Value_Capped': ('decrease', 0.2, 0.2),
            'Air_Change_Rate_Capped': ('decrease', 0.5, 0.5),
            'Renewable_Energy_Usage': ('increase', 0.2, 1.0),
        }
        
        # Dictionary to store improvements and their impact
        improvements = {}
        
        # Try each improvement individually
        for feature, (direction, step, limit) in feature_improvements.items():
            # Skip if feature not in params or if it's None
            if feature not in params or params[feature] is None:
                continue
            
            # Create copy of params
            new_params = params.copy()
            
            # Apply improvement step
            if direction == 'increase':
                if new_params[feature] < limit:
                    new_params[feature] = min(new_params[feature] + step, limit)
                else:
                    continue  # Already at or above limit
            else:  # 'decrease'
                if new_params[feature] > limit:
                    new_params[feature] = max(new_params[feature] - step, limit)
                else:
                    continue  # Already at or below limit
            
            # Predict EUI with improved parameter
            new_eui = predict_eui(new_params, model, preprocessor)
            
            # Calculate impact
            impact = eui - new_eui
            if impact > 0:
                improvements[feature] = (impact, new_params[feature], params[feature])
        
        # Sort improvements by impact
        sorted_improvements = sorted(improvements.items(), key=lambda x: x[1][0], reverse=True)
        
        # Generate recommendations
        for feature, (impact, new_value, old_value) in sorted_improvements:
            # Format values for display
            if 'Efficiency' in feature or 'Usage' in feature:
                old_formatted = f"{old_value:.2f} (efficiency ratio)"
                new_formatted = f"{new_value:.2f} (efficiency ratio)"
            elif 'U-Value' in feature:
                old_formatted = f"{old_value:.2f} W/m²K"
                new_formatted = f"{new_value:.2f} W/m²K"
            elif 'Density' in feature and 'Lighting' in feature:
                old_formatted = f"{old_value:.2f} W/m²"
                new_formatted = f"{new_value:.2f} W/m²"
            elif 'Density' in feature and 'Equipment' in feature:
                old_formatted = f"{old_value:.2f} W/m²"
                new_formatted = f"{new_value:.2f} W/m²"
            elif 'Temperature' in feature:
                old_formatted = f"{old_value:.1f}°C"
                new_formatted = f"{new_value:.1f}°C"
            elif 'Rate' in feature:
                old_formatted = f"{old_value:.2f} ACH"
                new_formatted = f"{new_value:.2f} ACH"
            else:
                old_formatted = f"{old_value:.2f}"
                new_formatted = f"{new_value:.2f}"
            
            # Generate human-readable improvement explanation
            feature_name = feature.replace('_', ' ')
            if 'U-Value' in feature:
                feature_name = feature_name.replace('U-Value Capped', 'Insulation')
                explanation = f"Improve {feature_name} from {old_formatted} to {new_formatted}"
                detail = "Better insulation reduces heat transfer through the building envelope."
            elif feature == 'HVAC_Efficiency':
                explanation = f"Upgrade HVAC system efficiency from {old_formatted} to {new_formatted}"
                detail = "A more efficient HVAC system uses less energy while maintaining comfort levels."
            elif feature == 'Lighting_Density':
                explanation = f"Reduce lighting power density from {old_formatted} to {new_formatted}"
                detail = "More efficient lighting systems and controls reduce energy consumption."
            elif feature == 'Equipment_Density':
                explanation = f"Reduce equipment power density from {old_formatted} to {new_formatted}"
                detail = "Energy-efficient equipment and proper power management reduce electricity use."
            elif feature == 'Air_Change_Rate_Capped':
                explanation = f"Improve building airtightness from {old_formatted} to {new_formatted}"
                detail = "Reducing air leakage improves energy efficiency by minimizing uncontrolled air exchange."
            elif feature == 'Renewable_Energy_Usage':
                explanation = f"Increase renewable energy usage from {old_formatted} to {new_formatted}"
                detail = "On-site renewable energy generation offsets grid electricity consumption."
            else:
                explanation = f"Improve {feature_name} from {old_formatted} to {new_formatted}"
                detail = "This change reduces overall energy consumption of the building."
            
            # Calculate percentage improvement in EUI
            percentage_improvement = (impact / eui) * 100
            
            recommendations.append({
                'feature': feature,
                'explanation': explanation,
                'detail': detail,
                'impact_eui': impact,
                'percentage_improvement': percentage_improvement,
                'old_value': old_value,
                'new_value': new_value,
                'formatted_old': old_formatted,
                'formatted_new': new_formatted,
            })
        
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []

# Calculate feature importance using model coefficients
def calculate_feature_importance(model, X, preprocessor=None):
    """Calculate feature importance based on the model"""
    try:
        # Get feature names
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Get feature importance from model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = np.mean(importances, axis=0)
        else:
            # If model doesn't support native importance, return equal importance
            importances = np.ones(len(feature_names)) / len(feature_names)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    except Exception as e:
        st.warning(f"Error calculating feature importance: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['Feature', 'Importance'])

# Calculate permutation importance
def calculate_permutation_importance(model, X, preprocessor=None):
    """Calculate permutation importance for the model"""
    try:
        if not HAS_ML:
            return pd.DataFrame(columns=['Feature', 'Importance'])
        
        # Get feature names
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # If preprocessor is provided and X is not preprocessed
        if preprocessor is not None and hasattr(preprocessor, 'transform'):
            X_processed = preprocessor.transform(X)
        else:
            X_processed = X
        
        # Get a sample for permutation importance (for efficiency)
        sample_size = min(500, X_processed.shape[0])
        indices = np.random.choice(X_processed.shape[0], sample_size, replace=False)
        X_sample = X_processed[indices] if isinstance(X_processed, np.ndarray) else X_processed.iloc[indices]
        
        # Generate fake target for permutation importance
        y_sample = model.predict(X_sample)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    except Exception as e:
        st.warning(f"Error calculating permutation importance: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['Feature', 'Importance'])

# Create a chart for feature importance
def create_feature_importance_chart(importance_df, top_n=10):
    """Create a horizontal bar chart for feature importance"""
    if importance_df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Feature Importance Not Available",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400
        )
        return fig
    
    # Take top N features
    df = importance_df.head(top_n).copy()
    
    # Create friendly feature names
    df['Feature_Display'] = df['Feature'].apply(lambda x: x.replace('_', ' ').title())
    
    # Sort for display (ascending for horizontal bar chart)
    df = df.sort_values('Importance')
    
    # Use fixed colors instead of dynamically calculated gradient to avoid NaN issues
    colors = [
        '#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', 
        '#66bb6a', '#4caf50', '#43a047', '#388e3c', 
        '#2e7d32', '#1b5e20'
    ]
    
    # Make sure we have enough colors for all features
    feature_colors = [colors[i % len(colors)] for i in range(len(df))]
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        y=df['Feature_Display'],
        x=df['Importance'],
        orientation='h',
        marker=dict(
            color=feature_colors,
            line=dict(color='rgba(0, 0, 0, 0.1)', width=1)
        )
    ))
    
    # Add value annotations
    for i, row in enumerate(df.itertuples()):
        fig.add_annotation(
            x=row.Importance + max(df['Importance'])*0.02,
            y=row.Feature_Display,
            text=f"{row.Importance:.3f}",
            showarrow=False,
            font=dict(size=10)
        )
    
    fig.update_layout(
        title={
            'text': "Feature Importance",
            'font': {'size': 18, 'color': '#2e7d32'},
            'y': 0.95
        },
        xaxis_title="Importance",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(
            autorange="reversed",
            gridcolor='rgba(0,0,0,0.05)'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.1)',
            zerolinewidth=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

# Calculate CO2 equivalence values
def calculate_co2_equivalence(co2_emissions):
    """Calculate CO2 equivalence in terms of cars, trees, etc."""
    # CO2 equivalence factors based on EPA and other sources
    # kg CO2 per year
    car_emissions_per_year = 4600  # kg CO2 for average passenger vehicle
    tree_sequestration_per_year = 25  # kg CO2 per tree per year
    flight_emissions_per_hour = 90  # kg CO2 per person per hour
    home_energy_emissions_per_year = 5500  # kg CO2 for average home
    
    # Calculate equivalences
    cars_equivalent = co2_emissions / car_emissions_per_year
    trees_needed = co2_emissions / tree_sequestration_per_year
    flight_hours_equivalent = co2_emissions / flight_emissions_per_hour
    homes_equivalent = co2_emissions / home_energy_emissions_per_year
    
    return {
        'cars': cars_equivalent,
        'trees': trees_needed,
        'flight_hours': flight_hours_equivalent,
        'homes': homes_equivalent
    }

# Create a visual representation of CO2 emissions
def create_co2_emissions_chart(annual_emissions, equivalence, key_suffix=""):
    """Create a chart for CO2 emissions with equivalence visualizations"""
    # Create figure with subplots
    fig = go.Figure()
    
    # Add main emissions bar
    fig.add_trace(go.Bar(
        x=['Annual CO₂ Emissions'],
        y=[annual_emissions],
        name='Emissions (kg CO₂/year)',
        marker_color='#2e7d32',  # Dark green
        text=[f"{annual_emissions:,.0f} kg"],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f"Annual CO₂ emissions: {annual_emissions:,.0f} kg"]
    ))
    
    # Equivalence bars
    fig.add_trace(go.Bar(
        x=['Cars'],
        y=[equivalence['cars']],
        name='Cars (annual)',
        marker_color='#388e3c',  # Medium green
        text=[f"{equivalence['cars']:.1f} cars"],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f"Equivalent to {equivalence['cars']:.1f} cars driven for one year"]
    ))
    
    fig.add_trace(go.Bar(
        x=['Trees'],
        y=[equivalence['trees']],
        name='Trees needed',
        marker_color='#4caf50',  # Light green
        text=[f"{equivalence['trees']:.0f} trees"],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f"{equivalence['trees']:.0f} trees needed to absorb this CO₂ annually"]
    ))
    
    fig.add_trace(go.Bar(
        x=['Flight Hours'],
        y=[equivalence['flight_hours']],
        name='Flight hours',
        marker_color='#81c784',  # Very light green
        text=[f"{equivalence['flight_hours']:.0f} hours"],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f"Equivalent to {equivalence['flight_hours']:.0f} hours of flying"]
    ))
    
    fig.update_layout(
        title='CO₂ Emissions and Equivalences',
        barmode='group',
        xaxis_title="",
        yaxis_title="Value (log scale)",
        yaxis_type="log",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Create CO2 emissions trend chart
def create_co2_trend_chart(emissions_forecast, key_suffix=""):
    """Create a chart showing annual and cumulative CO2 emissions over time"""
    fig = go.Figure()
    
    # Add annual emissions bars
    fig.add_trace(go.Bar(
        x=emissions_forecast['Year'],
        y=emissions_forecast['Annual_Emissions'],
        name='Annual Emissions',
        marker_color='#4caf50',  # Green
        hovertemplate='Year %{x}: %{y:,.0f} kg CO₂<extra></extra>'
    ))
    
    # Add cumulative emissions line
    fig.add_trace(go.Scatter(
        x=emissions_forecast['Year'],
        y=emissions_forecast['Cumulative_Emissions'],
        name='Cumulative Emissions',
        mode='lines+markers',
        line=dict(color='#2e7d32', width=3),  # Dark green
        marker=dict(size=8, symbol='circle', color='#1b5e20'),
        hovertemplate='Year %{x}: %{y:,.0f} kg CO₂ (cumulative)<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='CO₂ Emissions Forecast',
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (kg)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Show gridlines only for y-axis
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    
    return fig

# Create partial dependence plot (PDP)
def create_pdp_plot(model, X, feature, preprocessor=None):
    """Create a partial dependence plot for a specific feature"""
    if not HAS_PDP:
        # Return empty figure if PDP functionality not available
        fig = go.Figure()
        fig.update_layout(
            title="Partial Dependence Plot Not Available",
            xaxis_title=feature,
            yaxis_title="Predicted EUI",
            height=400
        )
        return fig
    
    try:
        # Apply preprocessing if provided
        if preprocessor is not None and hasattr(preprocessor, 'transform'):
            X_processed = preprocessor.transform(X)
        else:
            X_processed = X
        
        # Get feature names and index of target feature
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
            feature_idx = feature_names.index(feature)
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            feature_idx = feature_names.index(feature) if feature in feature_names else int(feature.split('_')[1])
        
        # Get feature values for grid points
        if hasattr(X, 'iloc'):
            # If X is a DataFrame
            feature_values = X.iloc[:, feature_idx].unique()
        else:
            # If X is a numpy array
            feature_values = np.unique(X[:, feature_idx])
        
        # Sort values and limit to reasonable number of points
        feature_values = np.sort(feature_values)
        if len(feature_values) > 20:
            feature_values = np.linspace(feature_values.min(), feature_values.max(), 20)
        
        # Create grid for partial dependence
        pd_results = []
        for value in feature_values:
            # Create a copy of a sample
            X_sample = X_processed[0:1].copy() if isinstance(X_processed, np.ndarray) else X_processed.iloc[0:1].copy()
            
            # Repeat the sample for stable prediction
            X_sample = np.repeat(X_sample, 10, axis=0)
            
            # Set the feature value
            X_sample[:, feature_idx] = value
            
            # Predict
            pred = model.predict(X_sample).mean()
            
            pd_results.append({'value': value, 'prediction': pred})
        
        # Create dataframe
        pdp_df = pd.DataFrame(pd_results)
        
        # Create plot
        fig = go.Figure()
        
        # Add PDP line - using only lines for a cleaner look without dots
        fig.add_trace(go.Scatter(
            x=pdp_df['value'],
            y=pdp_df['prediction'],
            mode='lines',
            line=dict(color='#2e7d32', width=4, shape='spline', smoothing=1.3),
            fill='tozeroy',
            fillcolor='rgba(46, 125, 50, 0.1)',
            name=feature
        ))
        
        # Format feature name for display
        display_name = feature.replace('_', ' ').title()
        
        # Update layout
        fig.update_layout(
            title=f"Partial Dependence Plot: {display_name}",
            xaxis_title=display_name,
            yaxis_title="Predicted Energy Use Intensity (EUI)",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Show gridlines
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        
        return fig
    except Exception as e:
        st.warning(f"Error creating PDP plot: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Error Creating Partial Dependence Plot for {feature}",
            xaxis_title=feature,
            yaxis_title="Predicted EUI",
            height=400
        )
        return fig

# Create multiple PDP plots
def create_multiple_pdp_plots(model, X, features, preprocessor=None):
    """Create multiple PDP plots for a list of features"""
    pdp_plots = {}
    for feature in features:
        pdp_plots[feature] = create_pdp_plot(model, X, feature, preprocessor)
    return pdp_plots

# Format table
def format_table(df, precision=2, hide_index=True):
    """Format a DataFrame for display as a Streamlit table"""
    # Format numeric columns with the specified precision
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col] = df[col].apply(lambda x: f"{x:,.{precision}f}" if not pd.isna(x) else "")
    
    # Apply styling to the DataFrame
    styled_df = df.style.set_properties(**{
        'text-align': 'left',
        'font-size': '14px',
        'border-color': '#ddd',
        'padding': '5px'
    })
    
    # Hide index if requested
    if hide_index:
        styled_df = styled_df.hide(axis='index')
    
    # Apply row background colors
    styled_df = styled_df.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#2e7d32'), ('color', 'white')]},
        {'selector': 'tbody tr:nth-of-type(even)', 'props': [('background-color', '#f0f7f0')]},
        {'selector': 'tbody tr:hover', 'props': [('background-color', '#e8f5e9')]},
    ])
    
    return styled_df

# Create an improved visualization for CO2 emissions and equivalences
def create_co2_emissions_visualization(annual_emissions, equivalence):
    """Create an improved visualization for CO2 emissions and equivalences"""
    # Create figure
    fig = go.Figure()
    
    # Data for visualization
    categories = ['CO₂ Emissions', 'Car Equivalents', 'Trees Needed', 'Home Energy']
    values = [
        annual_emissions,
        equivalence['cars'] * 4600,  # convert back to kg CO2
        equivalence['trees'] * 25,   # convert back to kg CO2
        equivalence['homes'] * 5500  # convert back to kg CO2
    ]
    
    # Colors
    colors = ['#2e7d32', '#388e3c', '#4caf50', '#81c784']
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[
            f"{annual_emissions:,.0f} kg/year",
            f"{equivalence['cars']:.1f} cars/year",
            f"{equivalence['trees']:.0f} trees",
            f"{equivalence['homes']:.2f} homes/year"
        ],
        textposition='auto',
        hoverinfo='text',
        hovertext=[
            f"Annual CO₂ emissions: {annual_emissions:,.0f} kg",
            f"Equivalent to {equivalence['cars']:.1f} cars driven for one year",
            f"{equivalence['trees']:.0f} trees needed to absorb this CO₂ annually",
            f"Equivalent to energy use of {equivalence['homes']:.2f} average homes"
        ]
    ))
    
    # Update layout
    fig.update_layout(
        title='CO₂ Emissions and Environmental Impact',
        xaxis_title="",
        yaxis_title="CO₂ (kg/year)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(type='log')  # Use log scale for better visualization
    )
    
    # Show gridlines only for y-axis
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    
    return fig

def main():
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor()
    
    # Initialize session state if not exists
    if 'eui' not in st.session_state:
        st.session_state.eui = None
    if 'params' not in st.session_state:
        st.session_state.params = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'annual_energy' not in st.session_state:
        st.session_state.annual_energy = None
    if 'annual_cost' not in st.session_state:
        st.session_state.annual_cost = None
    if 'co2_emissions' not in st.session_state:
        st.session_state.co2_emissions = None
    if 'has_results' not in st.session_state:
        st.session_state.has_results = False
    if 'building_portfolio' not in st.session_state:
        st.session_state.building_portfolio = pd.DataFrame(columns=[
            'Building_Name', 'Building_Type', 'Area_m2', 'EUI', 'Annual_Energy', 'Annual_Cost', 'CO2_Emissions'
        ])
    
    # ===== App Header =====
    st.title("🏢 Building Energy Analysis Dashboard")
    
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h3 style='color: #2e7d32; margin-top: 0;'>Analyze your building's energy performance and get recommendations for improvement</h3>
        <p>This dashboard helps you understand the energy performance of your building, estimate carbon emissions, and identify improvement opportunities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections of the dashboard
    tabs = st.tabs([
        "📊 Building Analysis", 
        "⚡ Energy Performance & XAI", 
        "🔄 What-If Analysis",
        "💰 Retrofit Analysis",
        "✅ Recommendations"
    ])
    
    # ===== Building Analysis Tab =====
    with tabs[0]:
        st.markdown("### Building Parameters")
        st.markdown("Enter your building details to analyze its energy performance.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Building type and general information
            building_type = st.selectbox(
                "Building Type",
                options=['Residential', 'Office', 'Commercial', 'Industrial', 'Educational'],
                help="Select the type of building you're analyzing."
            )
            
            building_orientation = st.selectbox(
                "Building Orientation",
                options=['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest'],
                help="Primary orientation of the building facade."
            )
            
            weather_file = st.selectbox(
                "Climate Zone",
                options=['Cold', 'Temperate', 'Hot', 'Tropical', 'Arctic'],
                help="Climate zone where the building is located."
            )
            
            total_area = st.number_input(
                "Total Building Area (m²)",
                min_value=10.0,
                max_value=100000.0,
                value=2000.0,
                step=100.0,
                help="Total floor area of the building in square meters."
            )
            
            window_to_wall_ratio = st.slider(
                "Window to Wall Ratio",
                min_value=0.1,
                max_value=0.9,
                value=0.4,
                step=0.05,
                help="Ratio of window area to wall area. Higher values mean more windows."
            )
            
        with col2:
            # Building envelope and systems
            hvac_efficiency = st.slider(
                "HVAC System Efficiency",
                min_value=0.6,
                max_value=0.95,
                value=0.8,
                step=0.05,
                help="Efficiency of the heating, ventilation, and air conditioning system. Higher is better."
            )
            
            lighting_density = st.slider(
                "Lighting Power Density (W/m²)",
                min_value=5.0,
                max_value=20.0,
                value=12.0,
                step=1.0,
                help="Lighting power per unit area. Lower values mean more efficient lighting."
            )
            
            equipment_density = st.slider(
                "Equipment Power Density (W/m²)",
                min_value=5.0,
                max_value=30.0,
                value=15.0,
                step=1.0,
                help="Equipment power per unit area. Lower values mean more efficient equipment."
            )
            
            occupancy_level = st.slider(
                "Occupancy Density (people/m²)",
                min_value=0.01,
                max_value=0.1,
                value=0.05,
                step=0.01,
                help="Number of occupants per unit area."
            )
            
            renewable_energy = st.slider(
                "Renewable Energy Usage",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Proportion of energy from renewable sources. Higher is better."
            )
        
        st.markdown("### Building Envelope Properties")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wall_insulation = st.number_input(
                "Wall Insulation U-Value (W/m²K)",
                min_value=0.1,
                max_value=3.0,
                value=0.5,
                step=0.1,
                help="U-value of the wall insulation. Lower values mean better insulation."
            )
            
            roof_insulation = st.number_input(
                "Roof Insulation U-Value (W/m²K)",
                min_value=0.1,
                max_value=3.0,
                value=0.25,
                step=0.05,
                help="U-value of the roof insulation. Lower values mean better insulation."
            )
        
        with col2:
            floor_insulation = st.number_input(
                "Floor Insulation U-Value (W/m²K)",
                min_value=0.1,
                max_value=3.0,
                value=0.3,
                step=0.1,
                help="U-value of the floor insulation. Lower values mean better insulation."
            )
            
            door_insulation = st.number_input(
                "Door Insulation U-Value (W/m²K)",
                min_value=0.1,
                max_value=3.0,
                value=2.0,
                step=0.1,
                help="U-value of the doors. Lower values mean better insulation."
            )
        
        with col3:
            window_insulation = st.number_input(
                "Window U-Value (W/m²K)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="U-value of the windows. Lower values mean better insulation."
            )
            
            air_change_rate = st.number_input(
                "Air Change Rate (ACH)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Air changes per hour. Lower values mean better airtightness."
            )
        
        st.markdown("### HVAC Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            heating_setpoint = st.slider(
                "Heating Setpoint Temperature (°C)",
                min_value=18.0,
                max_value=24.0,
                value=21.0,
                step=0.5,
                help="Temperature at which heating is activated."
            )
            
            hot_water_usage = st.slider(
                "Hot Water Usage (L/person/day)",
                min_value=20.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                help="Average hot water usage per person per day."
            )
        
        with col2:
            heating_setback = st.slider(
                "Heating Setback Temperature (°C)",
                min_value=15.0,
                max_value=21.0,
                value=18.0,
                step=0.5,
                help="Reduced temperature setting during unoccupied hours."
            )
        
        st.markdown("### Cost Analysis Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            energy_rate = st.number_input(
                "Energy Rate ($/kWh)",
                min_value=0.05,
                max_value=0.5,
                value=0.15,
                step=0.01,
                help="Cost of electricity per kilowatt-hour."
            )
        
        # Button to analyze building energy performance
        analyze_button = st.button("Analyze Building Energy Performance", type="primary")
        
        if analyze_button:
            st.session_state.params = {
                'Building_Type': building_type,
                'Building_Orientation': building_orientation,
                'Weather_File': weather_file,
                'Total_Building_Area': total_area,
                'Window_to_Wall_Ratio': window_to_wall_ratio,
                'HVAC_Efficiency': hvac_efficiency,
                'Lighting_Density': lighting_density,
                'Equipment_Density': equipment_density,
                'Occupancy_Level': occupancy_level,
                'Renewable_Energy_Usage': renewable_energy,
                'Wall_Insulation_U-Value_Capped': wall_insulation,
                'Roof_Insulation_U-Value_Capped': roof_insulation,
                'Floor_Insulation_U-Value_Capped': floor_insulation,
                'Door_Insulation_U-Value_Capped': door_insulation,
                'Window_Insulation_U-Value_Capped': window_insulation,
                'Air_Change_Rate_Capped': air_change_rate,
                'Heating_Setpoint_Temperature': heating_setpoint,
                'Heating_Setback_Temperature': heating_setback,
                'Domestic_Hot_Water_Usage': hot_water_usage
            }
            
            # Display a spinner while calculating
            with st.spinner("Calculating building energy performance..."):
                # Predict EUI
                st.session_state.eui = predict_eui(st.session_state.params, model, preprocessor)
                
                # Calculate annual energy and cost
                st.session_state.annual_energy, st.session_state.annual_cost = calculate_energy_cost(
                    st.session_state.eui, total_area, energy_rate
                )
                
                # Calculate CO2 emissions
                st.session_state.co2_emissions = calculate_co2_emissions(st.session_state.annual_energy)
                
                # Generate recommendations
                st.session_state.recommendations = generate_recommendations(
                    st.session_state.params, st.session_state.eui, model, preprocessor
                )
                
                st.session_state.has_results = True
        
        # Display results if available
        if st.session_state.has_results:
            st.markdown("### Energy Performance Results")
            
            # Calculate energy rating
            rating, color = get_efficiency_rating(st.session_state.eui)
            
            # CO2 equivalence
            co2_equivalence = calculate_co2_equivalence(st.session_state.co2_emissions)
            
            # Create metric columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Improved energy rating display with better proportions
                st.markdown(f"""
                <div style='text-align: center; padding: 12px; background-color: {color}; color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='display: flex; align-items: center; justify-content: center;'>
                        <div style='height: 50px; width: 50px; display: flex; align-items: center; justify-content: center; 
                                  border: 2px solid white; border-radius: 50%; margin-right: 15px;'>
                            <span style='font-size: 1.8rem; font-weight: bold;'>{rating}</span>
                        </div>
                        <div style='text-align: left;'>
                            <div style='font-size: 0.9rem; margin-bottom: 3px;'>Energy Rating</div>
                            <div style='font-weight: 500;'>EUI: {st.session_state.eui:.1f} kWh/m²/year</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "Annual Energy Use",
                    f"{st.session_state.annual_energy/1000:.1f} MWh/year",
                    f"{st.session_state.annual_energy:,.0f} kWh/year"
                )
            
            with col3:
                st.metric(
                    "Annual Energy Cost",
                    f"${st.session_state.annual_cost:,.2f}/year",
                    f"at ${energy_rate:.2f}/kWh"
                )
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Annual CO₂ Emissions",
                    f"{st.session_state.co2_emissions/1000:.1f} tons CO₂/year",
                    f"{st.session_state.co2_emissions:,.0f} kg CO₂/year"
                )
            
            with col2:
                st.metric(
                    "Equivalent to Cars",
                    f"{co2_equivalence['cars']:.1f} cars/year",
                    "driven for one year"
                )
            
            with col3:
                st.metric(
                    "Trees Needed to Offset",
                    f"{co2_equivalence['trees']:.0f} trees",
                    "for annual absorption"
                )
            
            # Create gauge chart for EUI
            gauge_chart = create_gauge_chart(st.session_state.eui, "main")
            st.plotly_chart(gauge_chart, use_container_width=True, key="eui_gauge_main")
            
            # Add Carbon Footprint Section
            st.markdown("### Carbon Footprint Analysis")
            
            # CO2 emissions chart
            co2_chart = create_co2_emissions_chart(st.session_state.co2_emissions, co2_equivalence, key_suffix="main")
            st.plotly_chart(co2_chart, use_container_width=True, key="co2_emissions_main")
            
            # Add Cost Analysis Section
            st.markdown("### Cost Analysis")
            
            # Generate cost forecast
            energy_price = energy_rate
            annual_price_increase = 3.0  # Default value
            forecast_years = 10  # Default value
            area = st.session_state.params['Total_Building_Area']
            
            # Generate energy cost forecast
            forecast_df = generate_energy_forecast(st.session_state.eui, area, energy_price, annual_price_increase, forecast_years)
            
            # Create forecast chart
            cost_fig = go.Figure()
            
            # Add annual cost bars
            cost_fig.add_trace(go.Bar(
                x=forecast_df['Year'],
                y=forecast_df['Annual_Cost'],
                name='Annual Cost',
                marker_color='#4caf50',
                hovertemplate='Year %{x}: $%{y:,.2f}<extra></extra>'
            ))
            
            # Add cumulative cost line
            cost_fig.add_trace(go.Scatter(
                x=forecast_df['Year'],
                y=forecast_df['Cumulative_Cost'],
                name='Cumulative Cost',
                mode='lines+markers',
                line=dict(color='#2e7d32', width=3),
                marker=dict(size=8, symbol='circle', color='#1b5e20'),
                hovertemplate='Year %{x}: $%{y:,.2f} (cumulative)<extra></extra>'
            ))
            
            # Update layout
            cost_fig.update_layout(
                title='Energy Cost Forecast',
                xaxis_title='Year',
                yaxis_title='Cost ($)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Show the forecast chart
            st.plotly_chart(cost_fig, use_container_width=True, key="cost_forecast_main")
            
            # Option to save building to portfolio
            # Save Building to Portfolio Section
            st.markdown("### Save Building to Portfolio")
            
            building_name = st.text_input("Building Name", "My Building")
            
            save_button = st.button("Save to Building Portfolio")
            
            if save_button:
                # Create new building entry
                new_building = pd.DataFrame([{
                    'Building_Name': building_name,
                    'Building_Type': st.session_state.params['Building_Type'],
                    'Area_m2': st.session_state.params['Total_Building_Area'],
                    'EUI': st.session_state.eui,
                    'Annual_Energy': st.session_state.annual_energy,
                    'Annual_Cost': st.session_state.annual_cost,
                    'CO2_Emissions': st.session_state.co2_emissions
                }])
                
                # Append to portfolio
                st.session_state.building_portfolio = pd.concat([st.session_state.building_portfolio, new_building], ignore_index=True)
                
                st.success(f"Building '{building_name}' added to portfolio!")
    
    # ===== Energy Performance Tab =====
    with tabs[1]:
        st.markdown("### Building Energy Performance Analysis")
        
        if not st.session_state.has_results:
            st.info("Please analyze a building first to see its energy performance.")
        else:
            # Show detailed energy performance metrics
            st.markdown("#### Energy Use Intensity (EUI)")
            st.markdown("""
            EUI measures the energy use per square meter per year. Lower values indicate better energy efficiency.
            Different building types have different typical EUI ranges.
            """)
            
            # Building type benchmarks
            benchmarks = {
                'Residential': {'Best': 50, 'Typical': 150, 'Poor': 250},
                'Office': {'Best': 100, 'Typical': 200, 'Poor': 300},
                'Commercial': {'Best': 120, 'Typical': 220, 'Poor': 320},
                'Industrial': {'Best': 150, 'Typical': 250, 'Poor': 350},
                'Educational': {'Best': 90, 'Typical': 180, 'Poor': 270}
            }
            
            # Get benchmarks for the current building type
            building_type = st.session_state.params['Building_Type']
            benchmark = benchmarks.get(building_type, {'Best': 100, 'Typical': 200, 'Poor': 300})
            
            # Create a benchmark comparison chart
            fig = go.Figure()
            
            # Add bars for benchmarks
            fig.add_trace(go.Bar(
                x=['Best Practice', 'Typical', 'Poor Practice'],
                y=[benchmark['Best'], benchmark['Typical'], benchmark['Poor']],
                marker_color=['#2e7d32', '#ffa726', '#d32f2f'],
                name='Benchmark'
            ))
            
            # Add a marker for the current building
            fig.add_trace(go.Scatter(
                x=['Your Building'],
                y=[st.session_state.eui],
                mode='markers',
                marker=dict(size=15, symbol='star', color='#1976d2'),
                name='Your Building'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"EUI Comparison for {building_type} Buildings",
                xaxis_title="Building Category",
                yaxis_title="Energy Use Intensity (kWh/m²/year)",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Show the chart
            st.plotly_chart(fig, use_container_width=True, key="benchmark_comparison")
            
            # Show current performance gauge
            st.markdown("#### Current Performance")
            current_gauge = create_gauge_chart(st.session_state.eui, "current")
            st.plotly_chart(current_gauge, use_container_width=True, key="eui_gauge_current")
    
    # ===== What-If Analysis Tab =====
    with tabs[2]:
        st.markdown("### What-If Analysis")
        st.markdown("Explore how specific changes to your building would impact energy performance.")
        
        if not st.session_state.has_results:
            st.info("Please analyze a building first to use What-If Analysis.")
        else:
            what_if_col1, what_if_col2 = st.columns(2)
            
            with what_if_col1:
                # Select which parameter to modify
                what_if_parameter = st.selectbox(
                    "Select Parameter to Modify",
                    options=[
                        'HVAC_Efficiency',
                        'Window_to_Wall_Ratio',
                        'Wall_Insulation_U-Value_Capped',
                        'Roof_Insulation_U-Value_Capped',
                        'Window_Insulation_U-Value_Capped',
                        'Air_Change_Rate_Capped',
                        'Renewable_Energy_Usage',
                        'Lighting_Density',
                        'Equipment_Density',
                        'Occupancy_Level'
                    ],
                    key="what_if_parameter",
                    help="Choose a building parameter to see how changes affect energy performance."
                )
                
                # Get current value
                current_value = st.session_state.params[what_if_parameter]
                
                # Define min and max values for the parameter with appropriate fixed steps
                if 'Efficiency' in what_if_parameter or 'Usage' in what_if_parameter:
                    min_val = max(current_value * 0.5, 0.1)
                    max_val = min(current_value * 1.5, 0.95)
                    step_val = 0.05
                elif 'U-Value' in what_if_parameter:
                    min_val = max(current_value * 0.5, 0.1)
                    max_val = current_value * 1.5
                    step_val = 0.05
                elif 'Density' in what_if_parameter and 'Lighting' in what_if_parameter:
                    min_val = max(current_value * 0.5, 5.0)
                    max_val = current_value * 1.5
                    step_val = 1.0
                elif 'Density' in what_if_parameter and 'Equipment' in what_if_parameter:
                    min_val = max(current_value * 0.5, 5.0)
                    max_val = current_value * 1.5
                    step_val = 1.0
                elif 'Occupancy' in what_if_parameter:
                    min_val = max(current_value * 0.5, 0.01)
                    max_val = current_value * 1.5
                    step_val = 0.01
                else:
                    min_val = max(current_value * 0.5, 0.1)
                    max_val = current_value * 1.5
                    step_val = 0.05
                
                # Slider for new value using fixed step
                new_value = st.slider(
                    f"Modified {what_if_parameter.replace('_', ' ').title()}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(current_value),
                    step=step_val,
                    key="what_if_value"
                )
                
                # Add informative text
                if new_value < current_value:
                    change_pct = ((current_value - new_value) / current_value) * 100
                    st.markdown(f"*You've decreased this parameter by {change_pct:.1f}%*")
                elif new_value > current_value:
                    change_pct = ((new_value - current_value) / current_value) * 100
                    st.markdown(f"*You've increased this parameter by {change_pct:.1f}%*")
                
                # Add contextual information about the parameter
                parameter_info = {
                    'HVAC_Efficiency': "Higher values mean more efficient heating, cooling, and ventilation systems that require less energy to operate.",
                    'Window_to_Wall_Ratio': "Represents the proportion of windows to wall area. More windows typically increase energy loss but provide more natural light.",
                    'Wall_Insulation_U-Value_Capped': "Lower U-values indicate better insulation and reduced heat transfer through walls.",
                    'Roof_Insulation_U-Value_Capped': "Lower U-values indicate better insulation and reduced heat transfer through the roof.",
                    'Window_Insulation_U-Value_Capped': "Lower U-values indicate better insulated windows with reduced heat transfer.",
                    'Air_Change_Rate_Capped': "Lower rates indicate a more airtight building with fewer air leaks, reducing energy loss.",
                    'Renewable_Energy_Usage': "Higher values mean more energy comes from renewable sources, reducing grid consumption.",
                    'Lighting_Density': "Lower values indicate more efficient lighting systems that consume less energy per area.",
                    'Equipment_Density': "Lower values indicate more efficient equipment that consumes less energy per area.",
                    'Occupancy_Level': "Represents the number of people per area, which affects internal heat gains and energy consumption."
                }
                
                if what_if_parameter in parameter_info:
                    st.info(parameter_info[what_if_parameter])
            
            # Calculate the impact of the parameter change
            modified_params = st.session_state.params.copy()
            modified_params[what_if_parameter] = new_value
            
            # Predict modified EUI
            modified_eui = predict_eui(modified_params, model, preprocessor)
            
            # Calculate energy, cost, and emissions
            energy_rate = 0.15  # Hardcoding for simplicity 
            modified_energy, modified_cost = calculate_energy_cost(
                modified_eui, modified_params['Total_Building_Area'], energy_rate
            )
            
            modified_emissions = calculate_co2_emissions(modified_energy)
            
            # Display impact in the second column
            with what_if_col2:
                # Calculate changes
                impact_eui = modified_eui - st.session_state.eui
                impact_pct = (impact_eui / st.session_state.eui) * 100 if st.session_state.eui > 0 else 0
                
                # Determine if this is an improvement or deterioration
                is_improvement = impact_eui < 0
                impact_color = "#4CAF50" if is_improvement else "#F44336"
                
                # Show impact card
                st.markdown(f"""
                <div style='background-color: rgba({impact_color.replace('#', '').replace(')', '').replace('rgb(', '')}, 0.1); 
                            padding: 1.2rem; border-radius: 0.7rem; margin: 1rem 0; 
                            border-left: 5px solid {impact_color};'>
                    <h4 style='margin-top: 0; color: {impact_color};'>
                        {"Improvement" if is_improvement else "Deterioration"} in Energy Performance
                    </h4>
                    <p>EUI Change: <b>{impact_eui:.1f} kWh/m²/year</b> ({impact_pct:.1f}%)</p>
                    <p>New EUI: <b>{modified_eui:.1f} kWh/m²/year</b></p>
                    <p>Annual Cost Change: <b>${modified_cost - st.session_state.annual_cost:,.2f}</b></p>
                    <p>CO₂ Emissions Change: <b>{(modified_emissions - st.session_state.co2_emissions)/1000:.2f} tons/year</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show new rating vs old rating
                new_rating, new_color = get_efficiency_rating(modified_eui)
                old_rating, old_color = get_efficiency_rating(st.session_state.eui)
                
                st.markdown(f"""
                <div style='padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; background-color: rgba(25, 118, 210, 0.1); border-left: 4px solid #1976D2;'>
                    <h5 style='margin-top: 0; color: #1976D2;'>Energy Performance Rating Change</h5>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                        <div style='text-align: center; padding: 8px; background-color: {old_color}; color: white; border-radius: 5px; width: 45%;'>
                            <div style='font-size: 1.5rem; font-weight: bold;'>{old_rating}</div>
                            <div>Current</div>
                        </div>
                        <div style='display: flex; align-items: center; font-size: 1.5rem;'>→</div>
                        <div style='text-align: center; padding: 8px; background-color: {new_color}; color: white; border-radius: 5px; width: 45%;'>
                            <div style='font-size: 1.5rem; font-weight: bold;'>{new_rating}</div>
                            <div>Modified</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Display before/after comparison charts
            st.markdown("### Before vs. After Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create EUI comparison chart
                compare_fig = go.Figure()
                
                # Add bars for comparison
                compare_fig.add_trace(go.Bar(
                    x=['Current', 'Modified'],
                    y=[st.session_state.eui, modified_eui],
                    marker_color=[old_color, new_color],
                    text=[f"{st.session_state.eui:.1f}", f"{modified_eui:.1f}"],
                    textposition='auto',
                    name='EUI (kWh/m²/year)'
                ))
                
                # Update layout
                compare_fig.update_layout(
                    title='Energy Use Intensity (EUI)',
                    yaxis_title='kWh/m²/year',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Show the comparison chart
                st.plotly_chart(compare_fig, use_container_width=True, key="eui_comparison")
            
            with col2:
                # Create cost comparison chart
                cost_fig = go.Figure()
                
                # Add bars for comparison
                cost_fig.add_trace(go.Bar(
                    x=['Current', 'Modified'],
                    y=[st.session_state.annual_cost, modified_cost],
                    marker_color=['#1976D2', '#4CAF50' if modified_cost < st.session_state.annual_cost else '#F44336'],
                    text=[f"${st.session_state.annual_cost:,.0f}", f"${modified_cost:,.0f}"],
                    textposition='auto',
                    name='Annual Cost ($)'
                ))
                
                # Update layout
                cost_fig.update_layout(
                    title='Annual Energy Cost',
                    yaxis_title='Cost ($)',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Show the cost comparison chart
                st.plotly_chart(cost_fig, use_container_width=True, key="cost_comparison")
            
            # CO2 emissions comparison
            st.markdown("### Carbon Footprint Comparison")
            
            # Create emissions comparison chart
            emissions_fig = go.Figure()
            
            # Calculate CO2 equivalents
            current_co2_equiv = calculate_co2_equivalence(st.session_state.co2_emissions)
            modified_co2_equiv = calculate_co2_equivalence(modified_emissions)
            
            # Add bars for CO2 emissions comparison
            emissions_fig.add_trace(go.Bar(
                x=['Current', 'Modified'],
                y=[st.session_state.co2_emissions/1000, modified_emissions/1000],
                marker_color=['#607D8B', '#4CAF50' if modified_emissions < st.session_state.co2_emissions else '#F44336'],
                text=[f"{st.session_state.co2_emissions/1000:.1f}", f"{modified_emissions/1000:.1f}"],
                textposition='auto',
                name='CO₂ Emissions (tons/year)'
            ))
            
            # Update layout
            emissions_fig.update_layout(
                title='Annual CO₂ Emissions',
                yaxis_title='CO₂ Emissions (tons/year)',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Show the emissions comparison chart
            st.plotly_chart(emissions_fig, use_container_width=True, key="emissions_comparison")
            
            # Create metrics for emissions equivalents
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Current Emissions Equivalence")
                st.markdown(f"""
                <div style='padding: 15px; background-color: #E0E0E0; border-radius: 10px;'>
                    <p><b>{current_co2_equiv['cars']:.1f} cars</b> driven for one year</p>
                    <p><b>{current_co2_equiv['trees']:.0f} trees</b> needed to absorb this CO₂</p>
                    <p><b>{current_co2_equiv['homes']:.1f} homes'</b> annual electricity use</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Modified Emissions Equivalence")
                st.markdown(f"""
                <div style='padding: 15px; background-color: {new_color}; color: white; border-radius: 10px;'>
                    <p><b>{modified_co2_equiv['cars']:.1f} cars</b> driven for one year</p>
                    <p><b>{modified_co2_equiv['trees']:.0f} trees</b> needed to absorb this CO₂</p>
                    <p><b>{modified_co2_equiv['homes']:.1f} homes'</b> annual electricity use</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ===== Retrofit Analysis Tab =====
    with tabs[3]:
        st.markdown("### Building Retrofit Cost-Benefit Analysis")
        st.markdown("Evaluate the financial viability of various energy efficiency upgrades.")
        
        if not st.session_state.has_results:
            st.info("Please analyze a building first to see retrofit options.")
        else:
            # Define common retrofit options
            retrofit_options = {
                "HVAC_System_Upgrade": {
                    "name": "HVAC System Upgrade",
                    "description": "Upgrade to a more efficient HVAC system with smart controls",
                    "param": "HVAC_Efficiency",
                    "current": st.session_state.params["HVAC_Efficiency"],
                    "improved": min(st.session_state.params["HVAC_Efficiency"] + 0.12, 0.95),
                    "cost_per_m2": 35.0,
                    "lifespan": 15
                },
                "LED_Lighting": {
                    "name": "LED Lighting Retrofit",
                    "description": "Replace all lighting with high-efficiency LED fixtures",
                    "param": "Lighting_Density",
                    "current": st.session_state.params["Lighting_Density"],
                    "improved": max(st.session_state.params["Lighting_Density"] * 0.6, 5.0),
                    "cost_per_m2": 12.0,
                    "lifespan": 10
                },
                "Window_Replacement": {
                    "name": "High-Performance Windows",
                    "description": "Replace windows with triple-glazed, low-E, argon-filled units",
                    "param": "Window_Insulation_U-Value_Capped",
                    "current": st.session_state.params["Window_Insulation_U-Value_Capped"],
                    "improved": max(st.session_state.params["Window_Insulation_U-Value_Capped"] * 0.5, 0.8),
                    "cost_per_m2": 150.0 * st.session_state.params["Window_to_Wall_Ratio"],  # Cost proportional to window area
                    "lifespan": 25
                },
                "Wall_Insulation": {
                    "name": "Improved Wall Insulation",
                    "description": "Add external insulation and vapor barriers to walls",
                    "param": "Wall_Insulation_U-Value_Capped",
                    "current": st.session_state.params["Wall_Insulation_U-Value_Capped"],
                    "improved": max(st.session_state.params["Wall_Insulation_U-Value_Capped"] * 0.6, 0.15),
                    "cost_per_m2": 40.0,
                    "lifespan": 30
                },
                "Roof_Insulation": {
                    "name": "Enhanced Roof Insulation",
                    "description": "Upgrade roof insulation and add reflective coating",
                    "param": "Roof_Insulation_U-Value_Capped",
                    "current": st.session_state.params["Roof_Insulation_U-Value_Capped"],
                    "improved": max(st.session_state.params["Roof_Insulation_U-Value_Capped"] * 0.6, 0.12),
                    "cost_per_m2": 25.0,
                    "lifespan": 30
                },
                "Air_Sealing": {
                    "name": "Building Air Sealing",
                    "description": "Comprehensive air sealing of the building envelope",
                    "param": "Air_Change_Rate_Capped",
                    "current": st.session_state.params["Air_Change_Rate_Capped"],
                    "improved": max(st.session_state.params["Air_Change_Rate_Capped"] * 0.6, 0.3),
                    "cost_per_m2": 8.0,
                    "lifespan": 15
                },
                "Solar_PV": {
                    "name": "Solar PV Installation",
                    "description": "Rooftop solar photovoltaic system",
                    "param": "Renewable_Energy_Usage",
                    "current": st.session_state.params["Renewable_Energy_Usage"],
                    "improved": min(st.session_state.params["Renewable_Energy_Usage"] + 0.4, 0.95),
                    "cost_per_m2": 120.0,  # Assumes partial roof coverage based on building size
                    "lifespan": 25
                }
            }
            
            # Calculate for each retrofit option
            building_area = st.session_state.params["Total_Building_Area"]
            energy_rate = 0.15  # Hardcoded for simplicity, can be updated from user input
            annual_increase = 0.03  # Annual energy price increase assumption
            discount_rate = 0.05  # For NPV calculations
            
            # Function to calculate NPV
            def calculate_npv(annual_savings, initial_cost, lifespan, discount_rate=0.05, annual_increase=0.03):
                npv = -initial_cost  # Initial investment (negative cash flow)
                for year in range(1, lifespan + 1):
                    # Annual savings increase with energy prices
                    year_savings = annual_savings * ((1 + annual_increase) ** (year - 1))
                    # Discount the savings to present value
                    npv += year_savings / ((1 + discount_rate) ** year)
                return npv
            
            # Calculate results for each option
            retrofit_results = []
            
            for key, option in retrofit_options.items():
                # Calculate initial investment
                initial_cost = option["cost_per_m2"] * building_area
                
                # Create modified parameters
                test_params = st.session_state.params.copy()
                test_params[option["param"]] = option["improved"]
                
                # Calculate energy use with retrofit
                retrofit_eui = predict_eui(test_params, model, preprocessor)
                
                # Calculate energy and cost savings
                original_energy = st.session_state.annual_energy
                retrofit_energy = retrofit_eui * building_area
                energy_savings = original_energy - retrofit_energy
                
                # Annual cost savings
                annual_cost_savings = energy_savings * energy_rate
                
                # Simple payback period
                simple_payback = initial_cost / annual_cost_savings if annual_cost_savings > 0 else float('inf')
                
                # NPV over lifespan
                npv = calculate_npv(annual_cost_savings, initial_cost, option["lifespan"], discount_rate, annual_increase)
                
                # Return on investment (ROI)
                total_savings = annual_cost_savings * option["lifespan"]
                roi = ((total_savings - initial_cost) / initial_cost) * 100 if initial_cost > 0 else 0
                
                # CO2 reduction
                co2_reduction = (original_energy - retrofit_energy) * 0.5 / 1000  # tons CO2 per year
                
                # Add to results
                retrofit_results.append({
                    "Retrofit": option["name"],
                    "Description": option["description"],
                    "Initial_Cost": initial_cost,
                    "Annual_Savings": annual_cost_savings,
                    "Energy_Reduction": energy_savings,
                    "Energy_Reduction_Pct": (energy_savings / original_energy * 100 if original_energy > 0 else 0),
                    "Payback_Years": simple_payback,
                    "NPV": npv,
                    "ROI": roi,
                    "CO2_Reduction": co2_reduction,
                    "Lifespan": option["lifespan"],
                    "EUI_Reduction": st.session_state.eui - retrofit_eui
                })
            
            # Convert to DataFrame for display
            retrofit_df = pd.DataFrame(retrofit_results)
            
            # Sort by best ROI
            retrofit_df = retrofit_df.sort_values(by="ROI", ascending=False)
            
            # Section for selected retrofit option details
            st.markdown("### Select a Retrofit Option to Analyze")
            
            selected_retrofit = st.selectbox(
                "Choose a retrofit option for detailed analysis:",
                options=retrofit_df["Retrofit"].tolist(),
                key="retrofit_select"
            )
            
            # Get the selected retrofit data
            selected_data = retrofit_df[retrofit_df["Retrofit"] == selected_retrofit].iloc[0]
            
            # Show detailed information for the selected retrofit
            col1, col2 = st.columns(2)
            
            with col1:
                # Financial metrics
                st.markdown(f"#### {selected_data['Retrofit']} Financial Analysis")
                st.markdown(selected_data['Description'])
                
                # Create financial metrics card
                st.markdown(f"""
                <div style='background-color: #E8F5E9; padding: 1.2rem; border-radius: 0.8rem; margin: 0.8rem 0;'>
                    <h5 style='margin-top: 0; color: #2E7D32;'>Financial Summary</h5>
                    <table style='width: 100%;'>
                        <tr>
                            <td style='padding: 8px 0;'><b>Initial Investment:</b></td>
                            <td style='text-align: right;'>${selected_data['Initial_Cost']:,.2f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 8px 0;'><b>Annual Savings:</b></td>
                            <td style='text-align: right;'>${selected_data['Annual_Savings']:,.2f}/year</td>
                        </tr>
                        <tr>
                            <td style='padding: 8px 0;'><b>Simple Payback:</b></td>
                            <td style='text-align: right;'>{selected_data['Payback_Years']:.1f} years</td>
                        </tr>
                        <tr>
                            <td style='padding: 8px 0;'><b>Return on Investment:</b></td>
                            <td style='text-align: right;'>{selected_data['ROI']:.1f}% (lifetime)</td>
                        </tr>
                        <tr>
                            <td style='padding: 8px 0;'><b>Net Present Value:</b></td>
                            <td style='text-align: right;'>${selected_data['NPV']:,.2f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 8px 0;'><b>Expected Lifespan:</b></td>
                            <td style='text-align: right;'>{selected_data['Lifespan']} years</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                # Financial recommendation
                if selected_data['NPV'] > 0:
                    recommendation = "Financially Recommended"
                    rec_color = "#4CAF50"
                elif selected_data['Payback_Years'] < selected_data['Lifespan']:
                    recommendation = "Marginally Viable"
                    rec_color = "#FFA726"
                else:
                    recommendation = "Not Financially Viable"
                    rec_color = "#F44336"
                
                st.markdown(f"""
                <div style='text-align: center; padding: 10px; margin-top: 10px; background-color: {rec_color}; color: white; border-radius: 5px;'>
                    <h4 style='margin: 0;'>{recommendation}</h4>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                # Energy and environmental metrics
                st.markdown(f"#### Environmental Impact")
                
                # Energy reduction
                energy_pct = selected_data['Energy_Reduction_Pct']
                energy_color = "#4CAF50" if energy_pct > 10 else "#FFA726" if energy_pct > 5 else "#F44336"
                
                st.markdown(f"""
                <div style='padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; background-color: rgba({energy_color.replace('#', '').replace(')', '').replace('rgb(', '')}, 0.1); border-left: 4px solid {energy_color};'>
                    <h5 style='margin-top: 0; color: {energy_color};'>Energy Reduction</h5>
                    <div style='font-size: 2rem; font-weight: bold; text-align: center;'>{energy_pct:.1f}%</div>
                    <div style='text-align: center;'>{selected_data['Energy_Reduction']/1000:.1f} MWh/year</div>
                </div>
                """, unsafe_allow_html=True)
                
                # CO2 reduction
                annual_co2 = selected_data['CO2_Reduction']
                lifetime_co2 = annual_co2 * selected_data['Lifespan']
                
                st.markdown(f"""
                <div style='padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; background-color: rgba(46, 125, 50, 0.1); border-left: 4px solid #2E7D32;'>
                    <h5 style='margin-top: 0; color: #2E7D32;'>Carbon Reduction</h5>
                    <div style='font-size: 1.2rem; font-weight: bold; margin-bottom: 5px;'>{annual_co2:.1f} tons CO₂/year</div>
                    <div style='font-size: 0.9rem;'>Equivalent to removing {annual_co2/4.6:.1f} cars from the road annually</div>
                    <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(46, 125, 50, 0.2);'>
                        <div style='font-size: 1.2rem; font-weight: bold;'>{lifetime_co2:.1f} tons CO₂ lifetime</div>
                        <div style='font-size: 0.9rem;'>Equivalent to planting {lifetime_co2*40:.0f} trees</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Performance improvement
                eui_reduction = selected_data['EUI_Reduction']
                new_rating, new_color = get_efficiency_rating(st.session_state.eui - eui_reduction)
                old_rating, old_color = get_efficiency_rating(st.session_state.eui)
                
                st.markdown(f"""
                <div style='padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; background-color: rgba(25, 118, 210, 0.1); border-left: 4px solid #1976D2;'>
                    <h5 style='margin-top: 0; color: #1976D2;'>Performance Rating Change</h5>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                        <div style='text-align: center; padding: 8px; background-color: {old_color}; color: white; border-radius: 5px; width: 45%;'>
                            <div style='font-size: 1.5rem; font-weight: bold;'>{old_rating}</div>
                            <div>Current</div>
                        </div>
                        <div style='display: flex; align-items: center; font-size: 1.5rem;'>→</div>
                        <div style='text-align: center; padding: 8px; background-color: {new_color}; color: white; border-radius: 5px; width: 45%;'>
                            <div style='font-size: 1.5rem; font-weight: bold;'>{new_rating}</div>
                            <div>After Retrofit</div>
                        </div>
                    </div>
                    <div style='text-align: center; margin-top: 5px;'>
                        EUI Reduction: {eui_reduction:.1f} kWh/m²/year
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Compare all retrofit options
            st.markdown("### Comparison of All Retrofit Options")
            
            # Create ROI comparison chart
            fig = go.Figure()
            
            # Add bars for ROI
            fig.add_trace(go.Bar(
                x=retrofit_df["Retrofit"],
                y=retrofit_df["ROI"],
                marker_color="#4CAF50",
                name="ROI (%)"
            ))
            
            # Update layout
            fig.update_layout(
                title="Return on Investment Comparison",
                xaxis_title="Retrofit Option",
                yaxis_title="Return on Investment (%)",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis={'categoryorder':'total descending'},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Show the chart
            st.plotly_chart(fig, use_container_width=True, key="roi_comparison")
            
            # Create payback period comparison chart
            fig = go.Figure()
            
            # Filter out infinite payback periods
            filtered_df = retrofit_df[retrofit_df["Payback_Years"] < 100]  # Exclude unreasonable paybacks
            
            if not filtered_df.empty:
                # Add bars for payback
                fig.add_trace(go.Bar(
                    x=filtered_df["Retrofit"],
                    y=filtered_df["Payback_Years"],
                    marker_color="#1976D2",
                    name="Payback (Years)"
                ))
                
                # Add line for typical acceptable payback period
                fig.add_trace(go.Scatter(
                    x=filtered_df["Retrofit"],
                    y=[7] * len(filtered_df),  # Typical acceptable payback threshold
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name="Typical Acceptance Threshold (7 years)"
                ))
                
                # Update layout
                fig.update_layout(
                    title="Payback Period Comparison",
                    xaxis_title="Retrofit Option",
                    yaxis_title="Payback Period (Years)",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis={'categoryorder':'total ascending'},
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Show the chart
                st.plotly_chart(fig, use_container_width=True, key="payback_comparison")
            else:
                st.info("No retrofit options with reasonable payback periods available.")
            
            # Create table with all options
            st.markdown("### Retrofit Options Summary Table")
            
            # Format table for display
            display_df = pd.DataFrame({
                'Retrofit': retrofit_df['Retrofit'],
                'Initial Cost': retrofit_df['Initial_Cost'].apply(lambda x: f"${x:,.2f}"),
                'Annual Savings': retrofit_df['Annual_Savings'].apply(lambda x: f"${x:,.2f}"),
                'Energy Reduction': retrofit_df['Energy_Reduction_Pct'].apply(lambda x: f"{x:.1f}%"),
                'Payback (Years)': retrofit_df['Payback_Years'].apply(lambda x: f"{x:.1f}" if x < 100 else "N/A"),
                'ROI': retrofit_df['ROI'].apply(lambda x: f"{x:.1f}%"),
                'CO₂ Reduction (tons/yr)': retrofit_df['CO2_Reduction'].apply(lambda x: f"{x:.1f}")
            })
            
            # Function to apply color highlighting to the table
            def highlight_npv(val, col_name):
                if col_name == 'ROI':
                    try:
                        value = float(val.strip('%'))
                        if value > 100:
                            return 'background-color: #C8E6C9; color: #2E7D32; font-weight: bold'
                        elif value > 50:
                            return 'background-color: #E8F5E9; color: #388E3C'
                        elif value > 0:
                            return 'background-color: #F1F8E9; color: #558B2F'
                        else:
                            return 'background-color: #FFEBEE; color: #C62828'
                    except:
                        return ''
                elif col_name == 'Payback (Years)':
                    try:
                        if val == 'N/A':
                            return 'background-color: #FFEBEE; color: #C62828'
                        value = float(val)
                        if value < 5:
                            return 'background-color: #C8E6C9; color: #2E7D32; font-weight: bold'
                        elif value < 10:
                            return 'background-color: #E8F5E9; color: #388E3C'
                        elif value < 15:
                            return 'background-color: #FFF8E1; color: #F57F17'
                        else:
                            return 'background-color: #FFF3E0; color: #E65100'
                    except:
                        return ''
                return ''
            
            # Apply the styling
            def apply_style(row):
                return [highlight_npv(val, col) for val, col in zip(row.values, row.index)]
            
            # Display the styled table
            st.dataframe(display_df.style.apply(apply_style, axis=1), use_container_width=True)
            
            # Combined retrofit analysis
            st.markdown("### Combined Retrofit Package Analysis")
            st.markdown("Analyze the impact of implementing multiple retrofit measures simultaneously.")
            
            # Let user select multiple retrofits to combine
            selected_retrofits = st.multiselect(
                "Select retrofit options to combine:",
                options=retrofit_df["Retrofit"].tolist(),
                default=[retrofit_df["Retrofit"].iloc[0]] if not retrofit_df.empty else [],
                key="combined_retrofits"
            )
            
            if selected_retrofits:
                # Button to calculate combined impact
                if st.button("Calculate Combined Impact", key="calculate_combined"):
                    # Create modified parameters for combined retrofit
                    combined_params = st.session_state.params.copy()
                    
                    # Total cost of combined retrofits
                    total_cost = 0
                    
                    # Apply all selected retrofits
                    for retrofit_name in selected_retrofits:
                        retrofit_data = retrofit_df[retrofit_df["Retrofit"] == retrofit_name].iloc[0]
                        retrofit_key = next((k for k, v in retrofit_options.items() if v["name"] == retrofit_name), None)
                        
                        if retrofit_key:
                            option = retrofit_options[retrofit_key]
                            combined_params[option["param"]] = option["improved"]
                            total_cost += retrofit_data["Initial_Cost"]
                    
                    # Calculate energy use with combined retrofits
                    combined_eui = predict_eui(combined_params, model, preprocessor)
                    
                    # Calculate energy and cost savings
                    original_energy = st.session_state.annual_energy
                    combined_energy = combined_eui * building_area
                    energy_savings = original_energy - combined_energy
                    
                    # Annual cost savings
                    annual_cost_savings = energy_savings * energy_rate
                    
                    # Simple payback period
                    simple_payback = total_cost / annual_cost_savings if annual_cost_savings > 0 else float('inf')
                    
                    # Average lifespan (weighted by cost)
                    weighted_lifespan = 0
                    for retrofit_name in selected_retrofits:
                        retrofit_data = retrofit_df[retrofit_df["Retrofit"] == retrofit_name].iloc[0]
                        retrofit_key = next((k for k, v in retrofit_options.items() if v["name"] == retrofit_name), None)
                        if retrofit_key:
                            option = retrofit_options[retrofit_key]
                            weighted_lifespan += (retrofit_data["Initial_Cost"] / total_cost) * option["lifespan"]
                    
                    # NPV over weighted lifespan
                    combined_lifespan = int(weighted_lifespan)
                    npv = calculate_npv(annual_cost_savings, total_cost, combined_lifespan, discount_rate, annual_increase)
                    
                    # Return on investment (ROI)
                    total_savings = annual_cost_savings * combined_lifespan
                    roi = ((total_savings - total_cost) / total_cost) * 100 if total_cost > 0 else 0
                    
                    # CO2 reduction
                    co2_reduction = (original_energy - combined_energy) * 0.5 / 1000  # tons CO2 per year
                    
                    # Display combined results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Financial summary
                        st.markdown("#### Financial Summary of Combined Package")
                        
                        st.markdown(f"""
                        <div style='background-color: #E8F5E9; padding: 1.2rem; border-radius: 0.8rem; margin: 0.8rem 0;'>
                            <h5 style='margin-top: 0; color: #2E7D32;'>Combined Financial Impact</h5>
                            <table style='width: 100%;'>
                                <tr>
                                    <td style='padding: 8px 0;'><b>Total Investment:</b></td>
                                    <td style='text-align: right;'>${total_cost:,.2f}</td>
                                </tr>
                                <tr>
                                    <td style='padding: 8px 0;'><b>Annual Savings:</b></td>
                                    <td style='text-align: right;'>${annual_cost_savings:,.2f}/year</td>
                                </tr>
                                <tr>
                                    <td style='padding: 8px 0;'><b>Simple Payback:</b></td>
                                    <td style='text-align: right;'>{simple_payback:.1f} years</td>
                                </tr>
                                <tr>
                                    <td style='padding: 8px 0;'><b>Return on Investment:</b></td>
                                    <td style='text-align: right;'>{roi:.1f}% (lifetime)</td>
                                </tr>
                                <tr>
                                    <td style='padding: 8px 0;'><b>Net Present Value:</b></td>
                                    <td style='text-align: right;'>${npv:,.2f}</td>
                                </tr>
                                <tr>
                                    <td style='padding: 8px 0;'><b>Weighted Lifespan:</b></td>
                                    <td style='text-align: right;'>{combined_lifespan} years</td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Financial recommendation
                        if npv > 0:
                            recommendation = "Financially Recommended"
                            rec_color = "#4CAF50"
                        elif simple_payback < combined_lifespan:
                            recommendation = "Marginally Viable"
                            rec_color = "#FFA726"
                        else:
                            recommendation = "Not Financially Viable"
                            rec_color = "#F44336"
                        
                        st.markdown(f"""
                        <div style='text-align: center; padding: 10px; margin-top: 10px; background-color: {rec_color}; color: white; border-radius: 5px;'>
                            <h4 style='margin: 0;'>{recommendation}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Energy and environmental impact
                        st.markdown("#### Environmental Impact")
                        
                        # Energy reduction
                        energy_pct = energy_savings / original_energy * 100 if original_energy > 0 else 0
                        energy_color = "#4CAF50" if energy_pct > 15 else "#FFA726" if energy_pct > 8 else "#F44336"
                        
                        st.markdown(f"""
                        <div style='padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; background-color: rgba({energy_color.replace('#', '').replace(')', '').replace('rgb(', '')}, 0.1); border-left: 4px solid {energy_color};'>
                            <h5 style='margin-top: 0; color: {energy_color};'>Energy Reduction</h5>
                            <div style='font-size: 2rem; font-weight: bold; text-align: center;'>{energy_pct:.1f}%</div>
                            <div style='text-align: center;'>{energy_savings/1000:.1f} MWh/year</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # CO2 reduction
                        lifetime_co2 = co2_reduction * combined_lifespan
                        
                        st.markdown(f"""
                        <div style='padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; background-color: rgba(46, 125, 50, 0.1); border-left: 4px solid #2E7D32;'>
                            <h5 style='margin-top: 0; color: #2E7D32;'>Carbon Reduction</h5>
                            <div style='font-size: 1.2rem; font-weight: bold; margin-bottom: 5px;'>{co2_reduction:.1f} tons CO₂/year</div>
                            <div style='font-size: 0.9rem;'>Equivalent to removing {co2_reduction/4.6:.1f} cars from the road annually</div>
                            <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(46, 125, 50, 0.2);'>
                                <div style='font-size: 1.2rem; font-weight: bold;'>{lifetime_co2:.1f} tons CO₂ lifetime</div>
                                <div style='font-size: 0.9rem;'>Equivalent to planting {lifetime_co2*40:.0f} trees</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Performance improvement
                        eui_reduction = st.session_state.eui - combined_eui
                        new_rating, new_color = get_efficiency_rating(combined_eui)
                        old_rating, old_color = get_efficiency_rating(st.session_state.eui)
                        
                        st.markdown(f"""
                        <div style='padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; background-color: rgba(25, 118, 210, 0.1); border-left: 4px solid #1976D2;'>
                            <h5 style='margin-top: 0; color: #1976D2;'>Performance Rating Change</h5>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                                <div style='text-align: center; padding: 8px; background-color: {old_color}; color: white; border-radius: 5px; width: 45%;'>
                                    <div style='font-size: 1.5rem; font-weight: bold;'>{old_rating}</div>
                                    <div>Current</div>
                                </div>
                                <div style='display: flex; align-items: center; font-size: 1.5rem;'>→</div>
                                <div style='text-align: center; padding: 8px; background-color: {new_color}; color: white; border-radius: 5px; width: 45%;'>
                                    <div style='font-size: 1.5rem; font-weight: bold;'>{new_rating}</div>
                                    <div>After Retrofits</div>
                                </div>
                            </div>
                            <div style='text-align: center; margin-top: 5px;'>
                                EUI Reduction: {eui_reduction:.1f} kWh/m²/year
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Please select at least one retrofit option to analyze combined impact.")

    # ===== Cost Analysis Tab =====
    with tabs[3]:
        st.markdown("### Building Energy Cost Analysis")
        
        if not st.session_state.has_results:
            st.info("Please analyze a building first to see cost analysis.")
        else:
            # Show annual energy cost details
            st.markdown("#### Annual Energy Cost Breakdown")
            
            # Create cost parameters
            col1, col2 = st.columns(2)
            
            with col1:
                energy_price = st.number_input(
                    "Energy Price ($/kWh)",
                    min_value=0.05,
                    max_value=0.5,
                    value=0.15,
                    step=0.01
                )
                annual_price_increase = st.slider(
                    "Annual Energy Price Increase (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5
                )
            
            with col2:
                forecast_years = st.slider(
                    "Forecast Period (years)",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=5
                )
                
                area = st.session_state.params['Total_Building_Area']
                st.metric(
                    "Building Area",
                    f"{area:,.0f} m²"
                )
            
            # Calculate current annual cost
            annual_energy, annual_cost = calculate_energy_cost(st.session_state.eui, area, energy_price)
            
            # Generate energy cost forecast
            forecast_df = generate_energy_forecast(st.session_state.eui, area, energy_price, annual_price_increase, forecast_years)
            
            # Create forecast chart
            fig = go.Figure()
            
            # Add annual cost bars
            fig.add_trace(go.Bar(
                x=forecast_df['Year'],
                y=forecast_df['Annual_Cost'],
                name='Annual Cost',
                marker_color='#4caf50',
                hovertemplate='Year %{x}: $%{y:,.2f}<extra></extra>'
            ))
            
            # Add cumulative cost line
            fig.add_trace(go.Scatter(
                x=forecast_df['Year'],
                y=forecast_df['Cumulative_Cost'],
                name='Cumulative Cost',
                mode='lines+markers',
                line=dict(color='#2e7d32', width=3),
                marker=dict(size=8, symbol='circle', color='#1b5e20'),
                hovertemplate='Year %{x}: $%{y:,.2f} (cumulative)<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title='Energy Cost Forecast',
                xaxis_title='Year',
                yaxis_title='Cost ($)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Show the forecast chart
            st.plotly_chart(fig, use_container_width=True, key="cost_forecast_chart")
            
            # Show cost savings from improvements
            st.markdown("#### Potential Cost Savings")
            
            if st.session_state.recommendations:
                # Calculate potential energy savings
                total_eui_reduction = sum(rec['impact_eui'] for rec in st.session_state.recommendations)
                improved_eui = max(st.session_state.eui - total_eui_reduction, 0)
                
                # Calculate improved annual cost
                improved_energy, improved_cost = calculate_energy_cost(improved_eui, area, energy_price)
                annual_savings = annual_cost - improved_cost
                
                # Generate improved cost forecast
                improved_forecast = generate_energy_forecast(improved_eui, area, energy_price, annual_price_increase, forecast_years)
                
                # Calculate cumulative savings
                savings_df = pd.DataFrame({
                    'Year': forecast_df['Year'],
                    'Original_Cost': forecast_df['Annual_Cost'],
                    'Improved_Cost': improved_forecast['Annual_Cost'],
                    'Annual_Savings': forecast_df['Annual_Cost'] - improved_forecast['Annual_Cost'],
                    'Cumulative_Savings': forecast_df['Cumulative_Cost'] - improved_forecast['Cumulative_Cost']
                })
                
                # Create savings metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Annual Energy Cost Savings",
                        f"${annual_savings:,.2f}/year",
                        f"{(annual_savings/annual_cost)*100:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "10-Year Cumulative Savings",
                        f"${savings_df['Cumulative_Savings'].iloc[-1]:,.2f}",
                        "over 10 years"
                    )
                
                with col3:
                    st.metric(
                        "Average Annual Savings",
                        f"${savings_df['Annual_Savings'].mean():,.2f}/year",
                        "average over 10 years"
                    )
                
                # Create savings chart
                fig = go.Figure()
                
                # Add annual savings bars
                fig.add_trace(go.Bar(
                    x=savings_df['Year'],
                    y=savings_df['Annual_Savings'],
                    name='Annual Savings',
                    marker_color='#4caf50',
                    hovertemplate='Year %{x}: $%{y:,.2f}<extra></extra>'
                ))
                
                # Add cumulative savings line
                fig.add_trace(go.Scatter(
                    x=savings_df['Year'],
                    y=savings_df['Cumulative_Savings'],
                    name='Cumulative Savings',
                    mode='lines+markers',
                    line=dict(color='#2e7d32', width=3),
                    marker=dict(size=8, symbol='circle', color='#1b5e20'),
                    hovertemplate='Year %{x}: $%{y:,.2f} (cumulative)<extra></extra>'
                ))
                
                # Update layout
                fig.update_layout(
                    title='Potential Cost Savings from Improvements',
                    xaxis_title='Year',
                    yaxis_title='Savings ($)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Show the savings chart
                st.plotly_chart(fig, use_container_width=True, key="cost_savings_chart")
                
                # Show ROI analysis if we have improvement costs
                st.markdown("#### Return on Investment (ROI) Analysis")
                st.info("Enter estimated costs for each improvement to calculate ROI.")
                
                # Create a DataFrame for ROI analysis
                roi_data = []
                
                for i, rec in enumerate(st.session_state.recommendations[:5]):
                    # Estimate a default implementation cost
                    default_cost = annual_savings * (i + 1)  # Simple placeholder
                    
                    # Add to ROI data
                    rec_name = rec['explanation'].split(' from ')[0]
                    impact_eui = rec['impact_eui']
                    annual_energy_saved = impact_eui * area
                    annual_cost_saved = annual_energy_saved * energy_price
                    
                    roi_data.append({
                        'Recommendation': rec_name,
                        'Implementation Cost': default_cost,
                        'Annual Savings': annual_cost_saved,
                        'Simple Payback (years)': default_cost / annual_cost_saved if annual_cost_saved > 0 else float('inf'),
                        '10-Year Savings': annual_cost_saved * 10,
                        'ROI (10 years)': (annual_cost_saved * 10 - default_cost) / default_cost * 100 if default_cost > 0 else float('inf')
                    })
                
                # Create DataFrame
                roi_df = pd.DataFrame(roi_data)
                
                # Allow user to edit costs
                edited_roi = st.data_editor(
                    roi_df,
                    column_config={
                        'Recommendation': st.column_config.TextColumn("Recommendation", width="large"),
                        'Implementation Cost': st.column_config.NumberColumn("Implementation Cost ($)", format="$%.2f"),
                        'Annual Savings': st.column_config.NumberColumn("Annual Savings ($)", format="$%.2f"),
                        'Simple Payback (years)': st.column_config.NumberColumn("Simple Payback (years)", format="%.1f years"),
                        '10-Year Savings': st.column_config.NumberColumn("10-Year Savings ($)", format="$%.2f"),
                        'ROI (10 years)': st.column_config.NumberColumn("ROI (10 years)", format="%.1f%%")
                    },
                    disabled=['Recommendation', 'Annual Savings', 'Simple Payback (years)', '10-Year Savings', 'ROI (10 years)'],
                    use_container_width=True,
                    hide_index=True,
                    key="roi_editor"
                )
            else:
                st.info("No improvement recommendations available for cost analysis.")
    
    # ===== Carbon Footprint Tab =====
    # ===== Recommendations Tab =====
    with tabs[3]:
        st.markdown("### Comprehensive Building Energy Recommendations")
        
        if not st.session_state.has_results:
            st.info("Please analyze a building first to see carbon footprint analysis.")
        else:
            # Carbon emissions parameters
            col1, col2 = st.columns(2)
            
            with col1:
                emission_factor = st.slider(
                    "CO₂ Emission Factor (kg CO₂/kWh)",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Carbon intensity of the electricity grid. Lower values indicate cleaner energy."
                )
            
            with col2:
                carbon_forecast_years = st.slider(
                    "Carbon Forecast Period (years)",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=5
                )
            
            # Calculate carbon emissions
            annual_emissions = calculate_co2_emissions(st.session_state.annual_energy, emission_factor)
            
            # Calculate equivalence values
            equivalence = calculate_co2_equivalence(annual_emissions)
            
            # Create emissions visualization
            co2_viz = create_co2_emissions_visualization(annual_emissions, equivalence)
            
            # Show the visualization
            st.plotly_chart(co2_viz, use_container_width=True, key="co2_emissions_viz")
            
            # Generate carbon forecast
            emissions_forecast = generate_emissions_forecast(
                st.session_state.annual_energy, emission_factor, carbon_forecast_years
            )
            
            # Create CO2 trend chart
            co2_forecast_chart = create_co2_trend_chart(emissions_forecast, "forecast")
            
            # Show the forecast chart
            st.plotly_chart(co2_forecast_chart, use_container_width=True, key="co2_forecast_chart")
            
            # Show CO2 reduction potential
            st.markdown("#### Carbon Reduction Potential")
            
            if st.session_state.recommendations:
                # Calculate potential energy savings
                total_eui_reduction = sum(rec['impact_eui'] for rec in st.session_state.recommendations)
                improved_eui = max(st.session_state.eui - total_eui_reduction, 0)
                
                # Calculate improved emissions
                improved_energy = improved_eui * st.session_state.params['Total_Building_Area']
                improved_emissions = calculate_co2_emissions(improved_energy, emission_factor)
                annual_emission_reduction = annual_emissions - improved_emissions
                
                # Generate improved emissions forecast
                improved_forecast = generate_emissions_forecast(
                    improved_energy, emission_factor, carbon_forecast_years
                )
                
                # Calculate cumulative savings
                emission_savings_df = pd.DataFrame({
                    'Year': emissions_forecast['Year'],
                    'Original_Emissions': emissions_forecast['Annual_Emissions'],
                    'Improved_Emissions': improved_forecast['Annual_Emissions'],
                    'Annual_Reduction': emissions_forecast['Annual_Emissions'] - improved_forecast['Annual_Emissions'],
                    'Cumulative_Reduction': emissions_forecast['Cumulative_Emissions'] - improved_forecast['Cumulative_Emissions']
                })
                
                # Create emissions reduction metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Annual CO₂ Reduction",
                        f"{annual_emission_reduction/1000:.1f} tons/year",
                        f"{(annual_emission_reduction/annual_emissions)*100:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "10-Year Cumulative Reduction",
                        f"{emission_savings_df['Cumulative_Reduction'].iloc[-1]/1000:.1f} tons",
                        "over 10 years"
                    )
                
                with col3:
                    trees_saved = calculate_co2_equivalence(annual_emission_reduction)['trees']
                    st.metric(
                        "Equivalent Trees Saved",
                        f"{trees_saved:.0f} trees",
                        "annual absorption"
                    )
                
                # Create emissions reduction chart
                fig = go.Figure()
                
                # Add original emissions
                fig.add_trace(go.Bar(
                    x=emission_savings_df['Year'],
                    y=emission_savings_df['Original_Emissions'],
                    name='Original Emissions',
                    marker_color='#d32f2f',
                    hovertemplate='Year %{x}: %{y:,.0f} kg CO₂<extra></extra>'
                ))
                
                # Add improved emissions
                fig.add_trace(go.Bar(
                    x=emission_savings_df['Year'],
                    y=emission_savings_df['Improved_Emissions'],
                    name='Improved Emissions',
                    marker_color='#2e7d32',
                    hovertemplate='Year %{x}: %{y:,.0f} kg CO₂<extra></extra>'
                ))
                
                # Update layout
                fig.update_layout(
                    title='Carbon Emission Reduction Potential',
                    xaxis_title='Year',
                    yaxis_title='CO₂ Emissions (kg/year)',
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Show the emissions reduction chart
                st.plotly_chart(fig, use_container_width=True, key="emission_reduction_chart")
                
                # Show emission sources breakdown
                st.markdown("#### Carbon Emissions Sources")
                
                # Estimate emissions breakdown based on building parameters
                hvac_factor = (1 - st.session_state.params['HVAC_Efficiency']) * 2
                lighting_factor = st.session_state.params['Lighting_Density'] / 15
                equipment_factor = st.session_state.params['Equipment_Density'] / 20
                envelope_factor = (st.session_state.params['Window_Insulation_U-Value_Capped'] + 
                                  st.session_state.params['Wall_Insulation_U-Value_Capped']) / 3
                
                # Calculate emission percentages
                total_factor = hvac_factor + lighting_factor + equipment_factor + envelope_factor
                hvac_pct = hvac_factor / total_factor * 100
                lighting_pct = lighting_factor / total_factor * 100
                equipment_pct = equipment_factor / total_factor * 100
                envelope_pct = envelope_factor / total_factor * 100
                
                # Create pie chart of emission sources
                fig = go.Figure(data=[go.Pie(
                    labels=['HVAC Systems', 'Lighting', 'Equipment', 'Building Envelope'],
                    values=[hvac_pct, lighting_pct, equipment_pct, envelope_pct],
                    hole=.4,
                    marker=dict(colors=['#2e7d32', '#4caf50', '#81c784', '#c8e6c9']),
                    textinfo='label+percent',
                    hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
                )])
                
                fig.update_layout(
                    title='Estimated Carbon Emissions by Source',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Show the pie chart
                st.plotly_chart(fig, use_container_width=True, key="emission_pie_chart")
            else:
                st.info("No improvement recommendations available for carbon analysis.")
    
    # XAI features are now in the Energy Performance & XAI tab (tab 1)
    with tabs[1]:
        st.markdown("### Building Energy Performance & XAI")
        
        if not st.session_state.has_results:
            st.info("Please analyze a building first to see energy performance analysis.")
        else:
            # Create a navigational tab bar within Model Insights
            xai_tabs = st.tabs([
                "Model Explainability",
                "SHAP Analysis", 
                "LIME Explanations", 
                "Partial Dependence Plots", 
                "Sensitivity Analysis"
            ])
            
            # Create sample data for analysis
            features = [f for f in st.session_state.params.keys()]
            X_sample = pd.DataFrame([st.session_state.params])
            
            # === Model Explainability Tab ===
            with xai_tabs[0]:
                st.markdown("#### Model Explainability")
                st.markdown("""
                Explainable AI (XAI) techniques help understand how the model makes predictions. This dashboard provides several methods to analyze and interpret the model:
                
                - **SHAP Analysis**: Shows how each feature contributes to the prediction for your building
                - **LIME Explanations**: Creates a simplified local model to explain individual predictions
                - **Partial Dependence Plots**: Shows how changing a feature affects the prediction
                - **Sensitivity Analysis**: Tests how changing a parameter impacts energy performance
                
                Explore each tab to gain different insights into your building's energy performance drivers.
                """)
                
                # Calculate permutation importance if available
                if HAS_ML:
                    # Calculate permutation importance
                    perm_df = calculate_permutation_importance(model, X_sample, preprocessor)
                    
                    # Create permutation importance chart
                    perm_chart = create_feature_importance_chart(perm_df)
                    
                    # Show permutation importance
                    st.markdown("#### Permutation Feature Importance")
                    st.markdown("This shows how much model performance drops when each feature is randomly shuffled, revealing which parameters have the greatest impact on energy performance predictions.")
                    
                    st.plotly_chart(perm_chart, use_container_width=True, key="permutation_importance_chart")
            
            # === SHAP Analysis Tab ===
            with xai_tabs[1]:
                if HAS_XAI:
                    st.markdown("#### SHAP (SHapley Additive exPlanations) Analysis")
                    st.markdown("SHAP values explain how each feature contributes to pushing the model output from the base value to the final prediction.")
                    
                    try:
                        # Get SHAP values
                        processed_X = preprocessor.transform(X_sample)
                        shap_values = get_shap_values(model, processed_X)
                        
                        if shap_values is not None:
                            # Create SHAP summary plot
                            st.markdown("##### SHAP Summary Plot")
                            summary_plot = plot_shap_summary(shap_values, features)
                            
                            if summary_plot is not None:
                                st.pyplot(summary_plot, key="shap_summary_plot")
                                plt.close()
                            
                            # Create SHAP waterfall plot
                            st.markdown("##### SHAP Waterfall Plot")
                            waterfall_plot = plot_shap_waterfall(shap_values, features)
                            
                            if waterfall_plot is not None:
                                st.pyplot(waterfall_plot, key="shap_waterfall_plot")
                                plt.close()
                    except Exception as e:
                        st.error(f"Error generating SHAP plots: {e}")
                else:
                    st.info("SHAP analysis requires the SHAP library which is not available in this environment.")
            
            # === LIME Explanations Tab ===
            with xai_tabs[2]:
                if HAS_XAI:
                    st.markdown("#### LIME (Local Interpretable Model-agnostic Explanations)")
                    st.markdown("LIME explains individual predictions by approximating the model locally with a simpler, interpretable model.")
                    
                    try:
                        # Get LIME explanation
                        lime_exp = get_lime_explanation(model, preprocessor.transform(X_sample), features)
                        
                        if lime_exp is not None:
                            # Plot LIME explanation
                            lime_plot = plot_lime_explanation(lime_exp)
                            
                            if lime_plot is not None:
                                st.pyplot(lime_plot, key="lime_explanation_plot")
                                plt.close()
                            
                            # Show feature contributions as a table
                            st.markdown("##### Feature Contributions")
                            lime_values = lime_exp.as_list()
                            lime_df = pd.DataFrame(lime_values, columns=['Feature', 'Contribution'])
                            
                            # Format the dataframe
                            lime_df['Feature'] = lime_df['Feature'].apply(lambda x: x.replace('=', ': '))
                            lime_df['Contribution'] = lime_df['Contribution'].round(2)
                            
                            # Color code based on contribution
                            def color_contribution(val):
                                color = '#4CAF50' if val > 0 else '#F44336'
                                return f'background-color: {color}; color: white; opacity: {min(abs(val)/max(abs(lime_df["Contribution"].max()), 0.1), 1.0)}'
                            
                            # Apply styling
                            styled_lime_df = lime_df.style.applymap(color_contribution, subset=['Contribution'])
                            
                            # Display the table
                            st.dataframe(styled_lime_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating LIME explanation: {e}")
                else:
                    st.info("LIME explanations require the LIME library which is not available in this environment.")
            
            # === Partial Dependence Plots Tab ===
            with xai_tabs[3]:
                if HAS_PDP:
                    st.markdown("#### Partial Dependence Plots (PDP)")
                    st.markdown("These plots show how changes in a feature affect the predicted outcome while keeping all other features constant.")
                    
                    # Select top features for PDP based on importance
                    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                        importance_df = calculate_feature_importance(model, X_sample)
                        top_features = importance_df.head(6)['Feature'].tolist()
                    else:
                        # If no feature importance available, use some common important features
                        top_features = [
                            'HVAC_Efficiency', 'Window_Insulation_U-Value_Capped', 
                            'Wall_Insulation_U-Value_Capped', 'Lighting_Density',
                            'Renewable_Energy_Usage', 'Equipment_Density'
                        ]
                        # Filter to keep only features present in the dataframe
                        top_features = [f for f in top_features if f in X_sample.columns][:6]
                    
                    # Create PDP plots
                    pdp_plots = {}
                    for feature in top_features:
                        pdp_plots[feature] = create_pdp_plot(model, X_sample, feature, preprocessor)
                    
                    # Display PDP plots in columns
                    cols = st.columns(2)
                    
                    for i, (feature, plot) in enumerate(pdp_plots.items()):
                        with cols[i % 2]:
                            # Format feature name for display
                            display_name = feature.replace('_', ' ').title()
                            st.markdown(f"##### {display_name}")
                            st.plotly_chart(plot, use_container_width=True, key=f"pdp_plot_{feature}")
                else:
                    st.info("Partial Dependence Plots require scikit-learn which is not available in this environment.")
            
            # === Sensitivity Analysis Tab ===
            with xai_tabs[4]:
                st.markdown("#### Sensitivity Analysis")
                st.markdown("Test how changes to individual parameters affect the building's energy performance.")
                
                # Select a parameter to analyze
                sensitivity_feature = st.selectbox(
                    "Select Parameter to Test",
                    options=list(st.session_state.params.keys()),
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                # Define test values based on the selected feature
                current_value = st.session_state.params[sensitivity_feature]
                
                # Convert current_value to float if it's not already
                try:
                    current_value = float(current_value)
                except (ValueError, TypeError):
                    # If conversion fails, use a default value
                    current_value = 1.0
                    st.warning(f"Could not perform sensitivity analysis on {sensitivity_feature} - not a numeric value")
                
                if 'Efficiency' in sensitivity_feature or 'Usage' in sensitivity_feature:
                    min_val = max(current_value * 0.5, 0)
                    max_val = min(current_value * 1.5, 1) if 'Efficiency' in sensitivity_feature else current_value * 1.5
                elif 'U-Value' in sensitivity_feature:
                    min_val = max(current_value * 0.5, 0.1)
                    max_val = current_value * 2
                elif 'Temperature' in sensitivity_feature:
                    min_val = current_value - 3
                    max_val = current_value + 3
                else:
                    min_val = max(current_value * 0.5, 0)
                    max_val = current_value * 1.5
                
                # Generate test values
                test_values = np.linspace(min_val, max_val, 10)
                
                # Calculate EUI for each test value
                results = []
                
                for value in test_values:
                    test_params = st.session_state.params.copy()
                    test_params[sensitivity_feature] = value
                    test_eui = predict_eui(test_params, model, preprocessor)
                    results.append({
                        'Value': value,
                        'EUI': test_eui
                    })
                
                # Create DataFrame
                sensitivity_df = pd.DataFrame(results)
                
                # Create sensitivity plot
                fig = go.Figure()
                
                # Add a more visually appealing line plot without dots
                fig.add_trace(go.Scatter(
                    x=sensitivity_df['Value'],
                    y=sensitivity_df['EUI'],
                    mode='lines',
                    line=dict(color='#2e7d32', width=4, shape='spline', smoothing=1.3),
                    fill='tozeroy',
                    fillcolor='rgba(46, 125, 50, 0.1)',
                    name=sensitivity_feature
                ))
                
                # Add point for current value
                fig.add_trace(go.Scatter(
                    x=[current_value],
                    y=[st.session_state.eui],
                    mode='markers',
                    marker=dict(size=12, symbol='star', color='#d32f2f'),
                    name='Current Value'
                ))
                
                # Format feature name for display
                display_name = sensitivity_feature.replace('_', ' ').title()
                
                # Update layout
                fig.update_layout(
                    title=f"Sensitivity Analysis: {display_name}",
                    xaxis_title=display_name,
                    yaxis_title="Energy Use Intensity (kWh/m²/year)",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Show the sensitivity plot
                st.plotly_chart(fig, use_container_width=True, key="sensitivity_plot")
                st.info("Feature importance analysis not available for this model.")
    # ===== Building Portfolio Tab =====
    with tabs[2]:
        st.markdown("### Building Portfolio Analysis")
        
        if st.session_state.building_portfolio.empty:
            st.info("Your building portfolio is empty. Analyze and save buildings to add them here.")
        else:
            # Show the building portfolio
            st.markdown("#### Building Portfolio")
            
            # Format the DataFrame
            display_df = st.session_state.building_portfolio.copy()
            
            # Format columns
            display_df['EUI'] = display_df['EUI'].apply(lambda x: f"{x:.1f}")
            display_df['Annual_Energy'] = display_df['Annual_Energy'].apply(lambda x: f"{x/1000:.1f} MWh")
            display_df['Annual_Cost'] = display_df['Annual_Cost'].apply(lambda x: f"${x:,.2f}")
            display_df['CO2_Emissions'] = display_df['CO2_Emissions'].apply(lambda x: f"{x/1000:.1f} tons")
            
            # Rename columns
            display_df.columns = [
                'Building Name', 'Building Type', 'Area (m²)', 'EUI (kWh/m²/year)',
                'Annual Energy', 'Annual Cost', 'CO₂ Emissions'
            ]
            
            # Show the table
            st.dataframe(format_table(display_df), use_container_width=True)
            
            # Comparative analysis
            if len(st.session_state.building_portfolio) > 1:
                st.markdown("#### Portfolio Comparison")
                
                # Create comparison charts
                portfolio_df = st.session_state.building_portfolio
                
                # EUI Comparison
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=portfolio_df['Building_Name'],
                    y=portfolio_df['EUI'],
                    marker_color=[get_efficiency_rating(eui)[1] for eui in portfolio_df['EUI']],
                    text=[f"{eui:.1f}" for eui in portfolio_df['EUI']],
                    textposition='auto',
                    name='EUI'
                ))
                
                fig.update_layout(
                    title='Energy Use Intensity Comparison',
                    xaxis_title='Building',
                    yaxis_title='EUI (kWh/m²/year)',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True, key="portfolio_eui_comparison")
                
                # Create comparison by building type
                st.markdown("#### Comparison by Building Type")
                
                # Group by building type
                type_grouped = portfolio_df.groupby('Building_Type').agg({
                    'EUI': 'mean',
                    'Annual_Energy': 'sum',
                    'Annual_Cost': 'sum',
                    'CO2_Emissions': 'sum'
                }).reset_index()
                
                # Create building type comparison chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=type_grouped['Building_Type'],
                    y=type_grouped['EUI'],
                    marker_color='#2e7d32',
                    text=[f"{eui:.1f}" for eui in type_grouped['EUI']],
                    textposition='auto',
                    name='Average EUI'
                ))
                
                fig.update_layout(
                    title='Average EUI by Building Type',
                    xaxis_title='Building Type',
                    yaxis_title='EUI (kWh/m²/year)',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True, key="building_type_comparison")
                
                # Total emissions by building type
                fig = go.Figure()
                
                fig.add_trace(go.Pie(
                    labels=type_grouped['Building_Type'],
                    values=type_grouped['CO2_Emissions'],
                    marker=dict(colors=px.colors.sequential.Greens),
                    textinfo='label+percent',
                    hovertemplate='%{label}: %{value:,.0f} kg CO₂<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Total CO₂ Emissions by Building Type',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True, key="emissions_by_type")
            
            # Portfolio management options
            st.markdown("#### Portfolio Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Option to delete a building
                if not st.session_state.building_portfolio.empty:
                    building_to_delete = st.selectbox(
                        "Select Building to Remove",
                        options=st.session_state.building_portfolio['Building_Name'].tolist()
                    )
                    
                    delete_button = st.button("Remove Building from Portfolio")
                    
                    if delete_button:
                        # Remove building from portfolio
                        st.session_state.building_portfolio = st.session_state.building_portfolio[
                            st.session_state.building_portfolio['Building_Name'] != building_to_delete
                        ]
                        st.success(f"Building '{building_to_delete}' removed from portfolio!")
            
            with col2:
                # Option to clear the entire portfolio
                clear_button = st.button("Clear Entire Portfolio")
                
                if clear_button:
                    # Clear the portfolio
                    st.session_state.building_portfolio = pd.DataFrame(columns=[
                        'Building_Name', 'Building_Type', 'Area_m2', 'EUI', 'Annual_Energy', 'Annual_Cost', 'CO2_Emissions'
                    ])
                    st.success("Building portfolio cleared!")
    
    # Add footer
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem;'>
        <h4 style='color: #2e7d32; margin-top: 0;'>Building Energy Analysis Dashboard</h4>
        <p>This dashboard provides insights into building energy performance, costs, and carbon emissions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
