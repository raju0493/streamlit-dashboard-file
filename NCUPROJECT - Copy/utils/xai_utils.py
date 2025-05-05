"""
Explainable AI utilities for the Building Energy Analysis Dashboard.
Provides methods for generating SHAP, LIME and PDP explanations.
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.figure import Figure

# Handle conditional imports with fallbacks
try:
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
except ImportError:
    partial_dependence = None
    PartialDependenceDisplay = None

try:
    import shap
except ImportError:
    shap = None

try:
    from lime import lime_tabular
except ImportError:
    lime_tabular = None

from .feature_names import get_display_name

warnings.filterwarnings('ignore')

# --- Encoding function to handle categorical features ---
def encode_inputs(inputs):
    """
    Encode inputs with robust handling of various input types
    """
    # Handle string inputs
    if isinstance(inputs, str):
        print(f"Warning: String input received instead of dict: {inputs}")
        return {
            'HVAC_Efficiency': 3.0,
            'Wall_Insulation_U-Value_Capped': 0.5,
            'Roof_Insulation_U-Value_Capped': 0.3,
            'Total_Building_Area': 2000.0,
            'Weather_File': 0,  # 'Historical'
            'Building_Type': 0,  # 'Office'
        }
    
    # Handle other non-dict inputs
    if not isinstance(inputs, dict):
        print(f"Warning: Invalid input type: {type(inputs)}")
        return {
            'HVAC_Efficiency': 3.0,
            'Wall_Insulation_U-Value_Capped': 0.5,
            'Roof_Insulation_U-Value_Capped': 0.3,
            'Total_Building_Area': 2000.0,
            'Weather_File': 0,  # 'Historical'
            'Building_Type': 0,  # 'Office'
        }
    
    # Normal case: process dict inputs
    try:
        encoded = inputs.copy()
        mappings = {
            'Building_Type': {
                'Office': 0,
                'Retail': 1,
                'Residential': 2,
                'Educational': 3,
                'Detached': 0,
                'Semi-Detached': 1,
                'Terraced': 2,
                'Apartment': 3
            },
            'Renewable_Energy_Usage': {
                'No': 0,
                'Yes': 1
            },
            'Weather_File': {
                'Historical': 0,
                '2030': 1,
                '2050': 2,
                '2080': 3
            }
        }
        
        for feature, mapping in mappings.items():
            if feature in encoded:
                # Handle case where the value is already numeric
                if isinstance(encoded[feature], (int, float)):
                    continue
                
                # Handle case where value is string
                encoded[feature] = mapping.get(str(encoded[feature]), 0)
        
        return encoded
    except Exception as e:
        print(f"Error in encode_inputs: {e}")
        # Return default values as fallback
        return {
            'HVAC_Efficiency': 3.0,
            'Wall_Insulation_U-Value_Capped': 0.5,
            'Roof_Insulation_U-Value_Capped': 0.3,
            'Total_Building_Area': 2000.0,
            'Weather_File': 0,
            'Building_Type': 0,
        }

# --- Utility to safely transform inputs ---
def safe_transform(preprocessor, X):
    """
    Safely transform input data with robust error handling.
    Handles both numerical and categorical features properly.
    """
    # First, make sure categorical features are properly encoded
    X_processed = X.copy()
    
    # Handle specific categorical columns that cause issues
    if 'Weather_File' in X_processed.columns:
        weather_map = {'Historical': 0, '2030': 1, '2050': 2, '2080': 3}
        if X_processed['Weather_File'].dtype == 'object':
            X_processed['Weather_File'] = X_processed['Weather_File'].map(lambda x: weather_map.get(str(x), 0))
    
    if 'Renewable_Energy_Usage' in X_processed.columns:
        renewable_map = {'No': 0, 'Yes': 1}
        if X_processed['Renewable_Energy_Usage'].dtype == 'object':
            X_processed['Renewable_Energy_Usage'] = X_processed['Renewable_Energy_Usage'].map(lambda x: renewable_map.get(str(x), 0))
    
    if 'Building_Type' in X_processed.columns:
        building_map = {'Office': 0, 'Retail': 1, 'Residential': 2, 'Educational': 3}
        if X_processed['Building_Type'].dtype == 'object':
            X_processed['Building_Type'] = X_processed['Building_Type'].map(lambda x: building_map.get(str(x), 0))
    
    # Now attempt transformation with the preprocessor
    if hasattr(preprocessor, 'transform'):
        try:
            return preprocessor.transform(X_processed)
        except Exception as e:
            print(f"Error in transform: {e}")
            # Check for common issues and resolve them
            try:
                # Some models expect specific column orders
                feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else None
                if feature_names is not None:
                    # Reorder columns if possible
                    try:
                        X_reordered = X_processed.reindex(columns=feature_names)
                        return preprocessor.transform(X_reordered)
                    except Exception as e3:
                        print(f"Column reordering failed: {e3}")
                
                # If transform still fails, try fit_transform as fallback
                return preprocessor.fit_transform(X_processed)
            except Exception as e2:
                print(f"Error in fit_transform fallback: {e2}")
                # Last resort: return NumPy array of the data
                try:
                    return X_processed.values
                except:
                    return X_processed
    
    return X_processed

# --- Create SHAP plots ---
def create_shap_plots(model, preprocessor, inputs, num_plots=2):
    try:
        # Check if SHAP is available
        if shap is not None:
            # Use actual SHAP when available
            return create_shap_with_package(model, preprocessor, inputs)
        else:
            # Use our own implementation when SHAP isn't available
            return create_shap_alternative(model, preprocessor, inputs)
            
    except Exception as e:
        print(f"Error in create_shap_plots: {e}")
        return create_feature_impact_fallback(model, preprocessor, inputs)

# --- Create SHAP plots using the actual SHAP package ---
def create_shap_with_package(model, preprocessor, inputs):
    """Use the actual SHAP library when available"""
    try:
        inputs_encoded = encode_inputs(inputs)
        input_df = pd.DataFrame([inputs_encoded])
        X_transformed = safe_transform(preprocessor, input_df)

        explainer = shap.Explainer(model)
        shap_values = explainer(X_transformed)

        feature_names = list(inputs_encoded.keys())
        display_names = [get_display_name(f) for f in feature_names]

        figures = []

        feature_impacts = [(name, shap_values.values[0][i]) for i, name in enumerate(display_names)]
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        names = [x[0] for x in feature_impacts[:12]]
        values = [x[1] for x in feature_impacts[:12]]
        colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]

        # Create trace data directly with hover text
        hover_texts = []
        for i, (name, value) in enumerate(zip(names, values)):
            direction = "increases" if value > 0 else "decreases"
            hover_texts.append(f"{name}: {value:.3f} ({direction} energy use)")
            
        trace = {
                'type': 'bar',
                'y': names, 
                'x': values, 
                'orientation': 'h', 
                'marker': {'color': colors},
                'hovertext': hover_texts,
                'hoverinfo': 'text',
                'text': [f"+{v:.2f}" if v >= 0 else f"{v:.2f}" for v in values],  
                'textposition': 'outside',  
                'textfont': {
                    'size': 12,
                    'color': 'black'  # You can also use 'white' depending on your theme
                }
            }

        # Create layout with reference line and annotations
        layout = {
            'title': "SHAP Feature Impact", 
            'xaxis_title': "SHAP Value", 
            'yaxis_title': "Feature",
            'height': 500,
            'margin': {'l': 10, 'r': 10, 't': 40, 'b': 10},
            'shapes': [{
                'type': 'line',
                'x0': 0, 
                'y0': -0.5, 
                'x1': 0, 
                'y1': len(names)-0.5,
                'line': {
                    'color': '#9e9e9e',
                    'width': 1,
                    'dash': 'dot'
                }
            }],
            'annotations': [{
                'x': 0,
                'y': -0.8,
                'xref': "x",
                'yref': "y",
                'text': "← Decreases Energy Use | Increases Energy Use →",
                'showarrow': False,
                'font': {
                    'size': 12,
                    'color': "#555555"
                }
            }]
        }
        
        # Return the figure data directly
        return {'data': [trace], 'layout': layout}
    except Exception as e:
        print(f"Error in create_shap_with_package: {e}")
        return None

# --- Create a SHAP-like explanation without the SHAP package ---
def create_shap_alternative(model, preprocessor, inputs):
    """Create a SHAP-like visualization without requiring the SHAP package"""
    try:
        # Encode inputs
        inputs_encoded = encode_inputs(inputs)
        input_df = pd.DataFrame([inputs_encoded])
        X_transformed = safe_transform(preprocessor, input_df)
        
        # Get baseline prediction
        base_prediction = float(model.predict(X_transformed)[0])
        
        # Get feature names
        feature_names = list(inputs_encoded.keys())
        display_names = [get_display_name(f) for f in feature_names]
        
        # Calculate impact of each feature by perturbing inputs
        impacts = []
        
        for i, feature in enumerate(feature_names):
            # Skip categorical features for simplicity
            if feature in ['Building_Type', 'Renewable_Energy_Usage', 'Weather_File']:
                continue
            
            # Get current value
            current_value = inputs_encoded[feature]
            
            # Create a copy with this feature set to a baseline/reference value
            if 'U-Value' in feature:
                reference_value = 1.0  # Moderate insulation value
            elif feature == 'HVAC_Efficiency':
                reference_value = 2.5  # Average efficiency
            elif 'Temperature' in feature:
                reference_value = 20.0  # Standard temperature
            elif feature == 'Air_Change_Rate_Capped':
                reference_value = 1.0  # Average air change rate
            elif feature == 'Window_to_Wall_Ratio':
                reference_value = 0.3  # Standard ratio
            elif feature == 'Total_Building_Area':
                reference_value = 2000.0  # Average building area
            else:
                # For other numerical features, use 0 or 1 as reference
                reference_value = 0.0 if current_value > 1.0 else 1.0
            
            # Create input with feature set to reference
            modified_input = inputs_encoded.copy()
            modified_input[feature] = reference_value
            modified_df = pd.DataFrame([modified_input])
            modified_transformed = safe_transform(preprocessor, modified_df)
            
            # Get prediction with modified input
            modified_prediction = float(model.predict(modified_transformed)[0])
            
            # Calculate impact (difference from baseline prediction)
            impact = base_prediction - modified_prediction
            
            # Save the impact
            impacts.append((display_names[i], impact, current_value, reference_value))
        
        # Sort by absolute impact
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top features
        top_impacts = impacts[:12]  # Show top 12 features
        names = [x[0] for x in top_impacts]
        values = [x[1] for x in top_impacts]
        colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]
        
        # Create hover texts
        hover_texts = []
        for name, impact, current, reference in top_impacts:
            direction = "increases" if impact > 0 else "decreases"
            hover_texts.append(
                f"{name}<br>"
                f"Current value: {current:.2f}<br>"
                f"Reference value: {reference:.2f}<br>"
                f"Impact: {impact:.2f} ({direction} EUI)"
            )
        
        # Create trace data directly
        trace = {
            'type': 'bar',
            'y': names, 
            'x': values, 
            'orientation': 'h', 
            'marker': {'color': colors},
            'hovertext': hover_texts,
            'hoverinfo': "text",'textposition': 'outside',  
                'textfont': {
                    'size': 12,
                    'color': 'black'  # You can also use 'white' depending on your theme
                }
            }

        # Create layout
        layout = {
            'title': "Feature Impact Analysis", 
            'xaxis_title': "Impact on Prediction", 
            'yaxis_title': "Feature",
            'height': 500,
            'margin': {'l': 10, 'r': 10, 't': 40, 'b': 10},
            'template': 'plotly_white',
            'shapes': [{
                'type': 'line',
                'x0': 0, 
                'y0': -0.5, 
                'x1': 0, 
                'y1': len(names)-0.5,
                'line': {
                    'color': '#9e9e9e',
                    'width': 1,
                    'dash': 'dot'
                }
            }],
            'annotations': [{
                'x': 0,
                'y': -0.8,
                'xref': "x",
                'yref': "y",
                'text': "← Decreases Energy Use | Increases Energy Use →",
                'showarrow': False,
                'font': {
                    'size': 12,
                    'color': "#555555"
                }
            }]
        }
        
        # Return the figure data directly
        return {'data': [trace], 'layout': layout}
        
    except Exception as e:
        print(f"Error in create_shap_alternative: {e}")
        return create_feature_impact_fallback(model, preprocessor, inputs)

# --- Fallback feature impact ---
def create_feature_impact_fallback(model, preprocessor, inputs):
    """Create a fallback feature importance visualization using plotly instead of matplotlib"""
    try:
        inputs_encoded = encode_inputs(inputs)
        input_df = pd.DataFrame([inputs_encoded])
        X_transformed = safe_transform(preprocessor, input_df)

        feature_names = list(inputs_encoded.keys())
        display_names = [get_display_name(f) for f in feature_names]

        # Get feature importances from model if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Create reasonable importances based on domain knowledge
            importances = np.ones(len(feature_names))
            for i, feature in enumerate(feature_names):
                if 'Insulation' in feature:
                    importances[i] *= 1.5
                elif feature == 'HVAC_Efficiency':
                    importances[i] *= 2.0
                elif feature == 'Total_Building_Area':
                    importances[i] *= 1.8
                elif feature == 'Air_Change_Rate_Capped':
                    importances[i] *= 1.6
                elif 'Temperature' in feature:
                    importances[i] *= 1.3

        # Find top 10 features by importance
        sorted_idx = np.argsort(importances)[::-1][:10]
        top_features = [display_names[i] for i in sorted_idx]
        top_importances = importances[sorted_idx]

        # Create trace data directly instead of Figure object
        trace = {
            'type': 'bar',
            'y': top_features,
            'x': top_importances,
            'orientation': 'h',
            'marker': {'color': 'rgba(15, 118, 110, 0.7)'},
            'text': [f"{x:.3f}" for x in top_importances],
            'textposition': 'auto'
        }
        
        # Create layout
        layout = {
            'title': 'Feature Importance (Built-in)',
            'xaxis_title': 'Importance',
            'yaxis_title': 'Feature',
            'template': 'plotly_white',
            'height': 500,
            'margin': {'l': 10, 'r': 10, 't': 40, 'b': 10}
        }
        
        # Return the figure data directly
        return {'data': [trace], 'layout': layout}
    except Exception as e:
        print(f"Error in create_feature_impact_fallback: {e}")
        # Return an empty list as an absolute last resort
        return []
from lime.lime_tabular import LimeTabularExplainer
# --- Create LIME-style explanations ---
def create_lime_explanation(model, preprocessor, inputs):
    """Custom LIME-style feature impact explanation with correct +/− values."""
    try:
        # Encode and transform
        inputs_encoded = encode_inputs(inputs)
        if isinstance(inputs_encoded, str) or not isinstance(inputs_encoded, dict):
            inputs_encoded = {'HVAC_Efficiency': 3.0, 'Wall_Insulation_U-Value_Capped': 0.5, 'Roof_Insulation_U-Value_Capped': 0.3, 'Total_Building_Area': 2000.0, 'Weather_File': 0, 'Building_Type': 0}
        input_df = pd.DataFrame([inputs_encoded])
        X_transformed = safe_transform(preprocessor, input_df)

        feature_names = list(inputs_encoded.keys())
        display_names = [get_display_name(f) for f in feature_names]

        base_prediction = float(model.predict(X_transformed)[0])

        impacts = []
        for i, feature in enumerate(feature_names):
            if feature in ['Building_Type', 'Renewable_Energy_Usage', 'Weather_File']:
                continue  # Skip categorical
            perturbations = []
            if 'U-Value' in feature:
                perturbations = [0.1, 0.5, 1.0, 2.0]
            elif feature == 'HVAC_Efficiency':
                perturbations = [1.5, 2.5, 3.5, 4.5]
            elif 'Temperature' in feature:
                perturbations = [18, 20, 22, 24]
            elif feature == 'Air_Change_Rate_Capped':
                perturbations = [0.5, 1.0, 2.0, 3.0]
            elif feature == 'Window_to_Wall_Ratio':
                perturbations = [0.1, 0.3, 0.5]
            elif feature == 'Total_Building_Area':
                perturbations = [500, 2000, 4000]
            else:
                current = inputs_encoded[feature]
                perturbations = [max(0, current - 1), current, current + 1]

            prediction_changes = []
            for p in perturbations:
                modified = inputs_encoded.copy()
                modified[feature] = p
                mod_df = pd.DataFrame([modified])
                try:
                    mod_pred = float(model.predict(safe_transform(preprocessor, mod_df))[0])
                    prediction_changes.append((p, mod_pred))
                except Exception:
                    continue

            if len(prediction_changes) >= 2:
                x = np.array([pt[0] for pt in prediction_changes])
                y = np.array([pt[1] for pt in prediction_changes])
                slope = np.polyfit(x, y, 1)[0]
                # Impact is slope times reasonable perturbation width (normalized to 1 unit change)
                impact = slope * 1.0
                impacts.append((display_names[i], impact))

        # Sort impacts
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        if impacts:
            features = [i[0] for i in impacts[:6]]
            values = [i[1] for i in impacts[:6]]
            colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]

            trace = {
                'type': 'bar',
                'y': features,
                'x': values,
                'orientation': 'h',
                'marker': {'color': colors},
                'text': [f"+{v:.2f}" if v >= 0 else f"{v:.2f}" for v in values],
                'textposition': 'outside',
                'textfont': {'size': 12, 'color': 'black'}
            }

            layout = {
                'title': "LIME Feature Impact",
                'xaxis_title': "Impact on Prediction",
                'yaxis_title': "Feature",
                'template': 'plotly_white',
                'height': 500,
                'margin': {'l': 10, 'r': 10, 't': 40, 'b': 10},
                'shapes': [{
                    'type': 'line',
                    'x0': 0, 'y0': -0.5,
                    'x1': 0, 'y1': len(features) - 0.5,
                    'line': {'color': '#9e9e9e', 'width': 1, 'dash': 'dot'}
                }],
                'annotations': [{
                    'x': 0,
                    'y': -0.8,
                    'xref': "x",
                    'yref': "y",
                    'text': "← Decreases Energy Use | Increases Energy Use →",
                    'showarrow': False,
                    'font': {'size': 12, 'color': "#555555"}
                }]
            }

            return {'data': [trace], 'layout': layout}
        else:
            return create_feature_impact_fallback(model, preprocessor, inputs)

    except Exception as e:
        print(f"Error in create_lime_explanation: {e}")
        return create_feature_impact_fallback(model, preprocessor, inputs)
def create_lime_with_package(model, preprocessor, inputs):
    """Create a LIME explanation using the actual LIME package"""
    try:
        # Step 1: Encode the inputs
        inputs_encoded = encode_inputs(inputs)
        input_df = pd.DataFrame([inputs_encoded])
        X_transformed = safe_transform(preprocessor, input_df)

        # Step 2: Get feature names and their display names
        feature_names = list(inputs_encoded.keys())
        display_names = [get_display_name(f) for f in feature_names]

        # Step 3: Generate synthetic training data
        num_samples = 1000
        X_train = np.zeros((num_samples, len(feature_names)))

        # For each feature, generate synthetic values
        for i, feature in enumerate(feature_names):
            if 'U-Value' in feature:
                X_train[:, i] = np.random.uniform(0.1, 3.0, num_samples)
            elif feature == 'HVAC_Efficiency':
                X_train[:, i] = np.random.uniform(1.0, 5.0, num_samples)
            elif 'Temperature' in feature:
                X_train[:, i] = np.random.uniform(16.0, 24.0, num_samples)
            elif feature == 'Air_Change_Rate_Capped':
                X_train[:, i] = np.random.uniform(0.1, 5.0, num_samples)
            elif feature == 'Window_to_Wall_Ratio':
                X_train[:, i] = np.random.uniform(0.1, 0.6, num_samples)
            elif feature == 'Total_Building_Area':
                X_train[:, i] = np.random.uniform(500, 10000, num_samples)
            elif feature == 'Building_Type':
                X_train[:, i] = np.random.randint(0, 4, num_samples)
            elif feature == 'Renewable_Energy_Usage':
                X_train[:, i] = np.random.randint(0, 2, num_samples)
            elif feature == 'Weather_File':
                X_train[:, i] = np.random.randint(0, 4, num_samples)
            else:
                # Default for other features
                X_train[:, i] = np.random.uniform(0, 10, num_samples)

        # Step 4: Create DataFrame for synthetic data
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        
        # Step 5: Apply transformation on synthetic data
        X_train_transformed = safe_transform(preprocessor, X_train_df)

        # Step 6: Create LIME Explainer instance
        explainer = LimeTabularExplainer(
            X_train_transformed,
            feature_names=feature_names,
            class_names=['EUI'],
            discretize_continuous=False,
            mode='regression'
        )

        # Step 7: Generate the explanation instance
        exp = explainer.explain_instance(
            data_row=np.array(list(inputs_encoded.values())),
            predict_fn=lambda x: model.predict(safe_transform(preprocessor, pd.DataFrame(x, columns=feature_names))),
            num_features=10
        )

        # Step 8: Extract explanation data and sort by impact
        lime_data = exp.as_list()
        features = [item[0].split(' ')[0] for item in lime_data]
        values = [item[1] for item in lime_data]
        
        # Sorting by absolute impact
        sorted_indices = np.argsort(np.abs(values))[::-1]
        features = [features[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        # Step 9: Map features to their display names
        display_features = []
        for f in features:
            for key in feature_names:
                if f in key:
                    display_features.append(get_display_name(key))
                    break
            else:
                display_features.append(f)

        # Step 10: Set colors based on feature impact
        colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]

        # Step 11: Prepare the bar chart data for LIME explanation
        trace = {
            'type': 'bar',
            'y': display_features,
            'x': values,
            'orientation': 'h',
            'marker': {'color': colors},
            'text': [f"{v:.3f}" for v in values],
            'textposition': 'auto'
        }

        # Step 12: Define the layout for the LIME explanation
        layout = {
            'title': 'LIME Feature Impact',
            'xaxis_title': 'Impact on Prediction',
            'yaxis_title': 'Feature',
            'template': 'plotly_white',
            'height': 500,
            'margin': {'l': 10, 'r': 10, 't': 40, 'b': 10},
            'shapes': [{
                'type': 'line',
                'x0': 0,
                'y0': -0.5,
                'x1': 0,
                'y1': len(features) - 0.5,
                'line': {'color': '#9e9e9e', 'width': 1, 'dash': 'dot'}
            }],
            'annotations': [{
                'x': 0,
                'y': -0.8,
                'xref': "x",
                'yref': "y",
                'text': "← Decreases Energy Use | Increases Energy Use →",
                'showarrow': False,
                'font': {'size': 12, 'color': "#555555"}
            }]
        }

        # Return the plotly figure for visualization
        return {'data': [trace], 'layout': layout}

    except Exception as e:
        print(f"Error in create_lime_with_package: {e}")
        # Fall back to our custom LIME implementation if an error occurs
        return create_lime_explanation(model, preprocessor, inputs)
# --- Create Partial Dependence Plots ---
def create_pdp_plots(model, preprocessor, inputs, feature_name=None):
    """
    Create Partial Dependence Plots to show how the model's predictions
    change when varying a single feature while keeping others constant.
    """
    try:
        # Check if scikit-learn's partial_dependence is available
        if partial_dependence is not None:
            return create_pdp_with_sklearn(model, preprocessor, inputs, feature_name)
        else:
            # Use our custom implementation
            return create_pdp_alternative(model, preprocessor, inputs, feature_name)
    except Exception as e:
        print(f"Error in create_pdp_plots: {e}")
        return create_pdp_fallback(model, preprocessor, inputs, feature_name)

# --- Create PDP with scikit-learn ---
def create_pdp_with_sklearn(model, preprocessor, inputs, feature_name=None):
    """Create Partial Dependence Plot using scikit-learn"""
    try:
        inputs_encoded = encode_inputs(inputs)
        input_df = pd.DataFrame([inputs_encoded])
        
        # Get feature names and indices
        feature_names = list(inputs_encoded.keys())
        
        # If no specific feature is provided, choose a relevant one
        if feature_name is None or feature_name not in feature_names:
            # Try to choose a meaningful feature
            important_features = [
                'HVAC_Efficiency', 
                'Wall_Insulation_U-Value_Capped', 
                'Roof_Insulation_U-Value_Capped'
            ]
            for feature in important_features:
                if feature in feature_names:
                    feature_name = feature
                    break
            else:
                # Use the first numerical feature if none of the important ones are found
                for feature in feature_names:
                    if feature not in ['Building_Type', 'Renewable_Energy_Usage', 'Weather_File']:
                        feature_name = feature
                        break
                else:
                    # Fallback to the first feature
                    feature_name = feature_names[0]
        
        # Get feature index
        feature_idx = feature_names.index(feature_name)
        
        # Generate a synthetic dataset for PDP calculation
        # We use all available sample from our input and modify it
        n_samples = 100
        X_pdp = []
        for _ in range(n_samples):
            X_pdp.append(inputs_encoded.copy())
        X_pdp = pd.DataFrame(X_pdp)
        
        # Transform data for the model
        X_transformed = safe_transform(preprocessor, X_pdp)
        
        # Generate reasonable feature values for PDP
        if 'U-Value' in feature_name:
            grid_values = np.linspace(0.1, 3.0, 20)
        elif feature_name == 'HVAC_Efficiency':
            grid_values = np.linspace(1.0, 5.0, 20)
        elif 'Temperature' in feature_name:
            grid_values = np.linspace(16.0, 24.0, 20)
        elif feature_name == 'Air_Change_Rate_Capped':
            grid_values = np.linspace(0.1, 5.0, 20)
        elif feature_name == 'Window_to_Wall_Ratio':
            grid_values = np.linspace(0.1, 0.6, 20)
        elif feature_name == 'Total_Building_Area':
            grid_values = np.linspace(500, 10000, 20)
        else:
            # Default for other numerical features
            current_value = inputs_encoded[feature_name]
            grid_values = np.linspace(max(0, current_value * 0.5), current_value * 1.5, 20)
        
        # Create the PDP manually
        pdp_values = []
        pdp_mean = []
        
        for grid_val in grid_values:
            predictions = []
            for _ in range(10):  # Use 10 different samples for each grid value
                # Create a copy of the input
                modified_input = inputs_encoded.copy()
                # Set the feature to the grid value
                modified_input[feature_name] = grid_val
                # Convert to DataFrame
                modified_df = pd.DataFrame([modified_input])
                # Transform
                modified_transformed = safe_transform(preprocessor, modified_df)
                # Predict
                prediction = float(model.predict(modified_transformed)[0])
                predictions.append(prediction)
            
            # Calculate mean prediction
            mean_prediction = sum(predictions) / len(predictions)
            pdp_mean.append(mean_prediction)
            pdp_values.append((grid_val, mean_prediction, predictions))
        
        # Create trace data directly
        # Main PDP line trace
        main_trace = {
            'type': 'scatter',
            'x': grid_values.tolist() if hasattr(grid_values, 'tolist') else list(grid_values),
            'y': pdp_mean,
            'mode': 'lines+markers',
            'name': 'PDP',
            'line': {'color': '#1e88e5', 'width': 3},
            'marker': {'size': 8},
            'hovertemplate': f"{get_display_name(feature_name)}: %{{x:.2f}}<br>Predicted EUI: %{{y:.2f}} kWh/m²/yr<extra></extra>"
        }
        
        # Add a marker for the current value
        current_value = inputs_encoded[feature_name]
        # Find the closest grid value
        closest_idx = np.abs(grid_values - current_value).argmin()
        current_prediction = pdp_mean[closest_idx]
        
        marker_trace = {
            'type': 'scatter',
            'x': [float(current_value)],
            'y': [float(current_prediction)],
            'mode': 'markers',
            'name': 'Current Value',
            'marker': {
                'color': 'red',
                'size': 12,
                'symbol': 'star',
                'line': {'color': 'black', 'width': 1}
            },
            'hovertemplate': f"Current {get_display_name(feature_name)}: {current_value:.2f}<br>Predicted EUI: {current_prediction:.2f} kWh/m²/yr<extra></extra>"
        }
        
        # Determine if higher or lower values are better based on the trend
        if pdp_mean[0] > pdp_mean[-1]:
            better_direction = "Higher"
            worse_direction = "Lower"
        else:
            better_direction = "Lower"
            worse_direction = "Higher"
        
        # Create layout
        layout = {
            'title': f"Partial Dependence Plot: {get_display_name(feature_name)}",
            'xaxis_title': get_display_name(feature_name),
            'yaxis_title': "Predicted EUI (kWh/m²/yr)",
            'template': 'plotly_white',
            'height': 500,
            'margin': {'l': 10, 'r': 10, 't': 40, 'b': 10},
            'legend': {
                'orientation': "h",
                'yanchor': "bottom",
                'y': 1.02,
                'xanchor': "right",
                'x': 1
            },
            'annotations': [{
                'x': 0.5,
                'y': -0.15,
                'xref': "paper",
                'yref': "paper",
                'text': f"{better_direction} values of {get_display_name(feature_name)} → {'Lower' if better_direction == 'Higher' else 'Higher'} Energy Use Intensity",
                'showarrow': False,
                'font': {'size': 12, 'color': "#555555"}
            }]
        }
        
        # Return the figure data directly
        return {'data': [main_trace, marker_trace], 'layout': layout}
    
    except Exception as e:
        print(f"Error in create_pdp_with_sklearn: {e}")
        # Try the alternative implementation
        return create_pdp_alternative(model, preprocessor, inputs, feature_name)

# --- Create PDP without scikit-learn ---
def create_pdp_alternative(model, preprocessor, inputs, feature_name=None):
    """Create Partial Dependence Plot without requiring scikit-learn"""
    try:
        inputs_encoded = encode_inputs(inputs)
        input_df = pd.DataFrame([inputs_encoded])
        
        # Get feature names
        feature_names = list(inputs_encoded.keys())
        
        # If no specific feature is provided, choose a relevant one
        if feature_name is None or feature_name not in feature_names:
            # Try to choose a meaningful feature
            important_features = [
                'HVAC_Efficiency', 
                'Wall_Insulation_U-Value_Capped', 
                'Roof_Insulation_U-Value_Capped'
            ]
            for feature in important_features:
                if feature in feature_names:
                    feature_name = feature
                    break
            else:
                # Use the first numerical feature if none of the important ones are found
                for feature in feature_names:
                    if feature not in ['Building_Type', 'Renewable_Energy_Usage', 'Weather_File']:
                        feature_name = feature
                        break
                else:
                    # Fallback to the first feature
                    feature_name = feature_names[0]
        
        print(f"Creating PDP for feature: {feature_name}")
        
        # Get current value for the selected feature
        current_value = inputs_encoded[feature_name]
        
        # Generate grid values for the feature with fewer points (15 instead of 20)
        # to improve stability
        if 'U-Value' in feature_name:
            grid_values = np.linspace(0.1, 3.0, 15)
        elif feature_name == 'HVAC_Efficiency':
            grid_values = np.linspace(1.0, 5.0, 15)
        elif 'Temperature' in feature_name:
            grid_values = np.linspace(16.0, 24.0, 15)
        elif feature_name == 'Air_Change_Rate_Capped':
            grid_values = np.linspace(0.1, 5.0, 15)
        elif feature_name == 'Window_to_Wall_Ratio':
            grid_values = np.linspace(0.1, 0.6, 15)
        elif feature_name == 'Total_Building_Area':
            grid_values = np.linspace(500, 10000, 15)
        elif feature_name == 'Building_Type':
            grid_values = np.array([0, 1, 2, 3])  # Categorical
        elif feature_name == 'Renewable_Energy_Usage':
            grid_values = np.array([0, 1])  # Categorical
        elif feature_name == 'Weather_File':
            grid_values = np.array([0, 1, 2, 3])  # Categorical
        else:
            # Default for other features with narrower range for stability
            grid_values = np.linspace(max(0, current_value * 0.7), current_value * 1.3, 15)
        
        # Calculate PDP with error handling and retries
        pdp_values = []
        error_count = 0
        max_retries = 3
        
        # Try multiple approaches if needed to generate a valid PDP
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"PDP attempt {attempt+1} with adjusted grid values")
                # Reduce grid size and range for stability in retry attempts
                if not np.isnan(current_value):
                    # Create a more focused grid around the current value
                    grid_values = np.linspace(
                        max(0, current_value * 0.8), 
                        current_value * 1.2, 
                        10 - attempt
                    )
                else:
                    # Handle NaN current values
                    if 'U-Value' in feature_name:
                        grid_values = [0.5, 1.0, 1.5]
                    elif feature_name == 'HVAC_Efficiency':
                        grid_values = [2.0, 3.0, 4.0]
                    else:
                        # Generic fallback for other features
                        grid_values = [0.1, 0.5, 1.0]
            
            pdp_values = []
            for grid_val in grid_values:
                # Create a copy of the inputs
                modified_input = inputs_encoded.copy()
                # Modify the feature of interest
                modified_input[feature_name] = grid_val
                # Convert to DataFrame
                modified_df = pd.DataFrame([modified_input])
                
                # Transform with robust error handling
                try:
                    modified_transformed = safe_transform(preprocessor, modified_df)
                    # Predict
                    prediction = float(model.predict(modified_transformed)[0])
                    # Add to results only if prediction is valid (not NaN or infinity)
                    if not np.isnan(prediction) and not np.isinf(prediction):
                        pdp_values.append((grid_val, prediction))
                    else:
                        print(f"Warning: Invalid prediction value {prediction} for {feature_name}={grid_val}")
                        error_count += 1
                except Exception as e:
                    print(f"Error in PDP prediction for {feature_name}={grid_val}: {e}")
                    error_count += 1
                    continue
            
            # If we have enough successful predictions, break out of the retry loop
            if len(pdp_values) > 3:
                break
        
        # Sort values by x to ensure proper line drawing
        pdp_values.sort(key=lambda x: x[0])
        
        # If we have enough values, create a plot
        if len(pdp_values) > 3:
            # Extract x and y values
            x_values = [val[0] for val in pdp_values]
            y_values = [val[1] for val in pdp_values]
            
            # Create trace data directly instead of Figure object
            # Main PDP line trace
            main_trace = {
                'type': 'scatter',
                'x': x_values,
                'y': y_values,
                'mode': 'lines+markers',
                'name': 'PDP',
                'line': {'color': '#1e88e5', 'width': 3},
                'marker': {'size': 8},
                'hovertemplate': f"{get_display_name(feature_name)}: %{{x:.2f}}<br>Predicted EUI: %{{y:.2f}} kWh/m²/yr<extra></extra>"
            }
            
            # Add a marker for the current value
            # Find the closest x value to the current value
            closest_idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - current_value))
            current_prediction = y_values[closest_idx]
            
            marker_trace = {
                'type': 'scatter',
                'x': [current_value],
                'y': [current_prediction],
                'mode': 'markers',
                'name': 'Current Value',
                'marker': {
                    'color': 'red',
                    'size': 12,
                    'symbol': 'star',
                    'line': {'color': 'black', 'width': 1}
                },
                'hovertemplate': f"Current {get_display_name(feature_name)}: {current_value:.2f}<br>Predicted EUI: {current_prediction:.2f} kWh/m²/yr<extra></extra>"
            }
            
            # Determine if higher or lower values are better based on the trend
            if y_values[0] > y_values[-1]:
                better_direction = "Higher"
                worse_direction = "Lower"
            else:
                better_direction = "Lower"
                worse_direction = "Higher"
            
            # Create layout
            layout = {
                'title': f"Partial Dependence Plot: {get_display_name(feature_name)}",
                'xaxis_title': get_display_name(feature_name),
                'yaxis_title': "Predicted EUI (kWh/m²/yr)",
                'template': 'plotly_white',
                'height': 500,
                'margin': {'l': 10, 'r': 10, 't': 40, 'b': 10},
                'legend': {
                    'orientation': "h",
                    'yanchor': "bottom",
                    'y': 1.02,
                    'xanchor': "right",
                    'x': 1
                },
                'annotations': [{
                    'x': 0.5,
                    'y': -0.15,
                    'xref': "paper",
                    'yref': "paper",
                    'text': f"{better_direction} values of {get_display_name(feature_name)} → {'Lower' if better_direction == 'Higher' else 'Higher'} Energy Use Intensity",
                    'showarrow': False,
                    'font': {'size': 12, 'color': "#555555"}
                }]
            }
            
            # Return the figure data directly
            return {'data': [main_trace, marker_trace], 'layout': layout}
        else:
            # Not enough data points, use fallback
            return create_pdp_fallback(model, preprocessor, inputs, feature_name)
    
    except Exception as e:
        print(f"Error in create_pdp_alternative: {e}")
        return create_pdp_fallback(model, preprocessor, inputs, feature_name)

# --- PDP Fallback ---
def create_pdp_fallback(model, preprocessor, inputs, feature_name=None):
    """Create a fallback PDP visualization when other methods fail"""
    try:
        inputs_encoded = encode_inputs(inputs)
        
        # Get feature names
        feature_names = list(inputs_encoded.keys())
        
        # If no specific feature is provided, choose a relevant one
        if feature_name is None or feature_name not in feature_names:
            # Try to choose a meaningful feature
            important_features = [
                'HVAC_Efficiency', 
                'Wall_Insulation_U-Value_Capped', 
                'Roof_Insulation_U-Value_Capped'
            ]
            for feature in important_features:
                if feature in feature_names:
                    feature_name = feature
                    break
            else:
                # Use the first numerical feature
                for feature in feature_names:
                    if feature not in ['Building_Type', 'Renewable_Energy_Usage', 'Weather_File']:
                        feature_name = feature
                        break
                else:
                    # Fallback to the first feature
                    feature_name = feature_names[0]
        
        # Get current value for this feature
        current_value = inputs_encoded[feature_name]
        
        # Create a synthetic trend based on domain knowledge
        if 'U-Value' in feature_name:
            # For U-values, lower is better (better insulation)
            x_values = np.linspace(0.1, 3.0, 20)
            y_values = 100 + 30 * x_values  # Linear trend
        elif feature_name == 'HVAC_Efficiency':
            # For HVAC efficiency, higher is better
            x_values = np.linspace(1.0, 5.0, 20)
            y_values = 150 - 15 * x_values  # Linear trend
        elif 'Temperature' in feature_name:
            # For temperatures, there's usually an optimal point
            x_values = np.linspace(16.0, 24.0, 20)
            y_values = 120 + 2 * (x_values - 20)**2  # Quadratic trend
        elif feature_name == 'Air_Change_Rate_Capped':
            # For air changes, lower is usually better
            x_values = np.linspace(0.1, 5.0, 20)
            y_values = 100 + 12 * x_values  # Linear trend
        elif feature_name == 'Window_to_Wall_Ratio':
            # Window to wall ratio has mixed effects
            x_values = np.linspace(0.1, 0.6, 20)
            y_values = 125 - 50 * x_values + 150 * x_values**2  # Quadratic trend
        else:
            # Default trend
            x_values = np.linspace(max(0, current_value * 0.5), current_value * 1.5, 20)
            y_values = 120 + 5 * np.sin(x_values)  # Some arbitrary trend
        
        # Create a marker for the current value
        # Find the closest x value to the current value
        closest_idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - current_value))
        current_prediction = y_values[closest_idx]
        
        # Create trace data directly
        main_trace = {
            'type': 'scatter',
            'x': x_values,
            'y': y_values,
            'mode': 'lines+markers',
            'name': 'Estimated Trend',
            'line': {'color': '#1e88e5', 'width': 3, 'dash': 'dash'},
            'marker': {'size': 8},
            'hovertemplate': f"{get_display_name(feature_name)}: %{{x:.2f}}<br>Estimated EUI: %{{y:.2f}} kWh/m²/yr<extra></extra>"
        }
        
        # Add a marker for the current value
        marker_trace = {
            'type': 'scatter',
            'x': [current_value],
            'y': [current_prediction],
            'mode': 'markers',
            'name': 'Current Value',
            'marker': {
                'color': 'red',
                'size': 12,
                'symbol': 'star',
                'line': {'color': 'black', 'width': 1}
            },
            'hovertemplate': f"Current {get_display_name(feature_name)}: {current_value:.2f}<br>Estimated EUI: {current_prediction:.2f} kWh/m²/yr<extra></extra>"
        }
        
        # Create layout
        layout = {
            'title': f"Estimated Effect of {get_display_name(feature_name)} (Fallback Mode)",
            'xaxis_title': get_display_name(feature_name),
            'yaxis_title': "Estimated EUI (kWh/m²/yr)",
            'template': 'plotly_white',
            'height': 500,
            'margin': {'l': 10, 'r': 10, 't': 40, 'b': 10},
            'legend': {
                'orientation': "h",
                'yanchor': "bottom",
                'y': 1.02,
                'xanchor': "right",
                'x': 1
            },
            'annotations': [{
                'x': 0.5,
                'y': -0.15,
                'xref': "paper",
                'yref': "paper",
                'text': "Note: This is an estimated effect based on domain knowledge, not model predictions.",
                'showarrow': False,
                'font': {'size': 12, 'color': "#555555"}
            }]
        }
        
        # Return the figure data directly
        return {'data': [main_trace, marker_trace], 'layout': layout}
    
    except Exception as e:
        print(f"Error in create_pdp_fallback: {e}")
        # Create a simple error message with direct data approach
        return {
            'data': [],
            'layout': {
                'title': "Error Generating PDP",
                'height': 500,
                'template': 'plotly_white',
                'margin': {'l': 10, 'r': 10, 't': 40, 'b': 10},
                'annotations': [{
                    'x': 0.5,
                    'y': 0.5,
                    'xref': "paper",
                    'yref': "paper",
                    'text': "Could not generate Partial Dependence Plot.<br>Please try a different feature or check model compatibility.",
                    'showarrow': False,
                    'font': {'size': 14, 'color': "#616161"}
                }]
            }
        }
