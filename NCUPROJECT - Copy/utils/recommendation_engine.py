"""
Recommendation engine for the Building Energy Analysis Dashboard.
Provides functions to generate actionable energy-saving recommendations
based on the building characteristics and the predicted EUI.
"""

import pandas as pd
import numpy as np
from .feature_names import get_display_name

def get_recommendations(inputs, baseline_eui, model, preprocessor):
    """
    Generate tailored energy-saving recommendations based on building inputs.
    
    Args:
        inputs (dict): Dictionary of building inputs
        baseline_eui (float): Predicted Energy Use Intensity
        model (object): Prediction model for simulating improvements
        preprocessor (object): Data preprocessor
        
    Returns:
        dict: Categorized recommendations with details
    """
    recommendations = {
        'quick_wins': [],
        'medium_term': [],
        'major_upgrades': []
    }
    
    # Copy inputs for modification
    area = inputs.get('Total_Building_Area', 1000)
    electricity_cost = 0.15  # $/kWh
    emissions_factor = 0.233  # kg CO2e/kWh
    
    # Function to predict EUI after an improvement
    def predict_with_improvement(modified_inputs):
        # Create DataFrame from inputs
        input_df = pd.DataFrame([modified_inputs])
        
        # Transform inputs using the preprocessor
        try:
            input_transformed = preprocessor.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_transformed)
            
            return float(prediction[0])
        except Exception as e:
            print(f"Error predicting with improvement: {e}")
            return baseline_eui  # Return baseline if prediction fails
    
    # Function to calculate annual savings
    def calculate_savings(eui_reduction):
        annual_energy_savings = eui_reduction * area  # kWh/year
        annual_cost_savings = annual_energy_savings * electricity_cost  # $/year
        annual_emissions_reduction = annual_energy_savings * emissions_factor  # kg CO2e/year
        
        return annual_energy_savings, annual_cost_savings, annual_emissions_reduction
    
    # 1. Check insulation values
    
    # Wall insulation
    if inputs.get('Wall_Insulation_U-Value_Capped', 0) > 0.3:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Wall_Insulation_U-Value_Capped'] = 0.3
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            if inputs.get('Wall_Insulation_U-Value_Capped', 0) > 0.5:
                cost_category = 'major_upgrades'
                estimated_cost = area * 100  # Approximate cost $100/m²
                description = "Add external wall insulation to dramatically improve thermal performance."
            else:
                cost_category = 'medium_term'
                estimated_cost = area * 50  # Approximate cost $50/m²
                description = "Improve wall insulation to reach a modern standard (U-value of 0.3 W/m²K)."
            
            recommendations[cost_category].append({
                'title': 'Improve Wall Insulation',
                'description': description,
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'Medium',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': round(estimated_cost, 0),
                'annual_savings': round(cost_savings, 0),
                'feature': 'Wall_Insulation_U-Value_Capped'
            })
    
    # Window insulation
    if inputs.get('Window_Insulation_U-Value_Capped', 0) > 1.8:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Window_Insulation_U-Value_Capped'] = 1.8
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            # Window area estimation based on window-to-wall ratio
            wwr = inputs.get('Window_to_Wall_Ratio', 0.3)
            estimated_window_area = area * 0.5 * wwr  # Rough estimate: external wall area ≈ 50% of floor area
            estimated_cost = estimated_window_area * 400  # Approximate cost $400/m² of window
            
            recommendations['major_upgrades'].append({
                'title': 'Upgrade to High-Performance Windows',
                'description': "Replace windows with high-performance triple glazing (U-value of 1.8 W/m²K or better).",
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'High',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': round(estimated_cost, 0),
                'annual_savings': round(cost_savings, 0),
                'feature': 'Window_Insulation_U-Value_Capped'
            })
    
    # Roof insulation
    if inputs.get('Roof_Insulation_U-Value_Capped', 0) > 0.16:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Roof_Insulation_U-Value_Capped'] = 0.16
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            estimated_cost = area * 25  # Approximate cost $25/m² of roof area
            
            recommendations['medium_term'].append({
                'title': 'Improve Roof Insulation',
                'description': "Add additional loft or roof insulation to reduce heat loss (U-value of 0.16 W/m²K).",
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'Medium',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': round(estimated_cost, 0),
                'annual_savings': round(cost_savings, 0),
                'feature': 'Roof_Insulation_U-Value_Capped'
            })
    
    # Floor insulation
    if inputs.get('Floor_Insulation_U-Value_Capped', 0) > 0.25:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Floor_Insulation_U-Value_Capped'] = 0.25
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            estimated_cost = area * 40  # Approximate cost $40/m²
            
            recommendations['medium_term'].append({
                'title': 'Improve Floor Insulation',
                'description': "Install floor insulation to reduce heat loss through the ground (U-value of 0.25 W/m²K).",
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'Medium to High',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': round(estimated_cost, 0),
                'annual_savings': round(cost_savings, 0),
                'feature': 'Floor_Insulation_U-Value_Capped'
            })
    
    # Air change rate / Air tightness
    if inputs.get('Air_Change_Rate_Capped', 0) > 1.0:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Air_Change_Rate_Capped'] = 1.0
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            if inputs.get('Air_Change_Rate_Capped', 0) > 3.0:
                category = 'quick_wins'
                difficulty = 'Low'
                description = "Seal obvious drafts around windows, doors, and other openings to improve airtightness."
                estimated_cost = 500  # Approximate cost for DIY draught-proofing
            else:
                category = 'medium_term'
                difficulty = 'Medium'
                description = "Implement professional air sealing to reduce infiltration rates to 1.0 air changes per hour."
                estimated_cost = area * 15  # Approximate cost $15/m²
            
            recommendations[category].append({
                'title': 'Improve Building Airtightness',
                'description': description,
                'potential_savings': round(eui_reduction, 1),
                'difficulty': difficulty,
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': round(estimated_cost, 0),
                'annual_savings': round(cost_savings, 0),
                'feature': 'Air_Change_Rate_Capped'
            })
    
    # 2. Check HVAC efficiency
    if inputs.get('HVAC_Efficiency', 0) < 4.0:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['HVAC_Efficiency'] = 4.0
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            if inputs.get('HVAC_Efficiency', 0) < 2.5:
                title = 'Upgrade to High-Efficiency Heat Pump'
                description = "Replace existing heating system with a modern air source heat pump (COP of 4.0)."
                estimated_cost = 8000 + (area * 15)  # Base cost + per m² cost
            else:
                title = 'Optimize HVAC System'
                description = "Upgrade components of the current HVAC system to improve efficiency (COP of 4.0)."
                estimated_cost = 3000 + (area * 5)  # Base cost + per m² cost
            
            recommendations['major_upgrades'].append({
                'title': title,
                'description': description,
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'High',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': round(estimated_cost, 0),
                'annual_savings': round(cost_savings, 0),
                'feature': 'HVAC_Efficiency'
            })
    
    # 3. Check heating setpoints
    if inputs.get('Heating_Setpoint_Temperature', 0) > 21.0:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Heating_Setpoint_Temperature'] = 21.0
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            recommendations['quick_wins'].append({
                'title': 'Lower Heating Setpoint Temperature',
                'description': "Reduce the heating setpoint temperature to 21°C, which is comfortable for most people while reducing energy use.",
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'Low',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': 0,  # No cost
                'annual_savings': round(cost_savings, 0),
                'feature': 'Heating_Setpoint_Temperature'
            })
    
    # 4. Check heating setback temperature
    if inputs.get('Heating_Setback_Temperature', 0) > 16.0:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Heating_Setback_Temperature'] = 16.0
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            recommendations['quick_wins'].append({
                'title': 'Optimize Heating Setback Temperature',
                'description': "Lower the nighttime/unoccupied heating temperature to 16°C to save energy while maintaining basic comfort.",
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'Low',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': 0,  # No cost for manual adjustment
                'annual_savings': round(cost_savings, 0),
                'feature': 'Heating_Setback_Temperature'
            })
    
    # 5. Check lighting density
    if inputs.get('Lighting_Density', 0) > 5.0:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Lighting_Density'] = 5.0
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            # Estimate cost based on area
            estimated_cost = area * 8  # Approximate cost $8/m² for LED upgrade
            
            recommendations['medium_term'].append({
                'title': 'Upgrade to LED Lighting',
                'description': "Replace existing lighting with high-efficiency LED fixtures to reduce lighting power density to 5 W/m² or less.",
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'Low',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': round(estimated_cost, 0),
                'annual_savings': round(cost_savings, 0),
                'feature': 'Lighting_Density'
            })
    
    # 6. Check equipment density
    if inputs.get('Equipment_Density', 0) > 10.0:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Equipment_Density'] = 10.0
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            recommendations['quick_wins'].append({
                'title': 'Reduce Equipment Energy Use',
                'description': "Use energy-efficient appliances and equipment, and enable power management features to reduce equipment power density.",
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'Low',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': 1000,  # Rough estimate for basic measures
                'annual_savings': round(cost_savings, 0),
                'feature': 'Equipment_Density'
            })
    
    # 7. Check renewable energy usage
    if inputs.get('Renewable_Energy_Usage', 'No') == 'No':
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Renewable_Energy_Usage'] = 'Yes'
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            # Estimate solar PV system cost
            system_size_kw = min(area / 10, 10)  # Rough sizing: 1kW per 10m² of floor area, max 10kW
            estimated_cost = system_size_kw * 1500  # Approximate cost $1500/kW installed
            
            recommendations['major_upgrades'].append({
                'title': 'Install Renewable Energy Systems',
                'description': f"Install solar PV panels (approximately {system_size_kw:.1f}kW system) to generate clean electricity and reduce energy costs.",
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'High',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': round(estimated_cost, 0),
                'annual_savings': round(cost_savings, 0),
                'feature': 'Renewable_Energy_Usage'
            })
    
    # 8. Check domestic hot water usage
    if inputs.get('Domestic_Hot_Water_Usage', 0) > 1.0:
        # Test improvement
        improved_inputs = inputs.copy()
        improved_inputs['Domestic_Hot_Water_Usage'] = 1.0
        
        new_eui = predict_with_improvement(improved_inputs)
        eui_reduction = baseline_eui - new_eui
        
        if eui_reduction > 0:
            _, cost_savings, emissions_reduction = calculate_savings(eui_reduction)
            
            recommendations['quick_wins'].append({
                'title': 'Reduce Hot Water Usage',
                'description': "Install low-flow fixtures, fix leaks, and consider water-efficient appliances to reduce hot water consumption.",
                'potential_savings': round(eui_reduction, 1),
                'difficulty': 'Low',
                'co2_reduction': round(emissions_reduction, 1),
                'estimated_cost': 300,  # Approximate cost for low-flow fixtures
                'annual_savings': round(cost_savings, 0),
                'feature': 'Domestic_Hot_Water_Usage'
            })
    
    # 9. If the building has a high window-to-wall ratio in a climate that needs heating
    if inputs.get('Window_to_Wall_Ratio', 0) > 0.4 and inputs.get('Weather_File', '') == 'Historical':
        # Window area estimation
        wwr = inputs.get('Window_to_Wall_Ratio', 0.4)
        estimated_window_area = area * 0.5 * wwr  # Rough estimate
        
        recommendations['quick_wins'].append({
            'title': 'Install Window Treatments',
            'description': "Add insulating blinds, curtains, or window films to reduce heat loss through windows during cold periods.",
            'potential_savings': 5.0,  # Approximate saving
            'difficulty': 'Low',
            'co2_reduction': 5.0 * area * emissions_factor,
            'estimated_cost': estimated_window_area * 30,  # $30/m² of window
            'annual_savings': 5.0 * area * electricity_cost,
            'feature': 'Window_to_Wall_Ratio'
        })
    
    # 10. Smart controls recommendation if heating setpoint is high
    if inputs.get('Heating_Setpoint_Temperature', 0) > 20.0:
        recommendations['medium_term'].append({
            'title': 'Install Smart Heating Controls',
            'description': "Add smart thermostats and zoning controls to optimize heating patterns based on occupancy and needs.",
            'potential_savings': 10.0,  # Approximate saving
            'difficulty': 'Medium',
            'co2_reduction': 10.0 * area * emissions_factor,
            'estimated_cost': 1000,  # Base cost for smart thermostat system
            'annual_savings': 10.0 * area * electricity_cost,
            'feature': 'Heating_Setpoint_Temperature'
        })
    
    return recommendations
