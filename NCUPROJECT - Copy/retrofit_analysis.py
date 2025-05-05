"""
Retrofit Analysis module for the Building Energy Analysis Dashboard.
Provides functionality for analyzing retrofit measures, their impacts on energy consumption,
cost savings, and carbon footprint reduction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
# Ensure the root directory is in the path
sys.path.append('.')  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import from different potential locations
try:
    from app import predict_eui, calculate_cost_and_emissions
except ImportError:
    try:
        from .app import predict_eui, calculate_cost_and_emissions
    except ImportError:
        # Create simplified versions if imports fail
        def predict_eui(inputs, model, preprocessor):
            # Simple prediction function for testing
            return 150.0  # Default EUI value
            
        def calculate_cost_and_emissions(eui, area, electricity_cost=0.34, emissions_factor=0.233):
            # Simple calculation function for testing
            annual_energy = eui * area
            annual_cost = annual_energy * electricity_cost
            annual_emissions = annual_energy * emissions_factor
            return annual_cost, annual_emissions

try:
    from utils.feature_names import get_display_name
except ImportError:
    try:
        from utils.feature_names import get_display_name
    except ImportError:
        # Simple fallback function
        def get_display_name(feature_name):
            return feature_name.replace('_Capped', '')

def display_retrofit_analysis_page(model, preprocessor):
    """Display comprehensive retrofit analysis with charts, graphs, tables, and carbon footprint analysis"""
    
    st.title("📊 Comprehensive Retrofit Analysis")
    
    with st.expander("About Retrofit Analysis", expanded=False):
        st.markdown("""
        This page allows you to analyze potential building retrofits and evaluate their impact on:
        
        - **Energy Savings**: Reduction in energy use intensity (EUI)
        - **Cost Savings**: Annual financial benefits from reduced energy consumption
        - **Carbon Footprint Reduction**: Environmental impact of retrofits
        - **Return on Investment**: Financial metrics including payback period, NPV, and ROI
        
        Explore different retrofit measures and compare their benefits to make informed decisions.
        """)
    
    # Check if we have baseline prediction
    if not hasattr(st.session_state, 'baseline_prediction') or st.session_state.baseline_prediction is None:
        st.warning("Please go to the Prediction page first and calculate your building's baseline energy performance.")
        return
    
    # General input parameters for analysis
    st.subheader("Analysis Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        study_period = st.slider("Study Period (years)", min_value=5, max_value=30, value=20, step=5)
    
    with col2:
        discount_rate = st.slider("Discount Rate (%)", min_value=1.0, max_value=10.0, value=3.5, step=0.5) / 100
    
    with col3:
        energy_price_escalation = st.slider("Annual Energy Price Escalation (%)", 
                                          min_value=0.0, max_value=8.0, value=2.5, step=0.5) / 100
    
    # Get building area from inputs
    building_area = st.session_state.inputs["Total_Building_Area"]
    energy_rate = 0.34  # Default electricity cost per kWh
    emissions_factor = 0.233  # kg CO2 per kWh
    
    # Calculate current values
    baseline_eui = st.session_state.baseline_prediction
    baseline_annual_energy = baseline_eui * building_area  # kWh
    baseline_annual_cost = st.session_state.baseline_cost
    baseline_annual_emissions = st.session_state.baseline_emissions
    
    # Convert the carbon equivalents to a DataFrame for display
    if hasattr(st.session_state, 'carbon_equivalents') and st.session_state.carbon_equivalents:
        carbon_eq = st.session_state.carbon_equivalents
    else:
        carbon_eq = {
            'trees_planted': round(baseline_annual_emissions / 21, 1),
            'car_miles': round(baseline_annual_emissions * 2.5, 0),
            'flights': round(baseline_annual_emissions / 255, 1),
            'smartphone_charges': round(baseline_annual_emissions / 0.005, 0)
        }
    
    # Display baseline metrics
    st.subheader("Baseline Building Performance")
    
    # Create a 3-column layout for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Energy Use Intensity (EUI)", f"{baseline_eui:.1f} kWh/m²/year")
        st.metric("Annual Energy Consumption", f"{baseline_annual_energy:.0f} kWh/year")
    
    with col2:
        st.metric("Annual Energy Cost", f"${baseline_annual_cost:.2f}")
        st.metric("Lifetime Energy Cost (no retrofits)", 
                 f"${sum([baseline_annual_cost * ((1 + energy_price_escalation) ** year) for year in range(study_period)]):.2f}")
    
    with col3:
        st.metric("Annual CO₂ Emissions", f"{baseline_annual_emissions:.1f} kg CO₂/year")
        st.metric("Equivalent to Trees Needed", f"{carbon_eq['trees_planted']:.1f} trees")
    
    # Create retrofit measure analysis tabs
    st.subheader("Retrofit Measure Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Building Envelope", 
        "HVAC System Upgrades", 
        "Lighting & Controls", 
        "Renewable Energy"
    ])
    
    # Helper function to calculate NPV
    def calculate_npv(annual_savings, initial_cost, lifespan, discount_rate=0.05, annual_increase=0.03):
        """Calculate Net Present Value of retrofit measure"""
        npv = -initial_cost  # Initial investment (negative cash flow)
        for year in range(1, lifespan + 1):
            # Annual savings increase with energy prices
            year_savings = annual_savings * ((1 + annual_increase) ** (year - 1))
            # Discount the savings to present value
            npv += year_savings / ((1 + discount_rate) ** year)
        return npv
    
    # Function to predict EUI with modified inputs
    def predict_with_modified_inputs(modifications):
        """Predict EUI with modified building parameters"""
        modified_inputs = st.session_state.inputs.copy()
        for param, value in modifications.items():
            modified_inputs[param] = value
        return predict_eui(modified_inputs, model, preprocessor)
    
    # Helper function to create analysis for a set of retrofit measures
    def analyze_retrofit_measures(measures):
        """Analyze a set of retrofit measures and return results as DataFrame"""
        results = []
        
        for measure in measures:
            # Calculate initial investment
            initial_cost = measure["base_cost"] + (measure["cost_per_m2"] * building_area)
            
            # Calculate energy savings based on measure type
            if "energy_savings_factor" in measure:
                # Direct energy savings factor
                new_eui = baseline_eui * (1 - measure["energy_savings_factor"])
                energy_savings = baseline_eui - new_eui
            elif "modifications" in measure:
                # Predict with modified building parameters
                new_eui = predict_with_modified_inputs(measure["modifications"])
                energy_savings = baseline_eui - new_eui
            else:
                # Default if no specific method provided
                energy_savings = 0
            
            # Calculate annual savings
            annual_energy_savings = energy_savings * building_area  # kWh
            annual_cost_savings = annual_energy_savings * energy_rate
            annual_emissions_reduction = annual_energy_savings * emissions_factor
            
            # Calculate simple payback
            simple_payback = initial_cost / annual_cost_savings if annual_cost_savings > 0 else float('inf')
            
            # Calculate financial metrics
            npv = calculate_npv(
                annual_cost_savings, 
                initial_cost, 
                measure["lifespan"], 
                discount_rate, 
                energy_price_escalation
            )
            
            # Calculate lifetime metrics
            lifetime_energy_savings = annual_energy_savings * measure["lifespan"]
            lifetime_emissions_reduction = annual_emissions_reduction * measure["lifespan"]
            
            # Calculate ROI
            total_savings = sum([annual_cost_savings * ((1 + energy_price_escalation) ** year) 
                              for year in range(measure["lifespan"])])
            roi = ((total_savings - initial_cost) / initial_cost) * 100 if initial_cost > 0 else 0
            
            # Add to results
            results.append({
                "Retrofit Measure": measure["name"],
                "Description": measure["description"],
                "Implementation Cost ($)": initial_cost,
                "Annual Energy Savings (kWh)": annual_energy_savings,
                "Annual Cost Savings ($)": annual_cost_savings,
                "Annual CO₂ Reduction (kg)": annual_emissions_reduction,
                "Simple Payback (years)": simple_payback,
                "NPV ($)": npv,
                "ROI (%)": roi,
                "Lifespan (years)": measure["lifespan"],
                "Lifetime CO₂ Reduction (kg)": lifetime_emissions_reduction
            })
        
        return pd.DataFrame(results)
    
    # Function to create comparison charts with enhanced styling
    def create_comparison_chart(df, value_col, title, color_sequence=None):
        """Create bar chart comparing retrofit measures with improved UI"""
        if color_sequence is None:
            color_sequence = ["#1e88e5", "#42a5f5", "#90caf9", "#e3f2fd"]  # Blue palette
        
        fig = go.Figure()
        
        # Make a copy to ensure we don't modify the original
        df_plot = df.copy()
        
        # Check if the dataframe is empty
        if df_plot.empty:
            # Return empty figure with a message
            fig.add_annotation(
                text="No data available for comparison",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="#616161")
            )
            return fig
        
        # Ensure the value column exists
        if value_col not in df_plot.columns:
            # Add warning annotation and return
            fig.add_annotation(
                text=f"Column '{value_col}' not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="#616161")
            )
            return fig
        
        # Sort data for better visualization if it's a metric like ROI or energy savings
        # For payback, we want ascending (lower is better)
        if "Payback" in value_col:
            df_plot = df_plot.sort_values(by=value_col, ascending=True)
        else:
            # For other metrics like ROI, savings, etc. higher is better
            df_plot = df_plot.sort_values(by=value_col, ascending=False)
        
        # Get min/max for color gradient
        min_val = df_plot[value_col].min()
        max_val = df_plot[value_col].max()
        
        # For each row in the dataframe
        for i, (index, row_data) in enumerate(df_plot.iterrows()):
            try:
                value = row_data[value_col]
                # Add trace only if value is not None or NaN
                if pd.notnull(value):
                    # Create gradient color based on value
                    # For payback, lower is better (green), for everything else, higher is better (green)
                    if "Payback" in value_col:
                        # Normalize between 0 and 1, but reverse since lower is better
                        normalized = 1 - ((value - min_val) / (max_val - min_val)) if max_val != min_val else 0.5
                    else:
                        normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                    
                    # Clamp to ensure we're in 0-1 range
                    normalized = max(0, min(normalized, 1))
                    
                    # Only use the color sequence if it has enough colors
                    if len(color_sequence) > i:
                        bar_color = color_sequence[i]
                    else:
                        # Otherwise generate a gradient color
                        if "Payback" in value_col:
                            r = int(255 * (1 - normalized))  # More red for higher payback
                            g = int(200 * normalized)        # More green for lower payback
                            b = 50                          # Fixed blue component
                        else:
                            r = int(100 * (1 - normalized))  # Less red for higher values
                            g = int(200 * normalized)        # More green for higher values
                            b = int(150 * (1 - normalized))  # Less blue for higher values
                        
                        bar_color = f"rgb({r},{g},{b})"
                    
                    # Format text differently based on value type
                    if "($)" in value_col:
                        text_value = f"${value:,.0f}"
                    elif "(%)" in value_col:
                        text_value = f"{value:.1f}%"
                    elif "Payback" in value_col:
                        text_value = f"{value:.1f} yrs"
                    elif "kWh" in value_col:
                        text_value = f"{value:,.0f} kWh"
                    elif "CO₂" in value_col or "CO2" in value_col:
                        text_value = f"{value:,.0f} kg"
                    else:
                        text_value = f"{value:.1f}"
                    
                    # Create hover text with more details
                    hover_text = f"<b>{index}</b><br>{value_col}: {text_value}"
                    
                    fig.add_trace(go.Bar(
                        x=[value],
                        y=[index],
                        orientation='h',
                        name=str(index),
                        marker_color=bar_color,
                        text=[text_value],
                        textposition='outside',
                        hoverinfo='text',
                        hovertext=hover_text,
                        textfont=dict(
                            family="Arial",
                            size=12,
                            color="black"
                        )
                    ))
            except Exception as e:
                # Skip this row if there's an error
                print(f"Error plotting row {index}: {str(e)}")
                continue
        
        # Add a more descriptive title based on the value column
        if "Simple Payback" in value_col:
            subtitle = "Lower values indicate faster return on investment"
        elif "ROI" in value_col:
            subtitle = "Higher values indicate better financial performance"
        elif "Energy Savings" in value_col:
            subtitle = "Higher values indicate greater energy efficiency"
        elif "CO₂ Reduction" in value_col or "CO2" in value_col:
            subtitle = "Higher values indicate greater environmental benefit"
        else:
            subtitle = ""
        
        fig.update_layout(
            title={
                'text': f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:gray'>{subtitle}</span>" if subtitle else ""),
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            xaxis=dict(
                showgrid=True, 
                gridcolor='lightgray',
                title=value_col,
                title_font=dict(size=12)
            ),
            yaxis=dict(
                showgrid=False,
                title="",
                tickfont=dict(size=12)
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig
    
    # Function to create emissions reduction chart over time with enhanced styling
    def create_emissions_reduction_chart(measure, annual_reduction):
        """Create chart showing emissions reduction over time with improved visualization"""
        years = list(range(measure["lifespan"] + 1))
        cumulative_emissions = [0]
        annual_emissions = [0]
        
        for year in range(1, measure["lifespan"] + 1):
            annual_emissions.append(annual_reduction)
            cumulative_emissions.append(cumulative_emissions[-1] + annual_reduction)
        
        fig = go.Figure()
        
        # Calculate total lifetime reduction for annotations
        total_reduction = cumulative_emissions[-1]
        
        # Add annual emissions bar with better coloring
        fig.add_trace(go.Bar(
            x=years,
            y=annual_emissions,
            name='Annual Reduction',
            marker_color='rgba(76, 175, 80, 0.7)',  # Transparent green
            hovertemplate='Year %{x}<br>Annual: %{y:.1f} kg CO₂<extra></extra>'
        ))
        
        # Add cumulative emissions line with improved styling
        fig.add_trace(go.Scatter(
            x=years,
            y=cumulative_emissions,
            name='Cumulative Reduction',
            mode='lines',
            line=dict(color='#2e7d32', width=4, shape='spline', smoothing=1.3),
            fill='tozeroy',
            fillcolor='rgba(46, 125, 50, 0.1)',
            hovertemplate='Year %{x}<br>Cumulative: %{y:.1f} kg CO₂<extra></extra>'
        ))
        
        # Add markers at specific points (start, middle, end)
        markers_x = [0, measure["lifespan"]//2, measure["lifespan"]]
        markers_y = [cumulative_emissions[i] for i in markers_x]
        
        fig.add_trace(go.Scatter(
            x=markers_x,
            y=markers_y,
            mode='markers',
            marker=dict(
                color='#2e7d32',
                size=10,
                line=dict(color='white', width=2)
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Add annotation at the end point
        fig.add_annotation(
            x=measure["lifespan"],
            y=cumulative_emissions[-1],
            text=f"Total: {total_reduction:,.0f} kg CO₂",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#2e7d32",
            ax=50,
            ay=-40,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#2e7d32",
            borderwidth=1,
            borderpad=4,
            font=dict(color="#2e7d32", size=12)
        )
        
        # Update layout with improved styling
        fig.update_layout(
            title={
                'text': f"<b>CO₂ Emissions Reduction Over Time</b><br><span style='font-size:12px;color:gray'>{measure['name']}, {measure['lifespan']} year lifespan</span>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='Year',
                titlefont=dict(size=12),
                showgrid=True,
                gridcolor='rgba(220, 220, 220, 0.4)',
                tickmode='linear',
                dtick=max(1, measure["lifespan"] // 10)  # Adjust tick spacing based on lifespan
            ),
            yaxis=dict(
                title='CO₂ Reduction (kg)',
                titlefont=dict(size=12),
                showgrid=True,
                gridcolor='rgba(220, 220, 220, 0.4)'
            ),
            legend=dict(
                orientation='h', 
                yanchor='bottom', 
                y=1.02, 
                xanchor='right', 
                x=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(220, 220, 220, 0.5)',
                borderwidth=1
            ),
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=80, b=40),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            hovermode="x unified"
        )
        
        return fig
    
    # Function to create carbon equivalence visualization with enhanced styling
    def create_carbon_equivalence_chart(total_emissions_reduction):
        """Create visually engaging visualization for carbon equivalence metrics"""
        # Calculate equivalences
        trees_equivalent = total_emissions_reduction / 21  # kg CO2 per tree per year
        car_miles_equivalent = total_emissions_reduction * 2.5  # miles per kg CO2
        flights_equivalent = total_emissions_reduction / 255  # kg CO2 per flight (short-haul)
        smartphone_charges_equivalent = total_emissions_reduction / 0.005  # kg CO2 per charge
        
        # Create figure with enhanced styling
        labels = ['Trees Planted', 'Car Miles Avoided', 'Flights Avoided', 'Smartphone Charges']
        values = [trees_equivalent, car_miles_equivalent, flights_equivalent, smartphone_charges_equivalent]
        
        # Create formatted hover texts with explanations
        hover_texts = [
            f"<b>Trees Planted:</b> {trees_equivalent:,.1f}<br>The CO₂ reduction is equivalent to planting this many trees and letting them grow for a year.<br>(Based on ~21kg CO₂ absorbed per tree per year)",
            f"<b>Car Miles Avoided:</b> {car_miles_equivalent:,.0f}<br>The CO₂ reduction is equivalent to not driving this many miles in an average car.<br>(Based on ~0.4kg CO₂ per mile)",
            f"<b>Flights Avoided:</b> {flights_equivalent:,.1f}<br>The CO₂ reduction is equivalent to avoiding this many short-haul flights.<br>(Based on ~255kg CO₂ per flight)",
            f"<b>Smartphone Charges:</b> {smartphone_charges_equivalent:,.0f}<br>The CO₂ reduction is equivalent to this many smartphone charges.<br>(Based on ~0.005kg CO₂ per charge)"
        ]
        
        # Use different visualization approaches based on scale
        if max(values) > 10000:
            # Use a more readable approach for large numbers
            fig = go.Figure()
            
            # Create a table-like visual with icons
            custom_data = list(zip(values, hover_texts))
            
            # Use horizontal bar chart with custom styling
            fig.add_trace(go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(
                    color=['rgba(46, 125, 50, 0.7)', 'rgba(25, 118, 210, 0.7)', 
                           'rgba(156, 39, 176, 0.7)', 'rgba(255, 152, 0, 0.7)'],
                    line=dict(color=['#2e7d32', '#1976d2', '#9c27b0', '#ff9800'], width=2)
                ),
                text=[f"{v:,.0f}" if v >= 100 else f"{v:.1f}" for v in values],
                textposition='outside',
                hoverinfo='text',
                hovertext=hover_texts,
                customdata=custom_data
            ))
            
            # Add icons annotation for each bar
            icons = ['🌳', '🚗', '✈️', '📱']
            for i, (label, icon) in enumerate(zip(labels, icons)):
                fig.add_annotation(
                    x=0,
                    y=i,
                    text=f"{icon} ",
                    showarrow=False,
                    xanchor='right',
                    yanchor='middle',
                    font=dict(size=18),
                    xshift=-10
                )
                
            fig.update_layout(
                title={
                    'text': "<b>Carbon Footprint Reduction Equivalents</b><br><span style='font-size:12px;color:gray'>Impact visualization of CO₂ reduction</span>",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis=dict(
                    type='log' if max(values) > 100000 else 'linear',
                    title="Equivalent Impact" + (" (logarithmic scale)" if max(values) > 100000 else ""),
                    titlefont=dict(size=12),
                ),
                yaxis=dict(
                    title="",
                    automargin=True
                )
            )
        else:
            # Use regular bar chart with enhanced styling for reasonable numbers
            fig = go.Figure()
            
            # Add icon-based bar chart
            for i, (label, value, hover) in enumerate(zip(labels, values, hover_texts)):
                fig.add_trace(go.Bar(
                    x=[label],
                    y=[value],
                    name=label,
                    marker_color=['rgba(46, 125, 50, 0.7)', 'rgba(25, 118, 210, 0.7)', 
                                  'rgba(156, 39, 176, 0.7)', 'rgba(255, 152, 0, 0.7)'][i],
                    text=f"{value:,.0f}" if value >= 100 else f"{value:.1f}",
                    textposition='outside',
                    hoverinfo='text',
                    hovertext=hover
                ))
            
            fig.update_layout(
                title={
                    'text': "<b>Carbon Footprint Reduction Equivalents</b><br><span style='font-size:12px;color:gray'>Impact visualization of CO₂ reduction</span>",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                showlegend=False
            )
        
        # Common layout settings with improved styling
        fig.update_layout(
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=40, r=40, t=80, b=20),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            xaxis=dict(showgrid=True, gridcolor='rgba(220, 220, 220, 0.4)'),
            yaxis=dict(showgrid=False)
        )
        
        return fig
    
    # ------ Tab 1: Building Envelope Measures ------
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Building Envelope Retrofit Measures")
        
        with col2:
            # Show key recommendation
            st.info("💡 **Top recommendation:** Insulation and air sealing typically offer the best ROI and should be prioritized before window replacements.")
        
        # Define envelope retrofit measures
        envelope_measures = [
            {
                "name": "Wall Insulation Upgrade",
                "description": "Upgrade wall insulation to reduce heat loss through walls.",
                "base_cost": 1000,
                "cost_per_m2": 75,
                "energy_savings_factor": 0.12,
                "lifespan": 25
            },
            {
                "name": "Roof Insulation Upgrade",
                "description": "Add additional roof insulation to reduce heat loss.",
                "base_cost": 1500,
                "cost_per_m2": 45,
                "energy_savings_factor": 0.10,
                "lifespan": 30
            },
            {
                "name": "Window Replacement (Double Glazing)",
                "description": "Replace single-glazed windows with double-glazing.",
                "base_cost": 3000,
                "cost_per_m2": 350,
                "modifications": {
                    "Window_Insulation_U-Value_Capped": max(
                        st.session_state.inputs["Window_Insulation_U-Value_Capped"] * 0.6, 
                        1.4  # Minimum achievable U-value
                    )
                },
                "lifespan": 25
            },
            {
                "name": "Air Sealing & Draft-proofing",
                "description": "Comprehensive air sealing to reduce air leakage.",
                "base_cost": 800,
                "cost_per_m2": 15,
                "modifications": {
                    "Air_Change_Rate_Capped": max(
                        st.session_state.inputs["Air_Change_Rate_Capped"] * 0.7,
                        0.5  # Minimum healthy air change rate
                    )
                },
                "lifespan": 15
            }
        ]
        
        # Analyze envelope measures
        envelope_df = analyze_retrofit_measures(envelope_measures)
        
        # Create tabs for different analysis views
        envelope_tab1, envelope_tab2, envelope_tab3 = st.tabs(["Comparison Table", "Charts", "Carbon Impact"])
        
        with envelope_tab1:
            # Format dataframe for display with enhanced styling
            display_df = envelope_df.copy()
            
            # Format numeric columns with appropriate formatting
            display_df["Implementation Cost ($)"] = display_df["Implementation Cost ($)"].map("${:,.2f}".format)
            display_df["Annual Cost Savings ($)"] = display_df["Annual Cost Savings ($)"].map("${:,.2f}".format)
            display_df["NPV ($)"] = display_df["NPV ($)"].map("${:,.2f}".format)
            display_df["Simple Payback (years)"] = display_df["Simple Payback (years)"].map("{:.1f}".format)
            display_df["ROI (%)"] = display_df["ROI (%)"].map("{:.1f}%".format)
            
            # Create a styled dataframe with color highlighting based on values
            def highlight_payback(val):
                """Color code payback periods: shorter is better (green)"""
                try:
                    val_float = float(val.replace(' years', '').strip())
                    if val_float < 5:
                        color = 'rgba(0, 200, 0, 0.2)'  # Very good - green
                    elif val_float < 10:
                        color = 'rgba(200, 200, 0, 0.2)'  # Good - yellow
                    else:
                        color = 'rgba(255, 165, 0, 0.1)'  # Average - light orange
                    return f'background-color: {color}'
                except:
                    return ''
            
            def highlight_roi(val):
                """Color code ROI values: higher is better (green)"""
                try:
                    val_float = float(val.replace('%', '').strip())
                    if val_float > 15:
                        color = 'rgba(0, 200, 0, 0.2)'  # Very good - green
                    elif val_float > 8:
                        color = 'rgba(200, 200, 0, 0.2)'  # Good - yellow
                    else:
                        color = 'rgba(255, 165, 0, 0.1)'  # Average - light orange
                    return f'background-color: {color}'
                except:
                    return ''
            
            def highlight_npv(val):
                """Color code NPV values: higher is better (green)"""
                try:
                    val_float = float(val.replace('$', '').replace(',', '').strip())
                    if val_float > 5000:
                        color = 'rgba(0, 200, 0, 0.2)'  # Very good - green
                    elif val_float > 0:
                        color = 'rgba(200, 200, 0, 0.2)'  # Good - yellow
                    else:
                        color = 'rgba(255, 165, 0, 0.1)'  # Average - light orange
                    return f'background-color: {color}'
                except:
                    return ''
            
            # Apply styles to specific columns
            styled_df = display_df[["Description", "Implementation Cost ($)", "Annual Cost Savings ($)", 
                                  "Simple Payback (years)", "ROI (%)", "NPV ($)"]].style.applymap(
                highlight_payback, subset=pd.IndexSlice[:, ['Simple Payback (years)']]
            ).applymap(
                highlight_roi, subset=pd.IndexSlice[:, ['ROI (%)']]
            ).applymap(
                highlight_npv, subset=pd.IndexSlice[:, ['NPV ($)']]
            )
            
            # Add styling properties
            styled_df = styled_df.set_properties(**{
                'text-align': 'left',
                'font-size': '14px',
                'border-color': '#888',
                'padding': '5px'
            })
            
            # Set caption and display the table
            st.write("**Financial Performance of Building Envelope Measures**")
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
        
        with envelope_tab2:
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                payback_fig = create_comparison_chart(
                    envelope_df.set_index("Retrofit Measure"), 
                    "Simple Payback (years)",
                    "Simple Payback Period (years)"
                )
                st.plotly_chart(payback_fig, use_container_width=True)
            
            with col2:
                roi_fig = create_comparison_chart(
                    envelope_df.set_index("Retrofit Measure"), 
                    "ROI (%)",
                    "Return on Investment (%)"
                )
                st.plotly_chart(roi_fig, use_container_width=True)
            
            # Energy savings chart
            energy_fig = create_comparison_chart(
                envelope_df.set_index("Retrofit Measure"), 
                "Annual Energy Savings (kWh)",
                "Annual Energy Savings (kWh)"
            )
            st.plotly_chart(energy_fig, use_container_width=True)
        
        with envelope_tab3:
            # Show carbon impact analysis for a selected measure
            selected_measure = st.selectbox(
                "Select a measure to view carbon impact details:",
                options=envelope_df["Retrofit Measure"].tolist(),
                key="envelope_carbon_select"
            )
            
            selected_row = envelope_df[envelope_df["Retrofit Measure"] == selected_measure].iloc[0]
            selected_measure_data = next(m for m in envelope_measures if m["name"] == selected_measure)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Annual CO₂ Reduction", 
                    f"{selected_row['Annual CO₂ Reduction (kg)']:.1f} kg CO₂"
                )
                st.metric(
                    "Lifetime CO₂ Reduction", 
                    f"{selected_row['Lifetime CO₂ Reduction (kg)']:.1f} kg CO₂"
                )
            
            with col2:
                trees_equivalent = selected_row['Lifetime CO₂ Reduction (kg)'] / 21
                car_miles_equivalent = selected_row['Lifetime CO₂ Reduction (kg)'] * 2.5
                
                st.metric(
                    "Equivalent to Trees Planted", 
                    f"{trees_equivalent:.1f} trees"
                )
                st.metric(
                    "Equivalent to Car Miles Avoided", 
                    f"{car_miles_equivalent:.1f} miles"
                )
            
            # Show CO2 reduction over time
            emissions_chart = create_emissions_reduction_chart(
                selected_measure_data,
                selected_row['Annual CO₂ Reduction (kg)']
            )
            st.plotly_chart(emissions_chart, use_container_width=True)
            
            # Show carbon equivalence visualization
            equivalence_chart = create_carbon_equivalence_chart(
                selected_row['Lifetime CO₂ Reduction (kg)']
            )
            st.plotly_chart(equivalence_chart, use_container_width=True)
    
    # ------ Tab 2: HVAC System Upgrades ------
    with tab2:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("HVAC System Upgrades")
        
        with col2:
            # Show key recommendation
            st.info("💡 **Key insight:** Heat recovery systems often provide the best combination of energy savings and CO₂ reduction for buildings with high ventilation needs.")
        
        # Define HVAC retrofit measures
        hvac_measures = [
            {
                "name": "High-Efficiency Heat Pump",
                "description": "Replace conventional heating/cooling with high-efficiency heat pump system.",
                "base_cost": 8000,
                "cost_per_m2": 75,
                "modifications": {
                    "HVAC_Efficiency": min(st.session_state.inputs["HVAC_Efficiency"] * 1.5, 5.0)
                },
                "lifespan": 15
            },
            {
                "name": "Smart Thermostat & Controls",
                "description": "Install smart thermostats and advanced controls for optimized operation.",
                "base_cost": 500,
                "cost_per_m2": 5,
                "energy_savings_factor": 0.08,
                "lifespan": 10
            },
            {
                "name": "Variable Speed Drives",
                "description": "Add variable speed drives to HVAC fans and pumps.",
                "base_cost": 2000,
                "cost_per_m2": 15,
                "energy_savings_factor": 0.07,
                "lifespan": 15
            },
            {
                "name": "Heat Recovery Ventilation",
                "description": "Install heat recovery ventilation system to recapture heat from exhaust air.",
                "base_cost": 3500,
                "cost_per_m2": 40,
                "energy_savings_factor": 0.11,
                "lifespan": 15
            }
        ]
        
        # Analyze HVAC measures
        hvac_df = analyze_retrofit_measures(hvac_measures)
        
        # Create tabs for different analysis views
        hvac_tab1, hvac_tab2, hvac_tab3 = st.tabs(["Comparison Table", "Charts", "Carbon Impact"])
        
        with hvac_tab1:
            # Format dataframe for display
            display_df = hvac_df.copy()
            display_df["Implementation Cost ($)"] = display_df["Implementation Cost ($)"].map("${:,.2f}".format)
            display_df["Annual Cost Savings ($)"] = display_df["Annual Cost Savings ($)"].map("${:,.2f}".format)
            display_df["NPV ($)"] = display_df["NPV ($)"].map("${:,.2f}".format)
            display_df["Simple Payback (years)"] = display_df["Simple Payback (years)"].map("{:.1f}".format)
            display_df["ROI (%)"] = display_df["ROI (%)"].map("{:.1f}%".format)
            
            st.dataframe(
                display_df[["Description", "Implementation Cost ($)", "Annual Cost Savings ($)", 
                          "Simple Payback (years)", "ROI (%)", "NPV ($)"]], 
                hide_index=True
            )
        
        with hvac_tab2:
            # Create comparison charts - similar to envelope tab
            col1, col2 = st.columns(2)
            
            with col1:
                payback_fig = create_comparison_chart(
                    hvac_df.set_index("Retrofit Measure"), 
                    "Simple Payback (years)",
                    "Simple Payback Period (years)",
                    color_sequence=["#1565c0", "#1976d2", "#42a5f5", "#bbdefb"]
                )
                st.plotly_chart(payback_fig, use_container_width=True)
            
            with col2:
                roi_fig = create_comparison_chart(
                    hvac_df.set_index("Retrofit Measure"), 
                    "ROI (%)",
                    "Return on Investment (%)",
                    color_sequence=["#1565c0", "#1976d2", "#42a5f5", "#bbdefb"]
                )
                st.plotly_chart(roi_fig, use_container_width=True)
        
        with hvac_tab3:
            # Similar to envelope tab3
            selected_measure = st.selectbox(
                "Select a measure to view carbon impact details:",
                options=hvac_df["Retrofit Measure"].tolist(),
                key="hvac_carbon_select"
            )
            
            selected_row = hvac_df[hvac_df["Retrofit Measure"] == selected_measure].iloc[0]
            selected_measure_data = next(m for m in hvac_measures if m["name"] == selected_measure)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Annual CO₂ Reduction", 
                    f"{selected_row['Annual CO₂ Reduction (kg)']:.1f} kg CO₂"
                )
                st.metric(
                    "Lifetime CO₂ Reduction", 
                    f"{selected_row['Lifetime CO₂ Reduction (kg)']:.1f} kg CO₂"
                )
            
            with col2:
                trees_equivalent = selected_row['Lifetime CO₂ Reduction (kg)'] / 21
                car_miles_equivalent = selected_row['Lifetime CO₂ Reduction (kg)'] * 2.5
                
                st.metric(
                    "Equivalent to Trees Planted", 
                    f"{trees_equivalent:.1f} trees"
                )
                st.metric(
                    "Equivalent to Car Miles Avoided", 
                    f"{car_miles_equivalent:.1f} miles"
                )
            
            # Show CO2 reduction over time
            emissions_chart = create_emissions_reduction_chart(
                selected_measure_data,
                selected_row['Annual CO₂ Reduction (kg)']
            )
            st.plotly_chart(emissions_chart, use_container_width=True)
    
    # ------ Tab 3: Lighting & Controls ------
    with tab3:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Lighting & Controls Upgrades")
        
        with col2:
            # Show key recommendation
            st.info("💡 **Quick win:** Smart lighting controls combined with LED fixtures can pay back in 3-4 years while dramatically reducing maintenance.")
        
        # Define lighting retrofit measures
        lighting_measures = [
            {
                "name": "LED Lighting Upgrade",
                "description": "Replace all lighting with high-efficiency LED fixtures.",
                "base_cost": 1200,
                "cost_per_m2": 25,
                "modifications": {
                    "Lighting_Density": max(st.session_state.inputs["Lighting_Density"] * 0.6, 5.0)
                },
                "lifespan": 10
            },
            {
                "name": "Occupancy & Daylight Sensors",
                "description": "Install occupancy sensors and daylight harvesting controls.",
                "base_cost": 800,
                "cost_per_m2": 10,
                "energy_savings_factor": 0.08,
                "lifespan": 10
            },
            {
                "name": "Smart Lighting Control System",
                "description": "Comprehensive networked lighting control system with scheduling, dimming, and zone control.",
                "base_cost": 2500,
                "cost_per_m2": 35,
                "energy_savings_factor": 0.15,
                "lifespan": 15
            }
        ]
        
        # Analyze lighting measures
        lighting_df = analyze_retrofit_measures(lighting_measures)
        
        # Display analysis similar to previous tabs
        lighting_tab1, lighting_tab2 = st.tabs(["Comparison Table", "Charts"])
        
        with lighting_tab1:
            # Format dataframe for display
            display_df = lighting_df.copy()
            display_df["Implementation Cost ($)"] = display_df["Implementation Cost ($)"].map("${:,.2f}".format)
            display_df["Annual Cost Savings ($)"] = display_df["Annual Cost Savings ($)"].map("${:,.2f}".format)
            display_df["NPV ($)"] = display_df["NPV ($)"].map("${:,.2f}".format)
            display_df["Simple Payback (years)"] = display_df["Simple Payback (years)"].map("{:.1f}".format)
            display_df["ROI (%)"] = display_df["ROI (%)"].map("{:.1f}%".format)
            
            st.dataframe(
                display_df[["Description", "Implementation Cost ($)", "Annual Cost Savings ($)", 
                          "Simple Payback (years)", "ROI (%)", "NPV ($)"]], 
                hide_index=True
            )
        
        with lighting_tab2:
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                energy_fig = create_comparison_chart(
                    lighting_df.set_index("Retrofit Measure"), 
                    "Annual Energy Savings (kWh)",
                    "Annual Energy Savings (kWh)",
                    color_sequence=["#c2185b", "#e91e63", "#f48fb1"]
                )
                st.plotly_chart(energy_fig, use_container_width=True)
            
            with col2:
                emissions_fig = create_comparison_chart(
                    lighting_df.set_index("Retrofit Measure"), 
                    "Annual CO₂ Reduction (kg)",
                    "Annual CO₂ Reduction (kg)",
                    color_sequence=["#c2185b", "#e91e63", "#f48fb1"]
                )
                st.plotly_chart(emissions_fig, use_container_width=True)
    
    # ------ Tab 4: Renewable Energy ------
    with tab4:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Renewable Energy Options")
        
        with col2:
            # Show key recommendation
            st.info("💡 **Long-term approach:** Focus on basic efficiency measures first, then consider renewables for maximum carbon reduction and long-term ROI.")
        
        # Define renewable energy options
        renewable_measures = [
            {
                "name": "Rooftop Solar PV System",
                "description": "Install solar photovoltaic panels on the building roof (3 kW system).",
                "base_cost": 5000,
                "cost_per_m2": 0,  # Fixed cost regardless of building size
                "energy_savings_factor": 0.25,  # 25% reduction in grid electricity
                "lifespan": 25
            },
            {
                "name": "Solar Water Heating",
                "description": "Install solar thermal system for water heating.",
                "base_cost": 3500,
                "cost_per_m2": 0,  # Fixed cost
                "energy_savings_factor": 0.10,
                "lifespan": 20
            },
            {
                "name": "Ground Source Heat Pump",
                "description": "Install ground source heat pump for heating and cooling.",
                "base_cost": 15000,
                "cost_per_m2": 100,
                "modifications": {
                    "HVAC_Efficiency": 5.0,  # Very high COP
                    "Renewable_Energy_Usage": "Yes"
                },
                "lifespan": 25
            }
        ]
        
        # Analyze renewable energy measures
        renewable_df = analyze_retrofit_measures(renewable_measures)
        
        # Display analysis
        renewable_tab1, renewable_tab2 = st.tabs(["Comparison Table", "Carbon Impact"])
        
        with renewable_tab1:
            # Format dataframe for display
            display_df = renewable_df.copy()
            display_df["Implementation Cost ($)"] = display_df["Implementation Cost ($)"].map("${:,.2f}".format)
            display_df["Annual Cost Savings ($)"] = display_df["Annual Cost Savings ($)"].map("${:,.2f}".format)
            display_df["NPV ($)"] = display_df["NPV ($)"].map("${:,.2f}".format)
            display_df["Simple Payback (years)"] = display_df["Simple Payback (years)"].map("{:.1f}".format)
            display_df["ROI (%)"] = display_df["ROI (%)"].map("{:.1f}%".format)
            
            st.dataframe(
                display_df[["Description", "Implementation Cost ($)", "Annual Cost Savings ($)", 
                          "Simple Payback (years)", "ROI (%)", "NPV ($)", "Lifetime CO₂ Reduction (kg)"]], 
                hide_index=True
            )
            
            # Create comparison charts
            emissions_fig = create_comparison_chart(
                renewable_df.set_index("Retrofit Measure"), 
                "Lifetime CO₂ Reduction (kg)",
                "Lifetime CO₂ Reduction (kg)",
                color_sequence=["#f57f17", "#ffa000", "#ffca28"]
            )
            st.plotly_chart(emissions_fig, use_container_width=True)
        
        with renewable_tab2:
            # Show cumulative carbon reduction over time for all renewable options
            fig = go.Figure()
            
            for i, row in renewable_df.iterrows():
                measure = next(m for m in renewable_measures if m["name"] == row["Retrofit Measure"])
                years = list(range(measure["lifespan"] + 1))
                cumulative_emissions = [0]
                
                for year in range(1, measure["lifespan"] + 1):
                    cumulative_emissions.append(cumulative_emissions[-1] + row["Annual CO₂ Reduction (kg)"])
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=cumulative_emissions,
                    name=row["Retrofit Measure"],
                    mode='lines',
                    line=dict(width=3)
                ))
            
            fig.update_layout(
                title='Cumulative CO₂ Reduction Over Time',
                xaxis_title='Year',
                yaxis_title='Cumulative CO₂ Reduction (kg)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=500,
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=60, b=40),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show carbon equivalence for highest impact renewable option
            best_measure = renewable_df.loc[renewable_df["Lifetime CO₂ Reduction (kg)"].idxmax()]
            
            st.subheader(f"Environmental Impact of {best_measure['Retrofit Measure']}")
            st.markdown(f"""
            Over its {best_measure['Lifespan (years)']} year lifespan, this measure would reduce CO₂ emissions by 
            **{best_measure['Lifetime CO₂ Reduction (kg)']:,.1f} kg**, which is equivalent to:
            """)
            
            equivalence_chart = create_carbon_equivalence_chart(
                best_measure['Lifetime CO₂ Reduction (kg)']
            )
            st.plotly_chart(equivalence_chart, use_container_width=True)
    
    # Create comprehensive retrofit package analysis
    st.subheader("Comprehensive Retrofit Package Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Analyze the combined impact of multiple retrofit measures as a package.")
    
    with col2:
        st.warning("💡 **Expert advice:** Implementing measures as a coordinated package often yields better results than piecemeal approaches. Consider financing options that bundle measures with different payback periods.")
    
    # Package selection
    st.markdown("#### Select Measures to Include in Package")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        envelope_selection = st.selectbox(
            "Building Envelope Measure",
            options=["None"] + envelope_df["Retrofit Measure"].tolist()
        )
    
    with col2:
        hvac_selection = st.selectbox(
            "HVAC System Upgrade",
            options=["None"] + hvac_df["Retrofit Measure"].tolist()
        )
    
    with col3:
        lighting_selection = st.selectbox(
            "Lighting & Controls",
            options=["None"] + lighting_df["Retrofit Measure"].tolist()
        )
    
    with col4:
        renewable_selection = st.selectbox(
            "Renewable Energy",
            options=["None"] + renewable_df["Retrofit Measure"].tolist()
        )
    
    # Calculate combined package impact
    selected_measures = []
    
    if envelope_selection != "None":
        selected_measures.append(envelope_df[envelope_df["Retrofit Measure"] == envelope_selection].iloc[0])
    
    if hvac_selection != "None":
        selected_measures.append(hvac_df[hvac_df["Retrofit Measure"] == hvac_selection].iloc[0])
    
    if lighting_selection != "None":
        selected_measures.append(lighting_df[lighting_df["Retrofit Measure"] == lighting_selection].iloc[0])
    
    if renewable_selection != "None":
        selected_measures.append(renewable_df[renewable_df["Retrofit Measure"] == renewable_selection].iloc[0])
    
    if selected_measures:
        # Calculate package totals
        total_cost = sum(measure["Implementation Cost ($)"] for measure in selected_measures)
        total_annual_savings = sum(measure["Annual Cost Savings ($)"] for measure in selected_measures)
        total_annual_energy_savings = sum(measure["Annual Energy Savings (kWh)"] for measure in selected_measures)
        total_annual_emissions_reduction = sum(measure["Annual CO₂ Reduction (kg)"] for measure in selected_measures)
        
        # Estimated new EUI after all measures
        estimated_new_eui = max(baseline_eui - (total_annual_energy_savings / building_area), 0)
        
        # Calculate simple payback
        simple_payback = total_cost / total_annual_savings if total_annual_savings > 0 else float('inf')
        
        # Calculate NPV - simplified approach
        package_npv = -total_cost
        for year in range(1, study_period + 1):
            year_savings = total_annual_savings * ((1 + energy_price_escalation) ** (year - 1))
            package_npv += year_savings / ((1 + discount_rate) ** year)
        
        # Calculate ROI
        total_lifetime_savings = sum([total_annual_savings * ((1 + energy_price_escalation) ** year) 
                                  for year in range(study_period)])
        package_roi = ((total_lifetime_savings - total_cost) / total_cost) * 100 if total_cost > 0 else 0
        
        # Calculate lifetime emissions reduction (simplified)
        lifetime_emissions_reduction = total_annual_emissions_reduction * study_period
        
        # Display package analysis results
        st.subheader("Retrofit Package Results")
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Implementation Cost", f"${total_cost:,.2f}")
            st.metric("Annual Energy Savings", f"{total_annual_energy_savings:,.0f} kWh")
            st.metric("New Estimated EUI", f"{estimated_new_eui:.1f} kWh/m²/year", 
                     delta=f"-{baseline_eui - estimated_new_eui:.1f} kWh/m²/year")
        
        with col2:
            st.metric("Annual Cost Savings", f"${total_annual_savings:,.2f}")
            st.metric("Simple Payback Period", f"{simple_payback:.1f} years")
            st.metric("Package NPV", f"${package_npv:,.2f}")
        
        with col3:
            st.metric("Annual CO₂ Reduction", f"{total_annual_emissions_reduction:.1f} kg CO₂")
            st.metric("Lifetime CO₂ Reduction", f"{lifetime_emissions_reduction:.1f} kg CO₂")
            st.metric("Return on Investment", f"{package_roi:.1f}%")
        
        # Create before/after comparison table
        st.subheader("Before vs. After Retrofit Package")
        
        comparison_data = {
            "Metric": ["Energy Use Intensity (kWh/m²/year)", 
                     "Annual Energy Consumption (kWh)", 
                     "Annual Energy Cost ($)", 
                     "Annual CO₂ Emissions (kg CO₂)"],
            "Before Retrofit": [baseline_eui, 
                              baseline_annual_energy, 
                              baseline_annual_cost, 
                              baseline_annual_emissions],
            "After Retrofit": [estimated_new_eui, 
                             baseline_annual_energy - total_annual_energy_savings,
                             baseline_annual_cost - total_annual_savings,
                             baseline_annual_emissions - total_annual_emissions_reduction],
            "Reduction": [baseline_eui - estimated_new_eui,
                        total_annual_energy_savings,
                        total_annual_savings,
                        total_annual_emissions_reduction],
            "Savings %": [(baseline_eui - estimated_new_eui) / baseline_eui * 100,
                        total_annual_energy_savings / baseline_annual_energy * 100,
                        total_annual_savings / baseline_annual_cost * 100,
                        total_annual_emissions_reduction / baseline_annual_emissions * 100]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index("Metric", inplace=True)
        
        # Format the values for display
        for col in comparison_df.columns:
            if col == "Savings %":
                comparison_df[col] = comparison_df[col].map("{:.1f}%".format)
            elif col == "After Retrofit" or col == "Before Retrofit" or col == "Reduction":
                if col == "Annual Energy Cost ($)" or "Cost" in comparison_df.index:
                    comparison_df.loc["Annual Energy Cost ($)", col] = "${:,.2f}".format(float(comparison_df.loc["Annual Energy Cost ($)", col]))
                else:
                    comparison_df[col] = comparison_df[col].map("{:,.1f}".format)
        
        st.dataframe(comparison_df)
        
        # Create a before/after comparison chart
        metrics = ["Energy Use Intensity", "Energy Consumption", "Energy Cost", "CO₂ Emissions"]
        before_values = [baseline_eui, 
                       baseline_annual_energy, 
                       baseline_annual_cost, 
                       baseline_annual_emissions]
        after_values = [estimated_new_eui, 
                      baseline_annual_energy - total_annual_energy_savings,
                      baseline_annual_cost - total_annual_savings,
                      baseline_annual_emissions - total_annual_emissions_reduction]
        
        fig = go.Figure()
        
        # Add before retrofit bars
        fig.add_trace(go.Bar(
            x=metrics,
            y=before_values,
            name="Before Retrofit",
            marker_color='#ef5350'  # Light red
        ))
        
        # Add after retrofit bars
        fig.add_trace(go.Bar(
            x=metrics,
            y=after_values,
            name="After Retrofit",
            marker_color='#66bb6a'  # Light green
        ))
        
        # Update layout
        fig.update_layout(
            title="Before vs. After Retrofit Package",
            xaxis_title="Metric",
            yaxis_title="Value",
            barmode='group',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add cumulative emissions reduction over time
        years = list(range(study_period + 1))
        cumulative_emissions = [0]
        
        for year in range(1, study_period + 1):
            cumulative_emissions.append(cumulative_emissions[-1] + total_annual_emissions_reduction)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=cumulative_emissions,
            mode='lines',
            line=dict(color='#2e7d32', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 187, 106, 0.3)'
        ))
        
        fig.update_layout(
            title=f'Cumulative CO₂ Emissions Reduction Over {study_period} Years',
            xaxis_title='Year',
            yaxis_title='CO₂ Reduction (kg)',
            height=400,
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=60, b=40),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add carbon equivalence information
        st.subheader("Environmental Impact")
        
        # Calculate carbon equivalences
        trees_planted = lifetime_emissions_reduction / 21  # kg CO₂ absorbed by one tree per year
        car_miles_avoided = lifetime_emissions_reduction * 2.5  # miles per kg CO₂
        flights_avoided = lifetime_emissions_reduction / 255  # kg CO₂ per flight
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Trees Planted Equivalent", f"{trees_planted:.1f} trees")
        
        with col2:
            st.metric("Car Miles Avoided", f"{car_miles_avoided:,.1f} miles")
        
        with col3:
            st.metric("Flights Avoided", f"{flights_avoided:.1f} flights")
        
        # Add carbon equivalence visualization
        equivalence_chart = create_carbon_equivalence_chart(lifetime_emissions_reduction)
        st.plotly_chart(equivalence_chart, use_container_width=True)
        
    else:
        st.info("Please select at least one retrofit measure to analyze the package.")