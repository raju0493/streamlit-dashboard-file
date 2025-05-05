"""
Utility module for generating PDF and Excel reports for the building energy dashboard.
"""

import io
import pandas as pd
import numpy as np
from datetime import datetime
import base64

# Try to import visualization and reporting libraries
try:
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.pdfgen import canvas
    import xlsxwriter
    HAS_REPORT_LIBS = True
except ImportError:
    HAS_REPORT_LIBS = False

from .feature_names import get_display_name

def generate_enhanced_pdf_report(inputs, eui, cost, emissions, rating, options, model=None, preprocessor=None):
    """
    Generate an enhanced PDF report with detailed insights and analysis.
    
    Args:
        inputs (dict): Dictionary of building inputs
        eui (float): Predicted Energy Use Intensity
        cost (float): Predicted annual energy cost
        emissions (float): Predicted annual CO2 emissions
        rating (str): Energy performance rating
        options (dict): Report options and configuration
        model (object, optional): Prediction model for additional analysis
        preprocessor (object, optional): Data preprocessor
        
    Returns:
        bytes: PDF file as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=1))
    
    # Modify the existing styles instead of adding new ones
    styles['Title'].fontName = 'Helvetica-Bold'
    styles['Title'].fontSize = 18
    styles['Title'].alignment = 1
    styles['Title'].spaceAfter = 12
    styles['Title'].textColor = colors.darkgreen
    
    # Check if 'Subtitle' style already exists, if not add it
    if 'Subtitle' not in styles:
        styles.add(ParagraphStyle(name='Subtitle', fontName='Helvetica-Bold', fontSize=14, spaceBefore=12, spaceAfter=6, textColor=colors.darkgreen))
    else:
        styles['Subtitle'].fontName = 'Helvetica-Bold'
        styles['Subtitle'].fontSize = 14
        styles['Subtitle'].spaceBefore = 12
        styles['Subtitle'].spaceAfter = 6
        styles['Subtitle'].textColor = colors.darkgreen
        
    # Check if 'Heading2' style already exists, if not add it
    if 'Heading2' not in styles:
        styles.add(ParagraphStyle(name='Heading2', fontName='Helvetica-Bold', fontSize=12, spaceBefore=10, spaceAfter=6, textColor=colors.darkgreen))
    else:
        styles['Heading2'].fontName = 'Helvetica-Bold'
        styles['Heading2'].fontSize = 12
        styles['Heading2'].spaceBefore = 10
        styles['Heading2'].spaceAfter = 6
        styles['Heading2'].textColor = colors.darkgreen
    
    # Title
    elements.append(Paragraph(options.get('title', "Building Energy Performance Report"), styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Organization and author info if provided
    if options.get('company_name'):
        elements.append(Paragraph(f"Organization: {options['company_name']}", styles['Center']))
    if options.get('prepared_by'):
        elements.append(Paragraph(f"Prepared by: {options['prepared_by']}", styles['Center']))
    
    # Date of report
    date_str = datetime.now().strftime("%d %B %Y, %H:%M")
    elements.append(Paragraph(f"Generated on: {date_str}", styles['Center']))
    elements.append(Spacer(1, 24))
    
    # Executive Summary section
    elements.append(Paragraph("Executive Summary", styles['Subtitle']))
    
    summary_text = f"""This report provides a comprehensive analysis of the building's energy performance. The building has an Energy Use Intensity (EUI) of {eui:.1f} kWh/m²/year, which corresponds to an energy efficiency rating of {rating}. The annual energy cost is estimated at ${cost:.2f}, with annual CO₂ emissions of {emissions:.1f} kg.

Based on this analysis, we have identified opportunities to improve energy efficiency, reduce operating costs, and lower the carbon footprint of the building."""
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Building Performance Metrics
    elements.append(Paragraph("Building Performance Metrics", styles['Subtitle']))
    
    # Performance metrics table
    data = [
        ["Metric", "Value", "Rating"],
        ["Energy Use Intensity (EUI)", f"{eui:.1f} kWh/m²/year", rating],
        ["Annual Energy Cost", f"${cost:.2f}", ""],
        ["CO₂ Emissions", f"{emissions:.1f} kg CO₂e/year", ""],
        ["Energy Cost per m²", f"${cost/inputs.get('Total_Building_Area', 1000):.2f}/m²", ""]
    ]
    
    table = Table(data, colWidths=[2*inch, 2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (2, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (2, 0), 12),
        ('BACKGROUND', (0, 1), (2, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (1, 1), (2, -1), 'CENTER'),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 24))
    
    # Building characteristics section
    elements.append(Paragraph("Building Characteristics", styles['Subtitle']))
    
    # Group parameters into categories
    param_categories = {
        "Building Envelope": [
            'Floor_Insulation_U-Value_Capped', 
            'Door_Insulation_U-Value_Capped',
            'Roof_Insulation_U-Value_Capped', 
            'Window_Insulation_U-Value_Capped',
            'Wall_Insulation_U-Value_Capped',
            'Air_Change_Rate_Capped',
            'Window_to_Wall_Ratio'
        ],
        "Building Systems": [
            'HVAC_Efficiency',
            'Domestic_Hot_Water_Usage',
            'Lighting_Density',
            'Equipment_Density',
            'Heating_Setpoint_Temperature',
            'Heating_Setback_Temperature',
            'Renewable_Energy_Usage'
        ],
        "General Information": [
            'Building_Type',
            'Total_Building_Area',
            'Weather_File'
        ]
    }
    
    # Create tables for each category
    for category, params in param_categories.items():
        elements.append(Paragraph(category, styles['Heading2']))
        
        param_data = [["Parameter", "Value"]]
        for param in params:
            if param in inputs:
                display_name = get_display_name(param)
                value = inputs[param]
                
                # Format value based on type
                if isinstance(value, float):
                    # Format with appropriate precision
                    if value < 0.1:
                        formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = f"{value:.2f}"
                        
                    # Add units where appropriate
                    if "U-Value" in param:
                        formatted_value += " W/m²K"
                    elif param == "Total_Building_Area":
                        formatted_value += " m²"
                    elif param == "Lighting_Density" or param == "Equipment_Density":
                        formatted_value += " W/m²"
                    elif "Temperature" in param:
                        formatted_value += " °C"
                    elif param == "Air_Change_Rate_Capped":
                        formatted_value += " ACH"
                else:
                    formatted_value = str(value)
                
                param_data.append([display_name, formatted_value])
        
        if len(param_data) > 1:  # Only create table if we have parameters
            param_table = Table(param_data, colWidths=[3*inch, 3*inch])
            param_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 6),
                ('BACKGROUND', (0, 1), (1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ]))
            
            elements.append(param_table)
            elements.append(Spacer(1, 12))
    
    # Add Energy Performance Insights
    elements.append(PageBreak())
    elements.append(Paragraph("Energy Performance Insights", styles['Subtitle']))
    
    insights_text = f"""Based on our analysis, your building demonstrates {rating} energy performance.

Key insights:

1. Energy Efficiency: Your building's EUI of {eui:.1f} kWh/m²/year places it in the {rating} category.

2. Cost Implications: The annual energy cost of ${cost:.2f} represents approximately ${cost/inputs.get('Total_Building_Area', 1000):.2f} per square meter.

3. Environmental Impact: The building produces {emissions:.1f} kg of CO₂ emissions annually, equivalent to approximately {emissions/21:.1f} trees needed to offset these emissions.

4. Comparative Performance: When compared to similar buildings in the industry, your building performs {'above' if 'A' in rating or 'B' in rating else 'below'} average."""
    
    # Fix for 'wrapOn' error: Make sure all strings are properly wrapped in Paragraph objects
    elements.append(Paragraph(insights_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Add cost analysis section if selected
    if options.get('include_cost_analysis', True):
        elements.append(Paragraph("Detailed Cost Analysis", styles['Subtitle']))
        
        area = inputs.get('Total_Building_Area', 1000)
        annual_energy = eui * area  # kWh
        monthly_cost = cost / 12
        daily_cost = cost / 365
        
        cost_analysis_text = f"""Your building's annual energy cost of ${cost:.2f} can be broken down as follows:

• Monthly cost: ${monthly_cost:.2f}
• Daily cost: ${daily_cost:.2f}
• Cost per square meter: ${cost/area:.2f}/m²
• Cost per kWh: ${cost/annual_energy:.2f}/kWh

With targeted energy efficiency improvements, these costs could be reduced by 20-30%."""
        elements.append(Paragraph(cost_analysis_text, styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # 5-year energy cost projection table
        elements.append(Paragraph("5-Year Energy Cost Projection", styles['Heading2']))
        
        projection_data = [["Year", "Projected Cost", "Cumulative Cost"]]
        cumulative = 0
        annual_increase = 0.03  # Assume 3% annual energy price increase
        
        for year in range(1, 6):
            projected_cost = cost * ((1 + annual_increase) ** (year - 1))
            cumulative += projected_cost
            projection_data.append([
                f"Year {year}", 
                f"${projected_cost:.2f}", 
                f"${cumulative:.2f}"
            ])
        
        projection_table = Table(projection_data, colWidths=[1.5*inch, 2*inch, 2.5*inch])
        projection_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (2, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (2, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (2, 0), 6),
            ('BACKGROUND', (0, 1), (2, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (2, -1), 'RIGHT'),
        ]))
        
        elements.append(projection_table)
        elements.append(Spacer(1, 12))
    
    # Add carbon footprint analysis if selected
    if options.get('include_carbon_footprint', True):
        elements.append(PageBreak())
        elements.append(Paragraph("Carbon Footprint Analysis", styles['Subtitle']))
        
        carbon_text = f"""Your building's annual carbon emissions of {emissions:.1f} kg CO₂e can be understood in terms of:

• Equivalent to {emissions/21:.1f} trees needed to offset these emissions annually
• Equivalent to driving approximately {emissions*2.5:.0f} miles in an average passenger car
• Equivalent to {emissions/255:.1f} roundtrip flights from London to Rome
• Equivalent to charging {emissions/0.005:.0f} smartphones

Reducing your building's energy consumption will directly reduce these emissions."""
        elements.append(Paragraph(carbon_text, styles['Normal']))
        elements.append(Spacer(1, 12))
    
    # Add benchmarking if selected
    if options.get('include_benchmark', True):
        elements.append(Paragraph("Industry Benchmarking", styles['Subtitle']))
        
        # Create benchmarking text based on building type
        building_type = inputs.get('Building_Type', 'Office')
        
        benchmark_text = f"""Based on industry standards for {building_type} buildings:

• Top performers (A+ rating): < 50 kWh/m²/yr
• Good performers (A & B ratings): 50-90 kWh/m²/yr
• Average performers (C rating): 90-110 kWh/m²/yr
• Below average performers (D, E, F ratings): > 110 kWh/m²/yr

Your building's EUI of {eui:.1f} kWh/m²/yr places it in the {rating} category."""
        elements.append(Paragraph(benchmark_text, styles['Normal']))
        elements.append(Spacer(1, 12))
    
    # Add recommendations if we have the recommendation engine
    if model is not None and preprocessor is not None and options.get('include_recommendations', True):
        elements.append(PageBreak())
        elements.append(Paragraph("Energy Efficiency Recommendations", styles['Subtitle']))
        
        try:
            from utils.recommendation_engine import get_recommendations
            recommendations = get_recommendations(inputs, eui, model, preprocessor)
            
            # Check if we got recommendations
            if recommendations and any(recommendations.values()):
                rec_intro = """Based on your building's characteristics, we recommend the following energy efficiency measures:"""
                elements.append(Paragraph(rec_intro, styles['Normal']))
                elements.append(Spacer(1, 12))
                
                # Process each category of recommendations
                for category, name in [
                    ('quick_wins', 'Quick Wins'), 
                    ('medium_term', 'Medium-Term Investments'), 
                    ('major_upgrades', 'Major Upgrades')
                ]:
                    if category in recommendations and recommendations[category]:
                        elements.append(Paragraph(name, styles['Heading2']))
                        
                        for i, rec in enumerate(recommendations[category]):
                            # Calculate payback
                            if rec['annual_savings'] > 0:
                                payback = rec['estimated_cost'] / rec['annual_savings']
                                if payback < 1:
                                    payback_text = f"{payback * 12:.1f} months"
                                else:
                                    payback_text = f"{payback:.1f} years"
                            else:
                                payback_text = "N/A"
                            
                            # Create recommendation text without extra indentation
                            rec_text = f"{i+1}. {rec['title']}\n"
                            rec_text += f"• Description: {rec['description']}\n"
                            rec_text += f"• Annual Savings: ${rec['annual_savings']:.0f}\n"
                            rec_text += f"• CO₂ Reduction: {rec['co2_reduction']:.0f} kg/year\n"
                            rec_text += f"• Implementation Cost: ${rec['estimated_cost']:.0f}\n"
                            rec_text += f"• Payback Period: {payback_text}"
                            
                            elements.append(Paragraph(rec_text, styles['Normal']))
                            elements.append(Spacer(1, 6))
            else:
                elements.append(Paragraph("No specific recommendations available for this building configuration.", styles['Normal']))
        except Exception as e:
            elements.append(Paragraph(f"Could not generate recommendations: {str(e)}", styles['Normal']))
    
    # Add footer
    elements.append(PageBreak())
    elements.append(Spacer(1, 24))
    elements.append(Paragraph.wrapOn("This report was generated using the Building Energy Analysis Dashboard.", styles['Normal']))
    elements.append(Paragraph.wrapOn("The predictions are based on a machine learning model and should be used as guidance only.", styles['Italic']))
    elements.append(Paragraph.wrapOn("For detailed energy assessments, please consult a qualified energy assessor.", styles['Italic']))
    
    # Build the PDF
    doc.build(elements)
    
    # Get the PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def generate_enhanced_excel_report(inputs, eui, cost, emissions, rating, options, model=None, preprocessor=None):
    """
    Generate an enhanced Excel report with detailed analysis and interactive elements.
    
    Args:
        inputs (dict): Dictionary of building inputs
        eui (float): Predicted Energy Use Intensity
        cost (float): Predicted annual energy cost
        emissions (float): Predicted annual CO2 emissions
        rating (str): Energy performance rating
        options (dict): Report options and configuration
        model (object, optional): Prediction model for additional analysis
        preprocessor (object, optional): Data preprocessor
        
    Returns:
        bytes: Excel file as bytes
    """
    # Create a BytesIO object to store the Excel file
    output = io.BytesIO()
    
    # Create a new Excel workbook with multiple worksheets
    workbook = xlsxwriter.Workbook(output)
    
    # Create formats
    title_format = workbook.add_format({
        'bold': True,
        'font_color': '#0f766e',
        'font_size': 16,
        'align': 'center',
        'valign': 'vcenter',
        'border': 0
    })
    
    header_format = workbook.add_format({
        'bold': True,
        'font_color': 'white',
        'bg_color': '#0f766e',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })
    
    subheader_format = workbook.add_format({
        'bold': True,
        'font_color': '#0f766e',
        'bg_color': '#e6fffa',
        'border': 1,
        'align': 'left',
        'valign': 'vcenter'
    })
    
    cell_format = workbook.add_format({
        'border': 1,
        'align': 'left',
        'valign': 'vcenter'
    })
    
    number_format = workbook.add_format({
        'border': 1,
        'align': 'right',
        'valign': 'vcenter',
        'num_format': '#,##0.00'
    })
    
    currency_format = workbook.add_format({
        'border': 1,
        'align': 'right',
        'valign': 'vcenter',
        'num_format': '$#,##0.00'
    })
    
    percent_format = workbook.add_format({
        'border': 1,
        'align': 'right',
        'valign': 'vcenter',
        'num_format': '0.0%'
    })
    
    good_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'font_color': 'white',
        'bg_color': '#10b981'  # Green
    })
    
    medium_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'font_color': 'white',
        'bg_color': '#f59e0b'  # Amber
    })
    
    poor_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'font_color': 'white',
        'bg_color': '#ef4444'  # Red
    })
    
    # --- Summary Worksheet ---
    summary_sheet = workbook.add_worksheet('Summary')
    
    # Set column widths
    summary_sheet.set_column('A:A', 30)
    summary_sheet.set_column('B:B', 20)
    summary_sheet.set_column('C:C', 20)
    
    # Title and date
    date_str = datetime.now().strftime("%d %B %Y, %H:%M")
    summary_sheet.merge_range('A1:C1', 'Building Energy Performance Report', title_format)
    summary_sheet.merge_range('A2:C2', f'Generated on: {date_str}', workbook.add_format({'align': 'center'}))
    
    # Key performance metrics
    summary_sheet.merge_range('A4:C4', 'Key Performance Metrics', subheader_format)
    
    summary_sheet.write('A6', 'Metric', header_format)
    summary_sheet.write('B6', 'Value', header_format)
    summary_sheet.write('C6', 'Rating', header_format)
    
    row = 6
    # Define metrics as dictionaries to avoid tuple issues
    metrics = [
        {'metric': 'Energy Use Intensity (EUI)', 'value': f'{eui:.1f} kWh/m²/year', 'rating': rating},
        {'metric': 'Annual Energy Cost', 'value': f'${cost:.2f}', 'rating': ''},
        {'metric': 'CO₂ Emissions', 'value': f'{emissions:.1f} kg CO₂e/year', 'rating': ''},
        {'metric': 'Energy Cost per m²', 'value': f'${cost/inputs.get("Total_Building_Area", 1000):.2f}/m²', 'rating': ''}
    ]
    
    for item in metrics:
        summary_sheet.write(row, 0, item['metric'], cell_format)
        summary_sheet.write(row, 1, item['value'], cell_format)
        
        if item['rating']:
            if 'A+' in item['rating'] or 'A' in item['rating']:
                summary_sheet.write(row, 2, item['rating'], good_format)
            elif 'B' in item['rating']:
                summary_sheet.write(row, 2, item['rating'], medium_format)
            else:
                summary_sheet.write(row, 2, item['rating'], poor_format)
        else:
            summary_sheet.write(row, 2, '', cell_format)
        
        row += 1
    
    # Building data section
    row += 2
    summary_sheet.merge_range(f'A{row}:C{row}', 'Building Characteristics', subheader_format)
    row += 2
    
    summary_sheet.write(row, 0, 'Parameter', header_format)
    summary_sheet.write(row, 1, 'Value', header_format)
    row += 1
    
    # Write building inputs
    for param, value in inputs.items():
        display_name = get_display_name(param)
        
        # Format value based on type
        if isinstance(value, float):
            # Format with appropriate precision
            if value < 0.1:
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = f"{value:.2f}"
                
            # Add units where appropriate
            if "U-Value" in param:
                formatted_value += " W/m²K"
            elif param == "Total_Building_Area":
                formatted_value += " m²"
            elif param == "Lighting_Density" or param == "Equipment_Density":
                formatted_value += " W/m²"
            elif "Temperature" in param:
                formatted_value += " °C"
            elif param == "Air_Change_Rate_Capped":
                formatted_value += " ACH"
        else:
            formatted_value = str(value)
        
        summary_sheet.write(row, 0, display_name, cell_format)
        summary_sheet.write(row, 1, formatted_value, cell_format)
        row += 1
    
    # --- Cost Analysis Worksheet ---
    if options.get('include_data_tables', True):
        cost_sheet = workbook.add_worksheet('Cost Analysis')
        
        # Set column widths
        cost_sheet.set_column('A:A', 25)
        cost_sheet.set_column('B:D', 15)
        
        # Title
        cost_sheet.merge_range('A1:D1', 'Energy Cost Analysis', title_format)
        
        # Current costs section
        cost_sheet.merge_range('A3:D3', 'Current Energy Costs', subheader_format)
        
        cost_sheet.write('A5', 'Cost Metric', header_format)
        cost_sheet.write('B5', 'Value', header_format)
        
        row = 5
        area = inputs.get('Total_Building_Area', 1000)
        annual_energy = eui * area  # kWh
        monthly_cost = cost / 12
        
        costs = [
            {'metric': 'Annual Energy Consumption', 'value': f'{annual_energy:,.0f} kWh'},
            {'metric': 'Energy Unit Cost', 'value': f'${cost/annual_energy:.4f}/kWh'},
            {'metric': 'Annual Energy Cost', 'value': f'${cost:,.2f}'},
            {'metric': 'Monthly Energy Cost', 'value': f'${monthly_cost:,.2f}'},
            {'metric': 'Cost per Square Meter', 'value': f'${cost/area:,.2f}/m²'}
        ]
        
        for cost_item in costs:
            cost_sheet.write(row, 0, cost_item['metric'], cell_format)
            cost_sheet.write(row, 1, cost_item['value'], cell_format)
            row += 1
        
        # Cost projection section
        row += 2
        cost_sheet.merge_range(f'A{row}:D{row}', 'Long-Term Cost Projection', subheader_format)
        row += 2
        
        cost_sheet.write(row, 0, 'Year', header_format)
        cost_sheet.write(row, 1, 'Annual Cost', header_format)
        cost_sheet.write(row, 2, 'Cumulative Cost', header_format)
        row += 1
        
        # Generate cost projection based on options
        annual_increase = options.get('energy_price_increase', 0.025)  # Default 2.5%
        projection_years = options.get('projection_years', 10)  # Default 10 years
        
        cumulative = 0
        for year in range(1, projection_years + 1):
            projected_cost = cost * ((1 + annual_increase) ** (year - 1))
            cumulative += projected_cost
            
            cost_sheet.write(row, 0, f'Year {year}', cell_format)
            cost_sheet.write(row, 1, projected_cost, currency_format)
            cost_sheet.write(row, 2, cumulative, currency_format)
            row += 1
    
    # --- Environmental Impact Worksheet ---
    if options.get('include_data_tables', True):
        env_sheet = workbook.add_worksheet('Environmental Impact')
        
        # Set column widths
        env_sheet.set_column('A:A', 25)
        env_sheet.set_column('B:C', 15)
        
        # Title
        env_sheet.merge_range('A1:C1', 'Carbon Footprint Analysis', title_format)
        
        # Carbon metrics section
        env_sheet.merge_range('A3:C3', 'Carbon Emissions Metrics', subheader_format)
        
        env_sheet.write('A5', 'Metric', header_format)
        env_sheet.write('B5', 'Value', header_format)
        
        row = 5
        carbon_metrics = [
            {'metric': 'Annual CO₂ Emissions', 'value': f'{emissions:,.1f} kg'},
            {'metric': 'Equivalent Trees Needed', 'value': f'{emissions/21:,.1f} trees'},
            {'metric': 'Equivalent Car Miles', 'value': f'{emissions*2.5:,.0f} miles'},
            {'metric': 'Equivalent Flights', 'value': f'{emissions/255:,.1f} flights'},
            {'metric': 'Equivalent Smartphone Charges', 'value': f'{emissions/0.005:,.0f} charges'}
        ]
        
        for item in carbon_metrics:
            env_sheet.write(row, 0, item['metric'], cell_format)
            env_sheet.write(row, 1, item['value'], cell_format)
            row += 1
    
    # --- ROI Calculator Worksheet ---
    if options.get('include_roi_calculator', True):
        roi_sheet = workbook.add_worksheet('ROI Calculator')
        
        # Set column widths
        roi_sheet.set_column('A:A', 25)
        roi_sheet.set_column('B:E', 15)
        
        # Title
        roi_sheet.merge_range('A1:E1', 'Return on Investment Calculator', title_format)
        
        # ROI Calculator
        roi_sheet.merge_range('A3:E3', 'Energy Efficiency Investment Analysis', subheader_format)
        
        roi_sheet.write('A5', 'Input Parameters', subheader_format)
        
        row = 6
        roi_sheet.write(row, 0, 'Initial Investment ($)', cell_format)
        roi_sheet.write(row, 1, 10000, number_format)  # Default value
        row += 1
        
        roi_sheet.write(row, 0, 'Annual Energy Savings ($)', cell_format)
        roi_sheet.write(row, 1, 2000, number_format)  # Default value
        row += 1
        
        roi_sheet.write(row, 0, 'Annual Energy Price Increase (%)', cell_format)
        roi_sheet.write(row, 1, options.get('energy_price_increase', 0.025), percent_format)
        row += 1
        
        roi_sheet.write(row, 0, 'Discount Rate (%)', cell_format)
        roi_sheet.write(row, 1, options.get('discount_rate', 0.035), percent_format)
        row += 1
        
        roi_sheet.write(row, 0, 'Investment Lifetime (years)', cell_format)
        roi_sheet.write(row, 1, 20, number_format)  # Default value
        row += 2
        
        # Results section
        roi_sheet.write(row, 0, 'Results', subheader_format)
        row += 1
        
        # Add formulas for ROI calculation
        roi_sheet.write(row, 0, 'Simple Payback Period (years)', cell_format)
        roi_sheet.write_formula(row, 1, '=B7/B8', number_format)
        row += 1
        
        roi_sheet.write(row, 0, 'Return on Investment (%)', cell_format)
        roi_sheet.write_formula(row, 1, '=(B8*B11)/B7*100', percent_format)
        row += 1
    
    # Close the workbook and return the data
    workbook.close()
    
    # Get the Excel data
    excel_data = output.getvalue()
    output.close()
    
    return excel_data