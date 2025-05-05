"""
Utility module for feature name handling and display.
Provides functionality for converting between raw feature names and
user-friendly display names without the "_Capped" suffix.
"""

# Mapping from raw feature names to human-readable display names
FEATURE_DISPLAY_NAMES = {
    'Floor_Insulation_U-Value_Capped': 'Floor Insulation U-Value',
    'Door_Insulation_U-Value_Capped': 'Door Insulation U-Value',
    'Roof_Insulation_U-Value_Capped': 'Roof Insulation U-Value',
    'Window_Insulation_U-Value_Capped': 'Window Insulation U-Value',
    'Wall_Insulation_U-Value_Capped': 'Wall Insulation U-Value',
    'Air_Change_Rate_Capped': 'Air Change Rate',
    'HVAC_Efficiency': 'HVAC Efficiency',
    'Domestic_Hot_Water_Usage': 'Domestic Hot Water Usage',
    'Lighting_Density': 'Lighting Density',
    'Occupancy_Level': 'Occupancy Level',
    'Equipment_Density': 'Equipment Density',
    'Heating_Setpoint_Temperature': 'Heating Setpoint Temperature',
    'Heating_Setback_Temperature': 'Heating Setback Temperature',
    'Window_to_Wall_Ratio': 'Window to Wall Ratio',
    'Total_Building_Area': 'Total Building Area',
    'Weather_File': 'Weather Scenario',
    'Building_Type': 'Building Type',
    'Building_Orientation': 'Building Orientation',
    'Renewable_Energy_Usage': 'Renewable Energy Usage'
}

# Reverse mapping (from display names to feature names)
DISPLAY_TO_FEATURE_NAMES = {v: k for k, v in FEATURE_DISPLAY_NAMES.items()}

# Default features to display in dropdowns
DEFAULT_FEATURES = DISPLAY_TO_FEATURE_NAMES

def get_display_name(feature_name):
    """
    Convert a raw feature name to a user-friendly display name.
    
    Args:
        feature_name (str): The raw feature name from the model
        
    Returns:
        str: A user-friendly display name without "_Capped" suffix
    """
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)

def get_feature_name(display_name):
    """
    Convert a display name back to the original feature name.
    
    Args:
        display_name (str): The user-friendly display name
        
    Returns:
        str: The original feature name with proper suffix
    """
    return DISPLAY_TO_FEATURE_NAMES.get(display_name, display_name)

def get_feature_display_names():
    """
    Get a dictionary of all feature display names.
    
    Returns:
        dict: Mapping of feature names to display names
    """
    return FEATURE_DISPLAY_NAMES

def get_feature_names():
    """
    Get a list of all raw feature names.
    
    Returns:
        list: List of all raw feature names
    """
    return list(FEATURE_DISPLAY_NAMES.keys())

def get_display_names():
    """
    Get a list of all display names.
    
    Returns:
        list: List of all display names
    """
    return list(FEATURE_DISPLAY_NAMES.values())
