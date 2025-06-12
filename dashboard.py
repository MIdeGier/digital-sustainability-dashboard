import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Tuple
from maturity_scoring import process_survey_response
import difflib

# Set page config
st.set_page_config(
    page_title="Twin Transition Maturity Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and Roboto font
st.markdown("""
<style>
    /* Apply Roboto to all elements */
    * {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Style the sidebar */
    section[data-testid="stSidebar"] {
        background-color: #295E4B !important;
    }
    
    /* Reset sidebar padding */
    section[data-testid="stSidebar"] > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Style sidebar image container */
    section[data-testid="stSidebar"] [data-testid="stImage"] {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        transform: scale(1.6) !important;
        transform-origin: top center !important;
    }
    
    /* Style the actual image */
    section[data-testid="stSidebar"] [data-testid="stImage"] img {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        border-radius: 0 !important;
        display: block !important;
    }
    
    /* Remove all margins and padding from the image containers */
    section[data-testid="stSidebar"] [data-testid="stImage"] > div,
    section[data-testid="stSidebar"] [data-testid="stImage"] > div > div {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }
    
    /* Ensure proper stacking context and remove margins */
    section[data-testid="stSidebar"] .element-container {
        position: relative !important;
        z-index: 1 !important;
        color: white !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }

    /* Remove any potential wrapper margins */
    section[data-testid="stSidebar"] > div > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Style sidebar text and elements */
    section[data-testid="stSidebar"] label {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
    }
    
    /* Style sidebar title */
    section[data-testid="stSidebar"] .element-container div[data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] h1 {
        color: white !important;
    }
    
    /* Style sidebar radio buttons */
    section[data-testid="stSidebar"] [role="radiogroup"] {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white !important;
    }
    
    /* Change radio button color to match sidebar */
    section[data-testid="stSidebar"] .stRadio > label > div[role="radiogroup"] > label > div:first-child {
        background-color: #295E4B !important;
        border-color: white !important;
    }
    
    /* Style for selected radio button in sidebar */
    section[data-testid="stSidebar"] .stRadio > label > div[role="radiogroup"] > label > div[data-baseweb="radio"] > div {
        background-color: white !important;
    }
    
    /* Style selectbox in sidebar */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #295E4B !important;
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div > div > div {
        color: white !important;
    }
    
    /* Custom styles for expanders */
    /* Base styles for all expanders */
    [data-testid="stExpander"] {
        border-radius: 0.5rem !important;
        margin: 0.3rem 0 !important;
    }

    /* Environmental section expanders */
    .environmental [data-testid="stExpander"] {
        border: 3px solid rgba(111, 166, 56, 0.8) !important;
    }
    
    /* Social section expanders */
    .social [data-testid="stExpander"] {
        border: 3px solid rgba(92, 169, 148, 0.8) !important;
    }
    
    /* Governance section expanders */
    .governance [data-testid="stExpander"] {
        border: 3px solid rgba(168, 228, 231, 0.8) !important;
    }

    /* Remove default styles from expander elements */
    [data-testid="stExpander"] > div {
        border: none !important;
    }
    
    [data-testid="stExpanderContent"] {
        border-top: none !important;
    }

    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .highlight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .category-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .category-box h4 {
        margin-top: 0;
        color: #233329 !important;
    }
    div.metric-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    div.category-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    div.score-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .score-box h4 {
        margin: 0 0 0.5rem 0;
        color: #233329 !important;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .score-line {
        margin: 0.3rem 0;
        color: #233329 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and validate the data files"""
    try:
        survey_data = pd.read_csv('Survey_data.csv')
        qa_weights = pd.read_csv('DST_Excel_QAweights.csv')
        
        # Clean survey data - remove empty rows
        survey_data = survey_data.dropna(how='all')
        survey_data = survey_data[survey_data.iloc[:,0].notna()]  # Remove rows where ID is null
        
        # Filter out header rows that might be mixed in data
        survey_data = survey_data[survey_data.iloc[:,0] != 'ID']
        survey_data = survey_data[~survey_data.iloc[:,0].astype(str).str.contains('people dimension', na=False)]
        
        # Convert ID column to numeric and sort
        survey_data.iloc[:,0] = pd.to_numeric(survey_data.iloc[:,0], errors='coerce')
        survey_data = survey_data.sort_values(by=survey_data.columns[0])
        
        # Reset index after filtering
        survey_data = survey_data.reset_index(drop=True)
        
        return survey_data, qa_weights, None
    except Exception as e:
        return None, None, str(e)

def parse_survey_columns(survey_data: pd.DataFrame) -> Dict:
    """Parse survey column names to understand the data structure"""
    columns = survey_data.columns.tolist()
    parsed_structure = {
        'basic_info': [],
        'sustainability_importance': [],
        'inclusion_questions': [],
        'weight_questions': [],
        'maturity_questions': []
    }
    
    for col in columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['id', 'time', 'email', 'name']):
            parsed_structure['basic_info'].append(col)
        elif 'how important' in col_lower and any(x in col_lower for x in ['environmental', 'social', 'governance']):
            parsed_structure['sustainability_importance'].append(col)
        elif 'do you want to include' in col_lower:
            parsed_structure['inclusion_questions'].append(col)
        elif 'how important' in col_lower and ('1 =' in col or 'essential' in col):
            parsed_structure['weight_questions'].append(col)
        elif any(x in col_lower for x in ['people dimension', 'process dimension', 'policy dimension']):
            parsed_structure['maturity_questions'].append(col)
    
    return parsed_structure

def extract_impact_from_question(question: str) -> str:
    """Extract the sustainability impact from question text"""
    question_lower = question.lower()
    
    impact_mapping = {
        'energy': 'energy consumption',
        'ghg': 'GHG emissions',
        'scope 1': 'GHG emissions',
        'air pollutant': 'air pollution',
        'hazardous chemical': 'hazardous chemical use',
        'resource': 'resource consumption',
        'waste': 'waste',
        'diversity': 'diversity & inclusion',
        'training': 'training & skills',
        'health': 'health & safety',
        'work-life': 'work-life balance',
        'ethical': 'ethical behaviour and culture'
    }
    
    for keyword, impact in impact_mapping.items():
        if keyword in question_lower:
            return impact
    
    return 'unknown'

def calculate_maturity_scores(survey_response: pd.Series, qa_weights: pd.DataFrame) -> Dict:
    """Calculate maturity scores from a single survey response using the QA weights matrix"""
    if survey_response.empty:
        return {}
    
    # Parse column structure
    structure = parse_survey_columns(survey_response.to_frame().T)
    
    # ESG categorization with impact categories
    esg_mapping = {
        'energy consumption': ('E', 'Energy'),
        'GHG emissions': ('E', 'GHG'),
        'air pollution': ('E', 'Air'),
        'hazardous chemical use': ('E', 'Chemical'),
        'resource consumption': ('E', 'Resource'),
        'waste': ('E', 'Waste'),
        'diversity & inclusion': ('S', 'Diversity'),
        'training & skills': ('S', 'Training'),
        'health & safety': ('S', 'Health'),
        'work-life balance': ('S', 'Work-life'),
        'ethical behaviour and culture': ('G', 'Ethics'),
        'ethical culture & behavior': ('G', 'Ethics')
    }
    
    # Initialize results structure to store raw scores and weights
    results = {
        'current': {
            'people': {'E': [], 'S': [], 'G': []},
            'process': {'E': [], 'S': [], 'G': []},
            'policy': {'E': [], 'S': [], 'G': []}
        },
        'future': {
            'people': {'E': [], 'S': [], 'G': []},
            'process': {'E': [], 'S': [], 'G': []},
            'policy': {'E': [], 'S': [], 'G': []}
        }
    }
    
    # Get weights for each impact
    weights = {}
    for question in structure['weight_questions']:
        if question in survey_response.index:
            impact = extract_impact_from_question(question)
            try:
                weight = int(survey_response[question])
                weights[impact] = weight
            except (ValueError, TypeError):
                weights[impact] = 1
    
    # Process maturity questions
    for question in structure['maturity_questions']:
        try:
            if question not in survey_response.index:
                continue
            
            response_text = str(survey_response[question]).strip()
            if response_text == 'nan' or response_text == '' or pd.isna(survey_response[question]):
                continue
            
            impact = extract_impact_from_question(question)
            if impact not in esg_mapping:
                continue
                
            esg_category, subcategory = esg_mapping[impact]
            
            # Determine dimension and timeframe
            question_lower = question.lower()
            if 'people dimension' in question_lower:
                dimension = 'people'
                if 'current situation' in question_lower:
                    qa_col = 'People current'
                    timeframe = 'current'
                else:
                    qa_col = 'People future'
                    timeframe = 'future'
            elif 'process dimension' in question_lower:
                dimension = 'process'
                if 'current situation' in question_lower:
                    qa_col = 'Process current'
                    timeframe = 'current'
                else:
                    qa_col = 'Process future'
                    timeframe = 'future'
            elif 'policy dimension' in question_lower:
                dimension = 'policy'
                if 'current situation' in question_lower:
                    qa_col = 'Policy current'
                    timeframe = 'current'
                else:
                    qa_col = 'Policy future'
                    timeframe = 'future'
            else:
                continue
            
            # Find matching row in qa_weights
            matching_rows = qa_weights[qa_weights[qa_col].notna()]
            
            if esg_category == 'G':
                matching_rows = matching_rows[matching_rows['ESG'] == 'Governance']
            else:
                if impact != 'unknown':
                    matching_rows = matching_rows[
                        (matching_rows['ESRS impact'].str.lower() == impact.lower()) |
                        (matching_rows['ESG'].str.lower() == impact.lower())
                    ]
            
            # Match response text
            response_matches = matching_rows[matching_rows[qa_col].str.strip() == response_text.strip()]
            
            if response_matches.empty and len(response_text) > 50:
                response_matches = matching_rows[
                    matching_rows[qa_col].str.strip().str.startswith(response_text[:50].strip())
                ]
            
            if not response_matches.empty:
                maturity_level = response_matches['Maturity level'].iloc[0]
                try:
                    maturity_level = int(maturity_level)
                    weight = weights.get(impact, 1)
                    
                    if esg_category == 'G':
                        # For governance, store maturity level directly
                        results[timeframe][dimension][esg_category].append(maturity_level)
                    else:
                        # For E and S, store both maturity level and weight
                        results[timeframe][dimension][esg_category].append((maturity_level, weight))
                except (ValueError, TypeError):
                    continue
                
        except Exception as e:
            continue
    
    # Calculate final scores
    final_results = {
        'current': {'people': {}, 'process': {}, 'policy': {}},
        'future': {'people': {}, 'process': {}, 'policy': {}}
    }
    
    for timeframe in ['current', 'future']:
        for dimension in ['people', 'process', 'policy']:
            final_results[timeframe][dimension] = {'E': 0, 'S': 0, 'G': 0}
            
            # Calculate E and S scores using weighted averages
            for esg in ['E', 'S']:
                scores = results[timeframe][dimension][esg]
                if scores:
                    total_weighted_score = sum(maturity * weight for maturity, weight in scores)
                    total_weights = sum(weight for _, weight in scores)
                    
                    if total_weights > 0:
                        avg_score = total_weighted_score / total_weights
                        # Round to nearest integer
                        final_results[timeframe][dimension][esg] = round(avg_score)
            
            # For governance, use the direct maturity level (already an integer)
            if results[timeframe][dimension]['G']:
                final_results[timeframe][dimension]['G'] = results[timeframe][dimension]['G'][0]
    
    return final_results

def map_response_to_maturity(response_text: str) -> int:
    """Map survey response text to maturity level 1-5"""
    if not response_text or response_text.strip() == '':
        return 0
    
    response_lower = response_text.lower()
    
    # Keywords indicating different maturity levels
    level_indicators = {
        5: ['autonomous', 'ai-driven', 'self-learning', 'adaptive', 'fully autonomous', 'continuous learning'],
        4: ['prescriptive', 'recommend', 'advanced analytics', 'automatically implement', 'real-time optimization'],
        3: ['predictive', 'forecast', 'system-generated alerts', 'proactive', 'anticipate'],
        2: ['dashboard', 'diagnostic', 'real-time data', 'analyze', 'identify', 'monitor'],
        1: ['basic', 'manual', 'limited', 'minimal', 'awareness', 'historical']
    }
    
    # Check for highest maturity level first
    for level in range(5, 0, -1):
        if any(keyword in response_lower for keyword in level_indicators[level]):
            return level
    
    return 1  # Default to level 1 if no clear indicators

def create_radar_chart(results: Dict, level: str = 'overview') -> go.Figure:
    """Create radar chart visualization"""
    if level == 'overview':
        # Calculate overall scores for each dimension
        dimensions = ['People', 'Process', 'Policy']
        current_values = []
        future_values = []
        
        for dim in ['people', 'process', 'policy']:
            current_scores = []
            future_scores = []
            
            for esg in ['E', 'S', 'G']:
                try:
                    current_score = float(results['current'][dim][esg])
                    future_score = float(results['future'][dim][esg])
                    
                    if current_score > 0:
                        current_scores.append(current_score)
                    if future_score > 0:
                        future_scores.append(future_score)
                except (KeyError, TypeError, ValueError):
                    continue
            
            # Round the averages to integers for visualization
            current_avg = round(np.mean(current_scores)) if current_scores else 0
            future_avg = round(np.mean(future_scores)) if future_scores else 0
            
            current_values.append(current_avg)
            future_values.append(future_avg)

        # Create a simplified circular visualization
        fig = go.Figure()

        # Define angles for each section (120 degrees each)
        angles = {
            'People': (-30, 90),     # Left section
            'Process': (90, 210),    # Top section
            'Policy': (210, 330)     # Right section
        }

        # Colors for overview
        colors = {
            'People': 'rgba(111, 166, 56, 0.6)',   # #6FA638 with 0.6 opacity
            'Process': 'rgba(92, 169, 148, 0.6)',  # #5CA994 with 0.6 opacity
            'Policy': 'rgba(168, 228, 231, 0.6)'   # #A8E4E7 with 0.6 opacity
        }

        # Add concentric circles
        for i in range(5):
            radius = (i + 1) * 0.2
            circle_points = np.linspace(0, 2*np.pi, 100)
            x = radius * np.cos(circle_points)
            y = radius * np.sin(circle_points)
            
            # Add circle
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='lightgray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add section dividing lines with arrows
        # Define the angles for the three dividing lines (in radians)
        division_angles = [
            -np.pi/6,    # -30 degrees (People-Process boundary)
            np.pi/2,     # 90 degrees (Process-Policy boundary)
            7*np.pi/6    # 210 degrees (Policy-People boundary)
        ]
        
        for angle in division_angles:
            # Calculate end point for the line (from center to level 5)
            radius = 1.0  # Exactly at level 5 (5 * 0.2)
            end_x = radius * np.cos(angle)
            end_y = radius * np.sin(angle)
            
            # Draw the main line from center to level 5
            fig.add_trace(go.Scatter(
                x=[0, end_x],
                y=[0, end_y],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add arrow at the outer end
            arrow_length = 0.08  # Length of arrow head
            arrow_width = 20  # Angle of arrow head in degrees
            
            # Calculate arrow angles (reversed direction for outward pointing)
            arrow_angle = angle  # Base angle for the arrow
            left_angle = arrow_angle - np.pi + np.radians(arrow_width)
            right_angle = arrow_angle - np.pi - np.radians(arrow_width)
            
            # Calculate arrow points (starting from the end point)
            arrow_left_x = end_x + arrow_length * np.cos(left_angle)
            arrow_left_y = end_y + arrow_length * np.sin(left_angle)
            arrow_right_x = end_x + arrow_length * np.cos(right_angle)
            arrow_right_y = end_y + arrow_length * np.sin(right_angle)
            
            # Add arrow head (pointing outward)
            fig.add_trace(go.Scatter(
                x=[arrow_left_x, end_x, arrow_right_x],
                y=[arrow_left_y, end_y, arrow_right_y],
                mode='lines',
                line=dict(color='black', width=1),
                fill='toself',
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add filled sections for current state
        for i, (dim, (start_angle, end_angle)) in enumerate(angles.items()):
            # Convert angles to radians
            start_rad = start_angle * np.pi/180
            end_rad = end_angle * np.pi/180
            
            # Current state (filled) - using exact integer values
            r = current_values[i] / 5  # current_values are now rounded integers
            if r > 0:
                theta = np.linspace(start_rad, end_rad, 50)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                x = np.concatenate([[0], x, [0]])
                y = np.concatenate([[0], y, [0]])
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    fill='toself',
                    fillcolor=colors[dim],
                    line=dict(color=colors[dim].replace('0.6', '1')),
                    name=f'Current {dim}',
                    showlegend=False
                ))

            # Future state (dashed outline) - using exact integer values
            r = future_values[i] / 5  # future_values are now rounded integers
            if r > 0:
                theta = np.linspace(start_rad, end_rad, 50)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                x = np.concatenate([[0], x, [0]])
                y = np.concatenate([[0], y, [0]])
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name=f'Future {dim}',
                    showlegend=False
                ))

        # Add dimension labels at the middle of each section on the outside
        middle_angles = {
            'People': np.radians(30),      # 30 degrees
            'Process': np.radians(150),    # 150 degrees
            'Policy': np.radians(270)       # 270 degrees
        }
        
        for cat, angle in middle_angles.items():
            # Position labels at 1.3 times the radius (outside the circles)
            radius = 1.3
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Adjust text alignment based on position
            if cat == 'People':
                align = 'left'
                x_offset = -0.1
            elif cat == 'Process':
                align = 'right'
                x_offset = 0.1
            else:  # Policy
                align = 'center'
                x_offset = 0
                y -= 0.1  # Move slightly down
            
            fig.add_annotation(
                x=x + x_offset,
                y=y,
                text=cat,
                showarrow=False,
                font=dict(
                    size=14,
                    color='black',
                    family='Roboto, -apple-system, BlinkMacSystemFont, sans-serif'
                ),
                xanchor='center' if cat == 'Policy' else ('left' if cat == 'People' else 'right'),
                yanchor='bottom' if cat == 'Policy' else 'middle'
            )

        # Update layout with Roboto font
        fig.update_layout(
            showlegend=False,
            width=600,
            height=600,
            title=dict(
                text="Results for your total ESG score",
                x=0.5,
                font=dict(
                    size=18,
                    family='Roboto, -apple-system, BlinkMacSystemFont, sans-serif'
                )
            ),
            xaxis=dict(
                range=[-1.5, 1.5],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                range=[-1.5, 1.5],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='white',
            font=dict(
                family='Roboto, -apple-system, BlinkMacSystemFont, sans-serif'
            )
        )

        return fig
    else:
        # ESG breakdown for specific dimension
        categories = ['Environmental', 'Social', 'Governance']
        current_values = []
        future_values = []
        
        dim_key = level.lower()
        for esg in ['E', 'S', 'G']:
            try:
                current = float(results['current'][dim_key][esg])
                future = float(results['future'][dim_key][esg])
                
                current_values.append(current)
                future_values.append(future)
            except (KeyError, TypeError, ValueError):
                current_values.append(0)
                future_values.append(0)

        fig = go.Figure()

        # Add concentric circles
        for i in range(5):
            radius = (i + 1) * 0.2
            circle_points = np.linspace(0, 2*np.pi, 100)
            x = radius * np.cos(circle_points)
            y = radius * np.sin(circle_points)
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='lightgray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Define angles for each section (120 degrees each)
        angles = {
            'Social': (-30, 90),     # Left section
            'Environmental': (90, 210),           # Top section
            'Governance': (210, 330)       # Right section
        }

        # Default box_colors to prevent 'not defined' errors
        box_colors = {
            'Environmental': 'rgba(200, 200, 200, 0.6)',
            'Social': 'rgba(200, 200, 200, 0.6)',
            'Governance': 'rgba(200, 200, 200, 0.6)'
        }

        # Add section dividing lines with arrows
        division_angles = [
            -np.pi/6,    # -30 degrees
            np.pi/2,     # 90 degrees
            7*np.pi/6    # 210 degrees
        ]
        
        for angle in division_angles:
            # Calculate end point for the line (from center to level 5)
            radius = 1.0  # Exactly at level 5 (5 * 0.2)
            end_x = radius * np.cos(angle)
            end_y = radius * np.sin(angle)
            
            # Draw the main line from center to level 5
            fig.add_trace(go.Scatter(
                x=[0, end_x],
                y=[0, end_y],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add arrow at the outer end
            arrow_length = 0.08
            arrow_width = 20  # degrees
            
            # Calculate arrow angles (reversed direction for outward pointing)
            arrow_angle = angle
            left_angle = arrow_angle - np.pi + np.radians(arrow_width)
            right_angle = arrow_angle - np.pi - np.radians(arrow_width)
            
            # Calculate arrow points (starting from the end point)
            arrow_left_x = end_x + arrow_length * np.cos(left_angle)
            arrow_left_y = end_y + arrow_length * np.sin(left_angle)
            arrow_right_x = end_x + arrow_length * np.cos(right_angle)
            arrow_right_y = end_y + arrow_length * np.sin(right_angle)
            
            # Add arrow head (pointing outward)
            fig.add_trace(go.Scatter(
                x=[arrow_left_x, end_x, arrow_right_x],
                y=[arrow_left_y, end_y, arrow_right_y],
                mode='lines',
                line=dict(color='black', width=1),
                fill='toself',
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add filled sections for current and future state
        for i, (cat, (start_angle, end_angle)) in enumerate(angles.items()):
            # Convert angles to radians
            start_rad = start_angle * np.pi/180
            end_rad = end_angle * np.pi/180
            
            # Current state (filled)
            r = current_values[i] / 5
            if r > 0:
                theta = np.linspace(start_rad, end_rad, 50)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                x = np.concatenate([[0], x, [0]])
                y = np.concatenate([[0], y, [0]])
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    fill='toself',
                    fillcolor=box_colors[cat],
                    line=dict(color=box_colors[cat].replace('0.6', '1')),
                    name=f'Current {cat}',
                    showlegend=False
                ))

            # Future state (dashed outline)
            r = future_values[i] / 5
            if r > 0:
                theta = np.linspace(start_rad, end_rad, 50)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                x = np.concatenate([[0], x, [0]])
                y = np.concatenate([[0], y, [0]])
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name=f'Future {cat}',
                    showlegend=False
                ))

        # Add dimension labels at the middle of each section on the outside
        middle_angles = {
            'Social': np.radians(30),      # 30 degrees
            'Environmental': np.radians(150),           # 150 degrees
            'Governance': np.radians(270)        # 270 degrees
        }
        
        for cat, angle in middle_angles.items():
            # Position labels at 1.3 times the radius (outside the circles)
            radius = 1.3
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Adjust text alignment based on position
            if cat == 'Social':  # Was Environmental
                align = 'left'
                x_offset = -0.1
            elif cat == 'Environmental':  # Was Social
                align = 'right'
                x_offset = 0.1
            else:  # Governance
                align = 'center'
                x_offset = 0
                y -= 0.1  # Move slightly down
            
            fig.add_annotation(
                x=x + x_offset,
                y=y,
                text=cat,
                showarrow=False,
                font=dict(
                    size=14,
                    color='black',
                    family='Roboto, -apple-system, BlinkMacSystemFont, sans-serif'
                ),
                xanchor='center' if cat == 'Governance' else ('left' if cat == 'Social' else 'right'),
                yanchor='bottom' if cat == 'Governance' else 'middle'
            )

        # Update layout
        fig.update_layout(
            showlegend=False,
            width=600,
            height=600,
            title=dict(
                text=f"{level.title()} dimension ESG breakdown",
                x=0.5,
                font=dict(
                    size=18,
                    family='Roboto, -apple-system, BlinkMacSystemFont, sans-serif'
                )
            ),
            xaxis=dict(
                range=[-1.5, 1.5],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                range=[-1.5, 1.5],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='white',
            font=dict(
                family='Roboto, -apple-system, BlinkMacSystemFont, sans-serif'
            )
        )

        return fig

def create_gap_analysis(results: Dict) -> pd.DataFrame:
    """Create gap analysis between current and future states"""
    gap_data = []
    
    dimensions = ['People', 'Process', 'Policy']
    esg_categories = ['Environmental', 'Social', 'Governance']
    esg_short = ['E', 'S', 'G']
    
    for i, dim in enumerate(['people', 'process', 'policy']):
        for j, esg in enumerate(esg_short):
            try:
                current = float(results['current'][dim][esg])
                future = float(results['future'][dim][esg])
                gap = future - current
                
                gap_data.append({
                    'Dimension': dimensions[i],
                    'ESG Category': esg_categories[j],
                    'Current': current,
                    'Future Goal': future,
                    'Gap': gap,
                    'Priority': 'High' if gap >= 2 else 'Medium' if gap >= 1 else 'Low'
                })
            except (KeyError, TypeError, ValueError) as e:
                print(f"Error processing {dimensions[i]} - {esg_categories[j]}: {str(e)}")
                gap_data.append({
                    'Dimension': dimensions[i],
                    'ESG Category': esg_categories[j],
                    'Current': 0,
                    'Future Goal': 0,
                    'Gap': 0,
                    'Priority': 'Low'
                })
    
    return pd.DataFrame(gap_data)

def extract_process_maturity(survey_response: pd.Series, qa_weights: pd.DataFrame) -> Dict[str, int]:
    """Extract maturity levels for process current dimension"""
    # Filter QA weights for process current dimension
    process_weights = qa_weights[qa_weights.iloc[:,0].str.lower() == 'process current']
    
    # Initialize results
    maturity_levels = {}
    
    # Process each mapping
    for _, weight_row in process_weights.iterrows():
        question = weight_row['Question']
        expected_answer = weight_row['Answer']
        esrs_impact = weight_row['ESRS impact']
        
        # Find matching question in survey response
        best_match = None
        best_ratio = 0
        
        for col in survey_response.index:
            ratio = difflib.SequenceMatcher(None, str(col).lower(), str(question).lower()).ratio()
            if ratio >= 0.95 and ratio > best_ratio:
                best_match = col
                best_ratio = ratio
        
        if best_match:
            # Compare answers
            response_answer = str(survey_response[best_match]).strip()
            expected_answer = str(expected_answer).strip()
            
            # Check for exact match first, then fuzzy match
            if response_answer == expected_answer or \
               difflib.SequenceMatcher(None, response_answer.lower(), expected_answer.lower()).ratio() >= 0.85:
                maturity_levels[esrs_impact] = int(weight_row['Maturity level'])
    
    return maturity_levels

def get_technology_recommendations(maturity_levels: Dict[str, int], tech2esrs_df: pd.DataFrame) -> List[Dict]:
    """Generate technology recommendations based on maturity levels"""
    recommendations = []
    
    # Load complexity data
    try:
        complexity_df = pd.read_csv('DST_Excel_Complexity.csv', index_col=0)
    except Exception as e:
        st.error(f"Error loading complexity data: {str(e)}")
        return recommendations
    
    for esrs_impact, current_level in maturity_levels.items():
        # Skip if already at max level
        if current_level >= 5:
            continue
            
        next_level = current_level + 1
        level_col = next_level  # Column names are numbers in the CSV
        
        # Find matching technologies
        matching_techs = tech2esrs_df[
            tech2esrs_df['ESRS number'].str.lower() == esrs_impact.lower()
        ]
        
        for _, tech_row in matching_techs.iterrows():
            tech_name = tech_row.iloc[1]  # Technology name is in second column
            tech_desc = tech_row[level_col]
            
            if pd.notna(tech_desc) and str(tech_desc).strip():
                # Get complexity for this technology at this maturity level
                complexity = "Unknown"
                try:
                    # Find the correct complexity column based on ESRS impact and level
                    complexity_cols = [col for col in complexity_df.columns if col.startswith(esrs_impact)]
                    if complexity_cols:
                        # Get the column for this maturity level
                        level_indices = range(len(complexity_cols))
                        col_index = next_level - 1  # Convert level to 0-based index
                        if col_index < len(complexity_cols):
                            complexity_col = complexity_cols[col_index]
                            complexity = complexity_df.loc[tech_name, complexity_col]
                            if pd.isna(complexity) or complexity == "N/A":
                                complexity = "Not applicable at this level"
                except Exception as e:
                    print(f"Error getting complexity for {tech_name}: {str(e)}")
                    complexity = "Unknown"
                
                recommendations.append({
                    'esrs_impact': esrs_impact,
                    'current_maturity_level': current_level,
                    'recommended_level': next_level,
                    'recommended_technology': tech_name,
                    'technology_description': tech_desc,
                    'implementation_complexity': complexity,
                    'used_for_esrs_impact': esrs_impact
                })
    
    return recommendations

def get_negative_impacts(technology: str, maturity_level: int, used_for_esrs_impact: str) -> List[Dict]:
    """Get negative impacts for a specific technology at a specific maturity level.
    
    Args:
        technology: The technology being recommended (e.g., "ERP")
        maturity_level: The maturity level at which it's being recommended (1-5)
        used_for_esrs_impact: The ESRS impact for which this technology was recommended
        
    Returns:
        List of dictionaries containing negative impact information
    """
    try:
        # Load ESRS2Tech data - skip the first row which contains "ESRS number"
        esrs2tech_df = pd.read_csv('DST_Excel_ESRS2Tech.csv', skiprows=[0])
        
        # Normalize the technology name
        tech_normalized = technology.strip().lower()
        
        # Find all rows where this technology appears (it may appear multiple times)
        tech_rows = esrs2tech_df[
            esrs2tech_df.iloc[:, 1].str.strip().str.lower() == tech_normalized
        ]
        
        negative_impacts = []
        
        # Process each row where this technology appears
        for _, row in tech_rows.iterrows():
            # Get the ESRS impact that might be negatively affected (Column 0)
            negatively_affected_esrs_impact = row.iloc[0]
            
            # Skip if this is the same ESRS impact we're trying to improve
            if negatively_affected_esrs_impact.lower() == used_for_esrs_impact.lower():
                continue
                
            # Get the description for this maturity level (Column 2-6 correspond to levels 1-5)
            # So for level 1, we want column 2 (index 2), for level 5 we want column 6 (index 6)
            level_col_index = maturity_level + 1
            description = row.iloc[level_col_index]
            
            # Only include if there's a description
            if pd.notna(description) and str(description).strip():
                negative_impacts.append({
                    'recommended_technology': technology,
                    'recommended_level': maturity_level,
                    'used_for_esrs_impact': used_for_esrs_impact,
                    'negatively_affected_esrs_impact': negatively_affected_esrs_impact,
                    'negative_impact_description': description,
                    'category': categorize_esrs_impact(negatively_affected_esrs_impact)
                })
        
        return negative_impacts
        
    except Exception as e:
        print(f"Error getting negative impacts: {str(e)}")
        return []

# Helper function to categorize ESRS impacts
def categorize_esrs_impact(impact: str) -> str:
    """Categorize an ESRS impact into E, S, or G category"""
    environmental_impacts = [
        'energy consumption', 'ghg emissions', 'air pollution',
        'hazardous chemical use', 'resource consumption', 'waste'
    ]
    social_impacts = [
        'diversity & inclusion', 'training & skills',
        'health & safety', 'work-life balance'
    ]
    governance_impacts = ['ethical behaviour and culture', 'ethical culture & behavior']
    
    impact_lower = impact.lower()
    
    if any(imp in impact_lower for imp in environmental_impacts):
        return 'Environmental'
    elif any(imp in impact_lower for imp in social_impacts):
        return 'Social'
    elif any(imp in impact_lower for imp in governance_impacts):
        return 'Governance'
    else:
        return 'Other'

def get_tradeoff_result(technology: str, esrs_impact: str, maturity_level: int) -> str:
    """Get the tradeoff result for a given technology, ESRS impact and maturity level."""
    try:
        # Load the trade-offs file
        tradeoffs_df = pd.read_csv('DST_Excel_TradeOffs.csv', header=[0, 1])
        
        # Normalize technology name and ESRS impact
        tech_normalized = technology.strip()
        impact_normalized = esrs_impact.strip()
        
        # Find the row for this technology
        tech_row = tradeoffs_df[tradeoffs_df.iloc[:, 0].str.strip() == tech_normalized]
        
        if not tech_row.empty:
            # Find the columns for this ESRS impact and maturity level
            for col in tradeoffs_df.columns:
                if col[0].strip() == impact_normalized and int(col[1]) == maturity_level:
                    result = tech_row[col].iloc[0]
                    if pd.notna(result) and str(result).strip():
                        return str(result).strip()
        
        return None
        
    except Exception as e:
        print(f"Error processing trade-offs: {str(e)}")
        return None

def format_tradeoff_statement(technology: str, esrs_impact: str, maturity_level: int, tradeoff_result: str) -> str:
    """Format a tradeoff statement for display."""
    if tradeoff_result:
        # Convert ESRS impact to lowercase for mid-sentence use
        impact_lowercase = esrs_impact.lower()
        return f"""
        <div style="background-color: rgba(229, 194, 26, 0.1); border-left: 4px solid #E5C21A; padding: 1rem; margin: 1rem 0; border-radius: 0.25rem;">
            <p style="margin: 0;">
                <strong>Trade-off analysis:</strong> There exists a trade-off for {technology}, because it can have both a positive 
                and negative impact on {impact_lowercase}. At this maturity level, the trade-off results in a netto 
                <strong>{tradeoff_result.lower()}</strong> of the {impact_lowercase}.
            </p>
        </div>
        """
    return ""

def get_maturity_descriptions(qa_weights: pd.DataFrame, dimension: str, esrs_impact: str) -> Dict[str, Dict[str, str]]:
    """Get maturity level descriptions for current and future states for a given dimension and ESRS impact."""
    descriptions = {
        'current': {str(i): '' for i in range(1, 6)},
        'future': {str(i): '' for i in range(1, 6)}
    }
    
    # Filter for the specific dimension and impact
    dimension_lower = dimension.lower()
    
    # Process both current and future states
    for state in ['current', 'future']:
        # Create the exact string to match in the Question column
        question_pattern = f"{dimension} {state}"
        
        # Filter rows for this dimension, state and ESRS impact
        mask = (
            qa_weights['Question'].str.contains(question_pattern, case=False, na=False) &
            (qa_weights['ESRS impact'].str.lower() == esrs_impact.lower())
        )
        relevant_rows = qa_weights[mask]
        
        # For each maturity level, get its description
        for _, row in relevant_rows.iterrows():
            level = str(int(row['Maturity level']))  # Convert to int first to handle potential float values
            descriptions[state][level] = row['Answer']
    
    return descriptions

def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and stripping whitespace."""
    return str(text).lower().strip()

def get_base_dimension(dimension: str) -> str:
    """Extract base dimension from a dimension string."""
    dimension = normalize_text(dimension)
    if dimension.startswith('people'):
        return 'people'
    elif dimension.startswith('process'):
        return 'process'
    elif dimension.startswith('policy'):
        return 'policy'
    return dimension

def process_maturity_data(qa_weights: pd.DataFrame, survey_data: pd.DataFrame, survey_id: int) -> pd.DataFrame:
    """
    Process QA weights and survey data to create a comprehensive maturity level overview.
    Returns a DataFrame with all maturity levels and their descriptions, marked with current and target levels.
    """
    # Step 1: Normalize the data
    qa_weights = qa_weights.copy()
    qa_weights['base_dimension'] = qa_weights['Question'].apply(lambda x: get_base_dimension(str(x)))
    qa_weights['ESRS impact'] = qa_weights['ESRS impact'].apply(normalize_text)
    
    # Get the survey response for the selected ID
    survey_response = survey_data[survey_data.iloc[:, 0] == survey_id].iloc[0]
    
    # Initialize the results list
    results = []
    
    # Define ESG categories and their impacts
    esg_impacts = {
        'Environmental': ['energy consumption', 'ghg emissions', 'air pollution', 'hazardous chemical use', 'resource use', 'waste management'],
        'Social': ['diversity & inclusion', 'training & skills', 'health & safety', 'work-life balance'],
        'Governance': ['ethical culture & behavior']
    }
    
    # Step 2 & 3: Process each base dimension and ESRS impact
    for base_dim in ['people', 'process', 'policy']:
        # Get all rows for this base dimension
        dim_mask = qa_weights['base_dimension'] == base_dim
        dim_data = qa_weights[dim_mask]
        
        # Process each ESRS impact that has data
        for esrs_impact in dim_data['ESRS impact'].unique():
            # Get all rows for this ESRS impact
            impact_mask = dim_data['ESRS impact'] == esrs_impact
            impact_data = dim_data[impact_mask]
            
            if impact_data.empty:
                continue
            
            # Get current and future descriptions
            current_data = impact_data[impact_data['Question'].str.contains('current', case=False)]
            future_data = impact_data[impact_data['Question'].str.contains('future', case=False)]
            
            if current_data.empty or future_data.empty:
                continue
            
            # Step 4: Get current and future levels from survey response
            try:
                current_level = int(round(float(survey_response[f"{base_dim}_current_{esrs_impact}"])))
                future_level = int(round(float(survey_response[f"{base_dim}_future_{esrs_impact}"])))
            except (KeyError, ValueError, TypeError):
                continue
            
            # Step 5: Build the overview for each maturity level
            for level in range(1, 6):
                current_desc = current_data[current_data['Maturity level'] == level]['Answer'].iloc[0] if not current_data[current_data['Maturity level'] == level].empty else ''
                future_desc = future_data[future_data['Maturity level'] == level]['Answer'].iloc[0] if not future_data[future_data['Maturity level'] == level].empty else ''
                
                if current_desc or future_desc:  # Only add if we have at least one description
                    results.append({
                        'survey_id': survey_id,
                        'base_dimension': base_dim,
                        'esrs_impact': esrs_impact,
                        'maturity_level': level,
                        'current_description': current_desc,
                        'future_description': future_desc,
                        'is_current_level': level == current_level,
                        'is_future_level': level == future_level
                    })
    
    return pd.DataFrame(results)

def get_esg_category(impact: str) -> str:
    """Get the ESG category for a given ESRS impact."""
    impact = normalize_text(impact)
    if impact in ['energy consumption', 'ghg emissions', 'air pollution', 'hazardous chemical use', 'resource use', 'waste management']:
        return 'Environmental'
    elif impact in ['diversity & inclusion', 'training & skills', 'health & safety', 'work-life balance']:
        return 'Social'
    elif impact in ['ethical culture & behavior']:
        return 'Governance'
    return None

def calculate_esg_scores(qa_weights: pd.DataFrame, survey_data: pd.DataFrame, survey_id: int) -> Tuple[Dict, Dict, Dict]:
    """
    Calculate Environmental, Social, and Governance scores for each dimension.
    Returns three dictionaries containing current and future scores for each dimension.
    """
    # Initialize score dictionaries
    env_scores = {dim: {'current': 0.0, 'future': 0.0} for dim in ['people', 'process', 'policy']}
    social_scores = {dim: {'current': 0.0, 'future': 0.0} for dim in ['people', 'process', 'policy']}
    gov_scores = {dim: {'current': 0.0, 'future': 0.0} for dim in ['people', 'process', 'policy']}
    
    # Get the survey response for the selected ID
    survey_response = survey_data[survey_data.iloc[:, 0] == survey_id].iloc[0]
    
    # Process each dimension
    for dimension in ['people', 'process', 'policy']:
        # Environmental impacts
        env_impacts = ['energy consumption', 'ghg emissions', 'air pollution', 
                      'hazardous chemical use', 'resource use', 'waste management']
        env_current = []
        env_future = []
        
        # Social impacts
        social_impacts = ['diversity & inclusion', 'training & skills', 
                         'health & safety', 'work-life balance']
        social_current = []
        social_future = []
        
        # Governance impacts
        gov_impacts = ['ethical culture & behavior']
        gov_current = []
        gov_future = []
        
        # Calculate scores for each impact category
        for impact in env_impacts:
            try:
                current = float(survey_response[f"{dimension}_current_{impact}"])
                future = float(survey_response[f"{dimension}_future_{impact}"])
                env_current.append(current)
                env_future.append(future)
            except (KeyError, ValueError, TypeError):
                continue
                
        for impact in social_impacts:
            try:
                current = float(survey_response[f"{dimension}_current_{impact}"])
                future = float(survey_response[f"{dimension}_future_{impact}"])
                social_current.append(current)
                social_future.append(future)
            except (KeyError, ValueError, TypeError):
                continue
                
        for impact in gov_impacts:
            try:
                current = float(survey_response[f"{dimension}_current_{impact}"])
                future = float(survey_response[f"{dimension}_future_{impact}"])
                gov_current.append(current)
                gov_future.append(future)
            except (KeyError, ValueError, TypeError):
                continue
        
        # Calculate average scores
        if env_current:
            env_scores[dimension]['current'] = sum(env_current) / len(env_current)
        if env_future:
            env_scores[dimension]['future'] = sum(env_future) / len(env_future)
            
        if social_current:
            social_scores[dimension]['current'] = sum(social_current) / len(social_current)
        if social_future:
            social_scores[dimension]['future'] = sum(social_future) / len(social_future)
            
        if gov_current:
            gov_scores[dimension]['current'] = sum(gov_current) / len(gov_current)
        if gov_future:
            gov_scores[dimension]['future'] = sum(gov_future) / len(gov_future)
    
    return env_scores, social_scores, gov_scores

def main():
    """Main dashboard application"""
    st.title("Your Twin Transition maturity results")
    
    # Load data
    survey_data, qa_weights, error = load_data()
    if error:
        st.error(f"Error loading data: {error}")
        return
    
    if survey_data is None or qa_weights is None:
        st.error("Could not load required data files. Please ensure Survey_data.csv and DST_Excel_QAweights.csv are present.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["📊 Overview", "🔍 Detailed analysis", "💡 Recommendations"]
    )
    
    # Survey response selector
    survey_ids = survey_data.iloc[:,0].unique()
    selected_id = st.sidebar.selectbox(
        "Select Survey Response ID:",
        options=survey_ids,
        format_func=lambda x: f"Response {x}"
    )
    
    # Store selected ID in session state
    st.session_state['selected_survey_id'] = selected_id

    # Add the transition image with updated styling
    st.sidebar.markdown('<div style="width: 100%; margin: 0; padding: 0;">', unsafe_allow_html=True)
    st.sidebar.image(
        "transition_image.png",
        use_container_width=True
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Calculate scores for selected response
    try:
        selected_response = survey_data[survey_data.iloc[:,0] == selected_id].iloc[0]
        results = calculate_maturity_scores(selected_response, qa_weights)
        
        if not results or all(all(all(scores == 0 for scores in dim.values()) for dim in timeframe.values()) for timeframe in results.values()):
            st.warning("No valid maturity scores found for this response.")
            return
            
        # Display selected page
        if page == "📊 Overview":
            show_overview(results)
        elif page == "🔍 Detailed analysis":
            show_detailed_analysis(results)
        else:
            show_recommendations(results)
            
    except Exception as e:
        st.error(f"Error processing survey response: {str(e)}")
        return

def show_detailed_kpis(qa_weights: pd.DataFrame, survey_data: pd.DataFrame, survey_id: int, dimension: str, esg_category: str):
    """Display detailed KPI information for a specific dimension and ESG category."""
    # Process the maturity data
    maturity_data = process_maturity_data(qa_weights, survey_data, survey_id)
    
    # Filter for the selected dimension and ESG category
    filtered_data = maturity_data[
        (maturity_data['base_dimension'] == dimension) & 
        (maturity_data['esrs_impact'].apply(lambda x: get_esg_category(x) == esg_category))
    ]
    
    if filtered_data.empty:
        st.info(f"No detailed KPI data available for {dimension} - {esg_category}")
        return
    
    # Group by ESRS impact
    for esrs_impact in filtered_data['esrs_impact'].unique():
        impact_data = filtered_data[filtered_data['esrs_impact'] == esrs_impact]
        
        st.markdown(f"#### {esrs_impact.title()}")
        
        # Create a table for all maturity levels
        table_data = []
        for _, row in impact_data.iterrows():
            level = row['maturity_level']
            current_desc = row['current_description']
            future_desc = row['future_description']
            
            status = []
            if row['is_current_level']:
                status.append("✓ Current")
            if row['is_future_level']:
                status.append("🎯 Target")
            
            table_data.append({
                "Level": f"Level {level}",
                "Current State Description": current_desc,
                "Target State Description": future_desc,
                "Status": " & ".join(status) if status else ""
            })
        
        # Display the table
        df = pd.DataFrame(table_data)
        st.dataframe(df.style.set_properties(**{'text-align': 'left'}))
        st.markdown("---")

def show_overview(results: Dict):
    """Show overview dashboard"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Your Twin Transition maturity")
        fig = create_radar_chart(results, 'overview')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show data processing meta info
        if 'meta' in results:
            meta = results['meta']
            with st.container():
                container = st.container()
                with container:
                    st.markdown('<div class="score-box">', unsafe_allow_html=True)
                    st.markdown("#### Data Processing Summary")
                    st.metric("Questions Processed", meta['processed_questions'])
                    st.metric("Valid Responses", meta['valid_responses'])
                    
                    if meta['valid_responses'] > 0:
                        response_rate = (meta['processed_questions'] / meta['total_questions']) * 100
                        st.metric("Response Rate", f"{response_rate:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Calculate averages per dimension
        dimensions = ['people', 'process', 'policy']
        dimension_display = {'people': 'People', 'process': 'Process', 'policy': 'Policy'}
        
        for dim in dimensions:
            current_scores = []
            future_scores = []
            
            for esg in ['E', 'S', 'G']:
                if results['current'][dim][esg] > 0:
                    current_scores.append(results['current'][dim][esg])
                if results['future'][dim][esg] > 0:
                    future_scores.append(results['future'][dim][esg])
            
            if current_scores:
                avg_current = round(sum(current_scores) / len(current_scores))
                avg_future = round(sum(future_scores) / len(future_scores)) if future_scores else avg_current
                
                st.markdown(f"""
                    <div class="score-box">
                        <h4>{dimension_display[dim]} dimension</h4>
                        <div class="score-line">Your current score is: {avg_current}/5</div>
                        <div class="score-line">Your target score is: {avg_future}/5</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="score-box">
                        <h4>{dimension_display[dim]} dimension</h4>
                        <div class="score-line">No valid scores available</div>
                    </div>
                """, unsafe_allow_html=True)

def show_detailed_analysis(results: Dict):
    """Show detailed analysis by dimension"""
    st.subheader("Zooming in on the ESG scores")
    
    dimension = st.selectbox(
        "Select dimension to see the ESG scores in more detail:",
        ["People", "Process", "Policy"]
    )
    box_colors = {
        'Environmental': 'rgba(200, 200, 200, 0.6)',
        'Social': 'rgba(200, 200, 200, 0.6)',
        'Governance': 'rgba(200, 200, 200, 0.6)'
    }
    # Define colors based on selected dimension
    if dimension.lower() == 'people':
        # Different shades of green
        box_colors = {
            'Environmental': 'rgba(111, 166, 56, 0.6)',    # #6FA638 with 0.6 opacity (matching overview)
            'Social': 'rgba(136, 208, 124, 0.6)',         # #88D07C with 0.6 opacity
            'Governance': 'rgba(217, 242, 208, 0.6)'      # #D9F2D0 with 0.6 opacity
        }
    elif dimension.lower() == 'process':
        # Different shades of green
        box_colors = {
            'Environmental': 'rgba(96, 177, 158, 0.6)',    # #60B19E with 0.6 opacity
            'Social': 'rgba(132, 207, 191, 0.6)',         # #84CFBF with 0.6 opacity
            'Governance': 'rgba(156, 225, 212, 0.6)'      # #9CE1D4 with 0.6 opacity
        }
    else:  # policy
        # Different shades of blue
        box_colors = {
            'Environmental': 'rgba(117, 196, 201, 0.6)',   # #75C4C9 with 0.6 opacity
            'Social': 'rgba(168, 228, 231, 0.6)',         # #A8E4E7 with 0.6 opacity
            'Governance': 'rgba(210, 248, 242, 0.6)'      # #D2F8F2 with 0.6 opacity
        }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_radar_chart(results, dimension)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"{dimension} Dimension Scores")
        dim_key = dimension.lower()
        
        for esg_full, esg_short in [("Environmental", "E"), ("Social", "S"), ("Governance", "G")]:
            try:
                current = float(results['current'][dim_key][esg_short])
                future = float(results['future'][dim_key][esg_short])
                
                # Create custom CSS for the score box with dynamic background color
                box_style = f"""
                    <div style="
                        background-color: {box_colors[esg_full]};
                        border: 1px solid rgba(0, 0, 0, 0.1);
                        border-radius: 0.5rem;
                        padding: 1rem;
                        margin: 0.5rem 0;
                    ">
                        <h4 style="
                            margin: 0 0 0.5rem 0;
                            color: rgba(0, 0, 0, 0.8);
                            font-size: 1.1rem;
                            font-weight: 600;
                        ">{esg_full}</h4>
                        <div style="color: rgba(0, 0, 0, 0.7);">Your current score is: {int(round(current))}/5</div>
                        <div style="color: rgba(0, 0, 0, 0.7);">Your target score is: {int(round(future))}/5</div>
                    </div>
                """
                st.markdown(box_style, unsafe_allow_html=True)
                
            except (KeyError, TypeError, ValueError) as e:
                st.markdown(f"""
                    <div class="score-box">
                        <h4>{esg_full}</h4>
                        Error displaying scores: {str(e)}
                    </div>
                """, unsafe_allow_html=True)

def show_recommendations(results: Dict):
    """Show recommendations based on maturity assessment"""
    st.subheader("Which digital manufacturing technologies can support your ESG goals?")
    box_colors = {
        'Environmental': 'rgba(200, 200, 200, 0.6)',
        'Social': 'rgba(200, 200, 200, 0.6)',
        'Governance': 'rgba(200, 200, 200, 0.6)'
    }

    # Add CSS for styling the expanders with category-specific colors
    st.markdown("""
        <style>
            /* Base styles for all expanders in recommendations */
            [data-testid="stExpander"] {
                border-radius: 0.5rem !important;
                margin: 0.3rem 0 !important;
                background-color: white !important;
            }

            /* Environmental section expanders */
            div.environmental [data-testid="stExpander"] {
                border: 3px solid rgba(111, 166, 56, 0.8) !important;
            }
            
            /* Social section expanders */
            div.social [data-testid="stExpander"] {
                border: 3px solid rgba(92, 169, 148, 0.8) !important;
            }
            
            /* Governance section expanders */
            div.governance [data-testid="stExpander"] {
                border: 3px solid rgba(168, 228, 231, 0.8) !important;
            }

            /* Remove default borders from expander elements */
            [data-testid="stExpander"] > div:first-child {
                border: none !important;
                box-shadow: none !important;
            }
            
            [data-testid="stExpanderContent"] {
                border: none !important;
                box-shadow: none !important;
            }

            /* Style for trade-off buttons */
            button[kind="secondary"] {
                background-color: #D99748 !important;
                border: none !important;
                color: white !important;
            }
            
            button[kind="secondary"]:hover {
                background-color: #c48841 !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Load required data files
    try:
        qa_weights = pd.read_csv('QA_weights.csv')
        tech2esrs = pd.read_csv('DST_Excel_Tech2ESRS.csv')
        survey_data = pd.read_csv('Survey_data.csv')
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        return
    
    # Get selected survey response
    selected_id = st.session_state.get('selected_survey_id')
    if not selected_id:
        st.warning("Please select a survey response to view recommendations.")
        return
        
    selected_response = survey_data[survey_data.iloc[:,0] == selected_id].iloc[0]
    
    # Extract process dimension maturity levels
    process_maturity = extract_process_maturity(selected_response, qa_weights)
    
    if not process_maturity:
        st.warning("No valid process dimension maturity levels found for this response.")
        return
    
    # Get technology recommendations
    recommendations = get_technology_recommendations(process_maturity, tech2esrs)
    
    if not recommendations:
        st.info("No technology recommendations available. This could be because the process is already at maximum maturity level.")
        return
    
    # Display recommendations by ESG category
    st.markdown("### Digital manufacturing technology recommendations for each impact")
    
    # Define ESG categories and their impacts
    esg_categories = {
        'Environmental': ['energy consumption', 'ghg emissions', 'air pollution', 'hazardous chemical use', 'resource use', 'waste management'],
        'Social': ['diversity & inclusion', 'training & skills', 'health & safety', 'work-life balance'],
        'Governance': ['ethical culture & behavior']
    }
    
    # Define complexity badge colors
    complexity_badges = {
        'Low': '🟢',
        'Medium': '🟡',
        'High': '🔴',
        'Unknown': '⚪',
        'Not applicable at this level': '⚫'
    }
    
    # Display recommendations by ESG category
    for category, impacts in esg_categories.items():
        # First, check if we have any recommendations for this category's impacts
        has_category_recs = False
        for impact in impacts:
            if any(r['esrs_impact'].lower() == impact.lower() for r in recommendations):
                has_category_recs = True
                break
        
        if has_category_recs:
            st.markdown(f"#### {category}")
            
            # Create a div wrapper for this category's expanders with explicit class
            st.markdown(f'<div class="{category.lower()}" style="display: block;">', unsafe_allow_html=True)
            
            # Display recommendations for each impact in this category
            for impact in impacts:
                # Get recommendations for this impact
                impact_recs = [r for r in recommendations if r['esrs_impact'].lower() == impact.lower()]
                if impact_recs:
                    # Get the maturity level from process_maturity using case-insensitive matching
                    maturity_key = next((k for k in process_maturity.keys() if k.lower() == impact.lower()), None)
                    if maturity_key is not None:
                        next_level = process_maturity[maturity_key] + 1
                        display_impact = impact.title()
                        
                        # Create the expander
                        with st.expander(f"**{display_impact}**: digital manufacturing technologies that can help you go to maturity level {next_level}"):
                            for rec in impact_recs:
                                st.markdown(f"""#### {rec['recommended_technology']}""")
                                
                                # Create two columns for the metadata and button
                                meta_col, button_col = st.columns([3, 1])
                                
                                with meta_col:
                                    complexity = rec.get('implementation_complexity', 'Unknown')
                                    complexity_badge = complexity_badges.get(complexity, '⚪')
                                    st.markdown(f"""
                                    **Target maturity level:** {rec['recommended_level']}/5  
                                    **Implementation complexity:** {complexity_badge} {complexity}
                                    """)
                                
                                with button_col:
                                    # Create a unique key for this button
                                    button_key = f"neg_impact_{rec['recommended_technology']}_{rec['recommended_level']}_{rec['esrs_impact']}"
                                    st.button("View trade-off with other ESG impacts", key=button_key, type="secondary")
                                
                                # Show the technology description
                                st.markdown(f"""{rec['technology_description']}""")
                                
                                # Create a container for negative impacts that only shows when button is clicked
                                if st.session_state.get(button_key, False):
                                    with st.container():
                                        negative_impacts = get_negative_impacts(
                                            rec['recommended_technology'],
                                            rec['recommended_level'],
                                            rec['esrs_impact']
                                        )
                                        
                                        if negative_impacts:
                                            st.markdown("---")
                                            st.markdown(f"""##### Potential negative sustainability impacts of {rec['recommended_technology']} at maturity level {rec['recommended_level']}""")
                                            
                                            # Check for trade-offs
                                            tradeoff_result = get_tradeoff_result(
                                                rec['recommended_technology'],
                                                rec['esrs_impact'],
                                                rec['recommended_level']
                                            )
                                            
                                            # If there's a trade-off, show it first
                                            if tradeoff_result:
                                                st.markdown(
                                                    format_tradeoff_statement(
                                                        rec['recommended_technology'],
                                                        rec['esrs_impact'],
                                                        rec['recommended_level'],
                                                        tradeoff_result
                                                    ),
                                                    unsafe_allow_html=True
                                                )
                                            
                                            # Group impacts by ESG category
                                            categorized_impacts = {
                                                'Environmental': [],
                                                'Social': [],
                                                'Governance': []
                                            }
                                            
                                            for neg_impact in negative_impacts:
                                                category = neg_impact['category']
                                                if category in categorized_impacts:
                                                    categorized_impacts[category].append(neg_impact)
                                            
                                            # Display impacts by category
                                            for category, impacts in categorized_impacts.items():
                                                if impacts:
                                                    st.markdown(f"""
                                                    <div style="
                                                        background-color: {box_colors.get(category, 'rgba(240, 240, 240, 0.6)')};
                                                        border: 1px solid rgba(0, 0, 0, 0.1);
                                                        border-radius: 0.5rem;
                                                        padding: 1rem;
                                                        margin: 0.5rem 0;
                                                    ">
                                                        <h4 style="
                                                            margin: 0 0 0.5rem 0;
                                                            color: rgba(0, 0, 0, 0.8);
                                                            font-size: 1.1rem;
                                                            font-weight: 600;
                                                        ">{category}</h4>
                                                        {''.join([f'<div style="margin: 0.5rem 0;"><strong>{impact["negatively_affected_esrs_impact"]}</strong>: {impact["negative_impact_description"]}</div>' for impact in impacts])}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                        else:
                                            st.info("No negative sustainability impacts identified for this technology at this maturity level.")
                                
                                st.markdown("---")
            
            # Close the wrapper div
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
