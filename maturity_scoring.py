import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional, Union

def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and stripping whitespace"""
    return str(text).strip().lower()

def string_similarity(a: str, b: str) -> float:
    """Calculate string similarity using SequenceMatcher"""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def find_matching_column(question: str, column_headers: List[str], min_similarity: float = 0.95) -> Optional[str]:
    """Find a matching column header for a given question"""
    normalized_question = normalize_text(question)
    
    # Try exact match first
    for header in column_headers:
        if normalize_text(header) == normalized_question:
            return header
    
    # If no exact match, try fuzzy matching
    best_match = None
    best_similarity = 0
    
    for header in column_headers:
        similarity = string_similarity(header, question)
        if similarity > best_similarity and similarity >= min_similarity:
            best_similarity = similarity
            best_match = header
    
    return best_match

def match_answer(actual_answer: str, expected_answer: str, min_similarity: float = 0.85) -> bool:
    """Check if actual answer matches expected answer using exact or fuzzy matching"""
    # Try exact match first
    if normalize_text(actual_answer) == normalize_text(expected_answer):
        return True
    
    # If no exact match, try fuzzy matching
    return string_similarity(actual_answer, expected_answer) >= min_similarity

def extract_esg_weights(survey_data: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    """Extract ESG weights for each respondent"""
    esg_weights = {}
    
    # ESG weight questions
    esg_questions = {
        'E': 'How important is Environmental sustainability to you?',
        'S': 'How important is Social sustainability to you?',
        'G': 'How important is Governance sustainability to you?'
    }
    
    for idx, row in survey_data.iterrows():
        survey_id = int(row.iloc[0])  # Assuming ID is first column
        weights = {'E': 0, 'S': 0, 'G': 0}
        
        for esg_cat, question in esg_questions.items():
            matching_col = find_matching_column(question, survey_data.columns)
            if matching_col:
                try:
                    weight = int(row[matching_col])
                    weights[esg_cat] = weight if 1 <= weight <= 3 else 0
                except (ValueError, TypeError):
                    weights[esg_cat] = 0
        
        esg_weights[survey_id] = weights
    
    return esg_weights

def extract_esrs_weights(survey_data: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    """Extract ESRS impact weights for each respondent"""
    esrs_weights = {}
    
    # ESRS impact questions mapping
    esrs_questions = {
        'energy consumption': 'How important is energy consumption reduction?',
        'GHG emissions': 'How important are GHG emission reductions?',
        'air pollution': 'How important are air pollutant emission reductions?',
        'hazardous chemical use': 'How important is hazardous chemical use reduction?',
        'resource use': 'How important are resource consumption reduction?',
        'waste management': 'How important is waste management & circularity?',
        'diversity & inclusion': 'How important is diversity & inclusion?',
        'training & skills': 'How important is training & skills?',
        'health & safety': 'How important is occupational health & safety?',
        'work-life balance': 'How important is work-life balance?',
        'ethical culture & behavior': None  # No survey question, will get default weight of 1
    }
    
    for idx, row in survey_data.iterrows():
        survey_id = int(row.iloc[0])
        weights = {}
        
        for impact, question in esrs_questions.items():
            if question:
                matching_col = find_matching_column(question, survey_data.columns)
                if matching_col:
                    try:
                        weight = int(row[matching_col])
                        weights[impact] = weight if 1 <= weight <= 4 else 0
                    except (ValueError, TypeError):
                        weights[impact] = 0
            else:
                # Special case for ethical culture & behavior
                weights[impact] = 1
        
        esrs_weights[survey_id] = weights
    
    return esrs_weights

def calculate_maturity_scores(
    survey_data: pd.DataFrame,
    qa_weights: pd.DataFrame
) -> Dict[int, Dict[str, Dict[str, Dict[str, int]]]]:
    """Calculate maturity scores for each respondent across all dimensions and ESRS impacts"""
    results = {}
    
    # Define dimensions and timeframes
    dimensions = ['people', 'process', 'policy']
    timeframes = ['current', 'future']
    
    for idx, survey_row in survey_data.iterrows():
        survey_id = int(survey_row.iloc[0])
        results[survey_id] = {
            'current': {dim: {} for dim in dimensions},
            'future': {dim: {} for dim in dimensions}
        }
        
        # Process each dimension and timeframe
        for dimension in dimensions:
            for timeframe in timeframes:
                dimension_key = f"{dimension} {timeframe}"
                
                # Filter QA weights for current dimension
                dimension_qa = qa_weights[
                    qa_weights.iloc[:, 0].str.lower().str.strip() == dimension_key
                ]
                
                # Group by ESRS impact
                for _, qa_group in dimension_qa.groupby('ESRS impact'):
                    if qa_group.empty:
                        continue
                        
                    esrs_impact = qa_group['ESRS impact'].iloc[0]
                    
                    # Try to find matching question and answer
                    for _, qa_row in qa_group.iterrows():
                        question = qa_row['Question']
                        expected_answer = qa_row['Answer']
                        maturity_level = qa_row['Maturity level']
                        
                        # Find matching survey column
                        matching_col = find_matching_column(question, survey_data.columns)
                        if not matching_col:
                            continue
                            
                        # Get respondent's answer
                        actual_answer = survey_row[matching_col]
                        
                        # Check if answer matches
                        if match_answer(str(actual_answer), str(expected_answer)):
                            # Store the maturity level
                            results[survey_id][timeframe][dimension][esrs_impact] = int(maturity_level)
                            break
    
    return results

def calculate_esg_scores(
    maturity_scores: Dict[int, Dict[str, Dict[str, Dict[str, int]]]], 
    esrs_weights: Dict[int, Dict[str, int]],
    esg_weights: Dict[int, Dict[str, int]]
) -> Dict[int, Dict[str, Dict[str, Dict[str, int]]]]:
    """Calculate ESG category scores (E, S, G) for each dimension"""
    
    # ESRS impact to ESG category mapping
    esrs_to_esg = {
        'energy consumption': 'E',
        'GHG emissions': 'E',
        'air pollution': 'E',
        'hazardous chemical use': 'E',
        'resource use': 'E',
        'waste management': 'E',
        'diversity & inclusion': 'S',
        'training & skills': 'S',
        'health & safety': 'S',
        'work-life balance': 'S',
        'ethical culture & behavior': 'G'
    }
    
    esg_scores = {}
    
    for survey_id, scores in maturity_scores.items():
        esg_scores[survey_id] = {
            'current': {'people': {}, 'process': {}, 'policy': {}},
            'future': {'people': {}, 'process': {}, 'policy': {}}
        }
        
        respondent_weights = esrs_weights.get(survey_id, {})
        esg_importance = esg_weights.get(survey_id, {'E': 1, 'S': 1, 'G': 1})
        
        for timeframe in ['current', 'future']:
            for dimension in ['people', 'process', 'policy']:
                # Initialize weighted sums for each ESG category
                weighted_sums = {'E': 0, 'S': 0, 'G': 0}
                weight_sums = {'E': 0, 'S': 0, 'G': 0}
                
                # Get maturity scores for this dimension
                dimension_scores = scores[timeframe][dimension]
                
                # Calculate weighted sums for each ESG category
                for esrs_impact, maturity in dimension_scores.items():
                    esg_cat = esrs_to_esg[esrs_impact]
                    weight = respondent_weights.get(esrs_impact, 1)
                    
                    weighted_sums[esg_cat] += maturity * weight
                    weight_sums[esg_cat] += weight
                
                # Calculate final scores for each ESG category
                for esg_cat in ['E', 'S', 'G']:
                    if weight_sums[esg_cat] > 0:
                        # Calculate weighted average and apply ESG importance weight
                        score = (weighted_sums[esg_cat] / weight_sums[esg_cat]) * (esg_importance[esg_cat] / 3)
                        # Round to nearest integer and ensure within 1-5 range
                        final_score = max(1, min(5, round(score)))
                        esg_scores[survey_id][timeframe][dimension][esg_cat] = final_score
                    else:
                        esg_scores[survey_id][timeframe][dimension][esg_cat] = 0
    
    return esg_scores

def calculate_overall_scores(
    esg_scores: Dict[int, Dict[str, Dict[str, Dict[str, int]]]], 
    esg_weights: Dict[int, Dict[str, int]]
) -> Dict[int, Dict[str, Dict[str, int]]]:
    """Calculate overall maturity score for each dimension"""
    
    overall_scores = {}
    
    for survey_id, scores in esg_scores.items():
        overall_scores[survey_id] = {
            'current': {'people': 0, 'process': 0, 'policy': 0},
            'future': {'people': 0, 'process': 0, 'policy': 0}
        }
        
        esg_importance = esg_weights.get(survey_id, {'E': 1, 'S': 1, 'G': 1})
        total_weight = sum(esg_importance.values())
        
        if total_weight == 0:
            continue
            
        for timeframe in ['current', 'future']:
            for dimension in ['people', 'process', 'policy']:
                dimension_scores = scores[timeframe][dimension]
                
                # Calculate weighted average of E, S, G scores
                weighted_sum = sum(
                    dimension_scores.get(esg_cat, 0) * esg_importance[esg_cat]
                    for esg_cat in ['E', 'S', 'G']
                )
                
                # Calculate and round final score
                if weighted_sum > 0:
                    final_score = round(weighted_sum / total_weight)
                    overall_scores[survey_id][timeframe][dimension] = max(1, min(5, final_score))
    
    return overall_scores

def process_survey_response(
    survey_data: pd.DataFrame,
    qa_weights: pd.DataFrame,
    survey_id: int
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, Dict[str, int]]]]:
    """Process a single survey response and return overall and ESG scores"""
    
    # Filter survey data for the specific respondent
    respondent_data = survey_data[survey_data.iloc[:, 0] == survey_id]
    
    if respondent_data.empty:
        raise ValueError(f"No survey response found for ID {survey_id}")
    
    # Calculate all scores
    maturity_scores = calculate_maturity_scores(respondent_data, qa_weights)
    esrs_weights = extract_esrs_weights(respondent_data)
    esg_weights = extract_esg_weights(respondent_data)
    
    esg_scores = calculate_esg_scores(maturity_scores, esrs_weights, esg_weights)
    overall_scores = calculate_overall_scores(esg_scores, esg_weights)
    
    return overall_scores[survey_id], esg_scores[survey_id]