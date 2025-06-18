import streamlit as st
import pandas as pd
import re
import json
import os
from google import genai
from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv


# Configure Streamlit page
st.set_page_config(
    page_title="🌟 Vibe Mapping Shopping Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .confidence-high { color: #4CAF50; font-weight: bold; }
    .confidence-medium { color: #FF9800; }
    .confidence-low { color: #F44336; }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini client
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key)

class VibeMapper:
    """Enhanced vibe mapping with comprehensive attribute mappings and confidence scoring"""
    
    def __init__(self):
        # Comprehensive vibe mappings with simple confidence values
        self.vibe_mappings = {
            # Style vibes
            "casual": {
                "fit": ["Relaxed", "Oversized"], 
                "fabric": ["Cotton", "Jersey", "Bamboo jersey"],
                "confidence": 0.75,  # High confidence - strong match
            },
            "elevated": {
                "fit": ["Body hugging", "Tailored"], 
                "fabric": ["Satin", "Silk", "Velvet"],
                "confidence": 0.75,  # High confidence - strong match
            },
            "cute": {
                "fit": ["Relaxed", "Flowy"], 
                "color_or_print": ["Pastel pink", "Pastel yellow", "Floral print"],
                "confidence": 0.65,  # Good confidence
            },
            "flowy": {
                "fit": ["Relaxed", "Flowy"], 
                "fabric": ["Chiffon", "Linen", "Viscose"],
                "confidence": 0.75,  # High confidence - strong match
            },
            "sleek": {
                "fit": ["Body hugging", "Tailored"], 
                "fabric": ["Satin", "Silk", "Crepe"],
                "confidence": 0.65,  # Good confidence
            },
            "relaxed": {
                "fit": ["Relaxed", "Oversized"],
                "confidence": 0.75,  # High confidence - strong match
            },
            "bodycon": {
                "fit": ["Body hugging", "Bodycon"],
                "confidence": 0.75,  # High confidence - strong match
            },
            
            # Seasonal vibes
            "summer": {
                "fabric": ["Linen", "Cotton", "Organic cotton", "Bamboo jersey", "Cotton gauze"],
                "confidence": 0.75,  # High confidence - strong match
            },
            "winter": {
                "fabric": ["Wool", "Cashmere", "Fleece", "Velvet"],
                "confidence": 0.75,  # High confidence - strong match
            },
            "spring": {
                "color_or_print": ["Pastel pink", "Pastel yellow", "Floral print"],
                "fabric": ["Cotton", "Linen"],
                "confidence": 0.65,  # Good confidence
            },
            
            # Occasion vibes
            "party": {
                "fabric": ["Satin", "Silk", "Velvet", "Sequined mesh"], 
                "occasion": ["Party", "Evening"],
                "confidence": 0.65,  # Good confidence
            },
            "work": {
                "fabric": ["Cotton poplin", "Wool-blend"], 
                "occasion": ["Work"], 
                "fit": ["Tailored"],
                "confidence": 0.75,  # High confidence - strong match
            },
            "vacation": {
                "fabric": ["Linen", "Cotton", "Cotton gauze"], 
                "occasion": ["Vacation"],
                "confidence": 0.65,  # Good confidence
            },
            "brunch": {
                "fit": ["Relaxed", "Flowy"], 
                "occasion": ["Everyday", "Vacation"],
                "confidence": 0.55,  # Medium confidence
            },
            
            # Color vibes
            "pastel": {
                "color_or_print": ["Pastel pink", "Pastel yellow", "Pastel coral"],
                "confidence": 0.55,  # Medium confidence
            },
            "bold": {
                "color_or_print": ["Ruby red", "Cobalt blue", "Emerald green"],
                "confidence": 0.55,  # Medium confidence
            },
            
            # Fabric/texture vibes
            "breathable": {
                "fabric": ["Linen", "Cotton", "Organic cotton", "Bamboo jersey"],
                "confidence": 0.75,  # High confidence - strong match
            },
            "luxurious": {
                "fabric": ["Silk", "Satin", "Velvet", "Cashmere"],
                "confidence": 0.65,  # Good confidence
            },
        }
        
        # Enhanced synonyms with context
        self.synonyms = {
            "dressy": "elevated",
            "fancy": "elevated", 
            "formal": "elevated",
            "comfy": "casual",
            "comfortable": "casual",
            "chill": "casual",
            "boho": "flowy",
            "bohemian": "flowy",
            "flowing": "flowy",
            "fitted": "bodycon",
            "tight": "bodycon",
            "loose": "relaxed",
            "baggy": "relaxed",
            "summery": "summer",
            "wintery": "winter",
            "springy": "spring",
            "professional": "work",
            "office": "work",
            "business": "work",
        }
        
        # Context enhancers - phrases that modify confidence
        self.context_enhancers = {
            "really": 1.2,
            "very": 1.1, 
            "super": 1.2,
            "kinda": 0.8,
            "maybe": 0.7,
            "somewhat": 0.8,
        }

    def extract_vibes_with_confidence(self, query: str) -> Tuple[Dict[str, List[str]], float, List[str]]:
        """Extract matching vibes from user query with simple confidence scoring"""
        
        query_lower = query.lower().strip()
        extracted_attrs = defaultdict(list)
        detected_vibes = []
        
        # Track confidence metrics
        total_confidence = 0.0
        matched_vibe_count = 0
        
        # Extract explicit vibes and attributes
        for vibe, mapping in self.vibe_mappings.items():
            if vibe in query_lower:
                # Get base confidence from the mapping
                confidence = mapping.get('confidence', 0.7)  # Default medium confidence
                
                # Simple position boost - terms at start of query are slightly more important
                if query_lower.startswith(vibe):
                    confidence = min(confidence * 1.1, 1.0)  # Slight boost, capped at 1.0
                
                # Add enhancers for common phrases
                for enhancer, multiplier in self.context_enhancers.items():
                    if f"{enhancer} {vibe}" in query_lower:
                        confidence = min(confidence * multiplier, 1.0)
                        break
                
                # Add to running totals
                total_confidence += confidence
                matched_vibe_count += 1
                
                # Save vibe with its confidence
                detected_vibes.append((vibe, confidence))
                
                # Extract attributes
                for attr_type, values in mapping.items():
                    if attr_type != 'confidence' and isinstance(values, list):
                        extracted_attrs[attr_type].extend(values)
        
        # Check synonyms
        for synonym, vibe in self.synonyms.items():
            if synonym in query_lower and vibe in self.vibe_mappings:
                # Get confidence from the original vibe but slightly reduced
                mapping = self.vibe_mappings[vibe]
                confidence = mapping.get('confidence', 0.7) * 0.9  # Slightly lower for synonyms
                
                # Add to running totals
                total_confidence += confidence
                matched_vibe_count += 1
                
                # Save synonym with its confidence
                detected_vibes.append((synonym, confidence))
                
                # Extract attributes from original vibe
                for attr_type, values in mapping.items():
                    if attr_type != 'confidence' and isinstance(values, list):
                        extracted_attrs[attr_type].extend(values)
        
        # Remove duplicates in attribute lists
        for attr_type in extracted_attrs:
            extracted_attrs[attr_type] = list(set(extracted_attrs[attr_type]))
        
        # Calculate overall confidence as simple average with scaling
        # Scale down confidence to be more realistic (max around 85%)
        if matched_vibe_count > 0:
            raw_confidence = total_confidence / matched_vibe_count
            # Apply a scaling factor to make the values more reasonable
            overall_confidence = raw_confidence * 0.85
        else:
            overall_confidence = 0.0
        
        # Sort detected vibes by confidence
        detected_vibes_sorted = [vibe for vibe, _ in sorted(detected_vibes, key=lambda x: x[1], reverse=True)]
        
        return dict(extracted_attrs), overall_confidence, detected_vibes_sorted


class FollowUpEngine:
    """Intelligent follow-up question generation using LLM and contextual information"""
    
    def __init__(self):
        # Define critical fields to check for
        self.critical_info = ["category", "size", "budget", "fit_preference", "occasion_season", "coverage_preference"]
        self.secondary_info = ["sleeve_length", "color_or_print", "fit", "occasion"]
        
        # Fallback contextual questions if LLM is unavailable
        self.contextual_questions = {
            "category": {
                "casual summer": "Are you thinking dresses, or more like tops and skirts for mixing and matching?",
                "party": "What type of party piece - a stunning dress or separates you can style up?",
                "work": "Are you looking for dresses, or professional separates like tops and skirts?",
                "default": "What type of pieces are you shopping for - dresses, tops, skirts, or pants?"
            },
            "size": {
                "default": "What size should I look for?",
                "multiple": "Any specific sizes you'd like me to check - or are you flexible?"
            },
            "budget": {
                "default": "What budget range should I focus on?"
            },
            "fit_preference": {
                "default": "How do you like things to fit - relaxed, tailored, or more bodycon?"
            },
            "occasion_season": {
                "default": "What occasion or season are you shopping for?"
            },
            "coverage_preference": {
                "default": "Any preferences on coverage, like sleeve length or piece length?"
            }
        }

    def generate_smart_questions(self, user_prefs: Dict[str, Any], inferred_attrs: Dict[str, List[str]], 
                                detected_vibes: List[str], confidence: float) -> List[str]:
        """Generate contextual questions using LLM based on missing information"""
        
        # Identify missing information
        missing_info = [info for info in self.critical_info if not user_prefs.get(info)]
        
        # If nothing is missing or if everything is provided, return empty list
        if not missing_info:
            return []
        
        # Use Gemini to generate dynamic follow-up questions
        if gemini_client:
            try:
                return self._generate_llm_questions(missing_info, user_prefs, inferred_attrs, detected_vibes)
            except Exception as e:
                print(f"LLM question generation failed: {e}")
                exit()
                # Fall back to template-based questions if LLM fails
                return self._generate_template_questions(missing_info, detected_vibes, user_prefs)
        else:
            # Use template-based questions if Gemini is not available
            return self._generate_template_questions(missing_info, detected_vibes, user_prefs)
    
    def _generate_llm_questions(self, missing_info: List[str], user_prefs: Dict[str, Any], 
                              inferred_attrs: Dict[str, List[str]], detected_vibes: List[str]) -> List[str]:
        """Generate enthusiastic opener and follow-up questions using Gemini LLM"""
        
        # Format vibes for prompt
        vibes_text = ", ".join(detected_vibes[:3]) if detected_vibes else ""  # Limit to top 3 vibes
        
        prompt = f"""
        As a helpful fashion shopping assistant, I need to respond to the user with an enthusiastic opener
        followed by 1-2 brief, conversational follow-up questions about missing information.
        
        Current user preferences: {json.dumps(user_prefs)}
        Detected style vibes: {vibes_text}
        Inferred attributes: {json.dumps(inferred_attrs)}
        
        Missing information that I need to ask about: {missing_info}
        
        Rules for generating response:
        1. Start with a brief, warm, and genuine opener that acknowledges their style vibe (max 10 words with relevant emoji)
        2. Then generate ONLY 1-2 brief, conversational questions about the missing information
        3. Be warm and friendly, but get straight to the point
        4. Questions should feel contextual to what they've already told me
        5. Focus only on the most important 1-2 missing items from: {missing_info}
        6. The question should be direct and specific to the missing info only. DO NOT ASK WHAT THEY ALREADY HAVE TOLD.
        
        Example good responses:
        - "Love that summer casual vibe! ☀️ What size should I look for?"
        - "Elevated looks are so chic! ✨ Any particular budget range you have in mind?"
        - "Party style is always fun! 🎉 Are you thinking tops, dresses, or something else?"
        
        Return ONLY the complete response (opener + questions) as the output, nothing else.
        """

        # Generate response from Gemini
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        # Clean and parse the response
        full_response = response.text.strip()
        
        # Return the full response including opener and questions
        return [full_response] if full_response else ["I love your style direction! ✨ What size should I look for?"]
    
    def _generate_template_questions(self, missing_info: List[str], detected_vibes: List[str], 
                                   user_prefs: Dict[str, Any]) -> List[str]:
        """Fallback to template-based questions if LLM is unavailable"""
        questions = []
        
        # Add enthusiasm based on detected vibes
        enthusiasm_openers = {
            "summer": "I love summer shopping! ☀️ ",
            "party": "Ooh, party pieces are so fun! ✨ ",
            "casual": "Perfect - casual pieces are so versatile! 😊 ",
            "work": "Great choice for professional pieces! 💼 ",
            "cute": "Aw, I love the cute vibe! 💕 "
        }
        
        opener = ""
        for vibe in detected_vibes:
            if vibe in enthusiasm_openers:
                opener = enthusiasm_openers[vibe]
                break
                
        if not opener and detected_vibes:
            opener = f"Love the {detected_vibes[0]} vibe! "
        
        # Generate questions using templates, limited to 2
        for info in missing_info[:2]:
            questions_dict = self.contextual_questions.get(info, {})
            
            # Try to match with detected vibes
            question = None
            for vibe in detected_vibes:
                if vibe in questions_dict:
                    question = questions_dict[vibe]
                    break
            
            # Check for vibe combinations
            if not question and len(detected_vibes) >= 2:
                vibe_combo = " ".join(detected_vibes[:2])
                if vibe_combo in questions_dict:
                    question = questions_dict[vibe_combo]
            
            # Default question if no match
            if not question:
                question = questions_dict.get("default", f"Could you tell me about your {info.replace('_', ' ')} preference?")
            
            # Add opener to first question only
            if not questions:
                questions.append(f"{opener}{question}")
            else:
                questions.append(question)
        
        return questions

class RecommendationEngine:
    """Enhanced recommendation engine with progressive filtering and smart matching"""
    
    def __init__(self, catalog_df: pd.DataFrame):
        self.catalog = self._prepare_catalog(catalog_df)
        self.fabric_families = self._build_fabric_families()
        
    def _prepare_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced catalog preparation with better data processing"""
        df_clean = df.copy()
        
        # Parse sizes more robustly
        df_clean['sizes_list'] = df_clean['available_sizes'].apply(
            lambda x: [size.strip().upper() for size in str(x).split(',')] if pd.notna(x) else []
        )
        
        # Clean and standardize fabric names
        df_clean['fabric'] = df_clean['fabric'].fillna('').str.strip()
        df_clean['fabric_clean'] = df_clean['fabric'].str.lower()
        
        # Clean other text fields
        text_fields = ['fit', 'color_or_print', 'sleeve_length', 'occasion']
        for field in text_fields:
            if field in df_clean.columns:
                df_clean[f'{field}_clean'] = df_clean[field].fillna('').str.lower().str.strip()
        
        return df_clean
    
    def _build_fabric_families(self) -> Dict[str, List[str]]:
        """Build fabric families for intelligent matching"""
        return {
            'linen': ['cotton', 'organic cotton', 'linen-blend', 'cotton gauze'],
            'cotton': ['linen', 'organic cotton', 'cotton-blend', 'bamboo jersey'],
            'silk': ['satin', 'chiffon', 'crepe'],
            'satin': ['silk', 'crepe'],
            'velvet': ['satin', 'silk', 'crushed velvet'],
            'jersey': ['modal jersey', 'bamboo jersey', 'ribbed jersey'],
            'wool': ['wool-blend', 'cashmere', 'tweed'],
        }

    def recommend_products(self, user_prefs: Dict[str, Any], inferred_attrs: Dict[str, List[str]], 
                          confidence: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Recommend products based on user preferences and inferred attributes"""
        
        # Start with full catalog  
        df = self.catalog.copy()
        filter_summary = {
            'applied_filters': {},
            'relaxed_filters': [],
        }
        
        # Apply explicit user preferences, but only apply category and size (the most important)
        essential_prefs = {}
        if user_prefs.get('category'):
            essential_prefs['category'] = user_prefs['category']
        if user_prefs.get('size'):
            essential_prefs['size'] = user_prefs['size']
            
        df = self._apply_explicit_filters(df, essential_prefs, filter_summary)
        
        # Only apply budget and other filters if we have enough products remaining
        if len(df) >= 5 and user_prefs.get('budget'):
            # Create a temporary dataframe with budget filter applied
            temp_df = df.copy()
            price_column = None
            for possible_name in ['price', 'Price', 'unit_price', 'cost', 'amount']:
                if possible_name in temp_df.columns:
                    price_column = possible_name
                    break
                    
            if price_column:
                temp_df = temp_df[temp_df[price_column] <= user_prefs['budget']]
                # Only apply budget filter if it doesn't reduce results too much
                if len(temp_df) >= 3:
                    df = temp_df
                    filter_summary['applied_filters']['budget'] = user_prefs['budget']
                else:
                    filter_summary['relaxed_filters'].append('budget')
        
        # Apply very relaxed inferred attribute filters if we have plenty of products
        if len(df) >= 6 and confidence > 0.5:
            df = self._apply_inferred_filters(df, inferred_attrs, confidence, filter_summary)
            
        # If we have too few results, expand search
        if len(df) < 3:
            df = self._expand_search(user_prefs, inferred_attrs, filter_summary)
        
        # Sort by relevance
        df = self._rank_results(df, user_prefs, inferred_attrs)
        
        # Return between 3-8 products, prioritizing the top matches
        min_results = 3
        max_results = 8
        result_count = min(max(min_results, len(df)), max_results)
        
        return df.head(result_count).reset_index(drop=True), filter_summary
    
    def _apply_explicit_filters(self, df: pd.DataFrame, user_prefs: Dict[str, Any], 
                               filter_summary: Dict[str, Any]) -> pd.DataFrame:
        """Apply explicit user preferences with strict matching"""
        
        # Category filter
        if user_prefs.get('category'):
            categories = user_prefs['category'] if isinstance(user_prefs['category'], list) else [user_prefs['category']]
            df = df[df['category'].isin(categories)]
            filter_summary['applied_filters']['category'] = categories
        
        # Size filter  
        if user_prefs.get('size'):
            sizes = user_prefs['size'] if isinstance(user_prefs['size'], list) else [user_prefs['size']]
            size_mask = df['sizes_list'].apply(lambda x: any(size in x for size in sizes))
            df = df[size_mask]
            filter_summary['applied_filters']['size'] = sizes
        
        # Budget filter
        if user_prefs.get('budget'):
            df = df[df['price'] <= user_prefs['budget']]
            filter_summary['applied_filters']['budget'] = user_prefs['budget']
        
        # Sleeve length filter
        if user_prefs.get('sleeve_length'):
            sleeve_values = user_prefs['sleeve_length'] if isinstance(user_prefs['sleeve_length'], list) else [user_prefs['sleeve_length']]
            sleeve_mask = df['sleeve_length'].apply(
                lambda x: any(sleeve.lower() in str(x).lower() for sleeve in sleeve_values) if pd.notna(x) else False
            )
            df = df[sleeve_mask]
            filter_summary['applied_filters']['sleeve_length'] = sleeve_values
        
        return df
    
    def _apply_inferred_filters(self, df: pd.DataFrame, inferred_attrs: Dict[str, List[str]], 
                               confidence: float, filter_summary: Dict[str, Any]) -> pd.DataFrame:
        """Apply inferred attributes with confidence-based flexibility"""
        
        for attr_type, values in inferred_attrs.items():
            if not values or attr_type not in df.columns:
                continue
            
            if attr_type == 'fabric':
                df = self._apply_fabric_filter(df, values, confidence, filter_summary)
            elif attr_type == 'fit':
                df = self._apply_fit_filter(df, values, confidence, filter_summary)
            elif attr_type in ['color_or_print', 'occasion']:
                df = self._apply_text_filter(df, attr_type, values, confidence, filter_summary)
        
        return df
    
    def _apply_fabric_filter(self, df: pd.DataFrame, fabrics: List[str], confidence: float, 
                            filter_summary: Dict[str, Any]) -> pd.DataFrame:
        """Smart fabric filtering with family matching"""
        
        # Start with exact matches
        exact_mask = df['fabric_clean'].apply(
            lambda x: any(fabric.lower() in x for fabric in fabrics)
        )
        exact_matches = df[exact_mask]
        
        # If confidence is high and we have good matches, use them
        if confidence > 0.7 and len(exact_matches) >= 3:
            filter_summary['applied_filters']['fabric'] = fabrics
            return exact_matches
        
        # Otherwise, expand to fabric families
        expanded_fabrics = fabrics.copy()
        for fabric in fabrics:
            fabric_key = fabric.lower()
            if fabric_key in self.fabric_families:
                expanded_fabrics.extend(self.fabric_families[fabric_key])
        
        expanded_mask = df['fabric_clean'].apply(
            lambda x: any(fabric.lower() in x for fabric in expanded_fabrics)
        )
        
        filter_summary['applied_filters']['fabric'] = fabrics
        filter_summary['relaxed_filters'].append('fabric')
        
        return df[expanded_mask]
    
    def _apply_fit_filter(self, df: pd.DataFrame, fits: List[str], confidence: float,
                         filter_summary: Dict[str, Any]) -> pd.DataFrame:
        """Apply fit filter with smart matching"""
        
        # Fit synonym mapping
        fit_synonyms = {
            'relaxed': ['flowy', 'oversized'],
            'flowy': ['relaxed'],
            'body hugging': ['bodycon', 'tailored'],
            'tailored': ['body hugging'],
        }
        
        expanded_fits = fits.copy()
        for fit in fits:
            fit_key = fit.lower()
            if fit_key in fit_synonyms:
                expanded_fits.extend(fit_synonyms[fit_key])
        
        fit_mask = df['fit_clean'].apply(
            lambda x: any(fit.lower() in x for fit in expanded_fits) if x else False
        )
        
        filter_summary['applied_filters']['fit'] = fits
        return df[fit_mask]
    
    def _apply_text_filter(self, df: pd.DataFrame, attr_type: str, values: List[str], 
                          confidence: float, filter_summary: Dict[str, Any]) -> pd.DataFrame:
        """Apply text-based filters with partial matching"""
        
        clean_field = f'{attr_type}_clean'
        if clean_field in df.columns:
            mask = df[clean_field].apply(
                lambda x: any(value.lower() in x for value in values) if x else False
            )
            filtered_df = df[mask]
            
            # If we get good results, use them
            if len(filtered_df) >= 2 or confidence > 0.8:
                filter_summary['applied_filters'][attr_type] = values
                return filtered_df
        
        # If strict filtering yields few results, return original
        return df
    
    def _expand_search(self, user_prefs: Dict[str, Any], inferred_attrs: Dict[str, List[str]], 
                      filter_summary: Dict[str, Any]) -> pd.DataFrame:
        """Expand search when initial results are too few"""
        
        df = self.catalog.copy()
        
        # Apply only the most critical filters
        if user_prefs.get('category'):
            categories = user_prefs['category'] if isinstance(user_prefs['category'], list) else [user_prefs['category']]
            df = df[df['category'].isin(categories)]
        
        if user_prefs.get('budget'):
            # Expand budget by 5%
            expanded_budget = user_prefs['budget'] * 1.05
            df = df[df['price'] <= expanded_budget]
            filter_summary['relaxed_filters'].append('budget')
        
        if user_prefs.get('size'):
            sizes = user_prefs['size'] if isinstance(user_prefs['size'], list) else [user_prefs['size']]
            size_mask = df['sizes_list'].apply(lambda x: any(size in x for size in sizes))
            df = df[size_mask]
        
        filter_summary['match_strategy'] = 'expanded'
        return df
    
    def _rank_results(self, df: pd.DataFrame, user_prefs: Dict[str, Any], 
                     inferred_attrs: Dict[str, List[str]]) -> pd.DataFrame:
        """Rank results by relevance score"""
        
        if len(df) == 0:
            return df
        
        # Calculate relevance scores
        scores = []
        for _, product in df.iterrows():
            score = 0
            
            # Price scoring (prefer items closer to budget)
            if user_prefs.get('budget'):
                price_ratio = product['price'] / user_prefs['budget']
                if price_ratio <= 1:
                    score += (1 - price_ratio) * 0.3  # Prefer items well within budget
            
            # Fabric match scoring
            if 'fabric' in inferred_attrs:
                fabric_text = str(product['fabric']).lower()
                for fabric in inferred_attrs['fabric']:
                    if fabric.lower() in fabric_text:
                        score += 0.4
                        break
            
            # Fit match scoring
            if 'fit' in inferred_attrs and pd.notna(product['fit']):
                fit_text = str(product['fit']).lower()
                for fit in inferred_attrs['fit']:
                    if fit.lower() in fit_text:
                        score += 0.3
                        break
            
            scores.append(score)
        
        df['relevance_score'] = scores
        return df.sort_values('relevance_score', ascending=False).drop('relevance_score', axis=1)

class GeminiAgent:
    """Enhanced Gemini API integration for natural responses"""
    
    def __init__(self, api_key: str):
        
        self.client = gemini_client
    
    def generate_justification(self, user_query: str, recommended_products: pd.DataFrame, 
                             user_prefs: Dict[str, Any], inferred_attrs: Dict[str, List[str]], 
                             filter_summary: Dict[str, Any]) -> str:
            """Generate personalized justification for recommendations"""
        
        # try:
            context = f"""
            User's original request: "{user_query}"r
            User preferences: {json.dumps(user_prefs, indent=2)}r
            Inferred style attributes: {json.dumps(inferred_attrs, indent=2)}
            
            Recommended products:
            {recommended_products[['name', 'category', 'fabric', 'fit', 'price', 'color_or_print']].to_string()}
            
            Filter strategy used: {filter_summary.get('match_strategy', 'standard')}
            Relaxed filters: {filter_summary.get('relaxed_filters', [])}
            """
            
            prompt = """
            As a knowledgeable fashion stylist, create a warm, personalized justification for these product recommendations. 
            
            Guidelines:
            - Start with enthusiasm about their style direction
            - Explain WHY these pieces work for their vibe (mention specific attributes like fabrics, fits)
            - If any filters were relaxed, explain the reasoning positively
            - Keep it conversational and under 30 words
            - End with encouragement about how great they'll look
            
            Example tone: "I love your casual summer brunch vibe! I selected these breathable linen and cotton pieces in relaxed fits that'll feel effortless yet put-together. The sleeveless styles are perfect for warm weather, and everything's under your $100 budget. These pieces will have you looking effortlessly chic!"
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"{context}\n\n{prompt}"
            )
            
            return response.text.strip()
            
        # except Exception as e:
        #     # Fallback to template-based justification
        #     return self._generate_template_justification(user_prefs, inferred_attrs, filter_summary)
    
    def _generate_template_justification(self, user_prefs: Dict[str, Any], 
                                       inferred_attrs: Dict[str, List[str]], 
                                       filter_summary: Dict[str, Any]) -> str:
        """Fallback template-based justification"""
        
        justification_parts = []
        
        # Opening based on inferred style
        if 'fabric' in inferred_attrs and any('linen' in f.lower() or 'cotton' in f.lower() for f in inferred_attrs['fabric']):
            justification_parts.append("I love your fresh, breathable style direction!")
        elif 'fabric' in inferred_attrs and any('silk' in f.lower() or 'satin' in f.lower() for f in inferred_attrs['fabric']):
            justification_parts.append("Perfect choice for an elevated, luxurious look!")
        else:
            justification_parts.append("Great style vision!")
        
        # Mention key attributes
        if 'fabric' in inferred_attrs:
            fabrics = ', '.join(inferred_attrs['fabric'][:2])
            justification_parts.append(f"I selected pieces in {fabrics}")
        
        if 'fit' in inferred_attrs:
            fits = ', '.join(inferred_attrs['fit'][:2])
            justification_parts.append(f"with {fits} fits")
        
        # Budget mention
        if user_prefs.get('budget'):
            justification_parts.append(f"all under ${user_prefs['budget']}")
        
        # Relaxed filters explanation
        if filter_summary.get('relaxed_filters'):
            justification_parts.append("I found some great alternatives that capture your style perfectly")
        
        return ' '.join(justification_parts) + ". You're going to look amazing! ✨"

class ShoppingAgent:
    """Main conversational shopping agent with enhanced natural flow"""
    
    def __init__(self, catalog_df: pd.DataFrame, gemini_api_key: str = None):
        self.vibe_mapper = VibeMapper()
        self.followup_engine = FollowUpEngine()
        self.rec_engine = RecommendationEngine(catalog_df)
        self.gemini_agent = GeminiAgent(gemini_api_key) if gemini_api_key else None
        
        # Initialize enhanced session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state with enhanced tracking"""
        defaults = {
            'conversation_state': "initial",
            'user_preferences': {},
            'inferred_attributes': {},
            'detected_vibes': [],
            'confidence_score': 0.0,
            'questions_asked': 0,
            'original_query': "",
            'current_recommendations': pd.DataFrame(),
            'filter_summary': {}
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def process_initial_query(self, query: str) -> str:
        """Enhanced initial query processing with better intelligence"""
        
        # Store original query for context
        st.session_state.original_query = query
        
        # Extract vibes with confidence scoring
        inferred_attrs, confidence, detected_vibes = self.vibe_mapper.extract_vibes_with_confidence(query)
        st.session_state.inferred_attributes = inferred_attrs
        st.session_state.confidence_score = confidence
        st.session_state.detected_vibes = detected_vibes
        
        # No need to extract detected vibes separately as the method now returns them directly
        
        # Parse explicit preferences
        user_prefs = self._parse_explicit_preferences(query)
        st.session_state.user_preferences.update(user_prefs)
        
        # Decide on follow-up strategy based on confidence and completeness
        missing_critical = self._get_missing_critical_info()
        
        # If confidence is high and we have most info, go straight to recommendations
        if confidence > 0.8 and len(missing_critical) <= 1:
            return self._generate_enhanced_recommendations()
        
        # If confidence is medium-high and only missing 1 critical item, ask 1 question
        elif confidence > 0.6 and len(missing_critical) == 1:
            questions = self.followup_engine.generate_smart_questions(
                st.session_state.user_preferences, 
                st.session_state.inferred_attributes,
                st.session_state.detected_vibes,
                st.session_state.confidence_score
            )
            
            if questions:
                st.session_state.conversation_state = "follow_up"
                st.session_state.questions_asked = 1
                
                # Use first question directly as it now includes an opener
                return questions[0]
        
        # Otherwise, ask strategic follow-up questions
        questions = self.followup_engine.generate_smart_questions(
            st.session_state.user_preferences, 
            inferred_attrs, 
            detected_vibes, 
            confidence
        )
        
        if questions:
            st.session_state.conversation_state = "follow_up"
            st.session_state.questions_asked = len(questions)
            
            # Questions now include an opener
            if len(questions) == 1:
                return questions[0]
            else:
                return f"{questions[0]}\n\nAlso: {questions[1]}"
        
        # Fallback to recommendations
        return self._generate_recommendations()

    def process_followup_response(self, response: str) -> str:
        """Enhanced followup response processing"""
        
        # Parse the response
        parsed_prefs = self._parse_followup_response(response)
        st.session_state.user_preferences.update(parsed_prefs)
        
        # Check if we have enough information now
        missing_critical = self._get_missing_critical_info()
        
        # If we still have critical missing info and haven't asked too many questions
        if missing_critical and st.session_state.questions_asked < 2:
            questions = self.followup_engine.generate_smart_questions(
                st.session_state.user_preferences,
                st.session_state.inferred_attributes,
                st.session_state.detected_vibes,
                st.session_state.confidence_score
            )
            
            if questions:
                st.session_state.questions_asked += 1
                return f"{questions[0]}"
        
        # Move to recommendations
        st.session_state.conversation_state = "recommendations"
        return self._generate_recommendations()

    def _get_missing_critical_info(self) -> List[str]:
        """Identify missing critical information"""
        critical = ["category", "size", "budget"]
        return [info for info in critical if not st.session_state.user_preferences.get(info)]

    def _parse_explicit_preferences(self, query: str) -> Dict[str, Any]:
        """Enhanced explicit preference parsing"""
        prefs = {}
        query_lower = query.lower()
        
        # Budget parsing (multiple patterns)
        budget_patterns = [
            r'under?\s*\$?(\d+)',
            r'below\s*\$?(\d+)', 
            r'less than\s*\$?(\d+)',
            r'budget\s*:?\s*\$?(\d+)',
            r'\$(\d+)\s*or\s*less',
            r'max\s*\$?(\d+)'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, query_lower)
            if match:
                prefs['budget'] = int(match.group(1))
                break
        
        # Size parsing (enhanced)
        size_pattern = r'\b(xs|s|m|l|xl|small|medium|large|extra small|extra large|\d+)\b'
        sizes = re.findall(size_pattern, query_lower)
        if sizes:
            size_map = {
                'xs': 'XS', 'extra small': 'XS',
                's': 'S', 'small': 'S', 
                'm': 'M', 'medium': 'M',
                'l': 'L', 'large': 'L',
                'xl': 'XL', 'extra large': 'XL'
            }
            prefs['size'] = [size_map.get(size, size.upper()) for size in sizes]
        
        # Category parsing (enhanced)
        categories = []
        category_patterns = {
            'dress': r'\b(dress|dresses)\b',
            'top': r'\b(top|tops|shirt|shirts|blouse|blouses|tee|tees)\b',
            'skirt': r'\b(skirt|skirts)\b', 
            'pants': r'\b(pants|trousers|jeans)\b'
        }
        
        for category, pattern in category_patterns.items():
            if re.search(pattern, query_lower):
                categories.append(category)
        
        if categories:
            prefs['category'] = categories
        
        # Sleeve length parsing
        if re.search(r'\b(sleeveless|no sleeves|tank)\b', query_lower):
            prefs['sleeve_length'] = ['Sleeveless', 'Spaghetti straps', 'Halter', 'Tank']
        elif re.search(r'\b(short sleeve|short sleeves)\b', query_lower):
            prefs['sleeve_length'] = ['Short sleeves', 'Short flutter sleeves', 'Cap sleeves']
        elif re.search(r'\b(long sleeve|long sleeves)\b', query_lower):
            prefs['sleeve_length'] = ['Long sleeves', 'Full sleeves']
        
        return prefs

    def _parse_followup_response(self, response: str) -> Dict[str, Any]:
        """Enhanced followup response parsing"""
        prefs = {}
        response_lower = response.lower()
        
        # Category parsing from responses
        if any(word in response_lower for word in ['dress', 'dresses']):
            prefs['category'] = prefs.get('category', []) + ['dress']
        if any(word in response_lower for word in ['top', 'tops', 'shirt', 'blouse']):
            prefs['category'] = prefs.get('category', []) + ['top']
        if any(word in response_lower for word in ['skirt', 'skirts']):
            prefs['category'] = prefs.get('category', []) + ['skirt']
        if any(word in response_lower for word in ['pants', 'jeans']):
            prefs['category'] = prefs.get('category', []) + ['pants']
        
        # Budget parsing
        budget_match = re.search(r'\$?(\d+)', response_lower)
        if budget_match:
            prefs['budget'] = int(budget_match.group(1))
        
        # Size parsing
        size_pattern = r'\b(xs|s|m|l|xl)\b'
        sizes = re.findall(size_pattern, response_lower)
        if sizes:
            prefs['size'] = [size.upper() for size in sizes]
        
        # Sleeve preferences
        if 'sleeveless' in response_lower or 'no sleeve' in response_lower:
            prefs['sleeve_length'] = ['Sleeveless', 'Spaghetti straps', 'Halter']
        
        return prefs

    def _generate_recommendations(self) -> str:
        """Generate personalized product recommendations"""
        
        # Get recommendations
        results, filter_summary = self.rec_engine.recommend_products(
            st.session_state.user_preferences,
            st.session_state.inferred_attributes,
            st.session_state.confidence_score
        )
        
        st.session_state.current_recommendations = results
        st.session_state.filter_summary = filter_summary
        
        # Handle no results gracefully
        if len(results) == 0:
            return self._generate_no_results_response()
        
        # Generate justification
        if self.gemini_agent:
            try:
                justification = self.gemini_agent.generate_justification(
                    st.session_state.original_query,
                    results,
                    st.session_state.user_preferences,
                    st.session_state.inferred_attributes,
                    filter_summary
                )
            except:
                justification = self._generate_template_justification(filter_summary)
        else:
            justification = self._generate_template_justification(filter_summary)
        
        # Create response
        count_text = f"{len(results)} perfect piece{'s' if len(results) != 1 else ''}"
        return f"{justification}\n\nI found {count_text} that match your style! 👗✨"

    def _generate_no_results_response(self) -> str:
        """Generate helpful response when no products match"""
        missing_info = self._get_missing_critical_info()
        
        if missing_info:
            return f"I'd love to find the perfect pieces for you! Could you help me with your {missing_info[0]} preference so I can show you some great options?"
        else:
            return "I don't see exact matches for those specific criteria, but I'd love to find alternatives! Would you like me to expand the search or adjust any of your preferences?"

    def _generate_template_justification(self, filter_summary: Dict[str, Any]) -> str:
        """Create justification from templates"""
        parts = []
        
        # Enthusiasm based on style
        if 'fabric' in st.session_state.inferred_attributes:
            fabrics = st.session_state.inferred_attributes['fabric']
            if any('linen' in f.lower() or 'cotton' in f.lower() for f in fabrics):
                parts.append("I love your fresh, breathable style choice!")
            elif any('silk' in f.lower() or 'satin' in f.lower() for f in fabrics):
                parts.append("Perfect - you have great taste in elevated pieces!")
            else:
                parts.append("Great style direction!")
        
        # Mention key selections
        selections = []
        if 'fabric' in st.session_state.inferred_attributes:
            fabrics = ', '.join(st.session_state.inferred_attributes['fabric'][:2])
            selections.append(f"selected {fabrics} pieces")
        
        if 'fit' in st.session_state.inferred_attributes:
            fits = ', '.join(st.session_state.inferred_attributes['fit'][:2])
            selections.append(f"in {fits} fits")
        
        if selections:
            parts.append(f"I {' '.join(selections)}")
        
        # Budget mention
        if st.session_state.user_preferences.get('budget'):
            parts.append(f"all under ${st.session_state.user_preferences['budget']}")
        
        # Relaxed filters mention
        if filter_summary.get('relaxed_filters'):
            parts.append("with some perfect alternatives that capture your vibe")
        
        parts.append("You're going to look amazing!")
        
        return ' '.join(parts) + " ✨"

def display_product_cards(products_df: pd.DataFrame):
    """Display product cards with formatted details"""
    if products_df.empty:
        return
    
    st.markdown("### 🛍️ Your Perfect Matches")
    
    # Display in a grid format
    cols = st.columns(2)
    
    for idx, (_, product) in enumerate(products_df.iterrows()):
        col = cols[idx % 2]
        
        with col:
            with st.container():
                # Product name and price header
                st.markdown(f"**{product['name']}** - `${product['price']}`")
                
                # Key details
                details = []
                details.append(f"📂 {product['category'].title()}")
                
                if pd.notna(product['fabric']):
                    details.append(f"🧵 {product['fabric']}")
                
                if pd.notna(product['fit']):
                    details.append(f"👗 {product['fit']}")
                
                if pd.notna(product['color_or_print']):
                    details.append(f"🎨 {product['color_or_print']}")
                
                if pd.notna(product['sleeve_length']):
                    details.append(f"👕 {product['sleeve_length']}")
                
                details.append(f"📏 Sizes: {product['available_sizes']}")
                
                for detail in details:
                    st.markdown(f"<small>{detail}</small>", unsafe_allow_html=True)
                
                st.markdown("---")

def main():
    """Enhanced main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌟 Vibe Mapping Shopping Assistant</h1>', unsafe_allow_html=True)
    st.markdown("*Tell me your style vibe, and I'll find your perfect pieces* ✨")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key
        api_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            help="Add your Gemini API key for enhanced AI responses!"
        )
        
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
        
        st.markdown("---")
        
        # Instructions
        st.markdown("### 💡 How to shop:")
        st.markdown("• **Share your vibe**: *'something cute for brunch'*")
        st.markdown("• **Answer 1-2 questions** (if I need to know more)")  
        st.markdown("• **Get personalized picks** with styling tips!")
        
        # Confidence indicator
        if 'confidence_score' in st.session_state and st.session_state.confidence_score > 0:
            confidence = st.session_state.confidence_score
            if confidence > 0.65:
                st.success(f"🎯 Style confidence: {confidence:.1%} - I've got your vibe!")
            elif confidence > 0.45:
                st.info(f"✨ Style confidence: {confidence:.1%} - Getting there!")
            else:
                st.warning(f"🤔 Style confidence: {confidence:.1%} - Tell me more!")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Start Fresh"):
            defaults = {
                'conversation_state': "initial",
                'user_preferences': {},
                'inferred_attributes': {},
                'detected_vibes': [],
                'confidence_score': 0.0,
                'questions_asked': 0,
                'original_query': "",
                'current_recommendations': pd.DataFrame(),
                'filter_summary': {},
                'messages': []
            }
            keys_to_clear = ['conversation_state', 'user_preferences', 'inferred_attributes', 
                           'detected_vibes', 'confidence_score', 'questions_asked', 'original_query',
                           'current_recommendations', 'filter_summary', 'messages']
            for key in keys_to_clear:
                if key in st.session_state:
                    st.session_state[key] = defaults.get(key, [])
            st.rerun()
    
    # Load catalog
    @st.cache_data
    def load_catalog():
        try:
            return pd.read_excel('Apparels_shared.xlsx')
        except FileNotFoundError:
            # Create sample data for demo
            return pd.DataFrame({
                'id': ['T001', 'D001'],
                'name': ['Sample Top', 'Sample Dress'],
                'category': ['top', 'dress'],
                'available_sizes': ['S,M,L', 'XS,S,M'],
                'fit': ['Relaxed', 'Flowy'],
                'fabric': ['Cotton', 'Linen'],
                'sleeve_length': ['Short sleeves', 'Sleeveless'],
                'color_or_print': ['Blue', 'White'],
                'occasion': ['Casual', 'Vacation'],
                'price': [50, 95]
            })
    
    try:
        catalog_df = load_catalog()
        
        if len(catalog_df) < 10:  # Sample data
            st.warning("📁 Using sample data. Upload 'Apparels_shared.xlsx' for full catalog.")
        
    except Exception as e:
        st.error(f"Error loading catalog: {str(e)}")
        return
    
    # Initialize agent
    agent = ShoppingAgent(catalog_df, api_key if api_key else None)
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show products if it's the assistant's final recommendation
            if (message["role"] == "assistant" and 
                'current_recommendations' in st.session_state and 
                len(st.session_state.current_recommendations) > 0):
                display_product_cards(st.session_state.current_recommendations)
    
    # Chat input
    if prompt := st.chat_input("Share your style vibe... ✨", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Finding your perfect style... 💫"):
                try:
                    if st.session_state.conversation_state == "initial":
                        response = agent.process_initial_query(prompt)
                    else:
                        response = agent.process_followup_response(prompt)
                    
                    st.markdown(response)
                    
                    # Show products if available
                    if ('current_recommendations' in st.session_state and 
                        len(st.session_state.current_recommendations) > 0):
                        display_product_cards(st.session_state.current_recommendations)
                
                except Exception as e:
                    print(f"Error processing query: {e}")
                    error_response = f"I'm having a small hiccup! 😅 Let me try a different approach. Could you tell me what type of clothing you're looking for and your size?"
                    st.markdown(error_response)
                    response = error_response
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
