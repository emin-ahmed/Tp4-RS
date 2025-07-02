import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mauritania Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .restaurant-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained recommender model"""
    try:
        with open('mauritania_restaurant_recommender.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("❌ Model file 'mauritania_restaurant_recommender.pkl' not found. Please make sure the file is in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def extract_keywords_from_name(restaurant_name):
    """Extract keywords and features from restaurant name"""
    name_lower = restaurant_name.lower()
    
    # Define cuisine/style keywords
    cuisine_keywords = {
        'italian': ['italian', 'pizza', 'pasta', 'trattoria', 'pizzeria', 'roma', 'napoli'],
        'french': ['french', 'cafe', 'bistro', 'brasserie', 'chez', 'le ', 'la ', 'les'],
        'chinese': ['chinese', 'dragon', 'golden', 'jade', 'peking', 'beijing'],
        'mexican': ['mexican', 'taco', 'cantina', 'hacienda', 'el ', 'la '],
        'indian': ['indian', 'taj', 'curry', 'spice', 'tandoor', 'maharaja'],
        'japanese': ['japanese', 'sushi', 'sake', 'tokyo', 'sakura', 'zen'],
        'american': ['grill', 'diner', 'burger', 'steakhouse', 'bbq', 'smokehouse'],
        'mediterranean': ['mediterranean', 'olive', 'greek', 'cyprus', 'santorini'],
        'seafood': ['seafood', 'fish', 'ocean', 'marina', 'catch', 'lobster', 'crab'],
        'fast_food': ['fast', 'quick', 'express', 'drive', 'burger', 'chicken']
    }
    
    # Define ambiance keywords
    ambiance_keywords = {
        'fine_dining': ['fine', 'elegant', 'luxury', 'premium', 'exclusive', 'royal'],
        'casual': ['casual', 'family', 'home', 'corner', 'neighborhood'],
        'romantic': ['romantic', 'intimate', 'cozy', 'candlelight', 'moonlight'],
        'sports': ['sports', 'game', 'stadium', 'victory', 'champion'],
        'coffee': ['coffee', 'cafe', 'espresso', 'bean', 'roast', 'brew']
    }
    
    detected_features = {
        'cuisine_type': 'general',
        'ambiance': 'casual',
        'price_level': 'medium',
        'keywords': []
    }
    
    # Extract cuisine type
    for cuisine, keywords in cuisine_keywords.items():
        if any(keyword in name_lower for keyword in keywords):
            detected_features['cuisine_type'] = cuisine
            detected_features['keywords'].extend([k for k in keywords if k in name_lower])
            break
    
    # Extract ambiance
    for amb, keywords in ambiance_keywords.items():
        if any(keyword in name_lower for keyword in keywords):
            detected_features['ambiance'] = amb
            detected_features['keywords'].extend([k for k in keywords if k in name_lower])
            break
    
    # Extract price indicators
    if any(word in name_lower for word in ['luxury', 'premium', 'fine', 'exclusive', 'royal']):
        detected_features['price_level'] = 'high'
    elif any(word in name_lower for word in ['budget', 'cheap', 'quick', 'fast', 'express']):
        detected_features['price_level'] = 'low'
    
    return detected_features

def get_restaurant_recommendations_by_name(model_package, query_name):
    """Get restaurant recommendations based on a user-provided restaurant name (not necessarily in dataset)"""
    
    if model_package is None:
        return []
    
    try:
        # Extract components from model package
        restaurant_features = model_package['restaurant_features']
        restaurant_averages = model_package['restaurant_averages']
        
        # Extract features from the query name
        query_features = extract_keywords_from_name(query_name)
        
        # Get all restaurant names for text similarity
        # Try different possible name columns
        name_column = None
        possible_name_cols = ['name', 'restaurant_name', 'business_name', 'title', 'resto_name']
        
        for col in possible_name_cols:
            if col in restaurant_features.columns:
                name_column = col
                break
        
        if name_column:
            restaurant_names = restaurant_features[name_column].fillna(f'Restaurant {restaurant_features.index}')
        else:
            restaurant_names = pd.Series([f'Restaurant {idx}' for idx in restaurant_features.index], 
                                       index=restaurant_features.index)
        
        # Create TF-IDF vectorizer for name similarity
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
        
        # Combine query with all restaurant names for vectorization
        all_names = [query_name] + restaurant_names.tolist()
        tfidf_matrix = tfidf.fit_transform(all_names)
        
        # Calculate similarity between query and all restaurants
        query_vector = tfidf_matrix[0]  # First row is our query
        restaurant_vectors = tfidf_matrix[1:]  # Rest are restaurant names
        
        # Calculate cosine similarity
        text_similarities = cosine_similarity(query_vector, restaurant_vectors)[0]
        
        # Create feature-based similarity scores
        feature_similarities = []
        
        for idx, rest_id in enumerate(restaurant_features.index):
            rest_info = restaurant_features.loc[rest_id]
            feature_score = 0.0
            
            # Check for cuisine type matches (if available in restaurant features)
            cuisine_cols = [col for col in restaurant_features.columns if 'cuisine' in col.lower() or 'category' in col.lower()]
            for col in cuisine_cols:
                if pd.notna(rest_info.get(col)) and query_features['cuisine_type'] in str(rest_info[col]).lower():
                    feature_score += 0.3
            
            # Check for keyword matches in restaurant name
            rest_name = str(rest_info.get('name', '')).lower()
            keyword_matches = sum(1 for keyword in query_features['keywords'] if keyword in rest_name)
            feature_score += keyword_matches * 0.2
            
            # Check for ambiance/style matches
            style_cols = [col for col in restaurant_features.columns if any(term in col.lower() for term in ['style', 'type', 'ambiance'])]
            for col in style_cols:
                if pd.notna(rest_info.get(col)) and query_features['ambiance'] in str(rest_info[col]).lower():
                    feature_score += 0.25
            
            feature_similarities.append(feature_score)
        
        # Combine text and feature similarities
        combined_similarities = []
        for i in range(len(text_similarities)):
            # Weighted combination: 60% text similarity, 40% feature similarity
            combined_score = 0.6 * text_similarities[i] + 0.4 * feature_similarities[i]
            combined_similarities.append(combined_score)
        
        # Get top 5 most similar restaurants
        top_indices = np.argsort(combined_similarities)[::-1][:5]
        
        recommendations = []
        for idx in top_indices:
            rest_id = restaurant_features.index[idx]
            text_sim = text_similarities[idx]
            feature_sim = feature_similarities[idx]
            combined_sim = combined_similarities[idx]
            
            # Get restaurant info
            rest_info = restaurant_features.loc[rest_id]
            avg_rating = restaurant_averages.get(rest_id, 0) if restaurant_averages is not None else 0
            
            # Get restaurant name - try multiple possible column names
            restaurant_name = None
            possible_name_cols = ['name', 'restaurant_name', 'business_name', 'title', 'resto_name']
            
            for col in possible_name_cols:
                if col in rest_info.index and pd.notna(rest_info[col]):
                    restaurant_name = str(rest_info[col])
                    break
            
            if not restaurant_name:
                restaurant_name = f'Restaurant {rest_id}'
            
            recommendations.append({
                'restaurant_id': rest_id,
                'name': restaurant_name,
                'text_similarity': text_sim,
                'feature_similarity': feature_sim,
                'combined_similarity': combined_sim,
                'avg_rating': avg_rating,
                'features': rest_info.to_dict(),
                'match_reasons': get_match_reasons(query_features, rest_info)
            })
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

def get_match_reasons(query_features, restaurant_info):
    """Get reasons why this restaurant matches the query"""
    reasons = []
    
    # Try to get restaurant name from multiple possible columns
    restaurant_name = None
    possible_name_cols = ['name', 'restaurant_name', 'business_name', 'title', 'resto_name']
    
    for col in possible_name_cols:
        if col in restaurant_info.index and pd.notna(restaurant_info[col]):
            restaurant_name = str(restaurant_info[col]).lower()
            break
    
    if not restaurant_name:
        restaurant_name = f'restaurant {restaurant_info.name}'.lower()
    
    # Check cuisine match
    if query_features['cuisine_type'] != 'general':
        cuisine_keywords = {
            'italian': ['italian', 'pizza', 'pasta', 'trattoria', 'pizzeria'],
            'french': ['french', 'cafe', 'bistro', 'brasserie'],
            'chinese': ['chinese', 'dragon', 'golden', 'jade'],
            'mexican': ['mexican', 'taco', 'cantina', 'hacienda'],
            'seafood': ['seafood', 'fish', 'ocean', 'marina', 'catch']
        }
        
        if query_features['cuisine_type'] in cuisine_keywords:
            for keyword in cuisine_keywords[query_features['cuisine_type']]:
                if keyword in restaurant_name:
                    reasons.append(f"Similar cuisine style ({keyword})")
                    break
    
    # Check keyword matches
    for keyword in query_features['keywords']:
        if keyword in restaurant_name:
            reasons.append(f"Name contains '{keyword}'")
    
    # Check ambiance match
    if query_features['ambiance'] != 'casual':
        ambiance_keywords = {
            'fine_dining': ['fine', 'elegant', 'luxury', 'premium'],
            'romantic': ['romantic', 'intimate', 'cozy'],
            'coffee': ['coffee', 'cafe', 'espresso', 'bean']
        }
        
        if query_features['ambiance'] in ambiance_keywords:
            for keyword in ambiance_keywords[query_features['ambiance']]:
                if keyword in restaurant_name:
                    reasons.append(f"Similar ambiance ({keyword})")
                    break
    
    if not reasons:
        reasons.append("Text similarity match")
    
    return reasons

def get_top_rated_restaurants(model_package):
    """Get top-rated restaurants"""
    try:
        restaurant_features = model_package['restaurant_features']
        restaurant_averages = model_package['restaurant_averages']
        
        if restaurant_averages is not None:
            top_restaurants = sorted(restaurant_averages.items(), key=lambda x: x[1], reverse=True)[:5]
            
            recommendations = []
            for rest_id, avg_rating in top_restaurants:
                if rest_id in restaurant_features.index:
                    rest_info = restaurant_features.loc[rest_id]
                    
                    # Get restaurant name - try multiple possible column names
                    restaurant_name = None
                    possible_name_cols = ['name', 'restaurant_name', 'business_name', 'title', 'resto_name']
                    
                    for col in possible_name_cols:
                        if col in rest_info.index and pd.notna(rest_info[col]):
                            restaurant_name = str(rest_info[col])
                            break
                    
                    if not restaurant_name:
                        restaurant_name = f'Restaurant {rest_id}'
                    
                    recommendations.append({
                        'restaurant_id': rest_id,
                        'name': restaurant_name,
                        'avg_rating': avg_rating,
                        'features': rest_info.to_dict()
                    })
            
            return recommendations
    except Exception as e:
        st.error(f"Error getting top restaurants: {str(e)}")
    
    return []

def display_recommendations(recommendations, show_similarity=True):
    """Display restaurant recommendations in a nice format"""
    if not recommendations:
        st.warning("No recommendations found.")
        return
    
    st.markdown("### 🎯 Top 5 Restaurant Recommendations")
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f"**#{i}**")
                if 'combined_similarity' in rec and show_similarity:
                    st.metric("Match Score", f"{rec['combined_similarity']:.2%}")
                if 'avg_rating' in rec:
                    st.metric("Avg Rating", f"{rec['avg_rating']:.1f}")
            
            with col2:
                st.markdown(f"### {rec['name']}")
                st.markdown(f"**Restaurant ID:** {rec['restaurant_id']}")
                
                # Show match reasons if available
                if 'match_reasons' in rec and rec['match_reasons']:
                    st.markdown("**Why this matches:**")
                    for reason in rec['match_reasons']:
                        st.markdown(f"• {reason}")
                
                # Show similarity breakdown if available
                if show_similarity and 'text_similarity' in rec:
                    with st.expander("📊 Similarity Details"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Name Similarity", f"{rec['text_similarity']:.2%}")
                        with col_b:
                            st.metric("Feature Match", f"{rec['feature_similarity']:.2%}")
                
                # Display additional features if available
                if 'features' in rec:
                    features = rec['features']
                    if 'address' in features and pd.notna(features['address']):
                        st.markdown(f"📍 **Address:** {features['address']}")
                    
                    # Display other relevant features
                    important_features = ['category', 'cuisine', 'price_range', 'atmosphere', 'specialties']
                    for key, value in features.items():
                        if (any(imp in key.lower() for imp in important_features) and 
                            pd.notna(value) and str(value) != '0' and key not in ['name', 'address']):
                            st.markdown(f"**{key.title()}:** {value}")
            
            with col3:
                st.markdown("🍽️")
            
            st.markdown("---")

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ Mauritania Restaurant Recommender</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading recommendation model..."):
        model_package = load_model()
    
    if model_package is None:
        st.stop()
    
    # Display model info
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.markdown(f"**Model Type:** {model_package.get('model_type', 'Unknown')}")
        st.markdown(f"**Training Date:** {model_package.get('training_date', 'Unknown')}")
        st.markdown(f"**Total Restaurants:** {model_package.get('total_restaurants', 'Unknown')}")
        st.markdown(f"**Total Users:** {model_package.get('total_users', 'Unknown')}")
        st.markdown(f"**Total Reviews:** {model_package.get('total_reviews', 'Unknown')}")
        
        if 'final_rmse' in model_package:
            st.markdown(f"**Model RMSE:** {model_package['final_rmse']:.3f}")
        if 'final_mae' in model_package:
            st.markdown(f"**Model MAE:** {model_package['final_mae']:.3f}")
    
    # Main interface
    st.markdown("## 🔍 Get Restaurant Recommendations")
    
    # Create tabs for different recommendation methods
    tab1, tab2 = st.tabs(["🎯 Find Similar Restaurants", "⭐ Top Rated"])
    
    with tab1:
        st.markdown("### Enter any restaurant name to find similar places")
        st.markdown("*You can enter any restaurant name - it doesn't need to be from our database!*")
        
        # Restaurant name input
        restaurant_name = st.text_input(
            "Restaurant Name:",
            placeholder="e.g., Pizza Palace, Le Bistro, Dragon Garden, Ocean View Seafood...",
            help="Enter any restaurant name. The system will analyze the name and find similar restaurants in Mauritania.",
            key="restaurant_name_input"
        )
        
        # Optional: Add some example buttons
        st.markdown("**Try these examples:**")
        col1, col2, col3, col4 = st.columns(4)
        

        final_restaurant_name = restaurant_name
        
        if st.button("🔍 Find Similar Restaurants", type="primary", key="main_search_button"):
            if final_restaurant_name and final_restaurant_name.strip():
                with st.spinner(f"Analyzing '{final_restaurant_name}' and finding similar restaurants..."):
                    # Show what the system detected
                    query_features = extract_keywords_from_name(final_restaurant_name)
                    
                    with st.expander("🔍 What we detected from your query"):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(f"**Cuisine Style:** {query_features['cuisine_type'].title()}")
                        with col_b:
                            st.markdown(f"**Ambiance:** {query_features['ambiance'].title()}")
                        with col_c:
                            st.markdown(f"**Price Level:** {query_features['price_level'].title()}")
                        
                        if query_features['keywords']:
                            st.markdown(f"**Keywords found:** {', '.join(query_features['keywords'])}")
                    
                    recommendations = get_restaurant_recommendations_by_name(model_package, final_restaurant_name)
                    display_recommendations(recommendations, show_similarity=True)
            else:
                st.warning("Please enter a restaurant name or click one of the examples.")
    
    with tab2:
        st.markdown("### Discover top-rated restaurants in Mauritania")
        
        if st.button("⭐ Show Top Rated Restaurants", type="primary", key="top_rated_button"):
            with st.spinner("Loading top-rated restaurants..."):
                recommendations = get_top_rated_restaurants(model_package)
                display_recommendations(recommendations, show_similarity=False)
    
    # Additional information
    st.markdown("---")
    st.markdown("## ℹ️ How it works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Smart Name Analysis:**
        - Analyzes any restaurant name you provide
        - Detects cuisine type (Italian, French, Chinese, etc.)
        - Identifies ambiance (fine dining, casual, romantic)
        - Extracts key features and style indicators
        - Uses TF-IDF for text similarity matching
        """)
    
    with col2:
        st.markdown("""
        **Recommendation Process:**
        - Combines text similarity (60%) with feature matching (40%)
        - Finds restaurants with similar names and characteristics
        - Considers cuisine type, ambiance, and style
        - Ranks by combined similarity score
        - Returns top 5 most relevant matches
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips for better recommendations:
    - Use descriptive names like "Ocean View Seafood" instead of just "Restaurant"
    - Include cuisine type in the name (e.g., "Mario's Italian Kitchen")
    - Mention style or ambiance (e.g., "Cozy Corner Cafe", "Elegant Dining Palace")
    - The system works with any language and restaurant name format
    """)

if __name__ == "__main__":
    main()