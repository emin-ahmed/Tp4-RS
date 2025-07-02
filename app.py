# # import streamlit as st
# # import pickle
# # import pandas as pd

# # @st.cache_resource
# # def load_model(path="mauritania_restaurant_recommender.pkl"):
# #     with open(path, "rb") as f:
# #         return pickle.load(f)

# # model = load_model()

# # st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
# # st.title("🍽️ Restaurant Recommender")

# # mode = st.sidebar.radio(
# #     "How would you like to get recommendations?",
# #     ("By User ID", "By Restaurant Name")
# # )

# # if mode == "By User ID":
# #     user_id = st.text_input("Enter User ID")
# #     if st.button("Recommend for User"):
# #         try:
# #             recs = model.recommend(user_id, top_k=5)
# #             df = pd.DataFrame(recs, columns=["Restaurant", "Score"])
# #             st.dataframe(df)
# #         except Exception as e:
# #             st.error(f"Error: {e}")
# # else:
# #     restaurant = st.text_input("Enter Restaurant Name")
# #     if st.button("Similar Restaurants"):
# #         try:
# #             sims = model.recommend_by_item(restaurant, top_k=5)
# #             df = pd.DataFrame(sims, columns=["Restaurant", "Similarity"])
# #             st.dataframe(df)
# #         except Exception as e:
# #             st.error(f"Error: {e}")

# # st.markdown("---")
# # st.markdown("### Batch Upload (CSV)")

# # uploaded = st.file_uploader("Upload CSV (column: user_id or restaurant)", type=["csv"])
# # if uploaded:
# #     df_in = pd.read_csv(uploaded)
# #     results = []
# #     key = "user_id" if mode == "By User ID" else "restaurant"

# #     for val in df_in[key].astype(str):
# #         if mode == "By User ID":
# #             recs = model.recommend(val, top_k=5)
# #             for item, score in recs:
# #                 results.append({key: val, "restaurant": item, "score": score})
# #         else:
# #             sims = model.recommend_by_item(val, top_k=5)
# #             for item, sim in sims:
# #                 results.append({key: val, "restaurant": item, "similarity": sim})

# #     out = pd.DataFrame(results)
# #     st.dataframe(out)  # display the full results in the app










# import streamlit as st
# import pickle
# import pandas as pd

# @st.cache_resource
# def load_model(path="mauritania_restaurant_recommender.pkl"):
#     with open(path, "rb") as f:
#         loaded_object = pickle.load(f)
    
#     # Check if it's a dictionary containing the model
#     if isinstance(loaded_object, dict):
#         # Common dictionary keys where model might be stored
#         possible_keys = ['model', 'recommender', 'algorithm', 'fitted_model']
        
#         for key in possible_keys:
#             if key in loaded_object:
#                 return loaded_object[key]
        
#         # If no common keys found, check all keys
#         for key, value in loaded_object.items():
#             if hasattr(value, 'recommend'):  # Found object with recommend method
#                 return value
        
#         # If still not found, raise informative error
#         st.error(f"Model not found in dictionary. Available keys: {list(loaded_object.keys())}")
#         st.stop()
    
#     return loaded_object

# try:
#     model = load_model()
# except Exception as e:
#     st.error(f"Failed to load model: {e}")
#     st.stop()

# st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
# st.title("🍽️ Restaurant Recommender")

# # Verify model has required methods
# if not hasattr(model, 'recommend'):
#     st.error("Model doesn't have 'recommend' method")
#     st.write(f"Model type: {type(model)}")
#     st.write(f"Available methods: {[method for method in dir(model) if not method.startswith('_')]}")
#     st.stop()

# mode = st.sidebar.radio(
#     "How would you like to get recommendations?",
#     ("By User ID", "By Restaurant Name")
# )

# if mode == "By User ID":
#     user_id = st.text_input("Enter User ID")
#     if st.button("Recommend for User"):
#         try:
#             recs = model.recommend(user_id, top_k=5)
#             df = pd.DataFrame(recs, columns=["Restaurant", "Score"])
#             st.dataframe(df)
#         except Exception as e:
#             st.error(f"Error: {e}")
# else:
#     restaurant = st.text_input("Enter Restaurant Name")
#     if st.button("Similar Restaurants"):
#         try:
#             # Check if method exists
#             if hasattr(model, 'recommend_by_item'):
#                 sims = model.recommend_by_item(restaurant, top_k=5)
#             else:
#                 st.error("Model doesn't have 'recommend_by_item' method")
#                 st.stop()
#             df = pd.DataFrame(sims, columns=["Restaurant", "Similarity"])
#             st.dataframe(df)
#         except Exception as e:
#             st.error(f"Error: {e}")

# st.markdown("---")
# st.markdown("### Batch Upload (CSV)")

# uploaded = st.file_uploader("Upload CSV (column: user_id or restaurant)", type=["csv"])
# if uploaded:
#     df_in = pd.read_csv(uploaded)
#     results = []
#     key = "user_id" if mode == "By User ID" else "restaurant"

#     for val in df_in[key].astype(str):
#         try:
#             if mode == "By User ID":
#                 recs = model.recommend(val, top_k=5)
#                 for item, score in recs:
#                     results.append({key: val, "restaurant": item, "score": score})
#             else:
#                 sims = model.recommend_by_item(val, top_k=5)
#                 for item, sim in sims:
#                     results.append({key: val, "restaurant": item, "similarity": sim})
#         except Exception as e:
#             st.error(f"Error processing {val}: {e}")
#             continue

#     if results:
#         out = pd.DataFrame(results)
#         st.dataframe(out)
#     else:
#         st.warning("No results generated.") 









import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

class RecommenderWrapper:
    def __init__(self, model_dict):
        self.model_dict = model_dict
        self.best_model_name = model_dict.get('best_model_name', 'svd_model')
        
        # Get the best model based on best_model_name
        if self.best_model_name == 'svd_model':
            self.model = model_dict['svd_model']
            self.reconstructed = model_dict.get('svd_reconstructed')
        elif self.best_model_name == 'ridge_model':
            self.model = model_dict['ridge_model']
        elif self.best_model_name == 'rf_model':
            self.model = model_dict['rf_model']
        else:
            # Default to SVD if available
            self.model = model_dict.get('svd_model', model_dict.get('ridge_model'))
        
        # Store other useful components
        self.train_matrix = model_dict.get('train_matrix')
        self.user_averages = model_dict.get('user_averages', {})
        self.restaurant_averages = model_dict.get('restaurant_averages', {})
        self.global_avg = model_dict.get('global_avg', 0)
        self.restaurant_features = model_dict.get('restaurant_features')
        
    def recommend(self, user_id, top_k=5):
        """Recommend restaurants for a user"""
        try:
            user_id_str = str(user_id)
            
            # If we have SVD reconstructed matrix, use it
            if self.reconstructed is not None and hasattr(self.train_matrix, 'index'):
                if user_id_str in self.train_matrix.index:
                    user_idx = self.train_matrix.index.get_loc(user_id_str)
                    user_scores = self.reconstructed[user_idx]
                    
                    # Get top recommendations (excluding already rated items)
                    rated_items = self.train_matrix.loc[user_id_str]
                    rated_indices = rated_items[rated_items > 0].index
                    
                    # Create scores dataframe
                    scores_df = pd.DataFrame({
                        'restaurant': self.train_matrix.columns,
                        'score': user_scores
                    })
                    
                    # Remove already rated restaurants
                    scores_df = scores_df[~scores_df['restaurant'].isin(rated_indices)]
                    
                    # Sort by score and get top k
                    top_recs = scores_df.nlargest(top_k, 'score')
                    return list(zip(top_recs['restaurant'], top_recs['score']))
            
            # Fallback: recommend popular restaurants
            if self.restaurant_averages:
                sorted_restaurants = sorted(self.restaurant_averages.items(), 
                                          key=lambda x: x[1], reverse=True)
                return sorted_restaurants[:top_k]
            
            return [("No recommendations available", 0.0)] * top_k
            
        except Exception as e:
            st.error(f"Error in recommend: {e}")
            return [("Error generating recommendations", 0.0)] * top_k
    
    def recommend_by_item(self, restaurant_name, top_k=5):
        """Find similar restaurants"""
        try:
            restaurant_name_str = str(restaurant_name)
            
            # If we have restaurant features, use them for similarity
            if (self.restaurant_features is not None and 
                restaurant_name_str in self.restaurant_features.index):
                
                target_features = self.restaurant_features.loc[restaurant_name_str].values.reshape(1, -1)
                similarities = cosine_similarity(target_features, self.restaurant_features.values)[0]
                
                # Create similarity dataframe
                sim_df = pd.DataFrame({
                    'restaurant': self.restaurant_features.index,
                    'similarity': similarities
                })
                
                # Remove the target restaurant itself
                sim_df = sim_df[sim_df['restaurant'] != restaurant_name_str]
                
                # Sort by similarity and get top k
                top_sims = sim_df.nlargest(top_k, 'similarity')
                return list(zip(top_sims['restaurant'], top_sims['similarity']))
            
            # Fallback using train matrix if available
            elif (self.train_matrix is not None and 
                  restaurant_name_str in self.train_matrix.columns):
                
                target_ratings = self.train_matrix[restaurant_name_str].fillna(0)
                similarities = []
                
                for col in self.train_matrix.columns:
                    if col != restaurant_name_str:
                        other_ratings = self.train_matrix[col].fillna(0)
                        # Calculate correlation as similarity measure
                        corr = np.corrcoef(target_ratings, other_ratings)[0, 1]
                        if not np.isnan(corr):
                            similarities.append((col, corr))
                
                # Sort by similarity and return top k
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]
            
            # Final fallback
            if self.restaurant_averages:
                sorted_restaurants = sorted(self.restaurant_averages.items(), 
                                          key=lambda x: x[1], reverse=True)
                return sorted_restaurants[:top_k]
            
            return [("No similar restaurants found", 0.0)] * top_k
            
        except Exception as e:
            st.error(f"Error in recommend_by_item: {e}")
            return [("Error finding similar restaurants", 0.0)] * top_k

@st.cache_resource
def load_model(path="mauritania_restaurant_recommender.pkl"):
    with open(path, "rb") as f:
        loaded_object = pickle.load(f)
    
    # Check if it's a dictionary containing the model components
    if isinstance(loaded_object, dict):
        # Wrap the dictionary in our custom recommender class
        return RecommenderWrapper(loaded_object)
    
    return loaded_object

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
st.title("🍽️ Restaurant Recommender")

# Display model information
if hasattr(model, 'model_dict'):
    st.sidebar.markdown("### Model Information")
    best_model = model.model_dict.get('best_model_name', 'Unknown')
    st.sidebar.write(f"**Best Model:** {best_model}")
    
    if 'final_rmse' in model.model_dict:
        st.sidebar.write(f"**RMSE:** {model.model_dict['final_rmse']:.4f}")
    if 'final_mae' in model.model_dict:
        st.sidebar.write(f"**MAE:** {model.model_dict['final_mae']:.4f}")
    
    total_users = model.model_dict.get('total_users', 'Unknown')
    total_restaurants = model.model_dict.get('total_restaurants', 'Unknown')
    st.sidebar.write(f"**Users:** {total_users}")
    st.sidebar.write(f"**Restaurants:** {total_restaurants}")

# Verify model has required methods
if not hasattr(model, 'recommend'):
    st.error("Model doesn't have 'recommend' method")
    st.write(f"Model type: {type(model)}")
    st.write(f"Available methods: {[method for method in dir(model) if not method.startswith('_')]}")
    st.stop()

mode = st.sidebar.radio(
    "How would you like to get recommendations?",
    ("By User ID", "By Restaurant Name")
)

if mode == "By User ID":
    user_id = st.text_input("Enter User ID")
    if st.button("Recommend for User"):
        if user_id:
            try:
                with st.spinner("Generating recommendations..."):
                    recs = model.recommend(user_id, top_k=5)
                    df = pd.DataFrame(recs, columns=["Restaurant", "Score"])
                    st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a User ID")
else:
    restaurant = st.text_input("Enter Restaurant Name")
    if st.button("Similar Restaurants"):
        if restaurant:
            try:
                with st.spinner("Finding similar restaurants..."):
                    sims = model.recommend_by_item(restaurant, top_k=5)
                    df = pd.DataFrame(sims, columns=["Restaurant", "Similarity"])
                    st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a Restaurant Name")

st.markdown("---")
st.markdown("### Batch Upload (CSV)")

uploaded = st.file_uploader("Upload CSV (column: user_id or restaurant)", type=["csv"])
if uploaded:
    try:
        df_in = pd.read_csv(uploaded)
        results = []
        key = "user_id" if mode == "By User ID" else "restaurant"
        
        if key not in df_in.columns:
            st.error(f"CSV must contain a column named '{key}'")
        else:
            progress_bar = st.progress(0)
            total_rows = len(df_in)
            
            for idx, val in enumerate(df_in[key].astype(str)):
                try:
                    if mode == "By User ID":
                        recs = model.recommend(val, top_k=5)
                        for item, score in recs:
                            results.append({key: val, "restaurant": item, "score": score})
                    else:
                        sims = model.recommend_by_item(val, top_k=5)
                        for item, sim in sims:
                            results.append({key: val, "restaurant": item, "similarity": sim})
                except Exception as e:
                    st.error(f"Error processing {val}: {e}")
                    continue
                
                # Update progress bar
                progress_bar.progress((idx + 1) / total_rows)
            
            progress_bar.empty()
            
            if results:
                out = pd.DataFrame(results)
                st.dataframe(out, use_container_width=True)
                
                # Add download button
                csv = out.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"recommendations_{mode.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No results generated.")
    except Exception as e:
        st.error(f"Error processing file: {e}")