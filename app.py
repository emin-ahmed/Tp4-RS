# # # # import streamlit as st
# # # # import pickle
# # # # import pandas as pd

# # # # @st.cache_resource
# # # # def load_model(path="mauritania_restaurant_recommender.pkl"):
# # # #     with open(path, "rb") as f:
# # # #         return pickle.load(f)

# # # # model = load_model()

# # # # st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
# # # # st.title("🍽️ Restaurant Recommender")

# # # # mode = st.sidebar.radio(
# # # #     "How would you like to get recommendations?",
# # # #     ("By User ID", "By Restaurant Name")
# # # # )

# # # # if mode == "By User ID":
# # # #     user_id = st.text_input("Enter User ID")
# # # #     if st.button("Recommend for User"):
# # # #         try:
# # # #             recs = model.recommend(user_id, top_k=5)
# # # #             df = pd.DataFrame(recs, columns=["Restaurant", "Score"])
# # # #             st.dataframe(df)
# # # #         except Exception as e:
# # # #             st.error(f"Error: {e}")
# # # # else:
# # # #     restaurant = st.text_input("Enter Restaurant Name")
# # # #     if st.button("Similar Restaurants"):
# # # #         try:
# # # #             sims = model.recommend_by_item(restaurant, top_k=5)
# # # #             df = pd.DataFrame(sims, columns=["Restaurant", "Similarity"])
# # # #             st.dataframe(df)
# # # #         except Exception as e:
# # # #             st.error(f"Error: {e}")

# # # # st.markdown("---")
# # # # st.markdown("### Batch Upload (CSV)")

# # # # uploaded = st.file_uploader("Upload CSV (column: user_id or restaurant)", type=["csv"])
# # # # if uploaded:
# # # #     df_in = pd.read_csv(uploaded)
# # # #     results = []
# # # #     key = "user_id" if mode == "By User ID" else "restaurant"

# # # #     for val in df_in[key].astype(str):
# # # #         if mode == "By User ID":
# # # #             recs = model.recommend(val, top_k=5)
# # # #             for item, score in recs:
# # # #                 results.append({key: val, "restaurant": item, "score": score})
# # # #         else:
# # # #             sims = model.recommend_by_item(val, top_k=5)
# # # #             for item, sim in sims:
# # # #                 results.append({key: val, "restaurant": item, "similarity": sim})

# # # #     out = pd.DataFrame(results)
# # # #     st.dataframe(out)  # display the full results in the app










# # # import streamlit as st
# # # import pickle
# # # import pandas as pd

# # # @st.cache_resource
# # # def load_model(path="mauritania_restaurant_recommender.pkl"):
# # #     with open(path, "rb") as f:
# # #         loaded_object = pickle.load(f)
    
# # #     # Check if it's a dictionary containing the model
# # #     if isinstance(loaded_object, dict):
# # #         # Common dictionary keys where model might be stored
# # #         possible_keys = ['model', 'recommender', 'algorithm', 'fitted_model']
        
# # #         for key in possible_keys:
# # #             if key in loaded_object:
# # #                 return loaded_object[key]
        
# # #         # If no common keys found, check all keys
# # #         for key, value in loaded_object.items():
# # #             if hasattr(value, 'recommend'):  # Found object with recommend method
# # #                 return value
        
# # #         # If still not found, raise informative error
# # #         st.error(f"Model not found in dictionary. Available keys: {list(loaded_object.keys())}")
# # #         st.stop()
    
# # #     return loaded_object

# # # try:
# # #     model = load_model()
# # # except Exception as e:
# # #     st.error(f"Failed to load model: {e}")
# # #     st.stop()

# # # st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
# # # st.title("🍽️ Restaurant Recommender")

# # # # Verify model has required methods
# # # if not hasattr(model, 'recommend'):
# # #     st.error("Model doesn't have 'recommend' method")
# # #     st.write(f"Model type: {type(model)}")
# # #     st.write(f"Available methods: {[method for method in dir(model) if not method.startswith('_')]}")
# # #     st.stop()

# # # mode = st.sidebar.radio(
# # #     "How would you like to get recommendations?",
# # #     ("By User ID", "By Restaurant Name")
# # # )

# # # if mode == "By User ID":
# # #     user_id = st.text_input("Enter User ID")
# # #     if st.button("Recommend for User"):
# # #         try:
# # #             recs = model.recommend(user_id, top_k=5)
# # #             df = pd.DataFrame(recs, columns=["Restaurant", "Score"])
# # #             st.dataframe(df)
# # #         except Exception as e:
# # #             st.error(f"Error: {e}")
# # # else:
# # #     restaurant = st.text_input("Enter Restaurant Name")
# # #     if st.button("Similar Restaurants"):
# # #         try:
# # #             # Check if method exists
# # #             if hasattr(model, 'recommend_by_item'):
# # #                 sims = model.recommend_by_item(restaurant, top_k=5)
# # #             else:
# # #                 st.error("Model doesn't have 'recommend_by_item' method")
# # #                 st.stop()
# # #             df = pd.DataFrame(sims, columns=["Restaurant", "Similarity"])
# # #             st.dataframe(df)
# # #         except Exception as e:
# # #             st.error(f"Error: {e}")

# # # st.markdown("---")
# # # st.markdown("### Batch Upload (CSV)")

# # # uploaded = st.file_uploader("Upload CSV (column: user_id or restaurant)", type=["csv"])
# # # if uploaded:
# # #     df_in = pd.read_csv(uploaded)
# # #     results = []
# # #     key = "user_id" if mode == "By User ID" else "restaurant"

# # #     for val in df_in[key].astype(str):
# # #         try:
# # #             if mode == "By User ID":
# # #                 recs = model.recommend(val, top_k=5)
# # #                 for item, score in recs:
# # #                     results.append({key: val, "restaurant": item, "score": score})
# # #             else:
# # #                 sims = model.recommend_by_item(val, top_k=5)
# # #                 for item, sim in sims:
# # #                     results.append({key: val, "restaurant": item, "similarity": sim})
# # #         except Exception as e:
# # #             st.error(f"Error processing {val}: {e}")
# # #             continue

# # #     if results:
# # #         out = pd.DataFrame(results)
# # #         st.dataframe(out)
# # #     else:
# # #         st.warning("No results generated.") 









# # import streamlit as st
# # import pickle
# # import pandas as pd
# # import numpy as np
# # from sklearn.metrics.pairwise import cosine_similarity
# # from sklearn.decomposition import TruncatedSVD

# # class RecommenderWrapper:
# #     def __init__(self, model_dict):
# #         self.model_dict = model_dict
# #         self.best_model_name = model_dict.get('best_model_name', 'svd_model')
        
# #         # Get the best model based on best_model_name
# #         if self.best_model_name == 'svd_model':
# #             self.model = model_dict['svd_model']
# #             self.reconstructed = model_dict.get('svd_reconstructed')
# #         elif self.best_model_name == 'ridge_model':
# #             self.model = model_dict['ridge_model']
# #         elif self.best_model_name == 'rf_model':
# #             self.model = model_dict['rf_model']
# #         else:
# #             # Default to SVD if available
# #             self.model = model_dict.get('svd_model', model_dict.get('ridge_model'))
        
# #         # Store other useful components
# #         self.train_matrix = model_dict.get('train_matrix')
# #         self.user_averages = model_dict.get('user_averages', {})
# #         self.restaurant_averages = model_dict.get('restaurant_averages', {})
# #         self.global_avg = model_dict.get('global_avg', 0)
# #         self.restaurant_features = model_dict.get('restaurant_features')
        
# #     def recommend(self, user_id, top_k=5):
# #         """Recommend restaurants for a user"""
# #         try:
# #             user_id_str = str(user_id)
            
# #             # If we have SVD reconstructed matrix, use it
# #             if self.reconstructed is not None and hasattr(self.train_matrix, 'index'):
# #                 if user_id_str in self.train_matrix.index:
# #                     user_idx = self.train_matrix.index.get_loc(user_id_str)
# #                     user_scores = self.reconstructed[user_idx]
                    
# #                     # Get top recommendations (excluding already rated items)
# #                     rated_items = self.train_matrix.loc[user_id_str]
# #                     rated_indices = rated_items[rated_items > 0].index
                    
# #                     # Create scores dataframe
# #                     scores_df = pd.DataFrame({
# #                         'restaurant': self.train_matrix.columns,
# #                         'score': user_scores
# #                     })
                    
# #                     # Remove already rated restaurants
# #                     scores_df = scores_df[~scores_df['restaurant'].isin(rated_indices)]
                    
# #                     # Sort by score and get top k
# #                     top_recs = scores_df.nlargest(top_k, 'score')
# #                     return list(zip(top_recs['restaurant'], top_recs['score']))
            
# #             # Fallback: recommend popular restaurants
# #             if self.restaurant_averages:
# #                 sorted_restaurants = sorted(self.restaurant_averages.items(), 
# #                                           key=lambda x: x[1], reverse=True)
# #                 return sorted_restaurants[:top_k]
            
# #             return [("No recommendations available", 0.0)] * top_k
            
# #         except Exception as e:
# #             st.error(f"Error in recommend: {e}")
# #             return [("Error generating recommendations", 0.0)] * top_k
    
# #     def recommend_by_item(self, restaurant_name, top_k=5):
# #         """Find similar restaurants"""
# #         try:
# #             restaurant_name_str = str(restaurant_name)
            
# #             # If we have restaurant features, use them for similarity
# #             if (self.restaurant_features is not None and 
# #                 restaurant_name_str in self.restaurant_features.index):
                
# #                 target_features = self.restaurant_features.loc[restaurant_name_str].values.reshape(1, -1)
# #                 similarities = cosine_similarity(target_features, self.restaurant_features.values)[0]
                
# #                 # Create similarity dataframe
# #                 sim_df = pd.DataFrame({
# #                     'restaurant': self.restaurant_features.index,
# #                     'similarity': similarities
# #                 })
                
# #                 # Remove the target restaurant itself
# #                 sim_df = sim_df[sim_df['restaurant'] != restaurant_name_str]
                
# #                 # Sort by similarity and get top k
# #                 top_sims = sim_df.nlargest(top_k, 'similarity')
# #                 return list(zip(top_sims['restaurant'], top_sims['similarity']))
            
# #             # Fallback using train matrix if available
# #             elif (self.train_matrix is not None and 
# #                   restaurant_name_str in self.train_matrix.columns):
                
# #                 target_ratings = self.train_matrix[restaurant_name_str].fillna(0)
# #                 similarities = []
                
# #                 for col in self.train_matrix.columns:
# #                     if col != restaurant_name_str:
# #                         other_ratings = self.train_matrix[col].fillna(0)
# #                         # Calculate correlation as similarity measure
# #                         corr = np.corrcoef(target_ratings, other_ratings)[0, 1]
# #                         if not np.isnan(corr):
# #                             similarities.append((col, corr))
                
# #                 # Sort by similarity and return top k
# #                 similarities.sort(key=lambda x: x[1], reverse=True)
# #                 return similarities[:top_k]
            
# #             # Final fallback
# #             if self.restaurant_averages:
# #                 sorted_restaurants = sorted(self.restaurant_averages.items(), 
# #                                           key=lambda x: x[1], reverse=True)
# #                 return sorted_restaurants[:top_k]
            
# #             return [("No similar restaurants found", 0.0)] * top_k
            
# #         except Exception as e:
# #             st.error(f"Error in recommend_by_item: {e}")
# #             return [("Error finding similar restaurants", 0.0)] * top_k

# # @st.cache_resource
# # def load_model(path="mauritania_restaurant_recommender.pkl"):
# #     with open(path, "rb") as f:
# #         loaded_object = pickle.load(f)
    
# #     # Check if it's a dictionary containing the model components
# #     if isinstance(loaded_object, dict):
# #         # Wrap the dictionary in our custom recommender class
# #         return RecommenderWrapper(loaded_object)
    
# #     return loaded_object

# # try:
# #     model = load_model()
# # except Exception as e:
# #     st.error(f"Failed to load model: {e}")
# #     st.stop()

# # st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
# # st.title("🍽️ Restaurant Recommender")

# # # Display model information
# # if hasattr(model, 'model_dict'):
# #     st.sidebar.markdown("### Model Information")
# #     best_model = model.model_dict.get('best_model_name', 'Unknown')
# #     st.sidebar.write(f"**Best Model:** {best_model}")
    
# #     if 'final_rmse' in model.model_dict:
# #         st.sidebar.write(f"**RMSE:** {model.model_dict['final_rmse']:.4f}")
# #     if 'final_mae' in model.model_dict:
# #         st.sidebar.write(f"**MAE:** {model.model_dict['final_mae']:.4f}")
    
# #     total_users = model.model_dict.get('total_users', 'Unknown')
# #     total_restaurants = model.model_dict.get('total_restaurants', 'Unknown')
# #     st.sidebar.write(f"**Users:** {total_users}")
# #     st.sidebar.write(f"**Restaurants:** {total_restaurants}")

# # # Verify model has required methods
# # if not hasattr(model, 'recommend'):
# #     st.error("Model doesn't have 'recommend' method")
# #     st.write(f"Model type: {type(model)}")
# #     st.write(f"Available methods: {[method for method in dir(model) if not method.startswith('_')]}")
# #     st.stop()

# # mode = st.sidebar.radio(
# #     "How would you like to get recommendations?",
# #     ("By User ID", "By Restaurant Name")
# # )

# # if mode == "By User ID":
# #     user_id = st.text_input("Enter User ID")
# #     if st.button("Recommend for User"):
# #         if user_id:
# #             try:
# #                 with st.spinner("Generating recommendations..."):
# #                     recs = model.recommend(user_id, top_k=5)
# #                     df = pd.DataFrame(recs, columns=["Restaurant", "Score"])
# #                     st.dataframe(df, use_container_width=True)
# #             except Exception as e:
# #                 st.error(f"Error: {e}")
# #         else:
# #             st.warning("Please enter a User ID")
# # else:
# #     restaurant = st.text_input("Enter Restaurant Name")
# #     if st.button("Similar Restaurants"):
# #         if restaurant:
# #             try:
# #                 with st.spinner("Finding similar restaurants..."):
# #                     sims = model.recommend_by_item(restaurant, top_k=5)
# #                     df = pd.DataFrame(sims, columns=["Restaurant", "Similarity"])
# #                     st.dataframe(df, use_container_width=True)
# #             except Exception as e:
# #                 st.error(f"Error: {e}")
# #         else:
# #             st.warning("Please enter a Restaurant Name")

# # st.markdown("---")
# # st.markdown("### Batch Upload (CSV)")

# # uploaded = st.file_uploader("Upload CSV (column: user_id or restaurant)", type=["csv"])
# # if uploaded:
# #     try:
# #         df_in = pd.read_csv(uploaded)
# #         results = []
# #         key = "user_id" if mode == "By User ID" else "restaurant"
        
# #         if key not in df_in.columns:
# #             st.error(f"CSV must contain a column named '{key}'")
# #         else:
# #             progress_bar = st.progress(0)
# #             total_rows = len(df_in)
            
# #             for idx, val in enumerate(df_in[key].astype(str)):
# #                 try:
# #                     if mode == "By User ID":
# #                         recs = model.recommend(val, top_k=5)
# #                         for item, score in recs:
# #                             results.append({key: val, "restaurant": item, "score": score})
# #                     else:
# #                         sims = model.recommend_by_item(val, top_k=5)
# #                         for item, sim in sims:
# #                             results.append({key: val, "restaurant": item, "similarity": sim})
# #                 except Exception as e:
# #                     st.error(f"Error processing {val}: {e}")
# #                     continue
                
# #                 # Update progress bar
# #                 progress_bar.progress((idx + 1) / total_rows)
            
# #             progress_bar.empty()
            
# #             if results:
# #                 out = pd.DataFrame(results)
# #                 st.dataframe(out, use_container_width=True)
                
# #                 # Add download button
# #                 csv = out.to_csv(index=False)
# #                 st.download_button(
# #                     label="Download Results as CSV",
# #                     data=csv,
# #                     file_name=f"recommendations_{mode.lower().replace(' ', '_')}.csv",
# #                     mime="text/csv"
# #                 )
# #             else:
# #                 st.warning("No results generated.")
# #     except Exception as e:
# #         st.error(f"Error processing file: {e}")





# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# class RecommenderWrapper:
#     def __init__(self, model_dict):
#         self.model_dict = model_dict
        
#         # Initialize all attributes to None first
#         self.model = None
#         self.reconstructed = None
#         self.train_matrix = None
#         self.user_averages = {}
#         self.restaurant_averages = {}
#         self.global_avg = 0
#         self.restaurant_features = None
#         self.best_model_name = 'unknown'
        
#         # Now safely extract values
#         try:
#             self.best_model_name = model_dict.get('best_model_name', 'svd_model')
            
#             # Get the best model based on best_model_name
#             if self.best_model_name == 'svd_model' and 'svd_model' in model_dict:
#                 self.model = model_dict['svd_model']
#             elif self.best_model_name == 'ridge_model' and 'ridge_model' in model_dict:
#                 self.model = model_dict['ridge_model']
#             elif self.best_model_name == 'rf_model' and 'rf_model' in model_dict:
#                 self.model = model_dict['rf_model']
#             else:
#                 # Default to any available model
#                 for model_key in ['svd_model', 'ridge_model', 'rf_model']:
#                     if model_key in model_dict:
#                         self.model = model_dict[model_key]
#                         self.best_model_name = model_key
#                         break
            
#             # Store other useful components safely
#             self.reconstructed = model_dict.get('svd_reconstructed')
#             self.train_matrix = model_dict.get('train_matrix')
#             self.user_averages = model_dict.get('user_averages', {})
#             self.restaurant_averages = model_dict.get('restaurant_averages', {})
#             self.global_avg = model_dict.get('global_avg', 0)
#             self.restaurant_features = model_dict.get('restaurant_features')
            
#         except Exception as e:
#             st.error(f"Error initializing RecommenderWrapper: {e}")
#             # Set safe defaults
#             self.model = None
#             self.reconstructed = None
        
#     def recommend(self, user_id, top_k=5):
#         """Recommend restaurants for a user"""
#         try:
#             user_id_str = str(user_id)
            
#             # Check if we have the required attributes
#             if not hasattr(self, 'reconstructed'):
#                 st.error("RecommenderWrapper not properly initialized")
#                 return [("Initialization error", 0.0)] * top_k
            
#             # Method 1: Try using reconstructed SVD matrix
#             if (self.reconstructed is not None and 
#                 self.train_matrix is not None and 
#                 hasattr(self.train_matrix, 'index') and 
#                 user_id_str in self.train_matrix.index):
                
#                 user_idx = self.train_matrix.index.get_loc(user_id_str)
#                 user_scores = self.reconstructed[user_idx]
                
#                 # Get top recommendations (excluding already rated items)
#                 rated_items = self.train_matrix.loc[user_id_str]
#                 rated_indices = rated_items[rated_items > 0].index
                
#                 # Create scores dataframe
#                 scores_df = pd.DataFrame({
#                     'restaurant': self.train_matrix.columns,
#                     'score': user_scores
#                 })
                
#                 # Remove already rated restaurants
#                 scores_df = scores_df[~scores_df['restaurant'].isin(rated_indices)]
                
#                 # Sort by score and get top k
#                 top_recs = scores_df.nlargest(top_k, 'score')
#                 return list(zip(top_recs['restaurant'], top_recs['score']))
            
#             # Method 2: Use user averages and collaborative filtering
#             elif (self.train_matrix is not None and 
#                   hasattr(self.train_matrix, 'index') and 
#                   user_id_str in self.train_matrix.index):
                
#                 user_ratings = self.train_matrix.loc[user_id_str]
#                 user_mean = user_ratings[user_ratings > 0].mean()
                
#                 # Find similar users based on common ratings
#                 similarities = []
#                 for other_user in self.train_matrix.index:
#                     if other_user != user_id_str:
#                         other_ratings = self.train_matrix.loc[other_user]
                        
#                         # Find common items
#                         common_items = (user_ratings > 0) & (other_ratings > 0)
#                         if common_items.sum() > 0:
#                             # Calculate similarity (correlation)
#                             user_common = user_ratings[common_items]
#                             other_common = other_ratings[common_items]
                            
#                             if len(user_common) > 1:
#                                 corr = np.corrcoef(user_common, other_common)[0, 1]
#                                 if not np.isnan(corr):
#                                     similarities.append((other_user, corr))
                
#                 # Get recommendations from similar users
#                 similarities.sort(key=lambda x: x[1], reverse=True)
#                 recommendations = {}
                
#                 for similar_user, similarity in similarities[:10]:  # Top 10 similar users
#                     similar_ratings = self.train_matrix.loc[similar_user]
#                     # Get items this user hasn't rated
#                     unrated_items = (user_ratings == 0) & (similar_ratings > 0)
                    
#                     for item in similar_ratings[unrated_items].index:
#                         if item not in recommendations:
#                             recommendations[item] = 0
#                         recommendations[item] += similarity * similar_ratings[item]
                
#                 if recommendations:
#                     sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
#                     return sorted_recs[:top_k]
            
#             # Method 3: Use user average preference
#             elif user_id_str in self.user_averages:
#                 user_avg = self.user_averages[user_id_str]
                
#                 # Recommend restaurants with ratings above user's average
#                 good_restaurants = []
#                 for restaurant, avg_rating in self.restaurant_averages.items():
#                     if avg_rating >= user_avg:
#                         good_restaurants.append((restaurant, avg_rating))
                
#                 good_restaurants.sort(key=lambda x: x[1], reverse=True)
#                 return good_restaurants[:top_k]
            
#             # Method 4: Fallback to most popular restaurants
#             if self.restaurant_averages:
#                 sorted_restaurants = sorted(self.restaurant_averages.items(), 
#                                           key=lambda x: x[1], reverse=True)
#                 return sorted_restaurants[:top_k]
            
#             return [("No recommendations available", 0.0)] * top_k
            
#         except Exception as e:
#             st.error(f"Error in recommend: {e}")
#             # Return popular restaurants as fallback
#             if self.restaurant_averages:
#                 sorted_restaurants = sorted(self.restaurant_averages.items(), 
#                                           key=lambda x: x[1], reverse=True)
#                 return sorted_restaurants[:top_k]
#             return [("Error generating recommendations", 0.0)] * top_k
    
#     def recommend_by_item(self, restaurant_name, top_k=5):
#         """Find similar restaurants"""
#         try:
#             restaurant_name_str = str(restaurant_name)
            
#             # If we have restaurant features, use them for similarity
#             if (self.restaurant_features is not None and 
#                 restaurant_name_str in self.restaurant_features.index):
                
#                 target_features = self.restaurant_features.loc[restaurant_name_str].values.reshape(1, -1)
                
#                 # Simple dot product similarity instead of cosine similarity
#                 similarities = []
#                 for idx, row in self.restaurant_features.iterrows():
#                     if idx != restaurant_name_str:
#                         # Calculate dot product similarity
#                         similarity = np.dot(target_features[0], row.values)
#                         similarities.append((idx, similarity))
                
#                 # Sort by similarity and get top k
#                 similarities.sort(key=lambda x: x[1], reverse=True)
#                 return similarities[:top_k]
            
#             # Fallback using train matrix if available
#             elif (self.train_matrix is not None and 
#                   restaurant_name_str in self.train_matrix.columns):
                
#                 target_ratings = self.train_matrix[restaurant_name_str].fillna(0)
#                 similarities = []
                
#                 for col in self.train_matrix.columns:
#                     if col != restaurant_name_str:
#                         other_ratings = self.train_matrix[col].fillna(0)
#                         # Calculate correlation as similarity measure
#                         corr = np.corrcoef(target_ratings, other_ratings)[0, 1]
#                         if not np.isnan(corr):
#                             similarities.append((col, corr))
                
#                 # Sort by similarity and return top k
#                 similarities.sort(key=lambda x: x[1], reverse=True)
#                 return similarities[:top_k]
            
#             # Final fallback
#             if self.restaurant_averages:
#                 sorted_restaurants = sorted(self.restaurant_averages.items(), 
#                                           key=lambda x: x[1], reverse=True)
#                 return sorted_restaurants[:top_k]
            
#             return [("No similar restaurants found", 0.0)] * top_k
            
#         except Exception as e:
#             st.error(f"Error in recommend_by_item: {e}")
#             return [("Error finding similar restaurants", 0.0)] * top_k

# @st.cache_resource
# def load_model(path="mauritania_restaurant_recommender.pkl"):
#     with open(path, "rb") as f:
#         loaded_object = pickle.load(f)
    
#     # Check if it's a dictionary containing the model components
#     if isinstance(loaded_object, dict):
#         # Wrap the dictionary in our custom recommender class
#         return RecommenderWrapper(loaded_object)
    
#     return loaded_object

# try:
#     model = load_model()
# except Exception as e:
#     st.error(f"Failed to load model: {e}")
#     st.stop()

# st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
# st.title("🍽️ Restaurant Recommender")

# # Display model information and debug details
# if hasattr(model, 'model_dict'):
#     st.sidebar.markdown("### Model Information")
#     best_model = model.model_dict.get('best_model_name', 'Unknown')
#     st.sidebar.write(f"**Best Model:** {best_model}")
    
#     if 'final_rmse' in model.model_dict:
#         st.sidebar.write(f"**RMSE:** {model.model_dict['final_rmse']:.4f}")
#     if 'final_mae' in model.model_dict:
#         st.sidebar.write(f"**MAE:** {model.model_dict['final_mae']:.4f}")
    
#     total_users = model.model_dict.get('total_users', 'Unknown')
#     total_restaurants = model.model_dict.get('total_restaurants', 'Unknown')
#     st.sidebar.write(f"**Users:** {total_users}")
#     st.sidebar.write(f"**Restaurants:** {total_restaurants}")
    
#     # Debug section
#     with st.sidebar.expander("Debug Info"):
#         st.write("Available keys in model:")
#         for key in sorted(model.model_dict.keys()):
#             value = model.model_dict[key]
#             if hasattr(value, 'shape'):
#                 st.write(f"- {key}: {type(value).__name__} {value.shape}")
#             elif isinstance(value, (dict, list)):
#                 st.write(f"- {key}: {type(value).__name__} (len: {len(value)})")
#             else:
#                 st.write(f"- {key}: {type(value).__name__}")


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
#         if user_id:
#             try:
#                 with st.spinner("Generating recommendations..."):
#                     recs = model.recommend(user_id, top_k=5)
#                     df = pd.DataFrame(recs, columns=["Restaurant", "Score"])
#                     st.dataframe(df, use_container_width=True)
#             except Exception as e:
#                 st.error(f"Error: {e}")
#         else:
#             st.warning("Please enter a User ID")
# else:
#     restaurant = st.text_input("Enter Restaurant Name")
#     if st.button("Similar Restaurants"):
#         if restaurant:
#             try:
#                 with st.spinner("Finding similar restaurants..."):
#                     sims = model.recommend_by_item(restaurant, top_k=5)
#                     df = pd.DataFrame(sims, columns=["Restaurant", "Similarity"])
#                     st.dataframe(df, use_container_width=True)
#             except Exception as e:
#                 st.error(f"Error: {e}")
#         else:
#             st.warning("Please enter a Restaurant Name")

# st.markdown("---")
# st.markdown("### Batch Upload (CSV)")

# uploaded = st.file_uploader("Upload CSV (column: user_id or restaurant)", type=["csv"])
# if uploaded:
#     try:
#         df_in = pd.read_csv(uploaded)
#         results = []
#         key = "user_id" if mode == "By User ID" else "restaurant"
        
#         if key not in df_in.columns:
#             st.error(f"CSV must contain a column named '{key}'")
#         else:
#             progress_bar = st.progress(0)
#             total_rows = len(df_in)
            
#             for idx, val in enumerate(df_in[key].astype(str)):
#                 try:
#                     if mode == "By User ID":
#                         recs = model.recommend(val, top_k=5)
#                         for item, score in recs:
#                             results.append({key: val, "restaurant": item, "score": score})
#                     else:
#                         sims = model.recommend_by_item(val, top_k=5)
#                         for item, sim in sims:
#                             results.append({key: val, "restaurant": item, "similarity": sim})
#                 except Exception as e:
#                     st.error(f"Error processing {val}: {e}")
#                     continue
                
#                 # Update progress bar
#                 progress_bar.progress((idx + 1) / total_rows)
            
#             progress_bar.empty()
            
#             if results:
#                 out = pd.DataFrame(results)
#                 st.dataframe(out, use_container_width=True)
                
#                 # Add download button
#                 csv = out.to_csv(index=False)
#                 st.download_button(
#                     label="Download Results as CSV",
#                     data=csv,
#                     file_name=f"recommendations_{mode.lower().replace(' ', '_')}.csv",
#                     mime="text/csv"
#                 )
#             else:
#                 st.warning("No results generated.")
#     except Exception as e:
#         st.error(f"Error processing file: {e}")







# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# @st.cache_resource
# def load_model(path="mauritania_restaurant_recommender.pkl"):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# model_pkg = load_model()

# # On récupère la matrice reconstituée users×restaurants
# svd_reconstructed = model_pkg['svd_reconstructed']  # numpy array

# # DataFrame des restaurants
# # On suppose que model_pkg['restaurant_features'] est un DataFrame
# # avec une colonne "name" pour le nom des restaurants
# restaurants_df = model_pkg['restaurant_features'].copy()
# if 'name' not in restaurants_df.columns:
#     # Si votre colonne s'appelle différemment, modifiez ici :
#     restaurants_df = restaurants_df.rename(columns={restaurants_df.columns[0]: 'name'})

# restaurant_names = restaurants_df['name'].tolist()

# st.set_page_config(page_title="🍽️ Recommandations Restaurants", layout="wide")
# st.title("🍽️ Recommander des restaurants similaires")

# selected = st.selectbox("Choisissez un restaurant :", restaurant_names)

# if st.button("Trouver 5 similaires"):
#     # Index du restaurant sélectionné
#     idx = restaurant_names.index(selected)
    
#     # Colonnes = restaurants, donc on transpose pour similarity sur colonnes
#     # svd_reconstructed : shape (n_users, n_restaurants)
#     # On extrait vecteur colonne i
#     item_vec = svd_reconstructed[:, idx].reshape(1, -1)
    
#     # Calcul des similarités cosinus entre cette colonne et toutes les autres
#     # On transpose svd_reconstructed pour obtenir (n_restaurants, n_users)
#     sims = cosine_similarity(item_vec, svd_reconstructed.T).flatten()
    
#     # On crée un DataFrame pour trier
#     sim_df = pd.DataFrame({
#         'name': restaurant_names,
#         'similarity': sims
#     })
#     # On retire lui-même, on trie et on prend top 6 (incluant lui, qu’on enlèvera)
#     top5 = (
#         sim_df
#         .sort_values('similarity', ascending=False)
#         .iloc[1:6]  # 1:6 pour sauter l'élément identique
#         .reset_index(drop=True)
#     )
    
#     st.write("### Les 5 restaurants les plus similaires à", selected)
#     st.dataframe(top5)
# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances

# # 1) Chargement du modèle
# @st.cache_resource
# def load_model(path="mauritania_restaurant_recommender.pkl"):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# model_pkg = load_model()

# # 2) Extraction des embeddings restaurants via le SVD
# #    svd_model.components_ a pour shape (n_components, n_restaurants)
# #    On transpose pour obtenir (n_restaurants, n_components)
# svd: "TruncatedSVD" = model_pkg['svd_model']
# item_embeddings = svd.components_.T

# # 3) Noms des restaurants
# restaurants_df = model_pkg['restaurant_features'].copy()
# if 'name' not in restaurants_df.columns:
#     restaurants_df = restaurants_df.rename(columns={restaurants_df.columns[0]: 'name'})
# restaurant_names = restaurants_df['name'].tolist()

# # 4) Interface Streamlit
# st.set_page_config(page_title="🍽️ Reco Restaurants (SVD)", layout="wide")
# st.title("🍽️ Recommander des restaurants similaires (SVD)")

# input_name = st.text_input("Entrez le nom du restaurant de référence :")

# if st.button("Trouver 5 similaires"):
#     if not input_name:
#         st.error("Veuillez saisir un nom de restaurant.")
#     elif input_name not in restaurant_names:
#         st.error("Ce restaurant n'existe pas dans notre base. Vérifiez l'orthographe !")
#     else:
#         idx = restaurant_names.index(input_name)
#         # 5) Calcul des distances euclidiennes dans l'espace latent
#         dists = euclidean_distances(
#             item_embeddings[idx].reshape(1, -1),
#             item_embeddings
#         ).flatten()
#         # 6) On met dans un DataFrame pour trier
#         df_sim = pd.DataFrame({
#             'Restaurant': restaurant_names,
#             'Distance': dists
#         })
#         top5 = (
#             df_sim
#             .sort_values('Distance', ascending=True)   # plus la distance est petite, plus c'est proche
#             .iloc[1:6]  # on enlève l'élément lui-même
#             .reset_index(drop=True)
#         )
#         st.write(f"### 5 restaurants les plus proches de « {input_name} »")
#         st.dataframe(top5)










# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Mauritania Restaurant Recommender",
#     page_icon="🍽️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .restaurant-card {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 4px solid #1f77b4;
#         margin: 1rem 0;
#     }
#     .metric-card {
#         background-color: #e3f2fd;
#         padding: 1rem;
#         border-radius: 8px;
#         text-align: center;
#     }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_data
# def load_model():
#     """Load the trained recommender model"""
#     try:
#         with open('mauritania_restaurant_recommender.pkl', 'rb') as f:
#             model_package = pickle.load(f)
#         return model_package
#     except FileNotFoundError:
#         st.error("❌ Model file 'mauritania_restaurant_recommender.pkl' not found. Please make sure the file is in the same directory as this app.")
#         return None
#     except Exception as e:
#         st.error(f"❌ Error loading model: {str(e)}")
#         return None

# def get_restaurant_recommendations(model_package, restaurant_id=None, restaurant_name=None, user_preferences=None):
#     """Get restaurant recommendations using the hybrid model"""
    
#     if model_package is None:
#         return []
    
#     try:
#         # Extract components from model package
#         restaurant_features = model_package['restaurant_features']
#         ridge_model = model_package['ridge_model']
#         scaler = model_package['scaler']
#         rf_model = model_package['rf_model']
#         restaurant_averages = model_package['restaurant_averages']
        
#         # If restaurant_name is provided, find the restaurant_id
#         if restaurant_name and restaurant_id is None:
#             # Assuming restaurant_features has a 'name' column
#             matching_restaurants = restaurant_features[
#                 restaurant_features['name'].str.contains(restaurant_name, case=False, na=False)
#             ]
#             if not matching_restaurants.empty:
#                 restaurant_id = matching_restaurants.index[0]
#             else:
#                 st.warning(f"Restaurant '{restaurant_name}' not found.")
#                 return []
        
#         # Method 1: Content-based similarity using restaurant features
#         if restaurant_id is not None:
#             # Get target restaurant features
#             if restaurant_id in restaurant_features.index:
#                 target_features = restaurant_features.loc[[restaurant_id]]
                
#                 # Calculate similarity with all restaurants
#                 feature_cols = [col for col in restaurant_features.columns if col not in ['name', 'address']]
                
#                 # Use only numerical features for similarity calculation
#                 numerical_features = restaurant_features[feature_cols].select_dtypes(include=[np.number])
#                 target_numerical = target_features[feature_cols].select_dtypes(include=[np.number])
                
#                 # Calculate cosine similarity
#                 similarities = cosine_similarity(target_numerical, numerical_features)[0]
                
#                 # Get top 6 similar restaurants (excluding the target restaurant itself)
#                 similar_indices = np.argsort(similarities)[::-1][1:6]  # Skip first (itself)
                
#                 recommendations = []
#                 for idx in similar_indices:
#                     rest_id = numerical_features.index[idx]
#                     similarity_score = similarities[idx]
                    
#                     # Get restaurant info
#                     rest_info = restaurant_features.loc[rest_id]
#                     avg_rating = restaurant_averages.get(rest_id, 0) if restaurant_averages is not None else 0
                    
#                     recommendations.append({
#                         'restaurant_id': rest_id,
#                         'name': rest_info.get('name', f'Restaurant {rest_id}'),
#                         'similarity_score': similarity_score,
#                         'avg_rating': avg_rating,
#                         'features': rest_info.to_dict()
#                     })
                
#                 return recommendations
        
#         # Method 2: General recommendations based on high ratings and popularity
#         else:
#             # Get top-rated restaurants
#             if restaurant_averages is not None:
#                 top_restaurants = sorted(restaurant_averages.items(), key=lambda x: x[1], reverse=True)[:5]
                
#                 recommendations = []
#                 for rest_id, avg_rating in top_restaurants:
#                     if rest_id in restaurant_features.index:
#                         rest_info = restaurant_features.loc[rest_id]
#                         recommendations.append({
#                             'restaurant_id': rest_id,
#                             'name': rest_info.get('name', f'Restaurant {rest_id}'),
#                             'avg_rating': avg_rating,
#                             'features': rest_info.to_dict()
#                         })
                
#                 return recommendations
            
#     except Exception as e:
#         st.error(f"Error generating recommendations: {str(e)}")
#         return []
    
#     return []

# def display_recommendations(recommendations):
#     """Display restaurant recommendations in a nice format"""
#     if not recommendations:
#         st.warning("No recommendations found.")
#         return
    
#     st.markdown("### 🎯 Top 5 Restaurant Recommendations")
    
#     for i, rec in enumerate(recommendations, 1):
#         with st.container():
#             col1, col2, col3 = st.columns([1, 3, 1])
            
#             with col1:
#                 st.markdown(f"**#{i}**")
#                 if 'similarity_score' in rec:
#                     st.metric("Similarity", f"{rec['similarity_score']:.2%}")
#                 if 'avg_rating' in rec:
#                     st.metric("Avg Rating", f"{rec['avg_rating']:.1f}")
            
#             with col2:
#                 st.markdown(f"### {rec['name']}")
#                 st.markdown(f"**Restaurant ID:** {rec['restaurant_id']}")
                
#                 # Display additional features if available
#                 if 'features' in rec:
#                     features = rec['features']
#                     if 'address' in features and pd.notna(features['address']):
#                         st.markdown(f"📍 **Address:** {features['address']}")
                    
#                     # Display other relevant features
#                     for key, value in features.items():
#                         if key not in ['name', 'address'] and pd.notna(value) and str(value) != '0':
#                             st.markdown(f"**{key.title()}:** {value}")
            
#             with col3:
#                 st.markdown("🍽️")
            
#             st.markdown("---")

# def main():
#     # Header
#     st.markdown('<h1 class="main-header">🍽️ Mauritania Restaurant Recommender</h1>', unsafe_allow_html=True)
    
#     # Load model
#     with st.spinner("Loading recommendation model..."):
#         model_package = load_model()
    
#     if model_package is None:
#         st.stop()
    
#     # Display model info
#     with st.sidebar:
#         st.markdown("## 📊 Model Information")
#         st.markdown(f"**Model Type:** {model_package.get('model_type', 'Unknown')}")
#         st.markdown(f"**Training Date:** {model_package.get('training_date', 'Unknown')}")
#         st.markdown(f"**Total Restaurants:** {model_package.get('total_restaurants', 'Unknown')}")
#         st.markdown(f"**Total Users:** {model_package.get('total_users', 'Unknown')}")
#         st.markdown(f"**Total Reviews:** {model_package.get('total_reviews', 'Unknown')}")
        
#         if 'final_rmse' in model_package:
#             st.markdown(f"**Model RMSE:** {model_package['final_rmse']:.3f}")
#         if 'final_mae' in model_package:
#             st.markdown(f"**Model MAE:** {model_package['final_mae']:.3f}")
    
#     # Main interface
#     st.markdown("## 🔍 Get Restaurant Recommendations")
    
#     # Create tabs for different recommendation methods
#     tab1, tab2 = st.tabs(["🎯 Similar Restaurants", "⭐ Top Rated"])
    
#     with tab1:
#         st.markdown("### Find restaurants similar to one you like")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Restaurant ID input
#             restaurant_id = st.text_input(
#                 "Enter Restaurant ID:",
#                 placeholder="e.g., REST123",
#                 help="Enter the ID of a restaurant you like to find similar ones"
#             )
        
#         with col2:
#             # Restaurant name input
#             restaurant_name = st.text_input(
#                 "Or enter Restaurant Name:",
#                 placeholder="e.g., La Taverne",
#                 help="Enter part of the restaurant name to search"
#             )
        
#         if st.button("🔍 Find Similar Restaurants", type="primary"):
#             if restaurant_id or restaurant_name:
#                 with st.spinner("Finding similar restaurants..."):
#                     recommendations = get_restaurant_recommendations(
#                         model_package, 
#                         restaurant_id=restaurant_id if restaurant_id else None,
#                         restaurant_name=restaurant_name if restaurant_name else None
#                     )
#                     display_recommendations(recommendations)
#             else:
#                 st.warning("Please enter either a Restaurant ID or Restaurant Name.")
    
#     with tab2:
#         st.markdown("### Discover top-rated restaurants")
        
#         if st.button("⭐ Show Top Rated Restaurants", type="primary"):
#             with st.spinner("Loading top-rated restaurants..."):
#                 recommendations = get_restaurant_recommendations(model_package)
#                 display_recommendations(recommendations)
    
#     # Additional information
#     st.markdown("---")
#     st.markdown("## ℹ️ How it works")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         **Similar Restaurants:**
#         - Uses content-based filtering
#         - Analyzes restaurant features and characteristics
#         - Finds restaurants with similar attributes
#         - Calculates cosine similarity scores
#         """)
    
#     with col2:
#         st.markdown("""
#         **Top Rated:**
#         - Shows restaurants with highest average ratings
#         - Based on historical review data
#         - Considers overall popularity and quality
#         - Perfect for discovering new places
#         """)

# if __name__ == "__main__":
#     main()











# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import StandardScaler
# import re
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Mauritania Restaurant Recommender",
#     page_icon="🍽️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .restaurant-card {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 4px solid #1f77b4;
#         margin: 1rem 0;
#     }
#     .metric-card {
#         background-color: #e3f2fd;
#         padding: 1rem;
#         border-radius: 8px;
#         text-align: center;
#     }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_data
# def load_model():
#     """Load the trained recommender model"""
#     try:
#         with open('mauritania_restaurant_recommender.pkl', 'rb') as f:
#             model_package = pickle.load(f)
#         return model_package
#     except FileNotFoundError:
#         st.error("❌ Model file 'mauritania_restaurant_recommender.pkl' not found. Please make sure the file is in the same directory as this app.")
#         return None
#     except Exception as e:
#         st.error(f"❌ Error loading model: {str(e)}")
#         return None

# def extract_keywords_from_name(restaurant_name):
#     """Extract keywords and features from restaurant name"""
#     name_lower = restaurant_name.lower()
    
#     # Define cuisine/style keywords
#     cuisine_keywords = {
#         'italian': ['italian', 'pizza', 'pasta', 'trattoria', 'pizzeria', 'roma', 'napoli'],
#         'french': ['french', 'cafe', 'bistro', 'brasserie', 'chez', 'le ', 'la ', 'les'],
#         'chinese': ['chinese', 'dragon', 'golden', 'jade', 'peking', 'beijing'],
#         'mexican': ['mexican', 'taco', 'cantina', 'hacienda', 'el ', 'la '],
#         'indian': ['indian', 'taj', 'curry', 'spice', 'tandoor', 'maharaja'],
#         'japanese': ['japanese', 'sushi', 'sake', 'tokyo', 'sakura', 'zen'],
#         'american': ['grill', 'diner', 'burger', 'steakhouse', 'bbq', 'smokehouse'],
#         'mediterranean': ['mediterranean', 'olive', 'greek', 'cyprus', 'santorini'],
#         'seafood': ['seafood', 'fish', 'ocean', 'marina', 'catch', 'lobster', 'crab'],
#         'fast_food': ['fast', 'quick', 'express', 'drive', 'burger', 'chicken']
#     }
    
#     # Define ambiance keywords
#     ambiance_keywords = {
#         'fine_dining': ['fine', 'elegant', 'luxury', 'premium', 'exclusive', 'royal'],
#         'casual': ['casual', 'family', 'home', 'corner', 'neighborhood'],
#         'romantic': ['romantic', 'intimate', 'cozy', 'candlelight', 'moonlight'],
#         'sports': ['sports', 'game', 'stadium', 'victory', 'champion'],
#         'coffee': ['coffee', 'cafe', 'espresso', 'bean', 'roast', 'brew']
#     }
    
#     detected_features = {
#         'cuisine_type': 'general',
#         'ambiance': 'casual',
#         'price_level': 'medium',
#         'keywords': []
#     }
    
#     # Extract cuisine type
#     for cuisine, keywords in cuisine_keywords.items():
#         if any(keyword in name_lower for keyword in keywords):
#             detected_features['cuisine_type'] = cuisine
#             detected_features['keywords'].extend([k for k in keywords if k in name_lower])
#             break
    
#     # Extract ambiance
#     for amb, keywords in ambiance_keywords.items():
#         if any(keyword in name_lower for keyword in keywords):
#             detected_features['ambiance'] = amb
#             detected_features['keywords'].extend([k for k in keywords if k in name_lower])
#             break
    
#     # Extract price indicators
#     if any(word in name_lower for word in ['luxury', 'premium', 'fine', 'exclusive', 'royal']):
#         detected_features['price_level'] = 'high'
#     elif any(word in name_lower for word in ['budget', 'cheap', 'quick', 'fast', 'express']):
#         detected_features['price_level'] = 'low'
    
#     return detected_features

# def get_restaurant_recommendations_by_name(model_package, query_name):
#     """Get restaurant recommendations based on a user-provided restaurant name (not necessarily in dataset)"""
    
#     if model_package is None:
#         return []
    
#     try:
#         # Extract components from model package
#         restaurant_features = model_package['restaurant_features']
#         restaurant_averages = model_package['restaurant_averages']
        
#         # Extract features from the query name
#         query_features = extract_keywords_from_name(query_name)
        
#         # Get all restaurant names for text similarity
#         restaurant_names = restaurant_features.get('name', restaurant_features.index.astype(str)).fillna('')
        
#         # Create TF-IDF vectorizer for name similarity
#         tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
        
#         # Combine query with all restaurant names for vectorization
#         all_names = [query_name] + restaurant_names.tolist()
#         tfidf_matrix = tfidf.fit_transform(all_names)
        
#         # Calculate similarity between query and all restaurants
#         query_vector = tfidf_matrix[0]  # First row is our query
#         restaurant_vectors = tfidf_matrix[1:]  # Rest are restaurant names
        
#         # Calculate cosine similarity
#         text_similarities = cosine_similarity(query_vector, restaurant_vectors)[0]
        
#         # Create feature-based similarity scores
#         feature_similarities = []
        
#         for idx, rest_id in enumerate(restaurant_features.index):
#             rest_info = restaurant_features.loc[rest_id]
#             feature_score = 0.0
            
#             # Check for cuisine type matches (if available in restaurant features)
#             cuisine_cols = [col for col in restaurant_features.columns if 'cuisine' in col.lower() or 'category' in col.lower()]
#             for col in cuisine_cols:
#                 if pd.notna(rest_info.get(col)) and query_features['cuisine_type'] in str(rest_info[col]).lower():
#                     feature_score += 0.3
            
#             # Check for keyword matches in restaurant name
#             rest_name = str(rest_info.get('name', '')).lower()
#             keyword_matches = sum(1 for keyword in query_features['keywords'] if keyword in rest_name)
#             feature_score += keyword_matches * 0.2
            
#             # Check for ambiance/style matches
#             style_cols = [col for col in restaurant_features.columns if any(term in col.lower() for term in ['style', 'type', 'ambiance'])]
#             for col in style_cols:
#                 if pd.notna(rest_info.get(col)) and query_features['ambiance'] in str(rest_info[col]).lower():
#                     feature_score += 0.25
            
#             feature_similarities.append(feature_score)
        
#         # Combine text and feature similarities
#         combined_similarities = []
#         for i in range(len(text_similarities)):
#             # Weighted combination: 60% text similarity, 40% feature similarity
#             combined_score = 0.6 * text_similarities[i] + 0.4 * feature_similarities[i]
#             combined_similarities.append(combined_score)
        
#         # Get top 5 most similar restaurants
#         top_indices = np.argsort(combined_similarities)[::-1][:5]
        
#         recommendations = []
#         for idx in top_indices:
#             rest_id = restaurant_features.index[idx]
#             text_sim = text_similarities[idx]
#             feature_sim = feature_similarities[idx]
#             combined_sim = combined_similarities[idx]
            
#             # Get restaurant info
#             rest_info = restaurant_features.loc[rest_id]
#             avg_rating = restaurant_averages.get(rest_id, 0) if restaurant_averages is not None else 0
            
#             recommendations.append({
#                 'restaurant_id': rest_id,
#                 'name': rest_info.get('name', f'Restaurant {rest_id}'),
#                 'text_similarity': text_sim,
#                 'feature_similarity': feature_sim,
#                 'combined_similarity': combined_sim,
#                 'avg_rating': avg_rating,
#                 'features': rest_info.to_dict(),
#                 'match_reasons': get_match_reasons(query_features, rest_info)
#             })
        
#         return recommendations
        
#     except Exception as e:
#         st.error(f"Error generating recommendations: {str(e)}")
#         return []

# def get_match_reasons(query_features, restaurant_info):
#     """Get reasons why this restaurant matches the query"""
#     reasons = []
    
#     rest_name = str(restaurant_info.get('name', '')).lower()
    
#     # Check cuisine match
#     if query_features['cuisine_type'] != 'general':
#         cuisine_keywords = {
#             'italian': ['italian', 'pizza', 'pasta', 'trattoria', 'pizzeria'],
#             'french': ['french', 'cafe', 'bistro', 'brasserie'],
#             'chinese': ['chinese', 'dragon', 'golden', 'jade'],
#             'mexican': ['mexican', 'taco', 'cantina', 'hacienda'],
#             'seafood': ['seafood', 'fish', 'ocean', 'marina', 'catch']
#         }
        
#         if query_features['cuisine_type'] in cuisine_keywords:
#             for keyword in cuisine_keywords[query_features['cuisine_type']]:
#                 if keyword in rest_name:
#                     reasons.append(f"Similar cuisine style ({keyword})")
#                     break
    
#     # Check keyword matches
#     for keyword in query_features['keywords']:
#         if keyword in rest_name:
#             reasons.append(f"Name contains '{keyword}'")
    
#     # Check ambiance match
#     if query_features['ambiance'] != 'casual':
#         ambiance_keywords = {
#             'fine_dining': ['fine', 'elegant', 'luxury', 'premium'],
#             'romantic': ['romantic', 'intimate', 'cozy'],
#             'coffee': ['coffee', 'cafe', 'espresso', 'bean']
#         }
        
#         if query_features['ambiance'] in ambiance_keywords:
#             for keyword in ambiance_keywords[query_features['ambiance']]:
#                 if keyword in rest_name:
#                     reasons.append(f"Similar ambiance ({keyword})")
#                     break
    
#     if not reasons:
#         reasons.append("Text similarity match")
    
#     return reasons

# def get_top_rated_restaurants(model_package):
#     """Get top-rated restaurants"""
#     try:
#         restaurant_features = model_package['restaurant_features']
#         restaurant_averages = model_package['restaurant_averages']
        
#         if restaurant_averages is not None:
#             top_restaurants = sorted(restaurant_averages.items(), key=lambda x: x[1], reverse=True)[:5]
            
#             recommendations = []
#             for rest_id, avg_rating in top_restaurants:
#                 if rest_id in restaurant_features.index:
#                     rest_info = restaurant_features.loc[rest_id]
#                     recommendations.append({
#                         'restaurant_id': rest_id,
#                         'name': rest_info.get('name', f'Restaurant {rest_id}'),
#                         'avg_rating': avg_rating,
#                         'features': rest_info.to_dict()
#                     })
            
#             return recommendations
#     except Exception as e:
#         st.error(f"Error getting top restaurants: {str(e)}")
    
#     return []

# def display_recommendations(recommendations, show_similarity=True):
#     """Display restaurant recommendations in a nice format"""
#     if not recommendations:
#         st.warning("No recommendations found.")
#         return
    
#     st.markdown("### 🎯 Top 5 Restaurant Recommendations")
    
#     for i, rec in enumerate(recommendations, 1):
#         with st.container():
#             col1, col2, col3 = st.columns([1, 3, 1])
            
#             with col1:
#                 st.markdown(f"**#{i}**")
#                 if 'combined_similarity' in rec and show_similarity:
#                     st.metric("Match Score", f"{rec['combined_similarity']:.2%}")
#                 if 'avg_rating' in rec:
#                     st.metric("Avg Rating", f"{rec['avg_rating']:.1f}")
            
#             with col2:
#                 st.markdown(f"### {rec['name']}")
#                 st.markdown(f"**Restaurant ID:** {rec['restaurant_id']}")
                
#                 # Show match reasons if available
#                 if 'match_reasons' in rec and rec['match_reasons']:
#                     st.markdown("**Why this matches:**")
#                     for reason in rec['match_reasons']:
#                         st.markdown(f"• {reason}")
                
#                 # Show similarity breakdown if available
#                 if show_similarity and 'text_similarity' in rec:
#                     with st.expander("📊 Similarity Details"):
#                         col_a, col_b = st.columns(2)
#                         with col_a:
#                             st.metric("Name Similarity", f"{rec['text_similarity']:.2%}")
#                         with col_b:
#                             st.metric("Feature Match", f"{rec['feature_similarity']:.2%}")
                
#                 # Display additional features if available
#                 if 'features' in rec:
#                     features = rec['features']
#                     if 'address' in features and pd.notna(features['address']):
#                         st.markdown(f"📍 **Address:** {features['address']}")
                    
#                     # Display other relevant features
#                     important_features = ['category', 'cuisine', 'price_range', 'atmosphere', 'specialties']
#                     for key, value in features.items():
#                         if (any(imp in key.lower() for imp in important_features) and 
#                             pd.notna(value) and str(value) != '0' and key not in ['name', 'address']):
#                             st.markdown(f"**{key.title()}:** {value}")
            
#             with col3:
#                 st.markdown("🍽️")
            
#             st.markdown("---")

# def main():
#     # Header
#     st.markdown('<h1 class="main-header">🍽️ Mauritania Restaurant Recommender</h1>', unsafe_allow_html=True)
    
#     # Load model
#     with st.spinner("Loading recommendation model..."):
#         model_package = load_model()
    
#     if model_package is None:
#         st.stop()
    
#     # Display model info
#     with st.sidebar:
#         st.markdown("## 📊 Model Information")
#         st.markdown(f"**Model Type:** {model_package.get('model_type', 'Unknown')}")
#         st.markdown(f"**Training Date:** {model_package.get('training_date', 'Unknown')}")
#         st.markdown(f"**Total Restaurants:** {model_package.get('total_restaurants', 'Unknown')}")
#         st.markdown(f"**Total Users:** {model_package.get('total_users', 'Unknown')}")
#         st.markdown(f"**Total Reviews:** {model_package.get('total_reviews', 'Unknown')}")
        
#         if 'final_rmse' in model_package:
#             st.markdown(f"**Model RMSE:** {model_package['final_rmse']:.3f}")
#         if 'final_mae' in model_package:
#             st.markdown(f"**Model MAE:** {model_package['final_mae']:.3f}")
    
#     # Main interface
#     st.markdown("## 🔍 Get Restaurant Recommendations")
    
#     # Create tabs for different recommendation methods
#     tab1, tab2 = st.tabs(["🎯 Find Similar Restaurants", "⭐ Top Rated"])
    
#     with tab1:
#         st.markdown("### Enter any restaurant name to find similar places")
#         st.markdown("*You can enter any restaurant name - it doesn't need to be from our database!*")
        
#         # Restaurant name input
#         restaurant_name = st.text_input(
#             "Restaurant Name:",
#             placeholder="e.g., Pizza Palace, Le Bistro, Dragon Garden, Ocean View Seafood...",
#             help="Enter any restaurant name. The system will analyze the name and find similar restaurants in Mauritania."
#         )
        
#         # Optional: Add some example buttons
#         st.markdown("**Try these examples:**")
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             if st.button("🍕 Pizza Corner"):
#                 restaurant_name = "Pizza Corner"
#         with col2:
#             if st.button("🍤 Ocean Breeze"):
#                 restaurant_name = "Ocean Breeze"
#         with col3:
#             if st.button("☕ Café Paris"):
#                 restaurant_name = "Café Paris"
#         with col4:
#             if st.button("🥘 Spice Garden"):
#                 restaurant_name = "Spice Garden"
        
#         if st.button("🔍 Find Similar Restaurants", type="primary"):
#             if restaurant_name.strip():
#                 with st.spinner(f"Analyzing '{restaurant_name}' and finding similar restaurants..."):
#                     # Show what the system detected
#                     query_features = extract_keywords_from_name(restaurant_name)
                    
#                     with st.expander("🔍 What we detected from your query"):
#                         col_a, col_b, col_c = st.columns(3)
#                         with col_a:
#                             st.markdown(f"**Cuisine Style:** {query_features['cuisine_type'].title()}")
#                         with col_b:
#                             st.markdown(f"**Ambiance:** {query_features['ambiance'].title()}")
#                         with col_c:
#                             st.markdown(f"**Price Level:** {query_features['price_level'].title()}")
                        
#                         if query_features['keywords']:
#                             st.markdown(f"**Keywords found:** {', '.join(query_features['keywords'])}")
                    
#                     recommendations = get_restaurant_recommendations_by_name(model_package, restaurant_name)
#                     display_recommendations(recommendations, show_similarity=True)
#             else:
#                 st.warning("Please enter a restaurant name.")
    
#     with tab2:
#         st.markdown("### Discover top-rated restaurants in Mauritania")
        
#         if st.button("⭐ Show Top Rated Restaurants", type="primary"):
#             with st.spinner("Loading top-rated restaurants..."):
#                 recommendations = get_top_rated_restaurants(model_package)
#                 display_recommendations(recommendations, show_similarity=False)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Restaurant ID input
#             restaurant_id = st.text_input(
#                 "Enter Restaurant ID:",
#                 placeholder="e.g., REST123",
#                 help="Enter the ID of a restaurant you like to find similar ones"
#             )
        
#         with col2:
#             # Restaurant name input
#             restaurant_name = st.text_input(
#                 "Or enter Restaurant Name:",
#                 placeholder="e.g., La Taverne",
#                 help="Enter part of the restaurant name to search"
#             )
        
#         if st.button("🔍 Find Similar Restaurants", type="primary"):
#             if restaurant_id or restaurant_name:
#                 with st.spinner("Finding similar restaurants..."):
#                     recommendations = get_restaurant_recommendations(
#                         model_package, 
#                         restaurant_id=restaurant_id if restaurant_id else None,
#                         restaurant_name=restaurant_name if restaurant_name else None
#                     )
#                     display_recommendations(recommendations)
#             else:
#                 st.warning("Please enter either a Restaurant ID or Restaurant Name.")
    
#     with tab2:
#         st.markdown("### Discover top-rated restaurants")
        
#         if st.button("⭐ Show Top Rated Restaurants", type="primary"):
#             with st.spinner("Loading top-rated restaurants..."):
#                 recommendations = get_restaurant_recommendations(model_package)
#                 display_recommendations(recommendations)
    
#     # Additional information
#     st.markdown("---")
#     st.markdown("## ℹ️ How it works")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         **Similar Restaurants:**
#         - Uses content-based filtering
#         - Analyzes restaurant features and characteristics
#         - Finds restaurants with similar attributes
#         - Calculates cosine similarity scores
#         """)
    
#     with col2:
#         st.markdown("""
#         **Top Rated:**
#         - Shows restaurants with highest average ratings
#         - Based on historical review data
#         - Considers overall popularity and quality
#         - Perfect for discovering new places
#         """)

# if __name__ == "__main__":
#     main()












# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import StandardScaler
# import re
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Mauritania Restaurant Recommender",
#     page_icon="🍽️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .restaurant-card {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 4px solid #1f77b4;
#         margin: 1rem 0;
#     }
#     .metric-card {
#         background-color: #e3f2fd;
#         padding: 1rem;
#         border-radius: 8px;
#         text-align: center;
#     }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_data
# def load_model():
#     """Load the trained recommender model"""
#     try:
#         with open('mauritania_restaurant_recommender.pkl', 'rb') as f:
#             model_package = pickle.load(f)
#         return model_package
#     except FileNotFoundError:
#         st.error("❌ Model file 'mauritania_restaurant_recommender.pkl' not found. Please make sure the file is in the same directory as this app.")
#         return None
#     except Exception as e:
#         st.error(f"❌ Error loading model: {str(e)}")
#         return None

# def extract_keywords_from_name(restaurant_name):
#     """Extract keywords and features from restaurant name"""
#     name_lower = restaurant_name.lower()
    
#     # Define cuisine/style keywords
#     cuisine_keywords = {
#         'italian': ['italian', 'pizza', 'pasta', 'trattoria', 'pizzeria', 'roma', 'napoli'],
#         'french': ['french', 'cafe', 'bistro', 'brasserie', 'chez', 'le ', 'la ', 'les'],
#         'chinese': ['chinese', 'dragon', 'golden', 'jade', 'peking', 'beijing'],
#         'mexican': ['mexican', 'taco', 'cantina', 'hacienda', 'el ', 'la '],
#         'indian': ['indian', 'taj', 'curry', 'spice', 'tandoor', 'maharaja'],
#         'japanese': ['japanese', 'sushi', 'sake', 'tokyo', 'sakura', 'zen'],
#         'american': ['grill', 'diner', 'burger', 'steakhouse', 'bbq', 'smokehouse'],
#         'mediterranean': ['mediterranean', 'olive', 'greek', 'cyprus', 'santorini'],
#         'seafood': ['seafood', 'fish', 'ocean', 'marina', 'catch', 'lobster', 'crab'],
#         'fast_food': ['fast', 'quick', 'express', 'drive', 'burger', 'chicken']
#     }
    
#     # Define ambiance keywords
#     ambiance_keywords = {
#         'fine_dining': ['fine', 'elegant', 'luxury', 'premium', 'exclusive', 'royal'],
#         'casual': ['casual', 'family', 'home', 'corner', 'neighborhood'],
#         'romantic': ['romantic', 'intimate', 'cozy', 'candlelight', 'moonlight'],
#         'sports': ['sports', 'game', 'stadium', 'victory', 'champion'],
#         'coffee': ['coffee', 'cafe', 'espresso', 'bean', 'roast', 'brew']
#     }
    
#     detected_features = {
#         'cuisine_type': 'general',
#         'ambiance': 'casual',
#         'price_level': 'medium',
#         'keywords': []
#     }
    
#     # Extract cuisine type
#     for cuisine, keywords in cuisine_keywords.items():
#         if any(keyword in name_lower for keyword in keywords):
#             detected_features['cuisine_type'] = cuisine
#             detected_features['keywords'].extend([k for k in keywords if k in name_lower])
#             break
    
#     # Extract ambiance
#     for amb, keywords in ambiance_keywords.items():
#         if any(keyword in name_lower for keyword in keywords):
#             detected_features['ambiance'] = amb
#             detected_features['keywords'].extend([k for k in keywords if k in name_lower])
#             break
    
#     # Extract price indicators
#     if any(word in name_lower for word in ['luxury', 'premium', 'fine', 'exclusive', 'royal']):
#         detected_features['price_level'] = 'high'
#     elif any(word in name_lower for word in ['budget', 'cheap', 'quick', 'fast', 'express']):
#         detected_features['price_level'] = 'low'
    
#     return detected_features

# def get_restaurant_recommendations_by_name(model_package, query_name):
#     """Get restaurant recommendations based on a user-provided restaurant name (not necessarily in dataset)"""
    
#     if model_package is None:
#         return []
    
#     try:
#         # Extract components from model package
#         restaurant_features = model_package['restaurant_features']
#         restaurant_averages = model_package['restaurant_averages']
        
#         # Extract features from the query name
#         query_features = extract_keywords_from_name(query_name)
        
#         # Get all restaurant names for text similarity
#         restaurant_names = restaurant_features.get('name', restaurant_features.index.astype(str)).fillna('')
        
#         # Create TF-IDF vectorizer for name similarity
#         tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
        
#         # Combine query with all restaurant names for vectorization
#         all_names = [query_name] + restaurant_names.tolist()
#         tfidf_matrix = tfidf.fit_transform(all_names)
        
#         # Calculate similarity between query and all restaurants
#         query_vector = tfidf_matrix[0]  # First row is our query
#         restaurant_vectors = tfidf_matrix[1:]  # Rest are restaurant names
        
#         # Calculate cosine similarity
#         text_similarities = cosine_similarity(query_vector, restaurant_vectors)[0]
        
#         # Create feature-based similarity scores
#         feature_similarities = []
        
#         for idx, rest_id in enumerate(restaurant_features.index):
#             rest_info = restaurant_features.loc[rest_id]
#             feature_score = 0.0
            
#             # Check for cuisine type matches (if available in restaurant features)
#             cuisine_cols = [col for col in restaurant_features.columns if 'cuisine' in col.lower() or 'category' in col.lower()]
#             for col in cuisine_cols:
#                 if pd.notna(rest_info.get(col)) and query_features['cuisine_type'] in str(rest_info[col]).lower():
#                     feature_score += 0.3
            
#             # Check for keyword matches in restaurant name
#             rest_name = str(rest_info.get('name', '')).lower()
#             keyword_matches = sum(1 for keyword in query_features['keywords'] if keyword in rest_name)
#             feature_score += keyword_matches * 0.2
            
#             # Check for ambiance/style matches
#             style_cols = [col for col in restaurant_features.columns if any(term in col.lower() for term in ['style', 'type', 'ambiance'])]
#             for col in style_cols:
#                 if pd.notna(rest_info.get(col)) and query_features['ambiance'] in str(rest_info[col]).lower():
#                     feature_score += 0.25
            
#             feature_similarities.append(feature_score)
        
#         # Combine text and feature similarities
#         combined_similarities = []
#         for i in range(len(text_similarities)):
#             # Weighted combination: 60% text similarity, 40% feature similarity
#             combined_score = 0.6 * text_similarities[i] + 0.4 * feature_similarities[i]
#             combined_similarities.append(combined_score)
        
#         # Get top 5 most similar restaurants
#         top_indices = np.argsort(combined_similarities)[::-1][:5]
        
#         recommendations = []
#         for idx in top_indices:
#             rest_id = restaurant_features.index[idx]
#             text_sim = text_similarities[idx]
#             feature_sim = feature_similarities[idx]
#             combined_sim = combined_similarities[idx]
            
#             # Get restaurant info
#             rest_info = restaurant_features.loc[rest_id]
#             avg_rating = restaurant_averages.get(rest_id, 0) if restaurant_averages is not None else 0
            
#             recommendations.append({
#                 'restaurant_id': rest_id,
#                 'name': rest_info.get('name', f'Restaurant {rest_id}'),
#                 'text_similarity': text_sim,
#                 'feature_similarity': feature_sim,
#                 'combined_similarity': combined_sim,
#                 'avg_rating': avg_rating,
#                 'features': rest_info.to_dict(),
#                 'match_reasons': get_match_reasons(query_features, rest_info)
#             })
        
#         return recommendations
        
#     except Exception as e:
#         st.error(f"Error generating recommendations: {str(e)}")
#         return []

# def get_match_reasons(query_features, restaurant_info):
#     """Get reasons why this restaurant matches the query"""
#     reasons = []
    
#     rest_name = str(restaurant_info.get('name', '')).lower()
    
#     # Check cuisine match
#     if query_features['cuisine_type'] != 'general':
#         cuisine_keywords = {
#             'italian': ['italian', 'pizza', 'pasta', 'trattoria', 'pizzeria'],
#             'french': ['french', 'cafe', 'bistro', 'brasserie'],
#             'chinese': ['chinese', 'dragon', 'golden', 'jade'],
#             'mexican': ['mexican', 'taco', 'cantina', 'hacienda'],
#             'seafood': ['seafood', 'fish', 'ocean', 'marina', 'catch']
#         }
        
#         if query_features['cuisine_type'] in cuisine_keywords:
#             for keyword in cuisine_keywords[query_features['cuisine_type']]:
#                 if keyword in rest_name:
#                     reasons.append(f"Similar cuisine style ({keyword})")
#                     break
    
#     # Check keyword matches
#     for keyword in query_features['keywords']:
#         if keyword in rest_name:
#             reasons.append(f"Name contains '{keyword}'")
    
#     # Check ambiance match
#     if query_features['ambiance'] != 'casual':
#         ambiance_keywords = {
#             'fine_dining': ['fine', 'elegant', 'luxury', 'premium'],
#             'romantic': ['romantic', 'intimate', 'cozy'],
#             'coffee': ['coffee', 'cafe', 'espresso', 'bean']
#         }
        
#         if query_features['ambiance'] in ambiance_keywords:
#             for keyword in ambiance_keywords[query_features['ambiance']]:
#                 if keyword in rest_name:
#                     reasons.append(f"Similar ambiance ({keyword})")
#                     break
    
#     if not reasons:
#         reasons.append("Text similarity match")
    
#     return reasons

# def get_top_rated_restaurants(model_package):
#     """Get top-rated restaurants"""
#     try:
#         restaurant_features = model_package['restaurant_features']
#         restaurant_averages = model_package['restaurant_averages']
        
#         if restaurant_averages is not None:
#             top_restaurants = sorted(restaurant_averages.items(), key=lambda x: x[1], reverse=True)[:5]
            
#             recommendations = []
#             for rest_id, avg_rating in top_restaurants:
#                 if rest_id in restaurant_features.index:
#                     rest_info = restaurant_features.loc[rest_id]
#                     recommendations.append({
#                         'restaurant_id': rest_id,
#                         'name': rest_info.get('name', f'Restaurant {rest_id}'),
#                         'avg_rating': avg_rating,
#                         'features': rest_info.to_dict()
#                     })
            
#             return recommendations
#     except Exception as e:
#         st.error(f"Error getting top restaurants: {str(e)}")
    
#     return []

# def display_recommendations(recommendations, show_similarity=True):
#     """Display restaurant recommendations in a nice format"""
#     if not recommendations:
#         st.warning("No recommendations found.")
#         return
    
#     st.markdown("### 🎯 Top 5 Restaurant Recommendations")
    
#     for i, rec in enumerate(recommendations, 1):
#         with st.container():
#             col1, col2, col3 = st.columns([1, 3, 1])
            
#             with col1:
#                 st.markdown(f"**#{i}**")
#                 if 'combined_similarity' in rec and show_similarity:
#                     st.metric("Match Score", f"{rec['combined_similarity']:.2%}")
#                 if 'avg_rating' in rec:
#                     st.metric("Avg Rating", f"{rec['avg_rating']:.1f}")
            
#             with col2:
#                 st.markdown(f"### {rec['name']}")
#                 st.markdown(f"**Restaurant ID:** {rec['restaurant_id']}")
                
#                 # Show match reasons if available
#                 if 'match_reasons' in rec and rec['match_reasons']:
#                     st.markdown("**Why this matches:**")
#                     for reason in rec['match_reasons']:
#                         st.markdown(f"• {reason}")
                
#                 # Show similarity breakdown if available
#                 if show_similarity and 'text_similarity' in rec:
#                     with st.expander("📊 Similarity Details"):
#                         col_a, col_b = st.columns(2)
#                         with col_a:
#                             st.metric("Name Similarity", f"{rec['text_similarity']:.2%}")
#                         with col_b:
#                             st.metric("Feature Match", f"{rec['feature_similarity']:.2%}")
                
#                 # Display additional features if available
#                 if 'features' in rec:
#                     features = rec['features']
#                     if 'address' in features and pd.notna(features['address']):
#                         st.markdown(f"📍 **Address:** {features['address']}")
                    
#                     # Display other relevant features
#                     important_features = ['category', 'cuisine', 'price_range', 'atmosphere', 'specialties']
#                     for key, value in features.items():
#                         if (any(imp in key.lower() for imp in important_features) and 
#                             pd.notna(value) and str(value) != '0' and key not in ['name', 'address']):
#                             st.markdown(f"**{key.title()}:** {value}")
            
#             with col3:
#                 st.markdown("🍽️")
            
#             st.markdown("---")

# def main():
#     # Header
#     st.markdown('<h1 class="main-header">🍽️ Mauritania Restaurant Recommender</h1>', unsafe_allow_html=True)
    
#     # Load model
#     with st.spinner("Loading recommendation model..."):
#         model_package = load_model()
    
#     if model_package is None:
#         st.stop()
    
#     # Display model info
#     with st.sidebar:
#         st.markdown("## 📊 Model Information")
#         st.markdown(f"**Model Type:** {model_package.get('model_type', 'Unknown')}")
#         st.markdown(f"**Training Date:** {model_package.get('training_date', 'Unknown')}")
#         st.markdown(f"**Total Restaurants:** {model_package.get('total_restaurants', 'Unknown')}")
#         st.markdown(f"**Total Users:** {model_package.get('total_users', 'Unknown')}")
#         st.markdown(f"**Total Reviews:** {model_package.get('total_reviews', 'Unknown')}")
        
#         if 'final_rmse' in model_package:
#             st.markdown(f"**Model RMSE:** {model_package['final_rmse']:.3f}")
#         if 'final_mae' in model_package:
#             st.markdown(f"**Model MAE:** {model_package['final_mae']:.3f}")
    
#     # Main interface
#     st.markdown("## 🔍 Get Restaurant Recommendations")
    
#     # Create tabs for different recommendation methods
#     tab1, tab2 = st.tabs(["🎯 Find Similar Restaurants", "⭐ Top Rated"])
    
#     with tab1:
#         st.markdown("### Enter any restaurant name to find similar places")
#         st.markdown("*You can enter any restaurant name - it doesn't need to be from our database!*")
        
#         # Restaurant name input
#         restaurant_name = st.text_input(
#             "Restaurant Name:",
#             placeholder="e.g., Pizza Palace, Le Bistro, Dragon Garden, Ocean View Seafood...",
#             help="Enter any restaurant name. The system will analyze the name and find similar restaurants in Mauritania.",
#             key="restaurant_name_input"
#         )
        
#         # Optional: Add some example buttons
#         st.markdown("**Try these examples:**")
#         col1, col2, col3, col4 = st.columns(4)
        
#         example_clicked = None
#         with col1:
#             if st.button("🍕 Pizza Corner", key="example_pizza"):
#                 example_clicked = "Pizza Corner"
#         with col2:
#             if st.button("🍤 Ocean Breeze", key="example_ocean"):
#                 example_clicked = "Ocean Breeze"
#         with col3:
#             if st.button("☕ Café Paris", key="example_cafe"):
#                 example_clicked = "Café Paris"
#         with col4:
#             if st.button("🥘 Spice Garden", key="example_spice"):
#                 example_clicked = "Spice Garden"
        
#         # Use example if clicked, otherwise use text input
#         final_restaurant_name = example_clicked if example_clicked else restaurant_name
        
#         if st.button("🔍 Find Similar Restaurants", type="primary", key="main_search_button"):
#             if final_restaurant_name and final_restaurant_name.strip():
#                 with st.spinner(f"Analyzing '{final_restaurant_name}' and finding similar restaurants..."):
#                     # Show what the system detected
#                     query_features = extract_keywords_from_name(final_restaurant_name)
                    
#                     with st.expander("🔍 What we detected from your query"):
#                         col_a, col_b, col_c = st.columns(3)
#                         with col_a:
#                             st.markdown(f"**Cuisine Style:** {query_features['cuisine_type'].title()}")
#                         with col_b:
#                             st.markdown(f"**Ambiance:** {query_features['ambiance'].title()}")
#                         with col_c:
#                             st.markdown(f"**Price Level:** {query_features['price_level'].title()}")
                        
#                         if query_features['keywords']:
#                             st.markdown(f"**Keywords found:** {', '.join(query_features['keywords'])}")
                    
#                     recommendations = get_restaurant_recommendations_by_name(model_package, final_restaurant_name)
#                     display_recommendations(recommendations, show_similarity=True)
#             else:
#                 st.warning("Please enter a restaurant name or click one of the examples.")
    
#     with tab2:
#         st.markdown("### Discover top-rated restaurants in Mauritania")
        
#         if st.button("⭐ Show Top Rated Restaurants", type="primary", key="top_rated_button"):
#             with st.spinner("Loading top-rated restaurants..."):
#                 recommendations = get_top_rated_restaurants(model_package)
#                 display_recommendations(recommendations, show_similarity=False)
    
#     # Additional information
#     st.markdown("---")
#     st.markdown("## ℹ️ How it works")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         **Smart Name Analysis:**
#         - Analyzes any restaurant name you provide
#         - Detects cuisine type (Italian, French, Chinese, etc.)
#         - Identifies ambiance (fine dining, casual, romantic)
#         - Extracts key features and style indicators
#         - Uses TF-IDF for text similarity matching
#         """)
    
#     with col2:
#         st.markdown("""
#         **Recommendation Process:**
#         - Combines text similarity (60%) with feature matching (40%)
#         - Finds restaurants with similar names and characteristics
#         - Considers cuisine type, ambiance, and style
#         - Ranks by combined similarity score
#         - Returns top 5 most relevant matches
#         """)
    
#     st.markdown("---")
#     st.markdown("""
#     ### 💡 Tips for better recommendations:
#     - Use descriptive names like "Ocean View Seafood" instead of just "Restaurant"
#     - Include cuisine type in the name (e.g., "Mario's Italian Kitchen")
#     - Mention style or ambiance (e.g., "Cozy Corner Cafe", "Elegant Dining Palace")
#     - The system works with any language and restaurant name format
#     """)

# if __name__ == "__main__":
#     main()




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
        
        example_clicked = None
        with col1:
            if st.button("🍕 Pizza Corner", key="example_pizza"):
                example_clicked = "Pizza Corner"
        with col2:
            if st.button("🍤 Ocean Breeze", key="example_ocean"):
                example_clicked = "Ocean Breeze"
        with col3:
            if st.button("☕ Café Paris", key="example_cafe"):
                example_clicked = "Café Paris"
        with col4:
            if st.button("🥘 Spice Garden", key="example_spice"):
                example_clicked = "Spice Garden"
        
        # Use example if clicked, otherwise use text input
        final_restaurant_name = example_clicked if example_clicked else restaurant_name
        
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