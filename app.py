# import streamlit as st
# import pickle
# import pandas as pd

# @st.cache_resource
# def load_model(path="mauritania_restaurant_recommender.pkl"):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# model = load_model()

# st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
# st.title("🍽️ Restaurant Recommender")

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
#             sims = model.recommend_by_item(restaurant, top_k=5)
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
#         if mode == "By User ID":
#             recs = model.recommend(val, top_k=5)
#             for item, score in recs:
#                 results.append({key: val, "restaurant": item, "score": score})
#         else:
#             sims = model.recommend_by_item(val, top_k=5)
#             for item, sim in sims:
#                 results.append({key: val, "restaurant": item, "similarity": sim})

#     out = pd.DataFrame(results)
#     st.dataframe(out)  # display the full results in the app










import streamlit as st
import pickle
import pandas as pd

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

@st.cache_resource
def load_model(path="mauritania_restaurant_recommender.pkl"):
    with open(path, "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        # step 1: get which model was best
        model_key = raw.get("best_model_name")
        if not model_key:
            st.error(f"No 'best_model_name' found in pickle. Keys: {list(raw.keys())}")
            st.stop()

        # step 2: grab that sub-model
        if model_key in raw:
            return raw[model_key]
        else:
            st.error(f"best_model_name='{model_key}' not in pickle. Available: {list(raw.keys())}")
            st.stop()

    # if it wasn’t a dict, assume it is the model itself
    return raw


try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
st.title("🍽️ Restaurant Recommender")

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
        try:
            recs = model.recommend(user_id, top_k=5)
            df = pd.DataFrame(recs, columns=["Restaurant", "Score"])
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error: {e}")
else:
    restaurant = st.text_input("Enter Restaurant Name")
    if st.button("Similar Restaurants"):
        try:
            # Check if method exists
            if hasattr(model, 'recommend_by_item'):
                sims = model.recommend_by_item(restaurant, top_k=5)
            else:
                st.error("Model doesn't have 'recommend_by_item' method")
                st.stop()
            df = pd.DataFrame(sims, columns=["Restaurant", "Similarity"])
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("### Batch Upload (CSV)")

uploaded = st.file_uploader("Upload CSV (column: user_id or restaurant)", type=["csv"])
if uploaded:
    df_in = pd.read_csv(uploaded)
    results = []
    key = "user_id" if mode == "By User ID" else "restaurant"

    for val in df_in[key].astype(str):
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

    if results:
        out = pd.DataFrame(results)
        st.dataframe(out)
    else:
        st.warning("No results generated.") 