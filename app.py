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



import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model(path="mauritania_restaurant_recommender.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

raw = load_model()

# If you pickled a dict of components, choose the one that has recommend()/recommend_by_item()
if isinstance(raw, dict):
    # e.g. if you stored under 'hybrid' or 'ridge', adjust the key to your case:
    model = raw.get('hybrid') or raw.get('ridge') or list(raw.values())[0]
else:
    model = raw

st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
st.title("🍽️ Restaurant Recommender")

# … rest of your code unchanged …
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
            sims = model.recommend_by_item(restaurant, top_k=5)
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
        if mode == "By User ID":
            recs = model.recommend(val, top_k=5)
            for item, score in recs:
                results.append({key: val, "restaurant": item, "score": score})
        else:
            sims = model.recommend_by_item(val, top_k=5)
            for item, sim in sims:
                results.append({key: val, "restaurant": item, "similarity": sim})

    out = pd.DataFrame(results)
    st.dataframe(out)  # display the full results in the app
