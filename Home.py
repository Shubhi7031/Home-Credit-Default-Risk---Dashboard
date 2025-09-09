# Home.py
import streamlit as st
import pandas as pd

from utils import load_data  # your flexible loader (handles UploadedFile or path)
from preprocessing import preprocess_application_train, summarize_missingness  # your preprocessor

st.set_page_config(page_title="Home Credit Dashboard", page_icon="🛍️", layout="wide")

st.title("Home Credit Analysis Dashboard")
st.markdown("""
Welcome to the *Home Credit Dashboard* built with *Streamlit*.  
Use the sidebar to navigate between modules:
* 📊 Overview and Data Quality
* 🎯 Target and Segmentation 
* 👨‍👩‍👧 Demographics and Household  
* 💰 Financial Health
* 📈 Correlations and Drivers
""")

st.markdown("---")
st.subheader("Upload / Use Default Dataset")

uploaded_file = st.file_uploader("D:/Home_credit/data/application_train", type=["csv"])

with st.spinner("Loading data..."):
    # If a file is uploaded, pass the UploadedFile object directly.
    # Otherwise, fall back to your repo-relative default path.
    if uploaded_file is not None:
        df = load_data.load_data(uploaded_file)
        st.success("Using uploaded file.")
    else:
        df = load_data.load_data("data/application_train.csv")
        # st.info("No file uploaded. Using default data at data/application_train.csv")


st.caption(f"Raw dataset — Rows: {len(df):,} | Columns: {len(df.columns):,}")

# ---------- Preprocess once and cache in session ----------
with st.spinner("Preprocessing (types, derived fields, imputation, winsorization, rare labels)..."):
    clean_df, artifacts = preprocess_application_train(df)

st.session_state["raw_df"] = df
st.session_state["clean_df"] = clean_df
st.session_state["artifacts"] = artifacts

# ---------- Quick QA views ----------
with st.expander("Data Quality — Missingness (Top 20)"):
    st.dataframe(summarize_missingness(df).head(20), use_container_width=True)

st.subheader("Sample of Cleaned Data (first 100 rows)")
st.dataframe(clean_df.head(100), use_container_width=True)

st.caption("Note: Numeric fields used in charts/KPIs are winsorized at 1% tails; ratios guard against zero/negative denominators.")
