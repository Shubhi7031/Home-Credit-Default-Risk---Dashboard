# utils/load_data.py
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path="D:/Home_credit/data/application_train.csv"):
    """
    Load dataset from a path or a Streamlit UploadedFile.
    """
    
    return pd.read_csv(file_path)
