import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    # Load the model from your local directory where it was saved
    return SentenceTransformer("fine_tuned_model") 
model = load_model() # Adjust this path to match where your model is saved