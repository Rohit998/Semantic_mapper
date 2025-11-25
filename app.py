import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize session state for threshold if it doesn't exist
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5

# Load fine-tuned model
@st.cache_resource  # Cache the model to prevent reloading
def load_model():
    # Try to load fine-tuned model, fall back to base model if not found
    model_path = "./fine_tuned_model"
    
    # Check if the fine-tuned model directory exists
    if os.path.exists(model_path) and os.path.isdir(model_path):
        try:
            return SentenceTransformer(model_path)
        except Exception as e:
            st.warning(f"Could not load fine-tuned model: {e}. Using base model instead.")
            return SentenceTransformer('all-mpnet-base-v2')
    else:
        # Use base model if fine-tuned model doesn't exist
        st.info("Fine-tuned model not found. Using base model 'all-mpnet-base-v2'. Train a model using Train_model2.py to use a custom fine-tuned model.")
        return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

st.title("Sentence-to-Sentence Semantic Matcher (Two Inputs)")
st.write("Compare each sentence from Box A with each sentence from Box B.")

# Move threshold slider before the input areas
threshold = st.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.threshold,
    step=0.01,
    key='threshold_slider'
)

# Input areas
sentences_a_input = st.text_area("Enter Sentences (Box A - one per line):")
sentences_b_input = st.text_area("Enter Sentences (Box B - one per line):")

if st.button("Compare Sentences"):
    if not sentences_a_input.strip() or not sentences_b_input.strip():
        st.warning("Please enter sentences in both boxes.")
    else:
        sentences_a = [s.strip() for s in sentences_a_input.strip().split("\n") if s]
        sentences_b = [s.strip() for s in sentences_b_input.strip().split("\n") if s]

        embeddings_a = model.encode(sentences_a)
        embeddings_b = model.encode(sentences_b)

        st.markdown("### Results:")
        for i, sentence_a in enumerate(sentences_a):
            matches_found = False
            st.write(f"Matches for: `{sentence_a}`")
            
            for j, sentence_b in enumerate(sentences_b):
                sim = cosine_similarity([embeddings_a[i]], [embeddings_b[j]])[0][0]
                if sim >= threshold:
                    matches_found = True
                    st.write(f"- `{sentence_b}` (Similarity: {sim:.2f})")
            
            if not matches_found:
                st.write("- No matches found above threshold")
            st.markdown("---")
