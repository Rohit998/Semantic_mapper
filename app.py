import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize session state for threshold if it doesn't exist
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5

# Load fine-tuned model
@st.cache_resource  # Cache the model to prevent reloading
def load_model():
    # Use the absolute path to your fine-tuned model
    model_path = "./fine_tuned_model"  # Adjust this path to where your model is actually saved
    return SentenceTransformer(model_path)

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
