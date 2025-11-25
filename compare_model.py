
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load models
base_model = SentenceTransformer("all-mpnet-base-v2")
fine_tuned_model = SentenceTransformer("fine_tuned_model")

# Load data
df = pd.read_csv("train_data.csv")
df["Label"] = df["Label"].astype(float)

results = []

print("Comparing base model vs fine-tuned model:\n")
for idx, row in df.iterrows():
    sent1 = row["Sentence1"]
    sent2 = row["Sentence2"]
    true_label = row["Label"]

    # Encode both models
    base_sim = util.cos_sim(
        base_model.encode(sent1, convert_to_tensor=True),
        base_model.encode(sent2, convert_to_tensor=True)
    ).item()

    tuned_sim = util.cos_sim(
        fine_tuned_model.encode(sent1, convert_to_tensor=True),
        fine_tuned_model.encode(sent2, convert_to_tensor=True)
    ).item()

    results.append({
        "Sentence 1": sent1,
        "Sentence 2": sent2,
        "True Label": true_label,
        "Base Score": round(base_sim, 3),
        "Fine-Tuned Score": round(tuned_sim, 3)
    })

    print(f"[{idx+1}]")
    print(f"Sentence 1: {sent1}")
    print(f"Sentence 2: {sent2}")
    print(f"True Label: {true_label:.2f} | Base: {base_sim:.2f} | Fine-Tuned: {tuned_sim:.2f}")
    print("-" * 60)

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved comparison results to 'model_comparison_results.csv'")
