import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import os

# Load and validate the dataset
df = pd.read_csv("train_data.csv")
if not all(col in df.columns for col in ["Sentence1", "Sentence2", "Label"]):
    print("Available columns:", df.columns.tolist())
    raise ValueError("CSV must contain 'Sentence1', 'Sentence2', and 'Label' columns")

# Convert labels to float if needed
df["Label"] = df["Label"].astype(float)

# Create training examples
train_examples = [
    InputExample(texts=[row["Sentence1"], row["Sentence2"]], label=row["Label"])
    for _, row in df.iterrows()
]

# Load base model
model = SentenceTransformer("all-mpnet-base-v2")

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
train_loss = losses.CosineSimilarityLoss(model=model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=10,
    show_progress_bar=True
)

# Save model
save_path = "fine_tuned_model"
os.makedirs(save_path, exist_ok=True)
model.save(save_path)
print(f"Model fine-tuned and saved to '{save_path}'")

# Quick test on one example
print("\nQuick test:")
test = train_examples[0]
emb1 = model.encode(test.texts[0], convert_to_tensor=True)
emb2 = model.encode(test.texts[1], convert_to_tensor=True)
score = util.cos_sim(emb1, emb2).item()
print(f"Similarity between:\n- {test.texts[0]}\n- {test.texts[1]}\n= {score:.2f}")
