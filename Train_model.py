from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# Load dataset and clean column names
df = pd.read_csv("train_data.csv")
df.columns = df.columns.str.strip()  # Remove any trailing/leading spaces from column names
print("Available columns:", df.columns.tolist())

# Convert to InputExample format
train_examples = [
    InputExample(texts=[row['Sentence1'], row['Sentence2']], label=float(row['Label']))
    for _, row in df.iterrows()
]

# Load base model
model = SentenceTransformer('all-mpnet-base-v2')

# Prepare data loader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

# Define loss
train_loss = losses.CosineSimilarityLoss(model)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=10,
    show_progress_bar=True
)

# Save fine-tuned model
model.save("fine_tuned_model")
