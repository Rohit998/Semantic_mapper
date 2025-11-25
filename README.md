# Sentence Matcher - Semantic Similarity Tool

A semantic sentence matching application that uses fine-tuned sentence transformers to compare and match sentences between two sets.
This project includes a Streamlit web interface for interactive sentence-to-sentence comparison and tools for training, testing, and comparing models.

**Repository**: [https://github.com/Rohit998/Semantic_mapper.git](https://github.com/Rohit998/Semantic_mapper.git)

## Features

- **Fine-tuned Sentence Transformer Model**: Custom-trained model based on `all-mpnet-base-v2` for domain-specific sentence similarity
- **Interactive Web Interface**: Streamlit app with dual input boxes for comparing sentence sets
- **Similarity Scoring**: Cosine similarity-based matching with adjustable threshold
- **Model Comparison**: Compare base model vs fine-tuned model performance
- **Testing Tools**: Scripts for testing and validating model performance

## Project Structure

```
sentence_matcher_dual_box/
├── app.py                          # Streamlit web application
├── Train_model.py                  # Training script (basic version)
├── Train_model2.py                 # Training script (enhanced version)
├── test_model.py                   # Model testing script
├── compare_model.py                # Base vs fine-tuned model comparison
├── load_module.py                  # Model loading utility
├── train_data.csv                  # Training dataset
├── requirements.txt                # Python dependencies
├── fine_tuned_model/               # Saved fine-tuned model
├── checkpoints/                    # Training checkpoints
└── model_comparison_results.csv    # Comparison results output
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rohit998/Semantic_mapper.git
   cd Semantic_mapper
   ```
   
   Or navigate to the project directory if already cloned:
   ```bash
   cd sentence_matcher_dual_box
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - `streamlit`
   - `sentence-transformers`
   - `scikit-learn`
   - `pandas`
   - `torch`

## Usage

### Running the Streamlit App

Launch the interactive web interface:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

**How to use the app:**
1. Adjust the similarity threshold using the slider (0.0 to 1.0)
2. Enter sentences in **Box A** (one per line)
3. Enter sentences in **Box B** (one per line)
4. Click "Compare Sentences" to see matches
5. Results show which sentences from Box B match each sentence in Box A above the threshold

### Training a Model

#### Option 1: Using Train_model2.py (Recommended)
```bash
python Train_model2.py
```

This script:
- Validates the training data format
- Fine-tunes the base model on your dataset
- Saves the model to `fine_tuned_model/`
- Performs a quick test on the first training example

#### Option 2: Using Train_model.py
```bash
python Train_model.py
```

**Training Data Format:**
Your `train_data.csv` should have the following columns:
- `Sentence1`: First sentence
- `Sentence2`: Second sentence
- `Label`: Similarity label (float, typically 0.0 to 1.0)

Example:
```csv
Sentence1,Sentence2,Label
"Accurately presents an organized summary of a patient case verbally and in writing","Communicate effectively with colleagues within one's profession or specialty, other health professionals, and health related agencies",1
```

### Testing the Model

Test the fine-tuned model on custom sentences:

```bash
python test_model.py
```

You can modify `test_model.py` to test your own sentence pairs.

### Comparing Models

Compare the base model (`all-mpnet-base-v2`) with your fine-tuned model:

```bash
python compare_model.py
```

This will:
- Evaluate both models on your training data
- Display similarity scores side-by-side
- Save results to `model_comparison_results.csv`

## Model Details

- **Base Model**: `all-mpnet-base-v2` (Microsoft's MPNet-based sentence transformer)
- **Fine-tuning**: Cosine Similarity Loss
- **Training Parameters**:
  - Epochs: 3
  - Batch Size: 4
  - Warmup Steps: 10

## Customization

### Adjusting Training Parameters

Edit `Train_model2.py` to modify:
- Number of epochs
- Batch size
- Warmup steps
- Learning rate (if specified)

### Changing the Model Path

If your model is saved in a different location, update the path in:
- `app.py` (line 13)
- `test_model.py` (line 3)
- `compare_model.py` (line 7)

## Requirements

- Python 3.7+
- PyTorch
- sentence-transformers
- streamlit
- scikit-learn
- pandas

## Notes

- The model is cached in the Streamlit app to improve performance
- Training requires a GPU for faster processing (CPU will work but slower)
- The fine-tuned model is saved in the `fine_tuned_model/` directory 
