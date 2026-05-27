# mlops_assignment2
# MLOps Assignment 2 — Fine-Tuning DistilBERT for Book Genre Classification

Fine-tuning a `distilbert-base-cased` model on Goodreads reviews to classify book genres across 7 categories, with a complete MLOps pipeline covering GPU-accelerated training on Kaggle, experiment tracking via Weights & Biases, and model deployment to Hugging Face Hub.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Pipeline Overview](#pipeline-overview)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [MLOps Components](#mlops-components)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Reproduction](#setup--reproduction)
- [Links](#links)

---

## Problem Statement

Given a Goodreads book review (free text), predict the **genre** of the book from 7 categories:

| Genre                      |
|----------------------------|
| Poetry                     |
| Comics & Graphic           |
| Fantasy & Paranormal       |
| History & Biography        |
| Mystery, Thriller & Crime  |
| Romance                    |
| Young Adult                |

This is a **multi-class text classification** task. The MLOps focus is on building a reproducible, tracked, and deployable pipeline — not on maximising accuracy.

---

## Dataset

- **Source**: [UCSD Goodreads Book Graph](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/) — one `.json.gz` file per genre
- **Sampling**: Reviews sampled per genre for manageable training on free GPU
- **Split**: 80% train / 20% test, stratified by genre
- **Input field**: `review_text` (truncated to 512 tokens by tokenizer)

---

## Pipeline Overview

```
Raw Goodreads Reviews (7 genres, UCSD Book Graph)
        │
        ▼
  Data Loading & Sampling
        │
        ▼
  DistilBERT Tokenization
  (distilbert-base-cased, max_length=512)
        │
        ▼
  Fine-Tuning on Kaggle GPU (T4)
  (HuggingFace Trainer API, 3 epochs)
        │
        ├──► Experiment Tracking (Weights & Biases)
        │         - Loss, Accuracy, F1 per epoch
        │         - Hyperparameters logged
        │         - Eval report as W&B Artifact
        ▼
  Evaluation on Test Set
  (Accuracy, F1, Classification Report)
        │
        ▼
  Model Deployment → Hugging Face Hub
  (publicly accessible for inference)
```

---

## Model Architecture

| Component        | Detail                                  |
|------------------|-----------------------------------------|
| Base model       | `distilbert-base-cased`                 |
| Task head        | `DistilBertForSequenceClassification`   |
| Number of labels | 7                                       |
| Max token length | 512                                     |
| Framework        | HuggingFace Transformers + PyTorch      |

**Why DistilBERT?**
DistilBERT is a distilled (compressed) version of BERT that retains ~97% of BERT's language understanding while being 40% smaller and 60% faster. For an MLOps assignment focused on workflow rather than state-of-the-art accuracy, DistilBERT offers the ideal balance between performance and training speed on free Kaggle GPU resources. Its well-documented HuggingFace integration also makes it straightforward to load, fine-tune, and deploy within a single pipeline.

---

## Training Configuration

| Hyperparameter       | Value                        |
|----------------------|------------------------------|
| Epochs               | 3                            |
| Train batch size     | 16                           |
| Eval batch size      | 32                           |
| Learning rate        | 3e-5                         |
| Warmup steps         | 100                          |
| Weight decay         | 0.01                         |
| Logging steps        | 50                           |
| Evaluation strategy  | Every epoch                  |
| Optimizer            | AdamW (HuggingFace default)  |
| Training platform    | Kaggle Notebook (GPU T4)     |

---

## MLOps Components

### 1. Experiment Tracking — Weights & Biases
- All runs logged automatically via `report_to="wandb"` in `TrainingArguments`
- Tracks: training loss, eval loss, accuracy, F1 score per epoch
- Hyperparameters captured in W&B config for full reproducibility
- Final evaluation report saved as a versioned **W&B Artifact**

### 2. Training Platform — Kaggle Notebooks
- Hardware: **GPU T4** (free tier, 30 hrs/week)
- Internet enabled for HuggingFace model downloads and Hub push
- API credentials stored securely via **Kaggle Secrets** (`WANDB_API_KEY`, `HF_TOKEN`)
- Zero hardcoded credentials in any code

### 3. Model Deployment — Hugging Face Hub
- Fine-tuned model and tokenizer pushed to HuggingFace Hub
- Publicly accessible — anyone can load and run inference with one line:
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="DishaSinghania/distilbert-goodreads-genres")
result = classifier("This fantasy novel had incredible world-building and magic systems.")
print(result)
```

---

## Results

| Metric      | Score |
|-------------|-------|
| Accuracy    |       |
| Weighted F1 |       |
| Eval Loss   |       |

> 

### What the W&B Charts Showed
- Training loss decreased steadily across all 3 epochs — model was learning
- Validation loss tracked training loss closely — no significant overfitting
- Accuracy and F1 improved each epoch, confirming the fine-tuning was effective

---

## Project Structure

```
mlops-assignment2/
├── mlops-assignment.ipynb      # Kaggle notebook — full training pipeline
├── mlops_assignment.py         # Python script exported from Kaggle notebook
├── requirements.txt            # All Python dependencies
└── README.md                   # This file
```

---

## Setup & Reproduction

### 1. Clone the repository
```bash
git clone https://github.com/Disha19/mlops-assignment2.git
cd mlops-assignment2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_huggingface_token>
```
> On Kaggle: use **Add-ons → Secrets** to store these securely instead of environment variables.

### 4. Run on Kaggle (Recommended)
- Import `mlops-assignment.ipynb` into Kaggle
- Enable GPU: Settings → Accelerator → GPU T4
- Enable Internet: Settings → Environment Preferences → Internet ON
- Add secrets: Add-ons → Secrets → add `WANDB_API_KEY` and `HF_TOKEN`
- Click **Run All**

### 5. Run locally (CPU only — slow)
```bash
python mlops_assignment.py
```
> For CPU runs, reduce sample size to 200 per genre and set `device = 'cpu'`

---

## Links

| Resource            | URL                                                                                   |
|---------------------|---------------------------------------------------------------------------------------|
| 🤗 Hugging Face Model | [DishaSinghania/distilbert-goodreads-genres](https://huggingface.co/DishaSinghania/distilbert-goodreads-genres) |
| 📊 W&B Dashboard      | [mlops-assignment2](https://wandb.ai/dishasinghania19/mlops-assignment2)              |
| 📓 Kaggle Notebook    | [mlops-assignment](https://www.kaggle.com/code/disha2031/mlops-assignment)            |
| 💻 GitHub Repository  | [mlops-assignment2](https://github.com/Disha19/mlops-assignment2)                    |
