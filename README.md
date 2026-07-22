# Hybrid Insurance Fraud Detection
**Integrating DistilBERT Textual Embeddings with Gradient Boosting**

This repository contains an end-to-end machine learning pipeline combining standard tabular data (XGBoost/Random Forest) with Natural Language Processing (DistilBERT)
to detect fraudulent insurance claims. Includes an interactive Streamlit dashboard for real-time claim triage.

## ⚙️ Setup & Installation
`git clone https://github.com/Professional-Team-Project-Group-25/Fraud-Detection.git && cd Fraud-Detection && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## 🚀 Execution Guide
1. **Train Tabular Baselines:** `python scripts/1_train_tabular.py`
2. **Fine-Tune DistilBERT:** `python scripts/2_train_distilbert.py`
3. **Generate Enriched Dataset:** `python scripts/3_generate_hybrid.py`
4. **Launch Dashboard:** `streamlit run app.py`

## 📊 Repository Structure
* `docs/`: Final Group Project Report PDF.
* `data/`: Raw and NLP-enriched datasets.
* `models/`: Trained models (`.pkl`) and DistilBERT weights.
* `notebooks/`: Exploratory Data Analysis.
* `scripts/`: 3-phase architecture execution scripts.

---
*MSc Data Science Team Project (7PAM2033) — Group 25, University of Hertfordshire*
