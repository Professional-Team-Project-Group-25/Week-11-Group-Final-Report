import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os

os.chdir(r"D:\python setup")
MODEL_DIR = r"D:\python setup\distilbert_claims_model"

def build_claim_narrative(row):
    return f"A {row['age']}-year-old customer from {row['policy_state']} reported a {row['incident_severity']} {row['incident_type']} incident. The authorities were {row['police_report_available']} to the scene. The total requested claim amount is ${row['total_claim_amount']}, which includes a vehicle damage claim of ${row['vehicle_claim']}."

print("Loading original dataset...")
df = pd.read_csv('insurance_claims.csv')
df['text'] = df.apply(build_claim_narrative, axis=1)

print(f"Loading local DistilBERT model from {MODEL_DIR}...")
# The 'r' before the string ensures Windows paths are read correctly
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Scoring claims (Generating deep learning probabilities)...")
dl_scores = []

# Process in small batches so we don't run out of memory
for text in df['text'].tolist():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        # Get the probability of class 1 (Fraud)
        fraud_prob = probs[0][1].item()
        dl_scores.append(fraud_prob)

# Attach scores and clean up text column
df['distilbert_fraud_score'] = dl_scores
df = df.drop(columns=['text'])

# Convert target back to 1s and 0s for XGBoost compatibility
df['fraud_reported'] = df['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0)

print("Saving enriched dataset...")
df.to_csv('insurance_claims_enriched.csv', index=False)

print("✅ Pipeline complete! Your Streamlit app is ready to run.")