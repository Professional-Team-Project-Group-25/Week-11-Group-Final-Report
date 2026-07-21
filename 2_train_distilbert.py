import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

os.chdir(r"D:\python setup")
MODEL_DIR = r"D:\python setup\distilbert_claims_model"

def build_claim_narrative(row):
    return f"A {row['age']}-year-old customer from {row['policy_state']} reported a {row['incident_severity']} {row['incident_type']} incident. The authorities were {row['police_report_available']} to the scene. The total requested claim amount is ${row['total_claim_amount']}, which includes a vehicle damage claim of ${row['vehicle_claim']}."

print("Preparing text data...")
df = pd.read_csv('insurance_claims.csv')
df['text'] = df.apply(build_claim_narrative, axis=1)
df['label'] = df['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./hf_results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_strategy="no", # We handle saving manually at the end
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

print("Training DistilBERT (This may take a few minutes)...")
trainer.train()

print(f"Saving model directly to {MODEL_DIR}...")
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("✅ DistilBERT trained and saved successfully! Proceed to step 3.")