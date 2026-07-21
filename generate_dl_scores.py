import pandas as pd

# 1. Load the preprocessed 1,000-row dataset
df = pd.read_csv('insurance_claims.csv')

# 2. Define the narrative template
def build_claim_narrative(row):
    """
    Translates a single row of tabular claims data into a natural language string.
    Note: Adjust the column names below to match your exact dataframe structure.
    """
    narrative = (
        f"A {row['age']}-year-old customer from {row['policy_state']} reported a "
        f"{row['incident_severity']} {row['incident_type']} incident. "
        f"The authorities were {row['police_report_available']} to the scene. "
        f"The total requested claim amount is ${row['total_claim_amount']}, "
        f"which includes a vehicle damage claim of ${row['vehicle_claim']}."
    )
    return narrative

# 3. Apply the function to create a new text feature
df['claim_narrative'] = df.apply(build_claim_narrative, axis=1)

# Preview the first narrative to ensure the conversion worked
print(df['claim_narrative'].iloc[0])

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# 1. Load your fine-tuned Hugging Face model and tokenizer
# Point this path to wherever your Trainer class saved the final weights
# Use an absolute path to be completely safe
model_path = "D:/python setup/distilbert_claims_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set the model to evaluation mode (disables dropout layers)
model.eval()

# 2. Define the scoring function
def extract_fraud_probability(text):
    # Tokenize the narrative text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    # Run the forward pass without tracking gradients (saves memory)
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract the raw logits for the classes
        logits = outputs.logits.numpy()[0]
        
    # Apply softmax to convert logits to probability percentages
    probabilities = softmax(logits)
    
    # Assuming Class 0 is 'Genuine' and Class 1 is 'Fraud'
    fraud_prob = probabilities[1] 
    return fraud_prob

# 3. Generate the deep learning scores
# For 1,000 rows, a standard pandas apply is efficient enough. 
df['distilbert_fraud_score'] = df['claim_narrative'].apply(extract_fraud_probability)

# 4. Save the enriched dataset for the XGBoost phase
df.drop(columns=['claim_narrative'], inplace=True) # Drop text to keep it purely tabular
df.to_csv('insurance_claims_enriched.csv', index=False)

print("DistilBERT scoring complete. Dataset ready for XGBoost.")

# Save the fine-tuned model and tokenizer to a specific folder
trainer.save_model("D:/python setup/distilbert_claims_model")
tokenizer.save_pretrained("D:/python setup/distilbert_claims_model")