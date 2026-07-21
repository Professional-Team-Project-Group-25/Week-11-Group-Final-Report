import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os

# Set working directory to ensure files save in the right place
os.chdir(r"D:\python setup")

print("Loading data...")
df = pd.read_csv('insurance_claims.csv')

# Drop non-informative columns
cols_to_drop = ['_c39', 'policy_number', 'incident_date', 'policy_bind_date']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Separate features and target
target_col = 'fraud_reported'
X = df.drop(columns=[target_col])
y = df[target_col].apply(lambda x: 1 if x == 'Y' else 0)

# Encode categorical variables and save encoders
print("Encoding features...")
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Save everything for the Streamlit dashboard
print("Saving tabular models...")
joblib.dump(rf_model, 'rf_model_2.pkl')
joblib.dump(xgb_model, 'xgb_model_2.pkl')
joblib.dump(label_encoders, 'encoders_2.pkl')

print("✅ Tabular setup complete! Proceed to step 2.")