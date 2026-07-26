import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

# 1. Page Configuration
st.set_page_config(page_title="Fraud Detection Pipeline", layout="wide")

# 2. Sidebar Navigation
st.sidebar.title("Project Experiments")
experiment = st.sidebar.radio(
    "Select Phase to Analyze:",
    (
        "Experiment 1: Fraud Oracle", 
        "Experiment 2: Insurance Claims",
        "Experiment 3: Hybrid Model (DistilBERT + XGBoost)"
    )
)

# ==========================================
# EXPERIMENTS 1 & 2: TABULAR COMPARISONS
# ==========================================
if experiment in ["Experiment 1: Fraud Oracle", "Experiment 2: Insurance Claims"]:
    
    st.title(f"Tabular Baselines: {experiment.split(':')[1]}")
    st.write("Compare standard machine learning architectures without NLP text enrichment.")

    # 3. Dynamic Model Loading
    @st.cache_resource
    def load_resources(exp_choice):
        if exp_choice == "Experiment 1: Fraud Oracle":
            rf = joblib.load('random_forest.pkl')
            xgb_model = joblib.load('xgboost_model.pkl')
            le_dict = joblib.load('label_encoders.pkl')
            target_col = 'FraudFound_P'
        else:
            rf = joblib.load('rf_model_2.pkl')
            xgb_model = joblib.load('xgb_model_2.pkl')
            le_dict = joblib.load('encoders_2.pkl')
            target_col = 'fraud_reported'
            
        return rf, xgb_model, le_dict, target_col

    try:
        rf_model, xgb_model, label_encoders, target_col = load_resources(experiment)
    except FileNotFoundError:
        st.error(f"Error: The .pkl files for {experiment} were not found in your directory.")
        st.stop()

    # 4. Data Loading Options (Bug Fixed with unique keys)
    st.markdown("---")
    
    use_local = st.checkbox(
        f"Use local {experiment.split(':')[1]} dataset automatically", 
        key=f"check_{experiment}" # Unique key prevents state bleed
    )
    
    uploaded_file = st.file_uploader(
        f"Or upload a custom CSV file", 
        type=['csv'], 
        key=f"upload_{experiment}" # Unique key prevents state bleed
    )

    df = None
    if use_local:
        try:
            file_path = 'insurance_claims.csv' if "Insurance Claims" in experiment else 'fraud_oracle.csv'
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            st.error(f"⚠️ Could not find '{file_path}'. Please check your folder.")
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    if df is not None:
        display_df = df.copy()
        
        # 1. Drop known identifiers and dates
        cols_to_drop = [target_col, '_c39', 'incident_date', 'policy_bind_date', 'policy_number']
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        # 2. Preprocess Data
        for col in list(X.columns):
            if col in label_encoders:
                le = label_encoders[col]
                X[col] = X[col].fillna('Unknown').astype(str)
                known_classes = set(le.classes_)
                X[col] = X[col].apply(lambda x: x if x in known_classes else le.classes_[0])
                X[col] = le.transform(X[col])
            elif X[col].dtype == 'object':
                X = X.drop(col, axis=1)
                
        X = X.fillna(0)
        
        # 3. Bulletproof Feature Alignment
        if hasattr(rf_model, 'feature_names_in_'):
            X = X[rf_model.feature_names_in_]
        
        # 4. Make Predictions
        try:
            rf_preds = rf_model.predict(X)
            xgb_preds = xgb_model.predict(X)
            
            results_df = display_df.copy()
            results_df['RF_Prediction'] = ["Fraud" if p == 1 else "Genuine" for p in rf_preds]
            results_df['XGB_Prediction'] = ["Fraud" if p == 1 else "Genuine" for p in xgb_preds]
            results_df['Models_Match'] = results_df['RF_Prediction'] == results_df['XGB_Prediction']
            
            st.subheader("Analysis Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Claims Analyzed", len(results_df))
            col2.metric("RF Fraud Detections", sum(rf_preds))
            col3.metric("XGB Fraud Detections", sum(xgb_preds))
            
            st.dataframe(results_df[['RF_Prediction', 'XGB_Prediction', 'Models_Match'] + [c for c in display_df.columns]])
            
        except ValueError as e:
            st.error(f"Prediction Error: Feature mismatch. Details: {e}")

# ==========================================
# EXPERIMENT 3: HYBRID ARCHITECTURE (MSc Level)
# ==========================================
else:
    st.title("Experiment 3: Hybrid AI Architecture")
    st.write("This phase demonstrates the dissertation's core hypothesis: enriching tabular gradient boosting models with Deep Learning (DistilBERT) textual embeddings yields superior fraud detection.")
    st.markdown("---")
    
    @st.cache_data
    def load_hybrid_data():
        try:
            return pd.read_csv('insurance_claims_enriched.csv')
        except FileNotFoundError:
            return None
            
    df_hybrid = load_hybrid_data()
    
    if df_hybrid is None:
        st.info("ℹ️ The Hybrid Architecture requires the 'insurance_claims_enriched.csv' file generated by the DistilBERT pipeline (Script 2). Please run that script first.")
    else:
        # Prepare Data
        target_col = 'fraud_reported'
        X = df_hybrid.drop(columns=[target_col, '_c39'], errors='ignore')
        y = df_hybrid[target_col]
        
        X = X.fillna(0)
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X, y, df_hybrid.index, test_size=0.2, random_state=42
        )
        
        # Train Meta-Learner
        with st.spinner("Compiling Hybrid Meta-Learner..."):
            hybrid_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='logloss')
            hybrid_model.fit(X_train, y_train)
            
        # Get Predictions AND Probabilities
        y_pred = hybrid_model.predict(X_test)
        y_prob = hybrid_model.predict_proba(X_test)[:, 1] # Probability of Fraud
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        
        # Build Interactive UI
        tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🧠 Explainable AI (XAI)", "🚨 Actionable Insights"])
        
        with tab1:
            st.subheader("Global Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Accuracy", f"{acc * 100:.1f}%", "Hybrid Edge")
            col2.metric("Precision (Quality)", f"{prec * 100:.1f}%")
            col3.metric("Recall (Capture Rate)", f"{rec * 100:.1f}%")
            
            st.text("Detailed Classification Report:")
            st.code(classification_report(y_test, y_pred))

        with tab2:
            st.subheader("Feature Importance & NLP Value")
            st.write("This chart isolates the features that drive the Hybrid Model's decision-making. Look for the DistilBERT probability/score feature to prove the value of textual analysis.")
            
            importance = hybrid_model.feature_importances_
            feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(12)
            
            # Display sorted bar chart
            st.bar_chart(feat_imp_df.set_index('Feature'))

        with tab3:
            st.subheader("Claims Triage & Risk Queue")
            st.write("By extracting the **probability scores** from the XGBoost meta-learner, we can triage claims from lowest to highest risk, providing actionable intelligence rather than just binary labels.")
            
            # Create a nice output dataframe
            results_df = df_hybrid.loc[indices_test].copy()
            results_df['Fraud_Probability'] = np.round(y_prob * 100, 1)
            
            # Categorize Risk
            conditions = [
                (results_df['Fraud_Probability'] >= 75),
                (results_df['Fraud_Probability'] >= 40) & (results_df['Fraud_Probability'] < 75),
                (results_df['Fraud_Probability'] < 40)
            ]
            choices = ['🔴 High Risk', '🟡 Medium Risk', '🟢 Low Risk']
            results_df['Risk_Tier'] = np.select(conditions, choices, default='Unknown')
            
            # Interactive Filter
            selected_tier = st.selectbox(
                "Filter by Risk Tier:", 
                ["View All", "🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"]
            )
            
            # Apply the filter
            if selected_tier == "View All":
                display_df = results_df.sort_values(by='Fraud_Probability', ascending=False)
            else:
                display_df = results_df[results_df['Risk_Tier'] == selected_tier].sort_values(by='Fraud_Probability', ascending=False)
            
            # Show how many claims match the filter
            st.metric(f"Total Claims ({selected_tier})", len(display_df))
            
            # Reorder columns to show the cool stuff first
            cols_to_show = ['Risk_Tier', 'Fraud_Probability', target_col] + [c for c in display_df.columns if c not in ['Risk_Tier', 'Fraud_Probability', target_col, '_c39']]
            st.dataframe(display_df[cols_to_show])
