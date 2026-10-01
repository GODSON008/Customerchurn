import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

def train_and_evaluate():
    os.makedirs("models", exist_ok=True)
    os.makedirs("static/images", exist_ok=True)
    
    # 1. Load Dataset
    data_path = "data/customer_churn_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found!")
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Preprocessing & Feature Engineering
    df_clean = df.copy()
    
    # Encode target
    df_clean['Churn_Numeric'] = (df_clean['Churn'] == 'Yes').astype(int)
    
    # Features selection
    feature_cols = [
        'Region', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'Tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'SupportTickets'
    ]
    
    X = df_clean[feature_cols]
    y = df_clean['Churn_Numeric']
    
    # One-Hot Encoding for categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    encoded_feature_names = X_encoded.columns.tolist()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Model Training & Comparison
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    results = {}
    best_model_name = "Gradient Boosting"
    best_model = None
    best_auc = 0.0
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4)
        }
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = model
            
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    # 4. Generate Visualizations for Report and Dashboard
    sns.set_theme(style="whitegrid")
    
    # Confusion Matrix
    y_pred_best = best_model.predict(X_test_scaled)
    y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retained (0)', 'Churned (1)'],
                yticklabels=['Retained (0)', 'Churned (1)'])
    plt.title(f"Confusion Matrix ({best_model_name})", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("static/images/confusion_matrix.png", dpi=300)
    plt.close()
    
    # ROC Curve Comparison
    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        probs = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['ROC_AUC']:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Baseline")
    plt.xlabel("False Positive Rate", fontweight='bold')
    plt.ylabel("True Positive Rate", fontweight='bold')
    plt.title("Model ROC Curves Comparison", fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("static/images/roc_curve.png", dpi=300)
    plt.close()
    
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])
        
    feat_imp = pd.DataFrame({
        'Feature': encoded_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=feat_imp.head(10), x='Importance', y='Feature', palette='crest')
    plt.title("Top 10 Churn Risk Drivers (Feature Importance)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Relative Importance Score", fontweight='bold')
    plt.tight_layout()
    plt.savefig("static/images/feature_importance.png", dpi=300)
    plt.close()
    
    # 5. Save Artifacts
    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "scaler": scaler,
        "encoded_columns": encoded_feature_names,
        "original_features": feature_cols,
        "categorical_columns": categorical_cols,
        "results": results,
        "top_features": feat_imp.head(10).to_dict(orient='records')
    }
    
    joblib.dump(artifact, "models/churn_model.joblib")
    print("\nSaved artifacts to models/churn_model.joblib and static/images/*.png successfully!")
    return artifact

if __name__ == "__main__":
    train_and_evaluate()
