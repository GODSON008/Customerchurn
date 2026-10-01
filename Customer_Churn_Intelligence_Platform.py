"""
================================================================================
AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship
Project: Customer Churn & Revenue Intelligence Platform
Domain: Business Intelligence, Machine Learning & Prescriptive Analytics
Author / Submitter: BharatCares Internship Submission
================================================================================
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List

# ==============================================================================
# SECTION 1: DATA PIPELINE & MODEL ENGINE
# ==============================================================================
DATA_FILE = "data/customer_churn_data.csv"
MODEL_FILE = "models/churn_model.joblib"

def ensure_dataset_and_model():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("static/images", exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print("[+] Generating customer subscription dataset...")
        np.random.seed(42)
        n = 3000
        
        c_ids = [f"CUST-{10000 + i}" for i in range(n)]
        regions = np.random.choice(["West", "East", "North", "South", "Central"], size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
        genders = np.random.choice(["Male", "Female"], size=n)
        seniors = np.random.choice([0, 1], size=n, p=[0.82, 0.18])
        partners = np.random.choice(["Yes", "No"], size=n, p=[0.52, 0.48])
        dependents = np.random.choice(["Yes", "No"], size=n, p=[0.33, 0.67])
        tenures = np.random.randint(1, 73, size=n)
        phones = np.random.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
        multilines = np.where(phones == "Yes", np.random.choice(["Yes", "No"], size=n, p=[0.45, 0.55]), "No phone service")
        internets = np.random.choice(["Fiber Optic", "DSL", "No"], size=n, p=[0.45, 0.35, 0.20])
        
        security = np.where(internets != "No", np.random.choice(["Yes", "No"], size=n, p=[0.38, 0.62]), "No internet service")
        backup = np.where(internets != "No", np.random.choice(["Yes", "No"], size=n, p=[0.42, 0.58]), "No internet service")
        protection = np.where(internets != "No", np.random.choice(["Yes", "No"], size=n, p=[0.40, 0.60]), "No internet service")
        tech_support = np.where(internets != "No", np.random.choice(["Yes", "No"], size=n, p=[0.35, 0.65]), "No internet service")
        tv = np.where(internets != "No", np.random.choice(["Yes", "No"], size=n, p=[0.48, 0.52]), "No internet service")
        movies = np.where(internets != "No", np.random.choice(["Yes", "No"], size=n, p=[0.50, 0.50]), "No internet service")
        
        contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.23, 0.22])
        paperless = np.random.choice(["Yes", "No"], size=n, p=[0.60, 0.40])
        payments = np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], size=n, p=[0.35, 0.20, 0.25, 0.20])
        
        base_charges = np.zeros(n)
        base_charges += np.where(phones == "Yes", 20.0, 0.0)
        base_charges += np.where(multilines == "Yes", 10.0, 0.0)
        base_charges += np.where(internets == "DSL", 30.0, 0.0)
        base_charges += np.where(internets == "Fiber Optic", 50.0, 0.0)
        base_charges += np.where(tech_support == "Yes", 6.0, 0.0)
        base_charges += np.where(tv == "Yes", 8.0, 0.0)
        monthly_charges = np.round(np.clip(base_charges + np.random.normal(5.0, 3.0, size=n), 18.5, 125.0), 2)
        total_charges = np.round(np.clip(monthly_charges * tenures + np.random.normal(0, 15.0, size=n), 18.5, None), 2)
        tickets = np.random.poisson(lam=1.5, size=n)
        
        churn_score = (
            - 0.05 * tenures
            + 1.8 * (contracts == "Month-to-month")
            - 1.2 * (contracts == "Two year")
            + 0.9 * (internets == "Fiber Optic")
            - 0.8 * (tech_support == "Yes")
            + 0.6 * (payments == "Electronic check")
            + 0.02 * monthly_charges
            + 0.45 * tickets
            + 0.5 * (regions == "West")
            + 0.3 * seniors
            - 0.4 * (partners == "Yes")
        )
        prob = 1 / (1 + np.exp(-(churn_score - 1.6)))
        churn = np.where(np.random.rand(n) < prob, "Yes", "No")
        
        df_gen = pd.DataFrame({
            "CustomerID": c_ids, "Region": regions, "Gender": genders, "SeniorCitizen": seniors,
            "Partner": partners, "Dependents": dependents, "Tenure": tenures, "PhoneService": phones,
            "MultipleLines": multilines, "InternetService": internets, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": tech_support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contracts,
            "PaperlessBilling": paperless, "PaymentMethod": payments, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges, "SupportTickets": tickets, "Churn": churn
        })
        df_gen.to_csv(DATA_FILE, index=False)

    if not os.path.exists(MODEL_FILE):
        print("[+] Training and serializing ML models...")
        df_train = pd.read_csv(DATA_FILE)
        df_train['Churn_Numeric'] = (df_train['Churn'] == 'Yes').astype(int)
        
        feature_cols = [
            'Region', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'Tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'SupportTickets'
        ]
        
        X = df_train[feature_cols]
        y = df_train['Churn_Numeric']
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        encoded_cols = X_encoded.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
        }
        
        results = {}
        best_name = "Gradient Boosting"
        best_model = None
        best_auc = 0.0
        
        for name, m in models.items():
            m.fit(X_train_s, y_train)
            pred = m.predict(X_test_s)
            probs = m.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, probs)
            results[name] = {
                "Accuracy": round(accuracy_score(y_test, pred), 4),
                "Precision": round(precision_score(y_test, pred), 4),
                "Recall": round(recall_score(y_test, pred), 4),
                "F1_Score": round(f1_score(y_test, pred), 4),
                "ROC_AUC": round(auc, 4)
            }
            if auc > best_auc:
                best_auc = auc
                best_name = name
                best_model = m
                
        importances = best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else np.abs(best_model.coef_[0])
        feat_imp = pd.DataFrame({'Feature': encoded_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        
        art = {
            "model": best_model,
            "model_name": best_name,
            "scaler": scaler,
            "encoded_columns": encoded_cols,
            "original_features": feature_cols,
            "categorical_columns": cat_cols,
            "results": results,
            "top_features": feat_imp.head(10).to_dict(orient='records')
        }
        joblib.dump(art, MODEL_FILE)

ensure_dataset_and_model()

# ==============================================================================
# SECTION 2: FASTAPI BACKEND SERVER & REST API ENDPOINTS
# ==============================================================================
artifact = joblib.load(MODEL_FILE)
model = artifact["model"]
scaler = artifact["scaler"]
encoded_columns = artifact["encoded_columns"]
model_results = artifact["results"]
top_features = artifact["top_features"]
df = pd.read_csv(DATA_FILE)

app = FastAPI(
    title="Customer Churn & Revenue Intelligence Platform",
    description="Executive Decision Intelligence Suite",
    version="1.0.0"
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

class CustomerInput(BaseModel):
    Region: str = "West"
    Gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    Tenure: int = 6
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber Optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 85.50
    TotalCharges: float = 513.00
    SupportTickets: int = 3

class SimulationInput(BaseModel):
    contract_upgrade_discount: float = 10.0
    free_tech_support_bundle: bool = True
    loyalty_retention_discount: float = 5.0

@app.get("/", response_class=HTMLResponse)
async def home():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Customer Churn & Revenue Intelligence API Running. Visit /docs for OpenAPI specifications.</h1>"

@app.get("/api/kpis")
async def get_kpis():
    total = int(len(df))
    churn_cnt = int((df['Churn'] == 'Yes').sum())
    retained_cnt = int(total - churn_cnt)
    churn_rate = float(round((churn_cnt / total) * 100, 2))
    tot_rev = float(round(df['TotalCharges'].sum(), 2))
    risk_rev = float(round(df[df['Churn'] == 'Yes']['MonthlyCharges'].sum() * 12, 2))
    savings = float(round(risk_rev * 0.35, 2))
    return {
        "total_customers": total,
        "active_customers": retained_cnt,
        "churned_customers": churn_cnt,
        "churn_rate_pct": churn_rate,
        "total_revenue": tot_rev,
        "annual_revenue_at_risk": risk_rev,
        "projected_retention_savings": savings
    }

@app.post("/api/predict")
async def predict(cust: CustomerInput):
    inp_df = pd.DataFrame([cust.dict()])
    cat_cols = artifact["categorical_columns"]
    encoded = pd.get_dummies(inp_df, columns=cat_cols, drop_first=True)
    final_df = pd.DataFrame(0, index=[0], columns=encoded_columns)
    for col in encoded.columns:
        if col in final_df.columns:
            final_df[col] = encoded[col]
            
    scaled = scaler.transform(final_df)
    prob = float(model.predict_proba(scaled)[0][1])
    risk_level = "Critical" if prob >= 0.70 else "High" if prob >= 0.50 else "Moderate" if prob >= 0.30 else "Low"
    
    actions = []
    if cust.Contract == "Month-to-month":
        actions.append("Offer 1-Year Contract Upgrade with 10% Loyalty Discount.")
    if cust.TechSupport == "No":
        actions.append("Attach Free 3-Month Priority 24/7 Tech Support.")
    if cust.SupportTickets >= 2:
        actions.append("Escalate account to Executive Customer Success Manager.")
    if not actions:
        actions.append("Customer healthy. Trigger loyalty rewards and cross-sell.")
        
    return {
        "churn_probability": round(prob, 4),
        "risk_level": risk_level,
        "risk_score_pct": round(prob * 100, 1),
        "estimated_annual_loss": round(cust.MonthlyCharges * 12, 2),
        "prescriptive_actions": actions
    }

if __name__ == "__main__":
    print("\n========================================================")
    print("  Customer Churn & Revenue Intelligence Platform")
    print("  AICTE | IBM SkillsBuild Internship Project")
    print("  Server launching on: http://localhost:8000")
    print("  API Docs available on: http://localhost:8000/docs")
    print("========================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
