"""
================================================================================
AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship
Project: Customer Churn & Revenue Intelligence Platform
Domain: Business Intelligence, Machine Learning & Prescriptive Analytics
Student Name: Heramb Krishna Arora
Format: Complete All-in-One Project Code File
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Set visualization theme
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)

# ==============================================================================
# 1. DATASET SYNTHESIS & INGESTION (Level 1: Executive KPIs)
# ==============================================================================
DATA_PATH = "data/customer_churn_data.csv"
os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("[+] Synthesizing 3,000 subscriber records with multi-variable churn dynamics...")
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
    payments = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
        p=[0.35, 0.20, 0.25, 0.20],
    )

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

    df = pd.DataFrame({
        "CustomerID": c_ids, "Region": regions, "Gender": genders, "SeniorCitizen": seniors,
        "Partner": partners, "Dependents": dependents, "Tenure": tenures, "PhoneService": phones,
        "MultipleLines": multilines, "InternetService": internets, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": tech_support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contracts,
        "PaperlessBilling": paperless, "PaymentMethod": payments, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges, "SupportTickets": tickets, "Churn": churn
    })
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

print(f"\n=======================================================")
print(f"📊 LEVEL 1: EXECUTIVE KPIs OVERVIEW")
print(f"=======================================================")
churn_rate = (df["Churn"] == "Yes").mean() * 100
total_rev = df["TotalCharges"].sum()
annual_risk_rev = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum() * 12
print(f"Total Subscriber Accounts : {len(df):,}")
print(f"Overall Churn Rate        : {churn_rate:.2f}%")
print(f"Total Cumulative Revenue  : ${total_rev:,.2f}")
print(f"Annualized Revenue at Risk: ${annual_risk_rev:,.2f}")

# ==============================================================================
# 2. PREPROCESSING & FEATURE ENCODING
# ==============================================================================
df_clean = df.copy()
df_clean["Churn_Numeric"] = (df_clean["Churn"] == "Yes").astype(int)

feature_cols = [
    "Region", "Gender", "SeniorCitizen", "Partner", "Dependents",
    "Tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "SupportTickets"
]

X = df_clean[feature_cols]
y = df_clean["Churn_Numeric"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 3. MACHINE LEARNING BENCHMARKING (LEVEL 4: RISK MATRIX)
# ==============================================================================
print(f"\n=======================================================")
print(f"🤖 MACHINE LEARNING MODEL BENCHMARKS")
print(f"=======================================================")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
}

results = []
for name, m in models.items():
    m.fit(X_train_scaled, y_train)
    pred = m.predict(X_test_scaled)
    prob = m.predict_proba(X_test_scaled)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": f"{accuracy_score(y_test, pred)*100:.2f}%",
        "Precision": f"{precision_score(y_test, pred)*100:.2f}%",
        "Recall": f"{recall_score(y_test, pred)*100:.2f}%",
        "F1-Score": f"{f1_score(y_test, pred)*100:.2f}%",
        "ROC-AUC": round(roc_auc_score(y_test, prob), 4)
    })

print(pd.DataFrame(results).to_string(index=False))

# Champion Model Selection & Detailed Report
champion = models["Logistic Regression"]
champion_preds = champion.predict(X_test_scaled)
print("\n--- Champion Model Classification Report (Logistic Regression) ---")
print(classification_report(y_test, champion_preds, target_names=["Retained", "Churned"]))

# ==============================================================================
# 4. FEATURE IMPORTANCE & ROOT-CAUSE ATTRIBUTION (LEVEL 3: DRIVERS)
# ==============================================================================
print(f"=======================================================")
print(f"🔍 LEVEL 3: ROOT-CAUSE DRIVERS & ATTRIBUTION")
print(f"=======================================================")
driver_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Coefficient": champion.coef_[0],
    "AbsImpact": np.abs(champion.coef_[0])
}).sort_values(by="AbsImpact", ascending=False)

print("Top 10 Churn Drivers (Magnitude):")
print(driver_importance[["Feature", "Coefficient", "AbsImpact"]].head(10).to_string(index=False))

# ==============================================================================
# 5. REAL-TIME PREDICTIVE SCORING & PRESCRIPTIVE PLAYBOOK (LEVEL 5: ACTIONS)
# ==============================================================================
print(f"\n=======================================================")
print(f"🎯 LEVEL 5: REAL-TIME INFERENCE & PRESCRIPTIVE ACTIONS")
print(f"=======================================================")

sample_subscriber = pd.DataFrame([{
    "Region": "West",
    "Gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "Tenure": 4,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber Optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 92.40,
    "TotalCharges": 369.60,
    "SupportTickets": 3
}])

sample_enc = pd.get_dummies(sample_subscriber, columns=cat_cols, drop_first=True)
aligned_sample = pd.DataFrame(0, index=[0], columns=X_encoded.columns)
for col in sample_enc.columns:
    if col in aligned_sample.columns:
        aligned_sample[col] = sample_enc[col]

scaled_sample = scaler.transform(aligned_sample)
churn_prob = champion.predict_proba(scaled_sample)[0][1]

print(f"Sample Subscriber Churn Risk : {churn_prob*100:.2f}%")
if churn_prob >= 0.70:
    print("Risk Classification          : 🔴 HIGH RISK")
    print("Prescriptive Action Bundle   :")
    print(" 1. Lock-in: Offer 1-year contract extension with 15% promotional rate.")
    print(" 2. Tech Support: Add complimentary premium Tech Support for 90 days.")
    print(" 3. Customer Care: Assign proactive Senior Success Rep to resolve pending tickets.")
elif churn_prob >= 0.35:
    print("Risk Classification          : 🟡 MEDIUM RISK")
    print("Prescriptive Action Bundle   :")
    print(" 1. Run satisfaction survey regarding billing accuracy and network performance.")
    print(" 2. Offer loyalty milestone credit.")
else:
    print("Risk Classification          : 🟢 LOW RISK (Healthy Account)")

# ==============================================================================
# 6. WHAT-IF STRATEGY & FINANCIAL ROI SIMULATOR
# ==============================================================================
print(f"\n=======================================================")
print(f"💼 FINANCIAL ROI & WHAT-IF STRATEGY SIMULATOR")
print(f"=======================================================")
target_churners = 300
avg_annual_revenue = 1000
retention_rate = 0.40
cost_per_customer = 60

retained_customers = target_churners * retention_rate
gross_revenue_saved = retained_customers * avg_annual_revenue
campaign_cost = target_churners * cost_per_customer
net_roi = ((gross_revenue_saved - campaign_cost) / campaign_cost) * 100

print(f"Target At-Risk Customers Contacted : {target_churners:,}")
print(f"Successfully Retained Customers    : {int(retained_customers):,}")
print(f"Gross Revenue Protected            : ${gross_revenue_saved:,.2f}")
print(f"Total Intervention Campaign Cost   : ${campaign_cost:,.2f}")
print(f"Projected Net Campaign ROI         : {net_roi:,.1f}%")
print(f"=======================================================\n")
