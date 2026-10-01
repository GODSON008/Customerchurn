import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 📊 Customer Churn & Revenue Intelligence Platform\n",
    "### **Executive Decision Intelligence Suite & Prescriptive ML Engine**\n",
    "> **AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship**  \n",
    "> *Conducted by BharatCares in association with IBM SkillsBuild*\n",
    "\n",
    "**Student Name:** Heramb Krishna Arora  \n",
    "**Submission Deliverable:** Complete All-in-One Project Code (`YourName_ProjectName.ipynb` / `.py`)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 1. Import Essential Libraries & Configure Environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve\n",
    "\n",
    "# Visualization Styling\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams[\"figure.figsize\"] = (9, 5)\n",
    "plt.rcParams[\"font.size\"] = 10\n",
    "print(\"✅ Libraries loaded successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 2. Dataset Synthesis & Executive Ingestion (Level 1: KPIs)\n",
    "Generates realistic customer telemetry with multi-factor churn dynamics if not already present on disk."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_path = 'data/customer_churn_data.csv'\n",
    "os.makedirs('data', exist_ok=True)\n",
    "\n",
    "if not os.path.exists(data_path):\n",
    "    print(\"Generating realistic subscriber dataset...\")\n",
    "    np.random.seed(42)\n",
    "    n = 3000\n",
    "    c_ids = [f\"CUST-{10000 + i}\" for i in range(n)]\n",
    "    regions = np.random.choice([\"West\", \"East\", \"North\", \"South\", \"Central\"], size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])\n",
    "    genders = np.random.choice([\"Male\", \"Female\"], size=n)\n",
    "    seniors = np.random.choice([0, 1], size=n, p=[0.82, 0.18])\n",
    "    partners = np.random.choice([\"Yes\", \"No\"], size=n, p=[0.52, 0.48])\n",
    "    dependents = np.random.choice([\"Yes\", \"No\"], size=n, p=[0.33, 0.67])\n",
    "    tenures = np.random.randint(1, 73, size=n)\n",
    "    phones = np.random.choice([\"Yes\", \"No\"], size=n, p=[0.90, 0.10])\n",
    "    multilines = np.where(phones == \"Yes\", np.random.choice([\"Yes\", \"No\"], size=n, p=[0.45, 0.55]), \"No phone service\")\n",
    "    internets = np.random.choice([\"Fiber Optic\", \"DSL\", \"No\"], size=n, p=[0.45, 0.35, 0.20])\n",
    "    security = np.where(internets != \"No\", np.random.choice([\"Yes\", \"No\"], size=n, p=[0.38, 0.62]), \"No internet service\")\n",
    "    backup = np.where(internets != \"No\", np.random.choice([\"Yes\", \"No\"], size=n, p=[0.42, 0.58]), \"No internet service\")\n",
    "    protection = np.where(internets != \"No\", np.random.choice([\"Yes\", \"No\"], size=n, p=[0.40, 0.60]), \"No internet service\")\n",
    "    tech_support = np.where(internets != \"No\", np.random.choice([\"Yes\", \"No\"], size=n, p=[0.35, 0.65]), \"No internet service\")\n",
    "    tv = np.where(internets != \"No\", np.random.choice([\"Yes\", \"No\"], size=n, p=[0.48, 0.52]), \"No internet service\")\n",
    "    movies = np.where(internets != \"No\", np.random.choice([\"Yes\", \"No\"], size=n, p=[0.50, 0.50]), \"No internet service\")\n",
    "    contracts = np.random.choice([\"Month-to-month\", \"One year\", \"Two year\"], size=n, p=[0.55, 0.23, 0.22])\n",
    "    paperless = np.random.choice([\"Yes\", \"No\"], size=n, p=[0.60, 0.40])\n",
    "    payments = np.random.choice([\"Electronic check\", \"Mailed check\", \"Bank transfer (automatic)\", \"Credit card (automatic)\"], size=n, p=[0.35, 0.20, 0.25, 0.20])\n",
    "    \n",
    "    base_charges = np.zeros(n)\n",
    "    base_charges += np.where(phones == \"Yes\", 20.0, 0.0)\n",
    "    base_charges += np.where(multilines == \"Yes\", 10.0, 0.0)\n",
    "    base_charges += np.where(internets == \"DSL\", 30.0, 0.0)\n",
    "    base_charges += np.where(internets == \"Fiber Optic\", 50.0, 0.0)\n",
    "    base_charges += np.where(tech_support == \"Yes\", 6.0, 0.0)\n",
    "    base_charges += np.where(tv == \"Yes\", 8.0, 0.0)\n",
    "    monthly_charges = np.round(np.clip(base_charges + np.random.normal(5.0, 3.0, size=n), 18.5, 125.0), 2)\n",
    "    total_charges = np.round(np.clip(monthly_charges * tenures + np.random.normal(0, 15.0, size=n), 18.5, None), 2)\n",
    "    tickets = np.random.poisson(lam=1.5, size=n)\n",
    "    \n",
    "    churn_score = (\n",
    "        - 0.05 * tenures\n",
    "        + 1.8 * (contracts == \"Month-to-month\")\n",
    "        - 1.2 * (contracts == \"Two year\")\n",
    "        + 0.9 * (internets == \"Fiber Optic\")\n",
    "        - 0.8 * (tech_support == \"Yes\")\n",
    "        + 0.6 * (payments == \"Electronic check\")\n",
    "        + 0.02 * monthly_charges\n",
    "        + 0.45 * tickets\n",
    "        + 0.5 * (regions == \"West\")\n",
    "        + 0.3 * seniors\n",
    "        - 0.4 * (partners == \"Yes\")\n",
    "    )\n",
    "    prob = 1 / (1 + np.exp(-(churn_score - 1.6)))\n",
    "    churn = np.where(np.random.rand(n) < prob, \"Yes\", \"No\")\n",
    "    \n",
    "    df = pd.DataFrame({\n",
    "        \"CustomerID\": c_ids, \"Region\": regions, \"Gender\": genders, \"SeniorCitizen\": seniors,\n",
    "        \"Partner\": partners, \"Dependents\": dependents, \"Tenure\": tenures, \"PhoneService\": phones,\n",
    "        \"MultipleLines\": multilines, \"InternetService\": internets, \"OnlineSecurity\": security,\n",
    "        \"OnlineBackup\": backup, \"DeviceProtection\": protection, \"TechSupport\": tech_support,\n",
    "        \"StreamingTV\": tv, \"StreamingMovies\": movies, \"Contract\": contracts,\n",
    "        \"PaperlessBilling\": paperless, \"PaymentMethod\": payments, \"MonthlyCharges\": monthly_charges,\n",
    "        \"TotalCharges\": total_charges, \"SupportTickets\": tickets, \"Churn\": churn\n",
    "    })\n",
    "    df.to_csv(data_path, index=False)\n",
    "else:\n",
    "    df = pd.read_csv(data_path)\n",
    "\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "churn_rate = (df['Churn'] == 'Yes').mean() * 100\n",
    "total_rev = df['TotalCharges'].sum()\n",
    "risk_rev = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum() * 12\n",
    "\n",
    "print(f\"📊 Executive Overview:\")\n",
    "print(f\" - Total Subscribers: {len(df):,}\")\n",
    "print(f\" - Churn Rate: {churn_rate:.2f}%\")\n",
    "print(f\" - Cumulative Revenue: ${total_rev:,.2f}\")\n",
    "print(f\" - Annual Revenue at Risk: ${risk_rev:,.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 3. Exploratory Data Analysis & Churn Drivers (Levels 2 & 3: Trends & Drivers)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Churn by Contract Type\n",
    "sns.countplot(data=df, x='Contract', hue='Churn', palette=['#3b82f6', '#ef4444'], ax=axes[0])\n",
    "axes[0].set_title('Churn Distribution by Contract Type', fontsize=12, fontweight='bold')\n",
    "axes[0].set_ylabel('Subscriber Count')\n",
    "\n",
    "# Churn by Internet Service\n",
    "sns.countplot(data=df, x='InternetService', hue='Churn', palette=['#3b82f6', '#ef4444'], ax=axes[1])\n",
    "axes[1].set_title('Churn Distribution by Internet Service', fontsize=12, fontweight='bold')\n",
    "axes[1].set_ylabel('Subscriber Count')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Support Tickets vs Monthly Charges Impact on Churn\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.boxplot(data=df, x='SupportTickets', y='MonthlyCharges', hue='Churn', palette=['#3b82f6', '#ef4444'])\n",
    "plt.title('Monthly Charges & Support Tickets vs. Churn Status', fontsize=12, fontweight='bold')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 4. Feature Engineering & Stratified Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_clean = df.copy()\n",
    "df_clean['Churn_Numeric'] = (df_clean['Churn'] == 'Yes').astype(int)\n",
    "\n",
    "feature_cols = [\n",
    "    'Region', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents',\n",
    "    'Tenure', 'PhoneService', 'MultipleLines', 'InternetService',\n",
    "    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',\n",
    "    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',\n",
    "    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'SupportTickets'\n",
    "]\n",
    "\n",
    "X = df_clean[feature_cols]\n",
    "y = df_clean['Churn_Numeric']\n",
    "\n",
    "cat_cols = X.select_dtypes(include=['object']).columns.tolist()\n",
    "X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_encoded, y, test_size=0.20, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print(f\"Training set shape: {X_train.shape} | Test set shape: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 5. Multi-Model ML Benchmarking & Evaluation (Level 4: Risk Matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    \"Logistic Regression\": LogisticRegression(max_iter=1000, random_state=42),\n",
    "    \"Random Forest\": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),\n",
    "    \"Gradient Boosting\": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)\n",
    "}\n",
    "\n",
    "results_table = []\n",
    "for name, m in models.items():\n",
    "    m.fit(X_train_scaled, y_train)\n",
    "    pred = m.predict(X_test_scaled)\n",
    "    prob = m.predict_proba(X_test_scaled)[:, 1]\n",
    "    \n",
    "    results_table.append({\n",
    "        \"Model\": name,\n",
    "        \"Accuracy\": f\"{accuracy_score(y_test, pred)*100:.2f}%\",\n",
    "        \"Precision\": f\"{precision_score(y_test, pred)*100:.2f}%\",\n",
    "        \"Recall\": f\"{recall_score(y_test, pred)*100:.2f}%\",\n",
    "        \"F1-Score\": f\"{f1_score(y_test, pred)*100:.2f}%\",\n",
    "        \"ROC-AUC\": round(roc_auc_score(y_test, prob), 4)\n",
    "    })\n",
    "\n",
    "metrics_df = pd.DataFrame(results_table)\n",
    "print(metrics_df.to_string(index=False))\n",
    "\n",
    "# Confusion Matrix for Champion Model\n",
    "champion = models[\"Logistic Regression\"]\n",
    "y_pred = champion.predict(X_test_scaled)\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "plt.figure(figsize=(6, 4.5))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])\n",
    "plt.title('Champion Model Confusion Matrix', fontsize=12, fontweight='bold')\n",
    "plt.ylabel('Actual Label')\n",
    "plt.xlabel('Predicted Label')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 6. Feature Importance & Root-Cause Attribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "importance = pd.DataFrame({\n",
    "    'Feature': X_encoded.columns,\n",
    "    'Importance': np.abs(champion.coef_[0])\n",
    "}).sort_values(by='Importance', ascending=False).head(10)\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.barplot(data=importance, x='Importance', y='Feature', palette='Blues_r')\n",
    "plt.title('Top 10 Churn Drivers (Standardized Absolute Coefficients)', fontsize=12, fontweight='bold')\n",
    "plt.xlabel('Impact Magnitude')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 7. Real-Time Inference & Prescriptive Retention Actions (Level 5: Actions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_subscriber = pd.DataFrame([{\n",
    "    'Region': 'West',\n",
    "    'Gender': 'Female',\n",
    "    'SeniorCitizen': 0,\n",
    "    'Partner': 'No',\n",
    "    'Dependents': 'No',\n",
    "    'Tenure': 5,\n",
    "    'PhoneService': 'Yes',\n",
    "    'MultipleLines': 'No',\n",
    "    'InternetService': 'Fiber Optic',\n",
    "    'OnlineSecurity': 'No',\n",
    "    'OnlineBackup': 'No',\n",
    "    'DeviceProtection': 'No',\n",
    "    'TechSupport': 'No',\n",
    "    'StreamingTV': 'Yes',\n",
    "    'StreamingMovies': 'Yes',\n",
    "    'Contract': 'Month-to-month',\n",
    "    'PaperlessBilling': 'Yes',\n",
    "    'PaymentMethod': 'Electronic check',\n",
    "    'MonthlyCharges': 89.90,\n",
    "    'TotalCharges': 449.50,\n",
    "    'SupportTickets': 3\n",
    "}])\n",
    "\n",
    "sample_enc = pd.get_dummies(sample_subscriber, columns=cat_cols, drop_first=True)\n",
    "aligned_sample = pd.DataFrame(0, index=[0], columns=X_encoded.columns)\n",
    "for col in sample_enc.columns:\n",
    "    if col in aligned_sample.columns:\n",
    "        aligned_sample[col] = sample_enc[col]\n",
    "\n",
    "scaled_sample = scaler.transform(aligned_sample)\n",
    "risk_probability = champion.predict_proba(scaled_sample)[0][1]\n",
    "\n",
    "print(f\"🎯 Individual Subscriber Churn Risk: {risk_probability * 100:.2f}%\")\n",
    "risk_band = 'High' if risk_probability >= 0.70 else 'Medium' if risk_probability >= 0.35 else 'Low'\n",
    "print(f\"🚦 Risk Classification: {risk_band} Risk\\n\")\n",
    "\n",
    "print(\"📦 Prescriptive Action Plan:\")\n",
    "if risk_band == 'High':\n",
    "    print(\"1. Contract Incentive: Offer 15% discount upon 1-year contract lock-in.\")\n",
    "    print(\"2. Technical Care: Issue 3-month complimentary Tech Support add-on.\")\n",
    "    print(\"3. Account Management: Route open support tickets directly to Tier-2 VIP queue.\")\n",
    "elif risk_band == 'Medium':\n",
    "    print(\"1. Proactive check-in survey regarding billing and connectivity quality.\")\n",
    "    print(\"2. Promote annual billing discount incentive.\")\n",
    "else:\n",
    "    print(\"1. Standard engagement; deliver loyalty milestone rewards.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 8. What-If Strategy & Financial ROI Simulator\n",
    "Calculates expected retained revenue vs. intervention cost to determine net campaign ROI."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def simulate_retention_campaign(target_churners=300, avg_annual_revenue=1000, retention_rate=0.40, cost_per_customer=60):\n",
    "    retained_customers = target_churners * retention_rate\n",
    "    gross_revenue_saved = retained_customers * avg_annual_revenue\n",
    "    campaign_cost = target_churners * cost_per_customer\n",
    "    net_roi = ((gross_revenue_saved - campaign_cost) / campaign_cost) * 100\n",
    "    \n",
    "    print(\"💼 Strategy Simulation Results:\")\n",
    "    print(f\" - Target Churners Contacted: {target_churners:,}\")\n",
    "    print(f\" - Expected Retained Subscribers: {int(retained_customers):,}\")\n",
    "    print(f\" - Gross Revenue Saved: ${gross_revenue_saved:,.2f}\")\n",
    "    print(f\" - Campaign Cost: ${campaign_cost:,.2f}\")\n",
    "    print(f\" - Net Campaign ROI: {net_roi:,.1f}%\")\n",
    "\n",
    "simulate_retention_campaign()"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("HerambKrishnaArora_CustomerChurnIntelligence.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("Generated complete all-in-one Jupyter Notebook: HerambKrishnaArora_CustomerChurnIntelligence.ipynb")
