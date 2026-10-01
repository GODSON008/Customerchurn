import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 📊 Customer Churn & Revenue Intelligence Platform\n",
    "### **AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship**\n",
    "**Student Name:** Heramb Krishna Arora  \n",
    "**Project Domain:** Business Intelligence, Machine Learning & Prescriptive Analytics  \n",
    "**Partner Organizations:** BharatCares & IBM SkillsBuild"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Essential Libraries & Setup"
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
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve\n",
    "\n",
    "# Styling\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams[\"figure.figsize\"] = (8, 5)\n",
    "print(\"Libraries loaded successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load & Explore Dataset (Level 1: Executive KPIs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('data/customer_churn_data.csv')\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "display(df.head())\n",
    "print(\"\\n--- Dataset Summary Statistics ---\")\n",
    "display(df.describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Exploratory Data Analysis & Churn Drivers (Levels 2 & 3: Trends & Drivers)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Level 1 & 2: Churn Distribution by Region & Contract\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "sns.countplot(data=df, x='Region', hue='Churn', palette=['#3b82f6', '#ef4444'], ax=axes[0])\n",
    "axes[0].set_title('Churn Volume by Region', fontsize=13, fontweight='bold')\n",
    "\n",
    "sns.countplot(data=df, x='Contract', hue='Churn', palette=['#3b82f6', '#ef4444'], ax=axes[1])\n",
    "axes[1].set_title('Churn Volume by Contract Type', fontsize=13, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Level 3: Tenure vs. Support Tickets Impact on Churn\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.boxplot(data=df, x='SupportTickets', y='MonthlyCharges', hue='Churn', palette=['#3b82f6', '#ef4444'])\n",
    "plt.title('Monthly Charges vs. Support Tickets by Churn Status', fontsize=13, fontweight='bold')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Preprocessing & Stratified Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
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
    "X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=42, stratify=y)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print(f\"Training set: {X_train.shape}, Test set: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Machine Learning Model Training & Comparative Benchmarks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
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
    "        \"Accuracy\": f\"{accuracy_score(y_test, pred)*100:.2f}%\开拓\",\n",
    "        \"Precision\": f\"{precision_score(y_test, pred)*100:.2f}%\开拓\",\n",
    "        \"Recall\": f\"{recall_score(y_test, pred)*100:.2f}%\开拓\",\n",
    "        \"F1-Score\": f\"{f1_score(y_test, pred)*100:.2f}%\开拓\",\n",
    "        \"ROC-AUC\": round(roc_auc_score(y_test, prob), 4)\n",
    "    })\n",
    "\n",
    "res_df = pd.DataFrame(results_table)\n",
    "display(res_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Diagnostic Visualizations & Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model = models[\"Logistic Regression\"]\n",
    "y_pred = best_model.predict(X_test_scaled)\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "plt.figure(figsize=(6, 5))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])\n",
    "plt.title('Champion Model Confusion Matrix', fontsize=13, fontweight='bold')\n",
    "plt.ylabel('Actual')\n",
    "plt.xlabel('Predicted')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Real-Time Inference & Prescriptive Decision Strategy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sample Customer Inference\n",
    "sample_customer = pd.DataFrame([{\n",
    "    'Region': 'West',\n",
    "    'Gender': 'Female',\n",
    "    'SeniorCitizen': 0,\n",
    "    'Partner': 'No',\n",
    "    'Dependents': 'No',\n",
    "    'Tenure': 6,\n",
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
    "    'MonthlyCharges': 85.50,\n",
    "    'TotalCharges': 513.00,\n",
    "    'SupportTickets': 3\n",
    "}])\n",
    "\n",
    "sample_encoded = pd.get_dummies(sample_customer, columns=cat_cols, drop_first=True)\n",
    "final_sample = pd.DataFrame(0, index=[0], columns=X_encoded.columns)\n",
    "for col in sample_encoded.columns:\n",
    "    if col in final_sample.columns:\n",
    "        final_sample[col] = sample_encoded[col]\n",
    "\n",
    "sample_scaled = scaler.transform(final_sample)\n",
    "churn_prob = best_model.predict_proba(sample_scaled)[0][1]\n",
    "\n",
    "print(f\"Predicted Churn Risk Probability: {churn_prob * 100:.1f}%\")\n",
    "print(\"Prescriptive Actions:\")\n",
    "print(\"- 1. Proactively offer 1-Year Contract Upgrade with 10% loyalty discount.\")\n",
    "print(\"- 2. Attach free 3-month priority Tech Support.\")\n",
    "print(\"- 3. Escalate account for VIP concierge onboarding.\")"
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

print("Created HerambKrishnaArora_CustomerChurnIntelligence.ipynb successfully!")
