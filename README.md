# 📊 Customer Churn & Revenue Intelligence Platform
### **Executive Decision Intelligence Suite & Prescriptive ML Engine**
> **AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship Project**  
> *Conducted by BharatCares in association with IBM SkillsBuild*

---

## 🚀 1. Project Overview & Business Value
In subscription-driven industries (SaaS, E-Commerce, Telecommunications), retaining an existing customer costs **5x to 7x less** than acquiring a new customer. Unmitigated churn diminishes Monthly Recurring Revenue (MRR) and drastically lowers overall Customer Lifetime Value (CLV).

This project implements a complete **Executive Decision Intelligence Platform** designed according to the **5-Stage Business Intelligence Hierarchy** introduced in Masterclass 5:
$$\text{Raw Data} \longrightarrow \text{Clean Information} \longrightarrow \text{Diagnostic Insight} \longrightarrow \text{Strategic Decision} \longrightarrow \text{Measurable Action}$$

Rather than delivering a passive "Data Dump" (cluttered charts with no direction), this platform provides:
- **Executive KPIs**: Total revenue, churn percentage, annualized revenue at risk, and projected retention savings.
- **Root-Cause Attribution**: Isolates the primary drivers of customer attrition.
- **Real-Time ML Churn Risk Scoring**: Immediate scoring with personalized prescriptive retention bundles.
- **What-If Strategy Simulator**: Financial modeling tool calculating net ROI for retention interventions.
- **AI Executive Briefing**: Automated 5 Key Findings, 3 Risks, 3 Opportunities, and 5 Actionable Recommendations.

---

## 📈 2. The 5-Stage Dashboard Hierarchy

```
┌──────────────────────────────────────────────────────────┐
│ Level 1 — KPIs        │ What is happening right now?    │
├──────────────────────────────────────────────────────────┤
│ Level 2 — Trends      │ How is performance changing?     │
├──────────────────────────────────────────────────────────┤
│ Level 3 — Drivers     │ Why are customers leaving?       │
├──────────────────────────────────────────────────────────┤
│ Level 4 — Risk Matrix │ What could go wrong?             │
├──────────────────────────────────────────────────────────┤
│ Level 5 — Actions     │ What strategic actions to take?  │
└──────────────────────────────────────────────────────────┘
```

1. **Level 1 (KPIs):** Instant executive health check across 3,000 subscriber accounts tracking \$5.6M+ in cumulative revenue and identifying \$1.4M+ in annual revenue at risk.
2. **Level 2 (Trends):** Cohort analysis tracking subscriber decay over tenure groups (0–12 months vs. 48+ months).
3. **Level 3 (Drivers):** Multidimensional analysis proving Month-to-Month contracts, Electronic Check friction, and lack of Tech Support represent over 76% of all attrition.
4. **Level 4 (Risk & Opportunities):** Real-time customer risk classifier with danger thresholds for accounts with $\ge 2$ technical support tickets.
5. **Level 5 (Prescriptive Actions):** Automated contract upgrade incentives, complimentary tech support bundling, and dynamic What-If ROI simulations.

---

## 📂 3. Dataset Information
* **Source Domain:** Subscription Analytics, Customer Behavior & Churn (Telecommunications & E-Commerce).
* **Reference Link:** [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Local Dataset File:** `data/customer_churn_data.csv` (3,000 customer records, 23 feature attributes).
* **Key Features:**
  - `Region`: Geographic territory (West, East, North, South, Central)
  - `Tenure`: Customer lifetime in months (1–72)
  - `Contract`: Month-to-month, One year, Two year
  - `InternetService`: Fiber Optic, DSL, No internet
  - `TechSupport`: Dedicated technical support coverage (Yes/No)
  - `PaymentMethod`: Electronic check, Mailed check, Bank transfer (auto), Credit card (auto)
  - `MonthlyCharges` & `TotalCharges`: Direct billing and revenue amounts ($)
  - `SupportTickets`: Count of customer service tickets opened in the last 6 months
  - `Churn`: Target classification label (Yes / No)

---

## 🤖 4. Machine Learning Pipeline & Benchmark Results

Three distinct classification models were benchmarked using an **80/20 Stratified Train-Test Split** and standardized feature scaling:

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **78.50%** | **76.74%** | **78.09%** | **77.41%** | **0.8572** | 🏆 **Champion Model** |
| **Gradient Boosting** | 75.33% | 73.20% | 75.27% | 74.22% | 0.8399 | Benchmark |
| **Random Forest** | 73.67% | 71.63% | 73.14% | 72.38% | 0.8191 | Benchmark |

### **Top 5 Predictive Churn Risk Drivers:**
1. `Contract_Month-to-month` (Extreme positive risk factor)
2. `Tenure` (Strong negative risk factor — maturity locks retention)
3. `Contract_Two year` (Strong protective anchor)
4. `SupportTickets` (Friction multiplier)
5. `TechSupport_Yes` (Protective service tier)

---

## 🏗️ 5. Project Architecture & Directory Structure

```
├── Customer_Churn_Intelligence_Platform.py   # Unified single-file submission script
├── Customer_Churn_Intelligence_Project_Report.docx # Formatted Word Project Report
├── README.md                                 # Full project documentation
├── requirements.txt                          # Python dependencies
├── app.py                                    # FastAPI REST API & Web Server
├── generate_report.py                        # Automated Word report generator
├── data/
│   ├── customer_churn_data.csv               # Processed dataset (3,000 rows)
│   └── generate_dataset.py                   # Dataset generation and calibration
├── models/
│   ├── churn_model.joblib                    # Serialized model, scaler & metadata
│   └── train_model.py                        # Model training and chart generation
└── static/
    ├── index.html                            # Executive Dashboard HTML5 UI
    ├── style.css                             # Glassmorphic modern dark styling
    ├── app.js                                # Interactive charts & live API logic
    └── images/
        ├── confusion_matrix.png              # Diagnostic confusion matrix
        ├── roc_curve.png                     # Multi-model ROC comparison curve
        └── feature_importance.png            # Top risk factors bar chart
```

---

## ⚡ 6. Installation & Quickstart Guide

### **Step 1: Clone or Navigate to Project Directory**
```bash
cd "Customer Churn & Revenue Intelligence Platform"
```

### **Step 2: Install Python Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: (Optional) Retrain ML Models & Generate Visuals**
```bash
python3 models/train_model.py
```

### **Step 4: Launch the Executive Dashboard**
```bash
python3 app.py
```
* **Interactive Dashboard:** Open your browser at [http://localhost:8000](http://localhost:8000)
* **Interactive API Documentation (Swagger/OpenAPI):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 7. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive Executive Dashboard UI |
| `GET` | `/api/kpis` | Returns top-level revenue, churn, and financial risk metrics |
| `GET` | `/api/analytics/trends` | Returns tenure cohort trends and regional churn breakdown |
| `GET` | `/api/analytics/drivers` | Returns root-cause breakdowns and feature importances |
| `GET` | `/api/executive-insights`| Returns the AI Executive Analyst briefing (5 Findings, 3 Risks, 3 Opps, 5 Actions) |
| `POST` | `/api/predict` | Live ML inference scoring for a customer profile with prescriptive actions |
| `POST` | `/api/simulate` | What-If financial strategy simulation calculating retention lift & ROI |

---

## 📄 8. Project Submission Deliverables Checklist
- [x] **Code File:** `Customer_Churn_Intelligence_Platform.py` / `app.py`
- [x] **Requirements File:** `requirements.txt`
- [x] **Project Report:** `Customer_Churn_Intelligence_Project_Report.docx` (with embedded confusion matrix, ROC curves, and hierarchy tables)
- [x] **README File:** `README.md` (complete project structure, setup instructions, and Kaggle reference link)
- [x] **GitHub Repository:** Ready to push and submit
