import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 3000

customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
regions = np.random.choice(["West", "East", "North", "South", "Central"], size=n_samples, p=[0.30, 0.25, 0.20, 0.15, 0.10])
genders = np.random.choice(["Male", "Female"], size=n_samples)
senior_citizens = np.random.choice([0, 1], size=n_samples, p=[0.82, 0.18])
partners = np.random.choice(["Yes", "No"], size=n_samples, p=[0.52, 0.48])
dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.33, 0.67])

# Tenure in months (1 to 72)
tenures = np.random.randint(1, 73, size=n_samples)

# Services
phone_services = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
multiple_lines = np.where(phone_services == "Yes", np.random.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55]), "No phone service")
internet_services = np.random.choice(["Fiber Optic", "DSL", "No"], size=n_samples, p=[0.45, 0.35, 0.20])

online_security = np.where(internet_services != "No", np.random.choice(["Yes", "No"], size=n_samples, p=[0.38, 0.62]), "No internet service")
online_backup = np.where(internet_services != "No", np.random.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58]), "No internet service")
device_protection = np.where(internet_services != "No", np.random.choice(["Yes", "No"], size=n_samples, p=[0.40, 0.60]), "No internet service")
tech_support = np.where(internet_services != "No", np.random.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65]), "No internet service")
streaming_tv = np.where(internet_services != "No", np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52]), "No internet service")
streaming_movies = np.where(internet_services != "No", np.random.choice(["Yes", "No"], size=n_samples, p=[0.50, 0.50]), "No internet service")

# Account information
contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.23, 0.22])
paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
payment_methods = np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], size=n_samples, p=[0.35, 0.20, 0.25, 0.20])

# Monthly charges base calculation based on services
base_charges = np.zeros(n_samples)
base_charges += np.where(phone_services == "Yes", 20.0, 0.0)
base_charges += np.where(multiple_lines == "Yes", 10.0, 0.0)
base_charges += np.where(internet_services == "DSL", 30.0, 0.0)
base_charges += np.where(internet_services == "Fiber Optic", 50.0, 0.0)
base_charges += np.where(online_security == "Yes", 5.0, 0.0)
base_charges += np.where(online_backup == "Yes", 5.0, 0.0)
base_charges += np.where(device_protection == "Yes", 5.0, 0.0)
base_charges += np.where(tech_support == "Yes", 6.0, 0.0)
base_charges += np.where(streaming_tv == "Yes", 8.0, 0.0)
base_charges += np.where(streaming_movies == "Yes", 8.0, 0.0)
monthly_charges = np.round(base_charges + np.random.normal(5.0, 3.0, size=n_samples), 2)
monthly_charges = np.clip(monthly_charges, 18.5, 125.0)

# Total charges
total_charges = np.round(monthly_charges * tenures + np.random.normal(0, 15.0, size=n_samples), 2)
total_charges = np.clip(total_charges, 18.5, None)

# Customer support tickets opened in past 6 months
support_tickets = np.random.poisson(lam=1.5, size=n_samples)

# Churn Probability modeling based on realistic business risk drivers
# Higher risk: Month-to-month contract, Fiber Optic without Tech Support, Electronic check, Short tenure, High monthly charges, High support tickets, West Region
churn_score = (
    - 0.05 * tenures
    + 1.8 * (contracts == "Month-to-month")
    - 1.2 * (contracts == "Two year")
    + 0.9 * (internet_services == "Fiber Optic")
    - 0.8 * (tech_support == "Yes")
    + 0.6 * (payment_methods == "Electronic check")
    + 0.02 * monthly_charges
    + 0.45 * support_tickets
    + 0.5 * (regions == "West")
    + 0.3 * senior_citizens
    - 0.4 * (partners == "Yes")
)

# Sigmoid function for probability
prob = 1 / (1 + np.exp(-(churn_score - 1.6)))
churn = np.where(np.random.rand(n_samples) < prob, "Yes", "No")

df = pd.DataFrame({
    "CustomerID": customer_ids,
    "Region": regions,
    "Gender": genders,
    "SeniorCitizen": senior_citizens,
    "Partner": partners,
    "Dependents": dependents,
    "Tenure": tenures,
    "PhoneService": phone_services,
    "MultipleLines": multiple_lines,
    "InternetService": internet_services,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contracts,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_methods,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "SupportTickets": support_tickets,
    "Churn": churn
})

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/customer_churn_data.csv", index=False)
print(f"Generated dataset successfully with {len(df)} rows. Churn rate: {(df['Churn'] == 'Yes').mean() * 100:.2f}%")
