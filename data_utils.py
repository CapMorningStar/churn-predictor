import pandas as pd
import numpy as np

def generate_churn_data(n_samples=2000, random_seed=42):
    """
    Generates a realistic synthetic customer churn dataset with structured correlations,
    similar to the famous Telco Churn dataset, for educational machine learning purposes.
    """
    np.random.seed(random_seed)
    
    # 1. Basic demographics
    customer_ids = [f"{np.random.randint(1000, 9999)}-{np.random.choice(['A', 'B', 'C', 'D', 'E'])}{np.random.randint(1000, 9999)}" for _ in range(n_samples)]
    genders = np.random.choice(["Female", "Male"], size=n_samples, p=[0.5, 0.5])
    senior_citizens = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    partners = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.3, 0.7])
    
    # 2. Account Information
    # Tenure in months (skewed slightly towards shorter and longer terms)
    tenures = np.random.choice(
        np.concatenate([
            np.random.randint(1, 12, size=int(n_samples * 0.3)),
            np.random.randint(12, 48, size=int(n_samples * 0.4)),
            np.random.randint(48, 73, size=int(n_samples * 0.3))
        ]),
        size=n_samples
    )
    
    contracts = []
    for t in tenures:
        if t < 12:
            contracts.append(np.random.choice(["Month-to-month", "One year"], p=[0.9, 0.1]))
        elif t < 36:
            contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.4, 0.4, 0.2]))
        else:
            contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.1, 0.3, 0.6]))
            
    paperless_billings = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.23, 0.21, 0.22]
    )
    
    # 3. Services
    phone_services = np.random.choice(["Yes", "No"], size=n_samples, p=[0.9, 0.1])
    multiple_lines = []
    for ps in phone_services:
        if ps == "Yes":
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
        else:
            multiple_lines.append("No phone service")
            
    internet_services = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.45, 0.20])
    
    online_securities = []
    online_backups = []
    device_protections = []
    tech_supports = []
    streaming_tvs = []
    streaming_movies = []
    
    for iserv in internet_services:
        if iserv != "No":
            online_securities.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            online_backups.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))
            device_protections.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))
            tech_supports.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            streaming_tvs.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
        else:
            msg = "No internet service"
            online_securities.append(msg)
            online_backups.append(msg)
            device_protections.append(msg)
            tech_supports.append(msg)
            streaming_tvs.append(msg)
            streaming_movies.append(msg)

    # 4. Charges
    # Calculate base charges dynamically depending on active services
    monthly_charges = []
    for i in range(n_samples):
        charge = 20.0  # Base line charge
        if phone_services[i] == "Yes":
            charge += 10.0
            if multiple_lines[i] == "Yes":
                charge += 10.0
                
        iserv = internet_services[i]
        if iserv == "DSL":
            charge += 25.0
        elif iserv == "Fiber optic":
            charge += 45.0
            
        if iserv != "No":
            if online_securities[i] == "Yes": charge += 8.0
            if online_backups[i] == "Yes": charge += 8.0
            if device_protections[i] == "Yes": charge += 8.0
            if tech_supports[i] == "Yes": charge += 8.0
            if streaming_tvs[i] == "Yes": charge += 12.0
            if streaming_movies[i] == "Yes": charge += 12.0
            
        # Add random noise to make charges less deterministic
        charge += np.random.normal(0, 3)
        monthly_charges.append(round(max(15.0, charge), 2))
        
    # Total charges (tenure * monthly charge, with minor variation)
    total_charges = []
    for i in range(n_samples):
        # Add random fluctuation to total charges
        val = tenures[i] * monthly_charges[i] * np.random.uniform(0.98, 1.02)
        total_charges.append(round(val, 2))
        
    # Introduce small missing data (empty strings / NaNs) in TotalCharges to teach imputation
    total_charges = np.array(total_charges, dtype=object)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    for idx in missing_indices:
        # Some are blank strings, some are NaN
        if np.random.rand() > 0.5:
            total_charges[idx] = " "
        else:
            total_charges[idx] = np.nan

    # 5. Target Variable: Churn (Probabilistic logic based on attributes)
    churn_labels = []
    for i in range(n_samples):
        # Base churn probability
        p_churn = 0.15
        
        # 1. Contract effects (Very strong)
        contract = contracts[i]
        if contract == "Month-to-month":
            p_churn += 0.35
        elif contract == "One year":
            p_churn -= 0.05
        elif contract == "Two year":
            p_churn -= 0.12
            
        # 2. Tenure effects (Longer tenure means less churn)
        t = tenures[i]
        if t < 6:
            p_churn += 0.20
        elif t < 12:
            p_churn += 0.10
        elif t > 36:
            p_churn -= 0.10
        elif t > 60:
            p_churn -= 0.15
            
        # 3. Internet Service & Support
        iserv = internet_services[i]
        if iserv == "Fiber optic":
            p_churn += 0.15 # Real-world fiber optic customers often churn more (due to aggressive promo rates ending)
        elif iserv == "No":
            p_churn -= 0.10
            
        if iserv != "No":
            if tech_supports[i] == "No":
                p_churn += 0.10
            else:
                p_churn -= 0.05
                
            if online_securities[i] == "No":
                p_churn += 0.05
                
        # 4. Billing and Demographics
        if paperless_billings[i] == "Yes":
            p_churn += 0.05
        if payment_methods[i] == "Electronic check":
            p_churn += 0.15
        if senior_citizens[i] == 1:
            p_churn += 0.08
            
        # Add random individual noise
        p_churn += np.random.normal(0, 0.05)
        
        # Clip probability
        p_churn = np.clip(p_churn, 0.01, 0.95)
        
        # Determine churn decision
        if np.random.rand() < p_churn:
            churn_labels.append("Yes")
        else:
            churn_labels.append("No")
            
    # Assemble DataFrame
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizens,
        "Partner": partners,
        "Dependents": dependents,
        "tenure": tenures,
        "PhoneService": phone_services,
        "MultipleLines": multiple_lines,
        "InternetService": internet_services,
        "OnlineSecurity": online_securities,
        "OnlineBackup": online_backups,
        "DeviceProtection": device_protections,
        "TechSupport": tech_supports,
        "StreamingTV": streaming_tvs,
        "StreamingMovies": streaming_movies,
        "Contract": contracts,
        "PaperlessBilling": paperless_billings,
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_labels
    })
    
    return df

if __name__ == "__main__":
    df = generate_churn_data(2500)
    df.to_csv("churn_data.csv", index=False)
    print(f"Generated data with shape {df.shape} and saved to 'churn_data.csv'.")
    print("Churn class balance:")
    print(df['Churn'].value_counts(normalize=True))
