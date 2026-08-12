import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_utils import generate_churn_data
from pipeline import ChurnPipeline

# Set page configuration with a modern title and layout
st.set_page_config(
    page_title="PrediChurn | Classical ML Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom premium styling
st.markdown("""
<style>
    /* CSS for cards and overall style */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .metric-val {
        font-size: 24px;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-lbl {
        font-size: 14px;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Section dividers */
    .section-header {
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 8px;
        margin-bottom: 20px;
        color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if "df_raw" not in st.session_state:
    # Pre-generate some default synthetic data
    st.session_state.df_raw = generate_churn_data(n_samples=2000, random_seed=42)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🔮 Model Settings")
st.sidebar.markdown("Configure the Classical ML pipeline parameters below.")

model_choice = st.sidebar.selectbox(
    "Choose Classifier",
    ["XGBoost", "Random Forest", "Logistic Regression"],
    index=0
)

# Render model-specific hyperparameters
model_params = {}
if model_choice == "XGBoost":
    st.sidebar.subheader("XGBoost Hyperparameters")
    model_params["n_estimators"] = st.sidebar.slider("Number of Estimators (Trees)", 20, 300, 100, 10)
    model_params["max_depth"] = st.sidebar.slider("Max Tree Depth", 2, 10, 5, 1)
    model_params["learning_rate"] = st.sidebar.slider("Learning Rate (eta)", 0.01, 0.50, 0.10, 0.01)
    
elif model_choice == "Random Forest":
    st.sidebar.subheader("Random Forest Hyperparameters")
    model_params["n_estimators"] = st.sidebar.slider("Number of Trees", 20, 300, 100, 10)
    model_params["max_depth"] = st.sidebar.slider("Max Tree Depth", 2, 20, 10, 1)
    model_params["min_samples_split"] = st.sidebar.slider("Min Samples to Split Node", 2, 20, 5, 1)

elif model_choice == "Logistic Regression":
    st.sidebar.subheader("Logistic Regression Hyperparameters")
    c_val = st.sidebar.number_input("Regularization Strength (C)", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
    model_params["C"] = c_val
    model_params["penalty"] = st.sidebar.selectbox("Penalty", ["l2", "none"])

balance_classes = st.sidebar.checkbox(
    "Handle Class Imbalance (Class Weighting)",
    value=True,
    help="Adjusts training loss dynamically to penalize minority class (Churn=Yes) errors more heavily."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Level 1: The Predictor
*Learn how handling messy tabular data fuels accurate context processing in GenAI RAG pipelines.*
""")

# ----------------- MAIN INTERFACE -----------------
st.title("🔮 PrediChurn: Churn Prediction & Tabular ML")
st.markdown("Build, train, visualize, and test classical ML pipelines for customer retention.")

# Create the top-level tabs
tab_data, tab_eda, tab_preproc, tab_train, tab_predict = st.tabs([
    "📁 Data Hub", 
    "📊 Exploratory Data Analysis", 
    "⚙️ Pipeline Preprocessing",
    "🤖 Model Training & Valuation", 
    "🔮 Individual Predictor"
])

# ================= TAB 1: DATA HUB =================
with tab_data:
    st.markdown("<h3 class='section-header'>📁 Ingest & Generate Data</h3>", unsafe_allow_html=True)
    
    col_opt_1, col_opt_2 = st.columns(2)
    with col_opt_1:
        data_source = st.radio(
            "Select Data Source:",
            ["Generate Synthetic Telecom Dataset", "Upload Custom CSV File"],
            index=0
        )
        
    with col_opt_2:
        if data_source == "Generate Synthetic Telecom Dataset":
            n_samples = st.slider("Number of synthetic customer records to generate:", 500, 10000, 2000, 100)
            if st.button("Generate Clean/Messy Dataset", width='stretch'):
                st.session_state.df_raw = generate_churn_data(n_samples=n_samples, random_seed=42)
                st.session_state.pipeline = None  # Reset trained model since dataset changed
                st.session_state.metrics = None
                st.success(f"Successfully generated {n_samples} customer records!")
        else:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    st.session_state.df_raw = uploaded_df
                    st.session_state.pipeline = None  # Reset trained model
                    st.session_state.metrics = None
                    st.success("Custom dataset uploaded successfully!")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

    st.markdown("### Raw Dataset Sample")
    st.dataframe(st.session_state.df_raw.head(10), width='stretch')
    
    # Missing statistics card
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Total Records</div>
            <div class="metric-val">{st.session_state.df_raw.shape[0]}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Columns (Features)</div>
            <div class="metric-val">{st.session_state.df_raw.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat3:
        # Sum of nulls plus any empty strings/whitespaces in TotalCharges
        nan_count = st.session_state.df_raw.isnull().sum().sum()
        if "TotalCharges" in st.session_state.df_raw.columns:
            nan_count += (st.session_state.df_raw["TotalCharges"] == " ").sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Messy Cells (Missing / Space)</div>
            <div class="metric-val" style="color: #DC2626;">{nan_count}</div>
        </div>
        """, unsafe_allow_html=True)

    # Let user download the current active dataset
    csv_bytes = st.session_state.df_raw.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Current Dataset (CSV)",
        data=csv_bytes,
        file_name="churn_dataset_active.csv",
        mime="text/csv",
        width='stretch'
    )


# ================= TAB 2: EXPLORATORY DATA ANALYSIS =================
with tab_eda:
    st.markdown("<h3 class='section-header'>📊 Visual Exploratory Analysis</h3>", unsafe_allow_html=True)
    df_eda = st.session_state.df_raw.copy()
    
    # Ensure standard types for mapping target
    if 'Churn' not in df_eda.columns:
        st.error("No 'Churn' column found in dataset! Cannot visualize churn metrics.")
    else:
        # Check target distribution
        churn_counts = df_eda['Churn'].value_counts()
        
        col_eda_1, col_eda_2 = st.columns([1, 2])
        
        with col_eda_1:
            st.markdown("#### Target Distribution (Churn)")
            fig_pie = px.pie(
                names=churn_counts.index, 
                values=churn_counts.values,
                color=churn_counts.index,
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                hole=0.4,
                title="Ratio of Retained vs Churned Customers"
            )
            fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, width='stretch')
            
        with col_eda_2:
            st.markdown("#### Customer Tenure Distribution")
            fig_hist = px.histogram(
                df_eda, 
                x="tenure", 
                color="Churn",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                barmode="overlay",
                title="Tenure Distribution (in Months) by Churn"
            )
            st.plotly_chart(fig_hist, width='stretch')

        st.markdown("---")
        
        col_eda_3, col_eda_4 = st.columns(2)
        
        with col_eda_3:
            st.markdown("#### Contract Type vs. Churn Rate")
            # Create a cross tab plot
            df_contract = df_eda.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
            fig_contract = px.bar(
                df_contract, 
                x="Contract", 
                y="Count", 
                color="Churn", 
                barmode="group",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                title="Number of Customers by Contract Type and Churn Status"
            )
            st.plotly_chart(fig_contract, width='stretch')
            
        with col_eda_4:
            st.markdown("#### Internet Service Type vs. Churn Rate")
            df_internet = df_eda.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
            fig_internet = px.bar(
                df_internet,
                x="InternetService",
                y="Count",
                color="Churn",
                barmode="group",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                title="Internet Service Tier Churn Comparison"
            )
            st.plotly_chart(fig_internet, width='stretch')


# ================= TAB 3: PIPELINE PREPROCESSING =================
with tab_preproc:
    st.markdown("<h3 class='section-header'>⚙️ Tabular Data Preprocessing</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    When dealing with real-world tabular data, it is rarely clean. To feed models like Random Forest or XGBoost, 
    we need to execute several transformations. This tab demonstrates what actions are applied inside our Scikit-Learn `ColumnTransformer` pipeline.
    """)
    
    col_prep_l, col_prep_r = st.columns(2)
    with col_prep_l:
        st.info("""
        #### 1. Handle Missing Values
        - **Numerical Features** (`tenure`, `MonthlyCharges`, `TotalCharges`): Imputed with the **median** values to handle empty fields or parsing failures.
        - **Categorical Features** (e.g. `InternetService`): Imputed with the **most frequent** value.
        """)
        
        st.info("""
        #### 2. Scale Numerical Features
        - Features like `TotalCharges` range in the thousands, while `tenure` ranges from 1 to 72.
        - **StandardScaler** transforms them to have a mean of 0 and variance of 1.
        """)

    with col_prep_r:
        st.info("""
        #### 3. One-Hot Encode Categoricals
        - Categorical inputs like `Contract` (Month-to-month, One year, Two year) are converted into distinct binary columns.
        - This prevents the model from assuming arbitrary numerical hierarchies.
        """)
        
        st.info("""
        #### 4. Train-Test Splitting
        - Data is stratified and split into **80% training** and **20% testing** sets to avoid data leakage and measure generalization.
        """)

    # Visual before-and-after demo
    st.markdown("### Pre-flight Data Cleaning Demonstration")
    pipeline_dummy = ChurnPipeline()
    cleaned_dummy = pipeline_dummy.clean_data(st.session_state.df_raw)
    
    col_demo_1, col_demo_2 = st.columns(2)
    with col_demo_1:
        st.markdown("**Original column types and NaN check:**")
        # Check TotalCharges parsing
        df_tc_demo = st.session_state.df_raw[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
        df_tc_demo["TotalCharges_CleanStatus"] = df_tc_demo["TotalCharges"].apply(lambda x: "Missing (whitespace/NaN)" if str(x).strip() == "" or pd.isna(x) else "Numeric")
        st.dataframe(df_tc_demo["TotalCharges_CleanStatus"].value_counts(), width='stretch')

    with col_demo_2:
        st.markdown("**Cleaned Numeric DataFrame (Ready for Scaler):**")
        st.dataframe(cleaned_dummy[["tenure", "MonthlyCharges", "TotalCharges"]].head(6), width='stretch')


# ================= TAB 4: MODEL TRAINING & VALUATION =================
with tab_train:
    st.markdown("<h3 class='section-header'>🤖 Model Training & Performance Dashboard</h3>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns([1, 3])
    
    with col_t1:
        st.write("Click below to train the model with your active hyperparameters (selected in the sidebar):")
        train_clicked = st.button("🚀 Train Model Pipeline", width='stretch', type="primary")
        
        if train_clicked:
            # Instantiate pipeline
            mapped_model = {
                "XGBoost": "xgboost",
                "Random Forest": "random_forest",
                "Logistic Regression": "logistic_regression"
            }[model_choice]
            
            with st.spinner(f"Training {model_choice} model..."):
                pipe = ChurnPipeline(model_type=mapped_model, model_params=model_params)
                metrics = pipe.train(st.session_state.df_raw, balance_weights=balance_classes)
                
                # Save to session state
                st.session_state.pipeline = pipe
                st.session_state.metrics = metrics
                st.success("Model trained successfully!")

    with col_t2:
        if st.session_state.metrics is not None:
            met = st.session_state.metrics
            
            # Display metrics cards
            st.markdown("#### Test Set Evaluation Metrics")
            card_col1, card_col2, card_col3, card_col4, card_col5 = st.columns(5)
            
            card_col1.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Accuracy</div>
                <div class="metric-val">{met['test_accuracy']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            card_col2.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Precision</div>
                <div class="metric-val">{met['test_precision']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            card_col3.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Recall</div>
                <div class="metric-val">{met['test_recall']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            card_col4.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">F1-Score</div>
                <div class="metric-val">{met['test_f1']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            card_col5.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">ROC-AUC</div>
                <div class="metric-val">{met['test_roc_auc']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.warning("Please click 'Train Model Pipeline' to fit a classifier and view validation metrics.")

    st.markdown("---")
    
    if st.session_state.metrics is not None:
        met = st.session_state.metrics
        
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("#### Confusion Matrix")
            cm = met["confusion_matrix"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Retained (No)", "Churned (Yes)"],
                y=["Retained (No)", "Churned (Yes)"],
                colorscale="Viridis",
                text=cm,
                texttemplate="%{text}",
                showscale=False
            ))
            fig_cm.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_cm, width='stretch')
            
        with col_plot2:
            st.markdown("#### ROC & PR Curves")
            # Build interactive ROC curve
            roc_data = met["roc_curve"]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_data["fpr"], y=roc_data["tpr"],
                mode='lines',
                name=f'ROC Curve (AUC = {met["test_roc_auc"]:.3f})',
                line=dict(color='darkorange', width=2)
            ))
            # Diagonal random line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random guess',
                line=dict(color='navy', width=1, dash='dash')
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.5, y=0.15),
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_roc, width='stretch')

        st.markdown("---")
        
        # Feature importances
        st.markdown("#### Feature Importances")
        feat_imp = met["feature_importance"]
        if len(feat_imp) > 0:
            df_imp = pd.DataFrame(feat_imp).head(15) # Show top 15
            fig_imp = px.bar(
                df_imp,
                y="feature",
                x="importance",
                orientation='h',
                title="Top 15 Predictive Features driving Churn Decision",
                color="importance",
                color_continuous_scale="Purples"
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=450)
            st.plotly_chart(fig_imp, width='stretch')
        else:
            st.info("Feature importance extraction is not supported or failed for this classifier.")


# ================= TAB 5: INDIVIDUAL PREDICTOR ("WHAT-IF") =================
with tab_predict:
    st.markdown("<h3 class='section-header'>🔮 What-If Customer Predictor</h3>", unsafe_allow_html=True)
    
    if st.session_state.pipeline is None:
        st.warning("Model pipeline has not been trained yet. Please train the model in the 'Model Training & Valuation' tab before using this predictor.")
    else:
        st.markdown("Configure a customer's contract, services, and charges to dynamically compute their probability of churn.")
        
        # Setup form fields
        col_inp1, col_inp2, col_inp3 = st.columns(3)
        
        with col_inp1:
            st.markdown("##### 👤 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner (Married/Cohabiting)", ["Yes", "No"])
            dependents = st.selectbox("Dependents (Children/Family)", ["Yes", "No"])
            
            st.markdown("##### 💵 Account & Billing")
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )

        with col_inp2:
            st.markdown("##### 📶 Core Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            
            # Sub-features contingent on Phone Service
            if phone_service == "Yes":
                multiple_lines = st.selectbox("Multiple Phone Lines", ["Yes", "No"])
            else:
                multiple_lines = "No phone service"
                
            internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
            
            # Contingent features on Internet Service
            if internet_service != "No":
                online_security = st.selectbox("Online Security Addon", ["Yes", "No"])
                online_backup = st.selectbox("Online Backup Addon", ["Yes", "No"])
                device_protection = st.selectbox("Device Protection Addon", ["Yes", "No"])
                tech_support = st.selectbox("Tech Support Addon", ["Yes", "No"])
            else:
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"

        with col_inp3:
            st.markdown("##### 🎬 Streaming Services")
            if internet_service != "No":
                streaming_tv = st.selectbox("Streaming TV Service", ["Yes", "No"])
                streaming_movies = st.selectbox("Streaming Movie Service", ["Yes", "No"])
            else:
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"

            st.markdown("##### 📝 Contract Status")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            tenure = st.slider("Tenure (Months Subscribed)", 1, 72, 12)
            
            monthly_charges = st.slider("Monthly Charges ($)", 15.0, 130.0, 50.0, 0.5)
            # Estimate TotalCharges dynamically but allow customization
            est_total = round(tenure * monthly_charges, 2)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(est_total))

        # Assemble the input dict
        input_record = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        st.markdown("---")
        
        # Prediction Output Section
        if st.button("🔮 Calculate Churn Risk Probability", type="primary", width='stretch'):
            res = st.session_state.pipeline.predict_single(input_record)
            
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                st.markdown("#### Retention Status Prediction")
                prob_yes = res["probability_yes"]
                
                # Visual gauge using Plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_yes * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability (%)", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#EF4444" if prob_yes > 0.5 else "#10B981"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#D1FAE5'},
                            {'range': [30, 70], 'color': '#FEF3C7'},
                            {'range': [70, 100], 'color': '#FEE2E2'}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=30, b=0, l=30, r=30))
                st.plotly_chart(fig_gauge, width='stretch')

            with col_res2:
                st.markdown("#### Actionable Churn Assessment")
                if prob_yes > 0.70:
                    st.error(f"### ⚠️ Critical Churn Risk ({prob_yes:.1%})")
                    st.markdown("""
                    **Risk profile details:**
                    - Customer has very high likelihood of leaving.
                    - **Recommended Retention Action**: Offer a discount/promotional rate on a **One or Two Year Contract** option, and propose **Tech Support** upgrades.
                    """)
                elif prob_yes > 0.40:
                    st.warning(f"### 🟡 Moderate Churn Risk ({prob_yes:.1%})")
                    st.markdown("""
                    **Risk profile details:**
                    - Customer shows early indicators of switching behavior.
                    - **Recommended Retention Action**: Proactive email outreach offering setup assistance or online security addon bundle deals.
                    """)
                else:
                    st.success(f"### ✅ Healthy Retention Profile ({prob_yes:.1%})")
                    st.markdown("""
                    **Risk profile details:**
                    - Customer is highly loyal and satisfied.
                    - **Recommended Retention Action**: Promote cross-selling of premium streaming features or device insurance.
                    """)
