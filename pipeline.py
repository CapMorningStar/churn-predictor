import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix, auc
)

class ChurnPipeline:
    def __init__(self, model_type="xgboost", model_params=None):
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.pipeline = None
        self.classes_ = None
        self.numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        self.target_col = 'Churn'

    def clean_data(self, df):
        """
        Cleans the dataframe by handling string-to-float conversions,
        whitespace issues, and casting datatypes correctly.
        """
        df_clean = df.copy()
        
        # 1. Drop customerID since it has no predictive power
        if "customerID" in df_clean.columns:
            df_clean = df_clean.drop(columns=["customerID"])
            
        # 2. Fix TotalCharges: convert empty strings/whitespaces to NaN, then float
        if "TotalCharges" in df_clean.columns:
            df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors='coerce')
            
        # 3. Ensure SeniorCitizen is treated as categorical/object for encoding
        if "SeniorCitizen" in df_clean.columns:
            df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(object)
            
        return df_clean

    def build_preprocessor(self):
        """
        Builds the ColumnTransformer for preprocessing.
        """
        # Numeric pipeline: Impute missing with median, scale features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline: Impute with most frequent, encode to dummy variables
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )
        return preprocessor

    def get_classifier(self, class_weight_ratio=1.0):
        """
        Initializes the classifier based on selection and hyperparams.
        Adjusts class weights to handle imbalance if requested.
        """
        # Class weight logic
        # ratio > 1 means penalizing Churn=Yes mistakes more heavily (since it is the minority class)
        if self.model_type == "logistic_regression":
            cw = {0: 1.0, 1: float(class_weight_ratio)} if class_weight_ratio != 1.0 else None
            params = {
                "max_iter": 1000,
                "class_weight": cw,
                "random_state": 42
            }
            params.update(self.model_params)
            return LogisticRegression(**params)
            
        elif self.model_type == "random_forest":
            cw = {0: 1.0, 1: float(class_weight_ratio)} if class_weight_ratio != 1.0 else None
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "class_weight": cw,
                "random_state": 42,
                "n_jobs": -1
            }
            params.update(self.model_params)
            return RandomForestClassifier(**params)
            
        elif self.model_type == "xgboost":
            # XGBoost uses scale_pos_weight (ratio of negative to positive samples)
            params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "scale_pos_weight": float(class_weight_ratio),
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss"
            }
            params.update(self.model_params)
            return XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, df, test_size=0.2, balance_weights=True):
        """
        Processes and splits data, builds pipeline, fits the model, and evaluates it.
        """
        # 1. Clean data
        df_clean = self.clean_data(df)
        
        # 2. Split X and Y
        X = df_clean.drop(columns=[self.target_col])
        # Convert target "Yes"/"No" to 1/0
        y = df_clean[self.target_col].map({"Yes": 1, "No": 0})
        self.classes_ = ["No", "Yes"]
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 3. Calculate class weights ratio
        class_weight_ratio = 1.0
        if balance_weights:
            # Count ratios: num_neg / num_pos
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            class_weight_ratio = neg_count / max(1, pos_count)
            
        # 4. Construct complete pipeline
        preprocessor = self.build_preprocessor()
        classifier = self.get_classifier(class_weight_ratio)
        
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        # 5. Fit pipeline
        self.pipeline.fit(X_train, y_train)
        
        # 6. Evaluate and return results
        metrics = self.evaluate(X_train, y_train, X_test, y_test)
        return metrics

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluates the trained pipeline on both train and test partitions.
        """
        # Predictions
        y_train_pred = self.pipeline.predict(X_train)
        y_train_proba = self.pipeline.predict_proba(X_train)[:, 1]
        
        y_test_pred = self.pipeline.predict(X_test)
        y_test_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics Calculation (Test partition)
        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Curves
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Calculate feature importances if possible
        feature_importance = self.get_feature_importances()
        
        return {
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_roc_auc": roc_auc,
            "test_pr_auc": pr_auc,
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_roc_auc": roc_auc_score(y_train, y_train_proba),
            "confusion_matrix": cm.tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()},
            "feature_importance": feature_importance
        }

    def get_feature_importances(self):
        """
        Extracts feature names from ColumnTransformer and importance coefficients from classifier.
        """
        if self.pipeline is None:
            return []
            
        try:
            # Get preprocessor & classifier steps
            preprocessor = self.pipeline.named_steps['preprocessor']
            classifier = self.pipeline.named_steps['classifier']
            
            # Extract feature names from OneHotEncoder and pass numeric columns
            cat_features = []
            # Traverse categorical pipeline to get one-hot category names
            for name, transformer, columns in preprocessor.transformers_:
                if name == 'cat':
                    ohe = transformer.named_steps['onehot']
                    cat_features = ohe.get_feature_names_out(self.categorical_cols).tolist()
            
            feature_names = self.numerical_cols + cat_features
            
            # Extract weights
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                importances = np.abs(classifier.coef_[0])
            else:
                return []
                
            # Create pairs
            feat_imp = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [{"feature": f, "importance": float(i)} for f, i in feat_imp]
        except Exception as e:
            print(f"Warning: could not extract feature importances: {e}")
            return []

    def predict_single(self, input_dict):
        """
        Accepts a single user record as a dict, cleans it, and returns prediction details.
        """
        if self.pipeline is None:
            raise ValueError("Model pipeline is not trained yet!")
            
        df_single = pd.DataFrame([input_dict])
        df_single_clean = self.clean_data(df_single)
        
        # Generate prediction and probabilities
        proba = self.pipeline.predict_proba(df_single_clean)[0]
        pred = self.pipeline.predict(df_single_clean)[0]
        
        return {
            "prediction": self.classes_[pred],
            "probability_no": float(proba[0]),
            "probability_yes": float(proba[1])
        }
