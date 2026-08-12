import sys
from data_utils import generate_churn_data
from pipeline import ChurnPipeline

def test_pipeline_flow():
    print("Testing data generation...")
    df = generate_churn_data(n_samples=500, random_seed=42)
    assert df.shape[0] == 500, "Generated dataset has wrong number of rows"
    print(f"Generated shape: {df.shape}")
    
    models = ["logistic_regression", "random_forest", "xgboost"]
    
    for model_name in models:
        print(f"\n--- Testing Model Pipeline: {model_name} ---")
        pipe = ChurnPipeline(model_type=model_name)
        
        print("Training model...")
        metrics = pipe.train(df, test_size=0.2, balance_weights=True)
        
        # Verify metrics keys
        required_keys = [
            "test_accuracy", "test_precision", "test_recall", "test_f1", 
            "test_roc_auc", "test_pr_auc", "train_accuracy", "train_roc_auc", 
            "confusion_matrix", "roc_curve", "pr_curve", "feature_importance"
        ]
        for key in required_keys:
            assert key in metrics, f"Metric key '{key}' missing from evaluation output"
            
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
        print(f"Feature Importances count: {len(metrics['feature_importance'])}")
        
        # Test individual prediction
        print("Testing single-record predictor...")
        sample_record = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 10,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.50,
            "TotalCharges": 855.0
        }
        pred_res = pipe.predict_single(sample_record)
        print(f"Prediction result: {pred_res}")
        assert "prediction" in pred_res
        assert "probability_no" in pred_res
        assert "probability_yes" in pred_res
        assert pred_res["prediction"] in ["Yes", "No"]
        
    print("\nAll pipeline flow tests passed successfully!")

if __name__ == "__main__":
    try:
        test_pipeline_flow()
        sys.exit(0)
    except AssertionError as e:
        print(f"Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(2)
