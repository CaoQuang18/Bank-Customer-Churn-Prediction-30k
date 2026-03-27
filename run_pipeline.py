"""
RUN PIPELINE
===========
Run the entire pipeline from start to finish with one command.

Usage:
    python run_pipeline.py

Pipeline includes:
    1. Load Data          -> data/raw/BankCustomer churn.csv
    2. Model Evaluation   -> Train + Evaluate + SHAP + Imbalanced
    3. Flask Dashboard    -> http://localhost:5000

Author: Auto-generated
Date: 2026
"""

import os
import sys
import time


def print_header(text):
    """Print header with nice format"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main():
    print("""
+======================================================================+
|                  BANK CHURN PREDICTION PIPELINE                     |
|                                                                      |
|  Step 1: Load Data (CSV)                                             |
|  Step 2: Model Evaluation (Train + Evaluate + SHAP + SMOTE)         |
|  Step 3: Flask Dashboard                                            |
+======================================================================+
    """)

    start_time = time.time()
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/shap_plots", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("powerbi", exist_ok=True)

    print_header("STEP 1: MODEL EVALUATION")
    print("Running model_evaluation.py (Load + Train + Evaluate)")
    from pipeline.model_evaluation import main as run_model_eval

    run_model_eval()

    print("\n[OK] Evaluation completed!")

    print_header("STEP 2: SHAP ANALYSIS")
    print("Running shap_analysis.py (Model Explainability)")
    from pipeline.shap_analysis import main as run_shap

    run_shap()
    print("\n[OK] SHAP Analysis completed!")

    print_header("STEP 3: IMBALANCED DATA ANALYSIS")
    print("Running imbalanced_analysis.py (SMOTE Analysis)")
    from pipeline.imbalanced_analysis import main as run_imbalance

    run_imbalance()
    print("\n[OK] Imbalanced Analysis completed!")

    total_time = time.time() - start_time

    print_header("[DONE] PIPELINE COMPLETED!")
    print(f"""
    Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)
    
    Output files created:
    """)

    output_files = [
        ("eda.json", "EDA analysis"),
        ("comparison.json", "Model comparison"),
        ("feat_imp.json", "Feature importance"),
        ("cv_results.json", "Cross-validation results"),
        ("cluster_profiles.json", "Cluster profiles"),
        ("cluster_strategies.json", "Cluster strategies"),
        ("results.json", "Evaluation results"),
        ("model_version.json", "Model version"),
        ("model_lr.pkl", "Logistic Regression model"),
        ("model_rf.pkl", "Random Forest model"),
        ("model_xgb.pkl", "XGBoost model"),
        ("model_kmeans.pkl", "K-Means model"),
        ("X_test.pkl", "Test features"),
        ("y_test.pkl", "Test labels"),
        ("X_train.pkl", "Train features"),
        ("y_train.pkl", "Train labels"),
        ("imbalance_analysis.json", "SMOTE analysis"),
        ("shap_analysis.json", "SHAP analysis"),
        ("shap_plots/", "SHAP visualizations"),
    ]

    for filename, desc in output_files:
        filepath = f"outputs/{filename}"
        if os.path.exists(filepath):
            if os.path.isdir(filepath):
                files = os.listdir(filepath)
                print(f"    [OK] {filename:<30} ({len(files)} files) - {desc}")
            else:
                size = os.path.getsize(filepath)
                size_str = (
                    f"{size / 1024:.1f} KB"
                    if size < 1024 * 1024
                    else f"{size / 1024 / 1024:.1f} MB"
                )
                print(f"    [OK] {filename:<30} ({size_str}) - {desc}")
        elif not os.path.isdir(filepath):
            print(f"    [!]  {filename:<30} (not created)")

    print(f"""
    
    To start Dashboard:
    
        python app.py
    
    Dashboard will run at: http://localhost:5000
    
    """)

    try:
        choice = input("Start Dashboard? (y/n): ").strip().lower()
        if choice == "y" or choice == "yes":
            print("\nStarting Dashboard...")
            os.system("python app.py")
    except KeyboardInterrupt:
        print("\n\nExited!")


if __name__ == "__main__":
    main()
