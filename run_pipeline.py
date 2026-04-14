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
import argparse
import subprocess
from datetime import datetime


def print_header(text):
    """Print header with nice format"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def _all_exist(paths):
    return all(os.path.exists(p) for p in paths)


def _has_files(dir_path):
    return os.path.isdir(dir_path) and len(os.listdir(dir_path)) > 0


def _fmt_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}m"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start-dashboard", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args(argv)

    print("""
+======================================================================+
|                  BANK CHURN PREDICTION PIPELINE                     |
|                                                                      |
|  Step 1: Load Data (CSV)                                             |
|  Step 2: Model Evaluation (Train + Evaluate + SHAP + SMOTE)         |
|  Step 3: Flask Dashboard                                            |
+======================================================================+
    """)

    start_time = time.perf_counter()
    step_times = {}
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/shap_plots", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("powerbi", exist_ok=True)

    print_header("STEP 1: MODEL EVALUATION")
    print("Running model_evaluation.py (Load + Train + Evaluate)")
    eval_outputs = [
        "outputs/results.json",
        "outputs/comparison.json",
        "outputs/model_lr.pkl",
        "outputs/model_rf.pkl",
        "outputs/model_xgb.pkl",
        "outputs/model_kmeans.pkl",
        "outputs/cluster_profiles.json",
        "outputs/cluster_strategies.json",
        "outputs/elbow_data.json",
    ]

    t0 = time.perf_counter()
    if not args.force and _all_exist(eval_outputs):
        print("[SKIP] Model evaluation artifacts already exist. Use --force to re-run.")
    else:
        from pipeline.model_evaluation import main as run_model_eval
        run_model_eval()
    step_times["Model Evaluation"] = time.perf_counter() - t0

    print("\n[OK] Evaluation completed!")

    print_header("STEP 2: SHAP ANALYSIS")
    print("Running shap_analysis.py (Model Explainability)")
    shap_outputs = [
        "outputs/shap_analysis.json",
    ]
    shap_plots_dir = "outputs/shap_plots"

    t0 = time.perf_counter()
    if not args.force and _all_exist(shap_outputs) and _has_files(shap_plots_dir):
        print("[SKIP] SHAP artifacts already exist. Use --force to re-run.")
    else:
        from pipeline.shap_analysis import main as run_shap
        run_shap()
    step_times["SHAP Analysis"] = time.perf_counter() - t0
    print("\n[OK] SHAP Analysis completed!")

    print_header("STEP 3: IMBALANCED DATA ANALYSIS")
    print("Running imbalanced_analysis.py (SMOTE Analysis)")
    imbalance_outputs = [
        "outputs/imbalance_analysis.json",
    ]

    t0 = time.perf_counter()
    if not args.force and _all_exist(imbalance_outputs):
        print("[SKIP] Imbalance analysis artifacts already exist. Use --force to re-run.")
    else:
        from pipeline.imbalanced_analysis import main as run_imbalance
        run_imbalance()
    step_times["Imbalanced Analysis"] = time.perf_counter() - t0
    print("\n[OK] Imbalanced Analysis completed!")

    if not args.no_audit:
        print_header("STEP 4: SANITY + SCHEMA AUDIT (STRICT)")
        t0 = time.perf_counter()
        cmds = [
            [sys.executable, "sanity_check.py"],
            [sys.executable, "schema_audit.py", "--strict"],
        ]
        for cmd in cmds:
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        step_times["Audit"] = time.perf_counter() - t0
        print("\n[OK] Audit completed!")

    total_time = time.perf_counter() - start_time

    print_header("[DONE] PIPELINE COMPLETED!")
    print(f"""
    Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)
    
    Output files created:
    """)


    print_header("TIMING SUMMARY")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Generated at: {now}")
    rows = list(step_times.items())
    width = max((len(name) for name, _ in rows), default=10)
    for name, sec in rows:
        print(f"- {name:<{width}} : {_fmt_seconds(sec)}")
    print(f"- {'Total':<{width}} : {_fmt_seconds(total_time)}")
    output_files = [
        ("eda.json", "EDA analysis"),
        ("comparison.json", "Model comparison"),
        ("feat_imp.json", "Feature importance"),
        ("cv_results.json", "Cross-validation results"),
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
    if args.start_dashboard:
        os.system("python app.py")


if __name__ == "__main__":
    main(sys.argv[1:])
