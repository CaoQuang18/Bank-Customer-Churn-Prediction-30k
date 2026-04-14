import warnings
import json
import sys
import pandas as pd
import numpy as np
import os

# FIX: Windows cp1252 không hỗ trợ tiếng Việt trong print() → reconfigure stdout sang UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass  # Python < 3.7 không có reconfigure, bỏ qua

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from pipeline.feature_engineering import run_feature_engineering
from pipeline.feature_engineering import Winsorizer, _make_one_hot_encoder
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# IMBALANCED ANALYSIS (SCIENTIFIC REBUILD)
# Mục tiêu: Đo lường chính xác sức mạnh của SMOTE thông qua Cross-Validation
# chống rò rỉ dữ liệu tuyệt đối (Zero Data Leakage).
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_method(X, y, pipeline_or_model, method_name, class_weight=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Nếu không phải pipeline thì cần fit trực tiếp
        if isinstance(pipeline_or_model, ImbPipeline) or hasattr(pipeline_or_model, "fit"):
            pipeline_or_model.fit(X_tr, y_tr)
            y_pred = pipeline_or_model.predict(X_val)
            y_prob = pipeline_or_model.predict_proba(X_val)[:, 1]
        
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred))
        metrics['recall'].append(recall_score(y_val, y_pred))
        metrics['f1'].append(f1_score(y_val, y_pred))
        metrics['roc_auc'].append(roc_auc_score(y_val, y_prob))
        
    return {
        "method": method_name,
        "accuracy": float(np.mean(metrics['accuracy'])),
        "precision": float(np.mean(metrics['precision'])),
        "recall": float(np.mean(metrics['recall'])),
        "f1": float(np.mean(metrics['f1'])),
        "roc_auc": float(np.mean(metrics['roc_auc'])),
        "description": "5-Fold CV metrics"
    }

def run_imbalanced_analysis():
    print("--- Phân tích Mất Cân bằng Dữ Liệu Khoa học ---")
    
    from pipeline.feature_engineering import run_feature_engineering
    fe = run_feature_engineering()
    X = fe["X_train"]
    y = fe["y_train"]
    num_features = fe["num_features"]
    cat_features = fe["cat_features"]

    winsorizer = Winsorizer(columns=['balance', 'monthly_ir', 'credit_sco', 'age', 'engagement_score', 'risk_score'])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', _make_one_hot_encoder(cat_features), cat_features)
        ]
    )

    # Đảm bảo index hợp lệ cho K-Fold
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # --- Thống kê Cơ bản ---
    total = len(y)
    churn = sum(y == 1)
    non_churn = sum(y == 0)
    
    imbalance_stats = {
        "total": int(total),
        "churn": int(churn),
        "non_churn": int(non_churn),
        "minority_pct": round((churn/total)*100, 1),
        "majority_pct": round((non_churn/total)*100, 1),
        "imbalance_ratio": round(non_churn/churn, 1),
        "severity": "Nghiêm trọng (Nguy cơ Model bị Mù)" if non_churn/churn > 3 else "Kiểm soát được"
    }
    
    print(f"Tổng mẫu: {total}. Rời bỏ: {churn} ({imbalance_stats['minority_pct']}%). Imbalance Ratio: {imbalance_stats['imbalance_ratio']}:1")

    # --- Phương thức 1: Baseline (Mù ráng học) ---
    print("\n--- 1. Baseline ---")
    rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    pipe_base = ImbPipeline([
        ('winsor', winsorizer),
        ('preprocess', preprocessor),
        ('rf', rf_baseline)
    ])
    res_base = evaluate_method(X, y, pipe_base, "Baseline (Bỏ mặc)")
    
    # --- Phương thức 2: Class Weights ---
    print("--- 2. Class Weights ---")
    rf_cw = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    pipe_cw = ImbPipeline([
        ('winsor', winsorizer),
        ('preprocess', preprocessor),
        ('rf', rf_cw)
    ])
    res_cw = evaluate_method(X, y, pipe_cw, "Class Weights (Tán xạ tự nhiên)")
    
    # --- Phương thức 3: SMOTE K-Fold ---
    print("--- 3. SMOTE Pipeline ---")
    pipe_smote = ImbPipeline([
        ('winsor', winsorizer),
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    res_smote = evaluate_method(X, y, pipe_smote, "SMOTE (Nội suy Không gian)")

    results_dict = {
        "Baseline": res_base,
        "Class Weights": res_cw,
        "SMOTE Pipeline": res_smote
    }
    
    # Xác định người chiến thắng
    best_f1 = max(results_dict.values(), key=lambda x: x['f1'])['method']
    best_recall = max(results_dict.values(), key=lambda x: x['recall'])['method']
    best_methods = {
        "best_f1": best_f1,
        "best_recall": best_recall
    }

    final_output = {
        "imbalance_analysis": imbalance_stats,
        "methods_comparison": results_dict,
        "best_methods": best_methods
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/imbalance_analysis.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print("\n[OK] Phân tích Data Leakage-Free hoàn thành. Đã xuất file JSON.")

def main():
    run_imbalanced_analysis()

if __name__ == "__main__":
    main()
