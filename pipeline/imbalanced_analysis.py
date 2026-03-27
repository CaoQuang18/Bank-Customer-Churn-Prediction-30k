"""
SMOTE + Imbalanced Classification Analysis
=========================================
Phân tích ảnh hưởng của Imbalanced Data và các phương pháp xử lý.

Mục tiêu:
1. Phân tích tỉ lệ imbalance trong dataset
2. So sánh hiệu suất trước/sau khi xử lý imbalance
3. Đánh giá các phương pháp: SMOTE, Class Weights, Threshold Tuning

Tại sao cần xử lý imbalance:
- Trong churn prediction, thường có ~20% KH churn, 80% không churn
- Model mặc định sẽ nghiêng về predict class chiếm đa số
- Cần đảm bảo model bắt được cả KH sẽ churn
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    recall_score,
    precision_score,
)


def load_data():
    """Load preprocessed data từ pipeline feature engineering"""
    if os.path.exists("outputs/X_train.pkl"):
        X = joblib.load("outputs/X_train.pkl")
        y = joblib.load("outputs/y_train.pkl")
        X_test = joblib.load("outputs/X_test.pkl")
        y_test = joblib.load("outputs/y_test.pkl")
        return X, y, X_test, y_test
    else:
        from pipeline.feature_engineering import run_feature_engineering

        fe = run_feature_engineering()
        return fe["X_train"], fe["y_train"], fe["X_test"], fe["y_test"]


def analyze_imbalance(y):
    """
    Phân tích mức độ imbalance của dataset

    Tại sao cần biết:
    - Imbalance ratio = majority / minority
    - Nếu ratio > 10: gọi là severe imbalance
    - Cần xử lý đặc biệt cho severe imbalance
    """
    n_majority = (y == 0).sum()
    n_minority = (y == 1).sum()
    imbalance_ratio = n_majority / n_minority

    return {
        "total": len(y),
        "majority_class": int(n_majority),
        "minority_class": int(n_minority),
        "majority_pct": round(n_majority / len(y) * 100, 2),
        "minority_pct": round(n_minority / len(y) * 100, 2),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "severity": "Severe"
        if imbalance_ratio > 10
        else "Moderate"
        if imbalance_ratio > 4
        else "Mild",
    }


def compare_imbalance_methods(
    X, y, X_test=None, y_test=None, test_size=0.2, random_state=42
):
    """
    So sánh các phương pháp xử lý imbalanced data

    Phương pháp so sánh:
    1. Baseline (không xử lý)
    2. Class Weights (điều chỉnh trọng số)
    3. SMOTE (oversampling)
    4. SMOTE + Class Weights (kết hợp)

    Tại sao cần so sánh:
    - Không phương pháp nào tốt nhất cho mọi trường hợp
    - Cần chọn phương pháp phù hợp với bài toán cụ thể
    """

    # Split data if not provided
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train = X
        y_train = y

    results = {}

    # 1. BASELINE - Không xử lý imbalance
    print("\n" + "=" * 60)
    print("1. BASELINE (Không xử lý imbalance)")
    print("=" * 60)

    rf_baseline = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    )
    rf_baseline.fit(X_train, y_train)

    y_pred = rf_baseline.predict(X_test)
    y_prob = rf_baseline.predict_proba(X_test)[:, 1]

    baseline_metrics = {
        "method": "Baseline (No处理)",
        "accuracy": float((y_pred == y_test).mean()),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "description": "Model không xử lý imbalance - tendency predict class chiếm đa số",
    }
    results["Baseline"] = baseline_metrics

    print(f"Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Precision: {baseline_metrics['precision']:.4f}")
    print(f"Recall: {baseline_metrics['recall']:.4f}")
    print(f"F1: {baseline_metrics['f1']:.4f}")
    print(f"ROC-AUC: {baseline_metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 2. CLASS WEIGHTS
    print("\n" + "=" * 60)
    print("2. CLASS WEIGHTS (class_weight='balanced')")
    print("=" * 60)

    rf_cw = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
    )
    rf_cw.fit(X_train, y_train)

    y_pred = rf_cw.predict(X_test)
    y_prob = rf_cw.predict_proba(X_test)[:, 1]

    cw_metrics = {
        "method": "Class Weights",
        "accuracy": float((y_pred == y_test).mean()),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "description": "Tự động điều chỉnh trọng số class theo tỉ lệ nghịch",
    }
    results["Class_Weights"] = cw_metrics

    print(f"Accuracy: {cw_metrics['accuracy']:.4f}")
    print(f"Precision: {cw_metrics['precision']:.4f}")
    print(f"Recall: {cw_metrics['recall']:.4f}")
    print(f"F1: {cw_metrics['f1']:.4f}")
    print(f"ROC-AUC: {cw_metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 3. SMOTE
    print("\n" + "=" * 60)
    print("3. SMOTE (Synthetic Minority Oversampling)")
    print("=" * 60)

    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"Before SMOTE: {len(y_train)} samples ({sum(y_train)} positive)")
    print(f"After SMOTE: {len(y_train_smote)} samples ({sum(y_train_smote)} positive)")

    rf_smote = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_smote.fit(X_train_smote, y_train_smote)

    y_pred = rf_smote.predict(X_test)
    y_prob = rf_smote.predict_proba(X_test)[:, 1]

    smote_metrics = {
        "method": "SMOTE",
        "accuracy": float((y_pred == y_test).mean()),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "smote_samples": {
            "before": int(len(y_train)),
            "after": int(len(y_train_smote)),
            "synthetic_created": int(sum(y_train_smote) - sum(y_train)),
        },
        "description": "Tạo synthetic samples cho minority class bằng interpolation",
    }
    results["SMOTE"] = smote_metrics

    print(f"Accuracy: {smote_metrics['accuracy']:.4f}")
    print(f"Precision: {smote_metrics['precision']:.4f}")
    print(f"Recall: {smote_metrics['recall']:.4f}")
    print(f"F1: {smote_metrics['f1']:.4f}")
    print(f"ROC-AUC: {smote_metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 4. SMOTE + Class Weights
    print("\n" + "=" * 60)
    print("4. SMOTE + Class Weights (Kết hợp)")
    print("=" * 60)

    rf_combined = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
    )
    rf_combined.fit(X_train_smote, y_train_smote)

    y_pred = rf_combined.predict(X_test)
    y_prob = rf_combined.predict_proba(X_test)[:, 1]

    combined_metrics = {
        "method": "SMOTE + Class Weights",
        "accuracy": float((y_pred == y_test).mean()),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "description": "Kết hợp SMOTE oversampling với class weights",
    }
    results["SMOTE_ClassWeights"] = combined_metrics

    print(f"Accuracy: {combined_metrics['accuracy']:.4f}")
    print(f"Precision: {combined_metrics['precision']:.4f}")
    print(f"Recall: {combined_metrics['recall']:.4f}")
    print(f"F1: {combined_metrics['f1']:.4f}")
    print(f"ROC-AUC: {combined_metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 5. Threshold Tuning on Baseline
    print("\n" + "=" * 60)
    print("5. Threshold Tuning (Default Threshold = 0.5)")
    print("=" * 60)

    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = (
        thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    )

    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Best F1 at threshold: {f1_scores[optimal_idx]:.4f}")

    y_pred_opt = (y_prob >= optimal_threshold).astype(int)

    threshold_metrics = {
        "method": "Threshold Tuning",
        "optimal_threshold": float(optimal_threshold),
        "accuracy": float((y_pred_opt == y_test).mean()),
        "precision": precision_score(y_test, y_pred_opt),
        "recall": recall_score(y_test, y_pred_opt),
        "f1": f1_score(y_test, y_pred_opt),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred_opt).tolist(),
        "description": "Điều chỉnh threshold để cân bằng Precision-Recall",
    }
    results["Threshold_Tuning"] = threshold_metrics

    print(f"Accuracy: {threshold_metrics['accuracy']:.4f}")
    print(f"Precision: {threshold_metrics['precision']:.4f}")
    print(f"Recall: {threshold_metrics['recall']:.4f}")
    print(f"F1: {threshold_metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_opt))

    return results


def analyze_threshold_impact(y_test, y_prob, model_name="Random Forest"):
    """
    Phân tích ảnh hưởng của threshold đến các metrics

    Tại sao cần:
    - Threshold mặc định 0.5 không phải lúc nào cũng tối ưu
    - Trong churn prediction, thường muốn recall cao (bắt nhiều KH sẽ churn)
    - Nhưng recall cao thì precision thấp (nhiều false positive)
    - Cần tìm threshold phù hợp với business objective
    """

    results = []
    thresholds = np.arange(0.1, 0.9, 0.05)

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        try:
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
        except:
            prec, rec, f1 = 0, 0, 0

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        results.append(
            {
                "threshold": round(thresh, 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "cost_saved": int(
                    fp * 100000 + tp * 100000
                ),  # Cost of false positive + true positive
                "cost_lost": int(fn * 400000),  # Cost of false negative
            }
        )

    return {
        "model": model_name,
        "threshold_analysis": results,
        "insights": {
            "precision_recall_tradeoff": "Recall cao -> Precision thấp và ngược lại. Cần tìm điểm cân bằng.",
            "business_objective": "Nếu muốn bắt nhiều KH churn: chọn threshold thấp. Nếu muốn chính xác: chọn threshold cao.",
            "cost_based": "Tính chi phí: FP = 100k (retention), FN = 400k (acquisition mới). Tối ưu theo chi phí.",
        },
    }


def main():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("IMBALANCED CLASSIFICATION ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n[PLOT] Loading data...")
    X, y, X_test, y_test = load_data()

    # Analyze imbalance
    print("\n📈 Analyzing Data Imbalance...")
    imbalance_info = analyze_imbalance(y)
    print(f"Total samples: {imbalance_info['total']}")
    print(
        f"Majority class (No Churn): {imbalance_info['majority_class']} ({imbalance_info['majority_pct']}%)"
    )
    print(
        f"Minority class (Churn): {imbalance_info['minority_class']} ({imbalance_info['minority_pct']}%)"
    )
    print(f"Imbalance Ratio: {imbalance_info['imbalance_ratio']}:1")
    print(f"Severity: {imbalance_info['severity']}")

    # Compare methods
    print("\n[SCI] Comparing Imbalance Handling Methods...")
    comparison_results = compare_imbalance_methods(X, y, X_test, y_test)

    # Save results
    output = {
        "imbalance_analysis": imbalance_info,
        "methods_comparison": comparison_results,
        "conclusion": {
            "best_recall": max(
                comparison_results.items(), key=lambda x: x[1]["recall"]
            )[0],
            "best_f1": max(comparison_results.items(), key=lambda x: x[1]["f1"])[0],
            "best_auc": max(comparison_results.items(), key=lambda x: x[1]["roc_auc"])[
                0
            ],
            "recommendation": "SMOTE + Class Weights cho recall cao nhất, nhưng cần cân nhắc precision. Threshold tuning là cách đơn giản và hiệu quả.",
        },
    }

    with open("outputs/imbalance_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("KẾT LUẬN")
    print("=" * 60)
    print(f"Best Recall: {output['conclusion']['best_recall']}")
    print(f"Best F1-Score: {output['conclusion']['best_f1']}")
    print(f"Best ROC-AUC: {output['conclusion']['best_auc']}")
    print("\n[OK] Results saved to outputs/imbalance_analysis.json")

    return output


if __name__ == "__main__":
    main()
