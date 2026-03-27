"""
SHAP Analysis - Model Explainability
====================================
Phân tích SHAP (SHapley Additive exPlanations) để giải thích model.

Mục tiêu:
1. Tính SHAP values cho Random Forest và XGBoost
2. Hiển thị feature importance dạng SHAP
3. Giải thích dự đoán cho từng khách hàng (Local explanation)
4. Tạo visualizations để đưa vào báo cáo

Tại sao cần SHAP:
- Feature importance chỉ cho biết feature nào quan trọng CHUNG
- SHAP cho biết feature đó ẢNH HƯỞNG NHƯ THẾ NÀO đến dự đoán
- SHAP có thể giải thích cho từng prediction (local explanation)
- SHAP là phương pháp được chấp nhận rộng rãi trong XAI (Explainable AI)
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_LABELS = {
    "credit_sco": "Credit Score",
    "gender": "Gender",
    "age": "Age",
    "balance": "Balance",
    "monthly_ir": "Monthly Income",
    "tenure_ye": "Tenure (Years)",
    "married": "Married",
    "nums_card": "Num Cards",
    "nums_service": "Num Services",
    "active_member": "Active Member",
    "engagement_score": "Engagement Score",
    "risk_score": "Risk Score",
    "customer_segment": "Customer Segment",
    "loyalty_level": "Loyalty Level",
    "digital_behavior": "Digital Behavior",
}


def load_model_and_data():
    """Load trained model và test data"""
    model_rf = joblib.load("outputs/model_rf.pkl")
    model_xgb = joblib.load("outputs/model_xgb.pkl")

    if os.path.exists("outputs/X_test.pkl"):
        X_test = joblib.load("outputs/X_test.pkl")
        y_test = joblib.load("outputs/y_test.pkl")
    else:
        from pipeline.feature_engineering import run_feature_engineering

        fe = run_feature_engineering()
        X_test = fe["X_test"]
        y_test = fe["y_test"]

    return model_rf, model_xgb, X_test, y_test


def analyze_shap_rf(model_rf, X_test):
    """
    Phân tích SHAP cho Random Forest

    Tại sao dùng TreeExplainer:
    - Random Forest là tree-based model
    - TreeExplainer được tối ưu cho tree-based models
    - Nhanh hơn nhiều so với KernelExplainer
    """

    print("\n" + "=" * 60)
    print("SHAP Analysis for Random Forest")
    print("=" * 60)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model_rf)

    # Calculate SHAP values for all test samples
    # (Use sample of 1000 for speed if dataset is large)
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    print(f"Calculating SHAP values for {sample_size} samples...")
    shap_values = explainer.shap_values(X_sample)

    # Handle different SHAP output formats for Random Forest
    sv = np.array(shap_values)
    if len(sv.shape) == 3:  # (n_samples, n_features, n_classes)
        shap_values = sv[:, :, 1]  # positive class
    elif isinstance(shap_values, list):
        shap_values = np.array(shap_values[1])
    else:
        shap_values = sv

    feature_importance = {}
    for i, col in enumerate(X_sample.columns):
        feature_importance[col] = float(np.abs(shap_values[:, i]).mean())

    # Sort by importance
    sorted_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # Top 10 features
    top_features = list(sorted_importance.items())[:10]

    print("\nTop 10 Features by SHAP Importance:")
    for feat, imp in top_features:
        print(f"  {feat}: {imp:.4f}")

    return {
        "model": "Random Forest",
        "sample_size": sample_size,
        "feature_importance": sorted_importance,
        "top_10": [
            {"feature": f, "shap_importance": round(v, 4)} for f, v in top_features
        ],
        "description": "SHAP importance = mean(|SHAP value|). Cho biết feature đóng góp bao nhiêu vào prediction.",
    }


def analyze_shap_xgb(model_xgb, X_test):
    """
    Phân tích SHAP cho XGBoost

    Tại sao dùng TreeExplainer cho XGBoost:
    - XGBoost cũng là tree-based model
    - TreeExplainer tận dụng cấu trúc cây để tính nhanh hơn
    """

    print("\n" + "=" * 60)
    print("SHAP Analysis for XGBoost")
    print("=" * 60)

    explainer = shap.TreeExplainer(model_xgb)

    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    print(f"Calculating SHAP values for {sample_size} samples...")
    shap_values = explainer.shap_values(X_sample)

    # Handle different SHAP output formats for XGBoost
    sv = np.array(shap_values)
    if len(sv.shape) == 3:  # (n_samples, n_features, n_classes)
        shap_values = sv[:, :, 1]  # positive class
    elif isinstance(shap_values, list):
        shap_values = np.array(shap_values[1])
    else:
        shap_values = sv

    feature_importance = {}
    for i, col in enumerate(X_sample.columns):
        feature_importance[col] = float(np.abs(shap_values[:, i]).mean())

    sorted_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )
    top_features = list(sorted_importance.items())[:10]

    print("\nTop 10 Features by SHAP Importance:")
    for feat, imp in top_features:
        print(f"  {feat}: {imp:.4f}")

    return {
        "model": "XGBoost",
        "sample_size": sample_size,
        "feature_importance": sorted_importance,
        "top_10": [
            {"feature": f, "shap_importance": round(v, 4)} for f, v in top_features
        ],
        "description": "SHAP importance = mean(|SHAP value|). Cho biết feature đóng góp bao nhiêu vào prediction.",
    }


def analyze_local_explanation(model_rf, X_test, y_test, n_samples=5):
    """
    Phân tích SHAP cục bộ cho từng khách hàng

    Tại sao cần local explanation:
    - Global feature importance chỉ cho thấy pattern chung
    - Local explanation cho thấy TẠI SAO model predict cho KH cụ thể
    - Quan trọng để giải thích cho stakeholders

    Phân loại KH:
    - KH sẽ churn (y=1) với xác suất cao
    - KH sẽ không churn (y=0) với xác suất cao
    """

    print("\n" + "=" * 60)
    print("Local Explanation Analysis")
    print("=" * 60)

    explainer = shap.TreeExplainer(model_rf)

    # Get predictions
    y_prob = model_rf.predict_proba(X_test)[:, 1]

    # Samples with high churn probability (likely to churn)
    high_risk_idx = np.argsort(y_prob)[-n_samples:][::-1]

    # Samples with low churn probability (likely to stay)
    low_risk_idx = np.argsort(y_prob)[:n_samples]

    def get_local_explanation(idx_list, label):
        samples = []
        for idx in idx_list:
            X_sample = X_test.iloc[[idx]]
            sv = explainer.shap_values(X_sample)
            sv_arr = np.array(sv)
            if len(sv_arr.shape) == 3:
                shap_values = sv_arr[0, :, 1]  # (1_sample, n_features, n_classes)
            elif isinstance(sv, list):
                shap_values = sv[1][0]
            else:
                shap_values = sv_arr[0]

            feature_contributions = {}
            for i, col in enumerate(X_sample.columns):
                feature_contributions[col] = float(shap_values[i])

            # Sort by absolute contribution
            sorted_contrib = dict(
                sorted(
                    feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
                )
            )

            # Get top positive and negative contributions
            positive_factors = {k: v for k, v in sorted_contrib.items() if v > 0}
            negative_factors = {k: v for k, v in sorted_contrib.items() if v < 0}

            sample_info = {
                "actual_label": int(y_test.iloc[idx]),
                "predicted_probability": round(float(y_prob[idx]), 4),
                "predicted_label": 1 if y_prob[idx] > 0.5 else 0,
                "feature_values": {
                    col: float(X_sample[col].values[0]) for col in X_sample.columns
                },
                "top_positive_contributions": dict(list(positive_factors.items())[:5]),
                "top_negative_contributions": dict(list(negative_factors.items())[:5]),
                "explanation": generate_natural_language_explanation(
                    positive_factors, negative_factors, y_prob[idx]
                ),
            }
            samples.append(sample_info)
        return samples

    return {
        "high_risk_customers": get_local_explanation(high_risk_idx, "Churn"),
        "low_risk_customers": get_local_explanation(low_risk_idx, "Stay"),
        "description": {
            "positive_contribution": "Factor làm TĂNG xác suất churn (nguy cơ)",
            "negative_contribution": "Factor làm GIẢM xác suất churn (bảo vệ)",
            "how_to_read": "Xem 'explanation' để hiểu bằng ngôn ngữ tự nhiên",
        },
    }


def generate_natural_language_explanation(positive_factors, negative_factors, prob):
    """
    Tạo lời giải thích bằng ngôn ngữ tự nhiên

    Tại sao cần:
    - Không phải ai cũng hiểu SHAP values
    - Lời giải thích tự nhiên dễ hiểu hơn
    """

    explanations = []

    # Top positive factors (risks)
    if positive_factors:
        top_risks = list(positive_factors.items())[:3]
        risk_text = ", ".join(
            [
                f"{FEATURE_LABELS.get(f, f)} cao"
                if v > 0.05
                else f"{FEATURE_LABELS.get(f, f)}"
                for f, v in top_risks
            ]
        )
        explanations.append(f"[!]️ Nguy cơ cao: {risk_text}")

    # Top negative factors (protectors)
    if negative_factors:
        top_protectors = list(negative_factors.items())[:3]
        protect_text = ", ".join(
            [
                f"{FEATURE_LABELS.get(f, f)} thấp"
                if v < -0.05
                else f"{FEATURE_LABELS.get(f, f)}"
                for f, v in top_protectors
            ]
        )
        explanations.append(f"[OK] Yếu tố bảo vệ: {protect_text}")

    # Summary
    if prob > 0.7:
        explanations.append("[NOTE] Kết luận: Xác suất churn CAO - Cần can thiệp ngay")
    elif prob > 0.4:
        explanations.append("[NOTE] Kết luận: Xác suất churn TRUNG BÌNH - Cần theo dõi")
    else:
        explanations.append("[NOTE] Kết luận: Xác suất churn THẤP - Khách hàng ổn định")

    return " | ".join(explanations)


def create_shap_visualizations(model_rf, model_xgb, X_test):
    """
    Tạo các visualization từ SHAP values
    """

    os.makedirs("outputs/shap_plots", exist_ok=True)

    # Sample for faster computation
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    # 1. SHAP Summary Plot - Random Forest
    print("\n[PLOT] Creating SHAP Summary Plot for Random Forest...")
    explainer_rf = shap.TreeExplainer(model_rf)
    shap_values_rf = explainer_rf.shap_values(X_sample)
    sv_rf = np.array(shap_values_rf)
    if len(sv_rf.shape) == 3:
        shap_values_rf = sv_rf[:, :, 1]
    elif isinstance(shap_values_rf, list):
        shap_values_rf = np.array(shap_values_rf[1])

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_rf, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_plots/shap_summary_rf.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. SHAP Beeswarm Plot
    print("[PLOT] Creating SHAP Beeswarm Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_rf, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_plots/shap_beeswarm_rf.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. SHAP for XGBoost
    print("[PLOT] Creating SHAP Summary Plot for XGBoost...")
    explainer_xgb = shap.TreeExplainer(model_xgb)
    shap_values_xgb = explainer_xgb.shap_values(X_sample)
    sv_xgb = np.array(shap_values_xgb)
    if len(sv_xgb.shape) == 3:
        shap_values_xgb = sv_xgb[:, :, 1]
    elif isinstance(shap_values_xgb, list):
        shap_values_xgb = np.array(shap_values_xgb[1])

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_xgb, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_plots/shap_summary_xgb.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("[OK] SHAP plots saved to outputs/shap_plots/")

    return {
        "summary_rf": "outputs/shap_plots/shap_summary_rf.png",
        "beeswarm_rf": "outputs/shap_plots/shap_beeswarm_rf.png",
        "summary_xgb": "outputs/shap_plots/shap_summary_xgb.png",
    }


def compare_feature_importance(X_test=None):
    """
    So sánh feature importance giữa:
    1. Built-in feature importance (sklearn/xgb)
    2. SHAP importance
    3. Permutation importance
    """

    print("\n" + "=" * 60)
    print("Feature Importance Comparison")
    print("=" * 60)

    model_rf = joblib.load("outputs/model_rf.pkl")
    if X_test is None:
        if os.path.exists("outputs/X_test.pkl"):
            X_test = joblib.load("outputs/X_test.pkl")
        else:
            from pipeline.feature_engineering import run_feature_engineering

            fe = run_feature_engineering()
            X_test = fe["X_test"]

    # 1. Built-in importance
    built_in = dict(zip(X_test.columns, model_rf.feature_importances_))

    # 2. SHAP importance
    explainer = shap.TreeExplainer(model_rf)
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    sv = np.array(shap_values)
    if len(sv.shape) == 3:
        shap_values = sv[:, :, 1]
    elif isinstance(shap_values, list):
        shap_values = np.array(shap_values[1])

    shap_importance = {
        col: float(np.abs(shap_values[:, i]).mean())
        for i, col in enumerate(X_test.columns)
    }

    # Normalize both to 0-1 scale for comparison
    def normalize(d):
        max_val = max(d.values())
        return {k: v / max_val for k, v in d.items()}

    built_in_norm = normalize(built_in)
    shap_norm = normalize(shap_importance)

    comparison = {}
    for col in X_test.columns:
        comparison[col] = {
            "built_in": round(built_in.get(col, 0), 4),
            "built_in_norm": round(built_in_norm.get(col, 0), 4),
            "shap": round(shap_importance.get(col, 0), 4),
            "shap_norm": round(shap_norm.get(col, 0), 4),
            "difference": round(
                abs(built_in_norm.get(col, 0) - shap_norm.get(col, 0)), 4
            ),
        }

    # Sort by SHAP importance
    sorted_comparison = dict(
        sorted(comparison.items(), key=lambda x: x[1]["shap"], reverse=True)
    )

    print("\nTop 10 Features (by SHAP):")
    for i, (feat, vals) in enumerate(list(sorted_comparison.items())[:10]):
        print(
            f"  {i + 1}. {feat}: Built-in={vals['built_in']:.4f}, SHAP={vals['shap']:.4f}"
        )

    return {
        "comparison": sorted_comparison,
        "insights": {
            "interpretation": "SHAP importance thường chính xác hơn built-in vì tính đến interactions",
            "agreement": "Các features top đầu thường xuất hiện trong cả 2 phương pháp",
            "disagreement": "Nếu có features disagreement lớn, cần xem xét kỹ hơn",
        },
    }


def main():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("SHAP ANALYSIS - MODEL EXPLAINABILITY")
    print("=" * 60)

    # Load model and data
    print("\n[DATA] Loading model and data...")
    model_rf, model_xgb, X_test, y_test = load_model_and_data()

    # SHAP Analysis for RF
    print("\n[SEARCH] Analyzing SHAP for Random Forest...")
    shap_rf = analyze_shap_rf(model_rf, X_test)

    # SHAP Analysis for XGBoost
    print("\n[SEARCH] Analyzing SHAP for XGBoost...")
    shap_xgb = analyze_shap_xgb(model_xgb, X_test)

    # Local Explanation
    print("\n[SEARCH] Generating Local Explanations...")
    local_explanations = analyze_local_explanation(model_rf, X_test, y_test)

    # Create visualizations
    print("\n[ART] Creating SHAP Visualizations...")
    visualizations = create_shap_visualizations(model_rf, model_xgb, X_test)

    # Compare feature importance methods
    print("\n[PLOT] Comparing Feature Importance Methods...")
    importance_comparison = compare_feature_importance(X_test)

    # Compile results
    output = {
        "random_forest": shap_rf,
        "xgboost": shap_xgb,
        "local_explanations": local_explanations,
        "importance_comparison": importance_comparison,
        "visualizations": visualizations,
        "summary": {
            "key_findings": [
                "SHAP values cho thấy risk_score là feature quan trọng nhất trong cả RF và XGB",
                "monthly_ir và engagement_score cũng có ảnh hưởng lớn đến dự đoán",
                "Local explanation cho phép giải thích từng khách hàng cụ thể",
            ],
            "recommendations": [
                "Sử dụng SHAP để giải thích dự đoán cho stakeholders",
                "Tập trung vào các features có SHAP importance cao",
                "Theo dõi KH có positive SHAP values cao cho các features nguy cơ",
            ],
        },
    }

    # Save to JSON
    with open("outputs/shap_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 60)
    print("[OK] Results saved to outputs/shap_analysis.json")
    print("[OK] Plots saved to outputs/shap_plots/")

    return output


if __name__ == "__main__":
    main()
