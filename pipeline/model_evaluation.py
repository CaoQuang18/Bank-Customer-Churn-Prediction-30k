import sys
import pandas as pd
import numpy as np
import joblib
import json
import os

# FIX: Windows cp1252 không hỗ trợ tiếng Việt → force UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from pipeline.feature_engineering import run_feature_engineering
from pipeline.model_training import (
    tune_and_train_models,
    train_kmeans,
)
from pipeline.eda import run_eda

# ══════════════════════════════════════════════════════════════════════════════
# ĐÁNH GIÁ & SO SÁNH MÔ HÌNH (ZERO LEAKAGE EDITION)
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(y_true, y_proba):
    """
    Tìm threshold tối ưu dựa trên F1-score. 
    QUAN TRỌNG: Hàm này nên được gọi trên tập VALIDATION, không phải TEST.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return best_threshold, best_f1, precisions.tolist(), recalls.tolist(), thresholds.tolist()

def evaluate_model(name, model, X_val, X_test, y_val, y_test):
    """
    Đánh giá mô hình:
    1. Tìm threshold tối ưu trên tập Validation.
    2. Kiểm chứng kết quả trên tập Test độc lập.
    """
    # 1. Tìm threshold trên Validation set
    y_proba_val = model.predict_proba(X_val)[:, 1]
    best_thresh, _, _, _, _ = find_optimal_threshold(y_val, y_proba_val)
    
    # 2. Đánh giá trên Test set với threshold vừa tìm được
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_opt = (y_proba_test >= best_thresh).astype(int)
    y_pred_def = (y_proba_test >= 0.5).astype(int)

    # ROC curve & PR curve trên Test set
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_test)

    result = {
        "name": name,
        "default_threshold": {
            "threshold": 0.5,
            "accuracy": round(accuracy_score(y_test, y_pred_def), 4),
            "precision": round(precision_score(y_test, y_pred_def, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_def, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred_def, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba_test), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_def).tolist(),
        },
        "optimal_threshold": {
            "threshold": round(best_thresh, 4),
            "accuracy": round(accuracy_score(y_test, y_pred_opt), 4),
            "precision": round(precision_score(y_test, y_pred_opt, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_opt, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred_opt, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba_test), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_opt).tolist(),
        },
        "roc": {"fpr": fpr[::10].tolist(), "tpr": tpr[::10].tolist()},
        "pr_curve": {
            "precisions": precisions[::10].tolist(),
            "recalls": recalls[::10].tolist(),
            "ap_score": round(average_precision_score(y_test, y_proba_test), 4),
        },
    }
    return result

def get_feature_importance(model_rf, model_xgb, model_lr, features_list):
    """Tổng hợp Feature Importance từ 3 mô hình."""
    preprocess = None
    if hasattr(model_lr, "named_steps"):
        preprocess = model_lr.named_steps.get("preprocess")
    if preprocess is None and hasattr(model_rf, "named_steps"):
        preprocess = model_rf.named_steps.get("preprocess")
    feature_names = []
    if preprocess is not None:
        try:
            feature_names = preprocess.get_feature_names_out().tolist()
        except Exception:
            feature_names = []
    if not feature_names:
        feature_names = list(features_list)

    rf_step = model_rf.named_steps['rf'] if hasattr(model_rf, "named_steps") and 'rf' in model_rf.named_steps else model_rf
    xgb_step = model_xgb.named_steps['xgb'] if hasattr(model_xgb, "named_steps") and 'xgb' in model_xgb.named_steps else model_xgb
    lr_step = model_lr.named_steps['lr'] if hasattr(model_lr, "named_steps") and 'lr' in model_lr.named_steps else model_lr

    rf_imp = dict(zip(feature_names, rf_step.feature_importances_))
    xgb_imp = dict(zip(feature_names, xgb_step.feature_importances_))
    lr_coef = dict(zip(feature_names, np.abs(lr_step.coef_[0])))
    
    lr_max = max(lr_coef.values()) if lr_coef.values() else 1
    lr_norm = {k: round(v / lr_max, 4) for k, v in lr_coef.items()}

    if not feature_names:
        n = 0
        try:
            n = len(rf_step.feature_importances_)
        except Exception:
            n = 0
        feature_names = [f"f{i}" for i in range(n)]
    combined = {f: round((rf_imp.get(f, 0) + xgb_imp.get(f, 0) + lr_norm.get(f, 0)) / 3, 4) for f in feature_names}
    combined_sorted = dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
    top3 = list(combined_sorted.keys())[:3]
    top_a = top3[0] if len(top3) > 0 else "—"
    top_b = top3[1] if len(top3) > 1 else "—"

    return {
        "combined": combined_sorted,
        "top3": top3,
        "rf": dict(sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)),
        "xgb": dict(sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)),
        "lr": dict(sorted(lr_norm.items(), key=lambda x: x[1], reverse=True)),
        "insight": {
            "so_lieu": f"Top 3 đặc trưng quan trọng nhất: {', '.join(top3) if top3 else '—'}",
            "nguyen_nhan": "Các đặc trưng này xuất hiện nhất quán quan trọng trên cả 3 mô hình",
            "ly_do": "Chúng phản ánh hành vi thực tế của khách hàng có nguy cơ rời bỏ cao nhất",
            "huong_xu_ly": f"Tập trung monitoring và can thiệp sớm dựa trên {top_a} và {top_b}",
        }
    }

def compare_models(results):
    comparison = []
    for name, res in results.items():
        opt = res["optimal_threshold"]
        comparison.append({
            "model": name,
            "accuracy": opt["accuracy"],
            "precision": opt["precision"],
            "recall": opt["recall"],
            "f1": opt["f1"],
            "roc_auc": opt["roc_auc"],
            "threshold": opt["threshold"],
        })

    df_cmp = pd.DataFrame(comparison).sort_values("roc_auc", ascending=False)
    best_model = df_cmp.iloc[0]["model"]
    
    insights = {
        "best_overall": best_model,
        "so_lieu": f"{best_model} đạt AUC={df_cmp.iloc[0]['roc_auc']} (kiểm chứng trên Test set độc lập)",
        # FIX I1: Giải thích đúng theo loại model, tránh mô tả sai ngữ nghĩa
        "tai_sao_nhu_vay": (
            f"{best_model} vượt trội nhờ regularization hiệu quả và "
            "dataset có tương quan tuyến tính rõ ràng (risk_score r=0.433, monthly_ir r=-0.301), "
            "phù hợp với bản chất phân loại linear của Logistic Regression."
            if "Logistic" in best_model else
            f"{best_model} vượt trội nhờ khả năng học các pattern phi tuyến và "
            "tương tác phức tạp giữa các đặc trưng mà mô hình ensemble xử lý hiệu quả."
        ),
    }
    return df_cmp.to_dict(orient="records"), insights

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("powerbi", exist_ok=True)

    print("... Running Feature Engineering (Zero Leakage)...")
    fe = run_feature_engineering()

    X_val, X_test, y_val, y_test = train_test_split(
        fe["X_test"], fe["y_test"], test_size=0.5, random_state=42, stratify=fe["y_test"]
    )

    print("... Training and Tuning models with ImbPipeline...")
    models, best_params, cv_results = tune_and_train_models(fe["X_train"], fe["y_train"], fe["num_features"], fe["cat_features"])
    
    print("... Evaluating models (Threshold from Val, Metrics from Test)...")
    results = {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        results[name] = evaluate_model(name, models[name], X_val, X_test, y_val, y_test)

    feat_imp = get_feature_importance(models['Random Forest'], models['XGBoost'], models['Logistic Regression'], [])
    comparison, compare_insights = compare_models(results)
    
    # Save artifacts
    joblib.dump(models['Logistic Regression'], "outputs/model_lr.pkl")
    joblib.dump(models['Random Forest'], "outputs/model_rf.pkl")
    joblib.dump(models['XGBoost'], "outputs/model_xgb.pkl")
    
    from pipeline.model_training import train_kmeans
    model_kmeans, scaler_km, df_clustered, cluster_profiles, elbow_data, cluster_strategies = train_kmeans(fe["df_clean"])
    
    # LƯU KẾT QUẢ PHÂN CỤM ĐỂ APP.PY DÙNG
    joblib.dump(model_kmeans, "outputs/model_kmeans.pkl")
    joblib.dump(scaler_km, "outputs/scaler_kmeans.pkl")
    
    # Lưu CSV cho Telesales
    os.makedirs("data/processed", exist_ok=True)
    df_clustered.to_csv("data/processed/clustered_data.csv", index=False)
    
    # Lưu JSONs bổ trợ cho Dashboard Clusters
    with open("outputs/cluster_profiles.json", "w", encoding="utf-8") as f:
        json.dump(cluster_profiles.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    
    with open("outputs/cluster_strategies.json", "w", encoding="utf-8") as f:
        json.dump(cluster_strategies, f, ensure_ascii=False, indent=2)
        
    with open("outputs/elbow_data.json", "w", encoding="utf-8") as f:
        json.dump(elbow_data, f, ensure_ascii=False, indent=2)

    with open("outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open("outputs/comparison.json", "w", encoding="utf-8") as f:
        json.dump({"table": comparison, "insights": compare_insights}, f, ensure_ascii=False, indent=2)

    print("\n[OK] Pipeline completed with Zero Leakage Strategy.")

if __name__ == "__main__":
    main()
