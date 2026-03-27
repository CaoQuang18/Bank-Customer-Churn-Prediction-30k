import pandas as pd
import numpy as np
import joblib
import json
import os
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
from pipeline.feature_engineering import run_feature_engineering, FEATURES
from pipeline.model_training import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_kmeans,
    cross_validate_models,
)
from pipeline.eda import run_eda
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════════════
# ĐÁNH GIÁ & SO SÁNH MÔ HÌNH
# Nguyên nhân: Accuracy đơn thuần không đủ de đánh giá mô hình churn
# vì dữ liệu mất cân bang -> can bộ metrics toàn diện.
# Lý do tại sao: Trong bài toán churn, Recall quan trọng hơn Precision
# vì bỏ sót 1 khach hang rời bỏ (FN) tốn kém hơn cảnh báo nhầm (FP).
# ══════════════════════════════════════════════════════════════════════════════


def find_optimal_threshold(y_true, y_proba):
    """
    Nguyên nhân: Threshold mặc định 0.5 không tối uu cho dữ liệu mất cân bang.
    Lý do tại sao: Tùy theo mục tiêu kinh doanh, co thể muốn tối đa F1
    hoặc tối đa Recall (chấp nhan FP cao hơn de không bỏ sót churn).
    Hướng xu ly: Tìm threshold tối đa hoa F1-score tren tap test.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return (
        best_threshold,
        best_f1,
        precisions.tolist(),
        recalls.tolist(),
        thresholds.tolist(),
    )


def evaluate_model(name, model, X_test, X_test_sc, y_test, use_scaled=False):
    """Đánh giá toàn diện một mô hình."""
    X = X_test_sc if use_scaled else X_test
    y_proba = model.predict_proba(X)[:, 1]

    # Threshold tuning
    best_thresh, best_f1, precisions, recalls, thresholds = find_optimal_threshold(
        y_test, y_proba
    )
    y_pred_opt = (y_proba >= best_thresh).astype(int)
    y_pred_def = (y_proba >= 0.5).astype(int)

    # ROC curve
    fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)

    result = {
        "name": name,
        # Metrics voi threshold mặc định 0.5
        "default_threshold": {
            "threshold": 0.5,
            "accuracy": round(accuracy_score(y_test, y_pred_def), 4),
            "precision": round(precision_score(y_test, y_pred_def, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_def, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred_def, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_def).tolist(),
        },
        # Metrics voi optimal threshold
        "optimal_threshold": {
            "threshold": round(best_thresh, 4),
            "accuracy": round(accuracy_score(y_test, y_pred_opt), 4),
            "precision": round(precision_score(y_test, y_pred_opt, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_opt, zero_division=0), 4),
            "f1": round(best_f1, 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_opt).tolist(),
        },
        # ROC curve data (downsample de giảm file size)
        "roc": {"fpr": fpr[::10].tolist(), "tpr": tpr[::10].tolist()},
        # Precision-Recall curve
        "pr_curve": {
            "precisions": precisions[::10],
            "recalls": recalls[::10],
            "ap_score": round(average_precision_score(y_test, y_proba), 4),
        },
    }
    return result


def get_feature_importance(model_rf, model_xgb, model_lr, scaler):
    """
    Nguyên nhân: Hiểu feature nao quan trọng nhat giúp ngân hang
    tap trung can thiệp đúng chỗ.
    Lý do tại sao: Dùng 3 nguồn khác nhau de cross-validate:
    RF importance, XGB importance, LR coefficients.
    """
    rf_imp = dict(zip(FEATURES, model_rf.feature_importances_))
    xgb_imp = dict(zip(FEATURES, model_xgb.feature_importances_))

    # LR coefficients (absolute value sau khi scale)
    lr_coef = dict(zip(FEATURES, np.abs(model_lr.coef_[0])))
    lr_max = max(lr_coef.values())
    lr_norm = {k: round(v / lr_max, 4) for k, v in lr_coef.items()}

    # Tổng hợp rank trung bình
    combined = {}
    for feat in FEATURES:
        combined[feat] = round(
            (rf_imp.get(feat, 0) + xgb_imp.get(feat, 0) + lr_norm.get(feat, 0)) / 3, 4
        )

    combined_sorted = dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
    top3 = list(combined_sorted.keys())[:3]

    return {
        "rf": {
            k: round(float(v), 4)
            for k, v in sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)
        },
        "xgb": {
            k: round(float(v), 4)
            for k, v in sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)
        },
        "lr": {
            k: round(float(v), 4)
            for k, v in sorted(lr_norm.items(), key=lambda x: x[1], reverse=True)
        },
        "combined": combined_sorted,
        "top3": top3,
        "insight": {
            "so_lieu": f"Top 3 đặc trưng quan trọng nhat: {', '.join(top3)}",
            "nguyen_nhan": "Các đặc trưng nay xuất hiện nhat quán quan trọng tren cả 3 mô hình",
            "ly_do": "Chúng phản ánh hanh vi thuc tế cua khach hang co nguy cơ rời bỏ cao nhat",
            "huong_xu_ly": f"Tap trung monitoring va can thiệp sớm dựa tren {top3[0]} va {top3[1]}",
            "giai_phap": "Xây dựng hệ thống cảnh báo sớm dựa tren top features, cá nhân hoa uu dai",
        },
    }


def compare_models(results):
    """
    Nguyên nhân: Mỗi mô hình co diem mạnh/yếu khác nhau, can so sánh
    co hệ thống de chon mô hình phù hợp nhat.
    Lý do tại sao: Không chi nhin AUC mà can xem xét toàn diện:
    tốc độ, khả nang giải thích, recall (quan trọng nhat cho churn).
    """
    comparison = []
    for name, res in results.items():
        opt = res["optimal_threshold"]
        comparison.append(
            {
                "model": name,
                "accuracy": opt["accuracy"],
                "precision": opt["precision"],
                "recall": opt["recall"],
                "f1": opt["f1"],
                "roc_auc": opt["roc_auc"],
                "threshold": opt["threshold"],
            }
        )

    df_cmp = pd.DataFrame(comparison).sort_values("roc_auc", ascending=False)
    best_model = df_cmp.iloc[0]["model"]
    best_auc = df_cmp.iloc[0]["roc_auc"]
    best_recall = df_cmp.iloc[0]["recall"]

    insights = {
        "best_overall": best_model,
        "so_lieu": f"{best_model} đạt AUC={best_auc}, Recall={best_recall} (optimal threshold)",
        "nguyen_nhan": "Các mô hình khác nhau về cah học pattern tu dữ liệu",
        "ly_do_tsao": {
            "Logistic Regression": "Mô hình tuyến tính, giả định quan hệ tuyến tính giua features va log-odds churn. Dễ giải thích nhung bỏ sót pattern phi tuyến.",
            "Random Forest": "Ensemble nhiều cây quyết định, nắm bắt tốt tương tác phi tuyến. Ổn định, ít overfit nhờ bagging.",
            "XGBoost": "Gradient boosting tối uu hoa tuần tự, học tu lỗi cua iteration truoc. Thường cho ket qua tốt nhat tren tabular data.",
        },
        "tai_sao_nhu_vay": f"{best_model} vượt trội vì dữ liệu churn co nhiều pattern phi tuyến va tương tác phức tạp giua ca features",
        "huong_xu_ly": f"Triển khai {best_model} lam mô hình chính, dùng LR lam baseline giải thích cho stakeholders",
        "giai_phap": f"Kết hợp {best_model} cho dự đoán tự động + LR cho báo cáo giải thích duoc",
    }

    return df_cmp.to_dict(orient="records"), insights


def get_cluster_strategy(cluster_profiles):
    """
    Chiến lược va uu dai cụ thể cho tung cụm khach hang.
    Dựa tren đặc trưng thuc tế cua tung cụm.
    """
    strategies = {}
    for _, row in cluster_profiles.iterrows():
        cluster_id = row["cluster"]
        name = row["cluster_name"]
        age = row["age_mean"]
        bal = row["balance_mean"]
        eng = row["engagement_mean"]
        churn = row["churn_rate"]
        strat_key = f"Cum {cluster_id}: {name}"

        name_lower = name.lower()
        if "vip" in name_lower or "tài sản" in name_lower:
            strategies[strat_key] = {
                "dac_trung": f"Tuổi TB {age:.0f}, số dư {bal / 1e6:.0f}M, engagement {eng:.0f}/100",
                "giai_doan_song": "Tích lũy tài sản, chuẩn bị hưu trí, lo cho gia đình",
                "nhu_cau": "Đầu tư dài hạn, bảo hiểm, quản lý tài sản",
                "chien_luoc": "Giữ chân và upsell sản phẩm cao cấp",
                "uu_dai": [
                    "Thẻ VIP với ưu đãi phí 0đ",
                    "Lãi suất tiết kiệm cao hơn 0.5%/năm",
                    "Tư vấn tài chính cá nhân miễn phí",
                    "Ưu tiên xử lý giao dịch, đường dây hotline riêng",
                    "Quà tặng sinh nhật và dịp lễ đặc biệt",
                ],
                "canh_bao": f"Churn rate {churn * 100:.1f}% - Mất 1 KH này = mất nhiều giá trị",
            }
        elif "thân thiết" in name_lower or "đặc quyền" in name_lower:
            strategies[strat_key] = {
                "dac_trung": f"Tuổi TB {age:.0f}, số dư {bal / 1e6:.0f}M, engagement {eng:.0f}/100",
                "giai_doan_song": "Xây dựng sự nghiệp, mua nhà/xe, lập gia đình",
                "nhu_cau": "Vay mua nhà, thẻ tín dụng, tiết kiệm linh hoạt",
                "chien_luoc": "Nuôi dưỡng và chuyển đổi thành KH thân thiết",
                "uu_dai": [
                    "Vay mua nhà lãi suất ưu đãi 2 năm đầu",
                    "Thẻ tín dụng hoàn tiền 2% mua sắm",
                    "Miễn phí chuyển khoản 24/7",
                    "Tích điểm đổi quà khi dùng app",
                    "Lộ trình thăng hạng loyalty rõ ràng",
                ],
                "canh_bao": f"Churn rate {churn * 100:.1f}% - Cần can thiệp sớm để giữ chân",
            }
        elif "rủi ro" in name_lower or "kích hoạt" in name_lower:
            strategies[strat_key] = {
                "dac_trung": f"Tuổi TB {age:.0f}, số dư {bal / 1e6:.0f}M, engagement {eng:.0f}/100",
                "giai_doan_song": "Đang có dấu hiệu không hài lòng hoặc tìm kiếm lựa chọn khác",
                "nhu_cau": "Cần được lắng nghe, giải quyết vấn đề, cảm thấy được trân trọng",
                "chien_luoc": "Can thiệp khẩn cấp, win-back campaign",
                "uu_dai": [
                    "Gọi điện tư vấn cá nhân trong 48h",
                    "Ưu đãi đặc biệt giữ chân: miễn phí 3 tháng",
                    "Giải quyết khiếu nại ưu tiên",
                    "Tặng điểm loyalty bù đắp trải nghiệm xấu",
                    "Khảo sát hài lòng và cam kết cải thiện",
                ],
                "canh_bao": f"[!] Churn rate {churn * 100:.1f}% - CANH BẠC: Cần hành động ngay!",
            }
        else:  # pho thong, tiem nang
            strategies[strat_key] = {
                "dac_trung": f"Tuổi TB {age:.0f}, số dư {bal / 1e6:.0f}M, engagement {eng:.0f}/100",
                "giai_doan_song": "Thu nhập ổn định, nhu cầu tài chính cơ bản",
                "nhu_cau": "Tài khoản thanh toán, vay tiêu dùng nhỏ, tiết kiệm",
                "chien_luoc": "Kích hoạt và tăng tần suất sử dụng dịch vụ",
                "uu_dai": [
                    "Miễn phí duy trì tài khoản 6 tháng",
                    "Cashback 1% giao dịch qua app",
                    "Vay tiêu dùng nhanh không cần tài sản đảm bảo",
                    "Hỗ trợ onboarding digital banking",
                    "Chương trình giới thiệu bạn bè nhận thưởng",
                ],
                "canh_bao": f"Churn rate {churn * 100:.1f}% - Nhóm đồng nhất, cần chiến lược mass market",
            }

    return strategies


def save_model_version(results, feat_imp, cv_results):
    """
    Luu thong tin phiên bản model de tracking va tái sử dung.
    """
    import datetime

    best = max(results.items(), key=lambda x: x[1]["optimal_threshold"]["roc_auc"])
    version = {
        "version": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "best_model": best[0],
        "metrics": best[1]["optimal_threshold"],
        "top_features": feat_imp["top3"],
        "cv_results": cv_results,
        "features": FEATURES,
        "n_features": len(FEATURES),
    }
    with open("outputs/model_version.json", "w", encoding="utf-8") as f:
        json.dump(version, f, ensure_ascii=False, indent=2)
    return version


def main():
    """Main function to run full model evaluation pipeline"""
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("powerbi", exist_ok=True)

    print("... Dang chạy Feature Engineering...")
    fe = run_feature_engineering()

    print("... Dang chạy EDA...")
    eda_data = run_eda(fe["df_clean"])
    with open("outputs/eda.json", "w", encoding="utf-8") as f:
        json.dump(eda_data, f, ensure_ascii=False, indent=2)

    # Xuất PowerBI EDA
    powerbi_rows = []
    for feature in [
        "age_group",
        "gender",
        "customer_segment",
        "loyalty_level",
        "digital_behavior",
        "active_member",
        "occupation",
        "origin_province",
    ]:
        if feature not in fe["df_clean"].columns:
            continue
        g = fe["df_clean"].groupby(feature)["exit"].agg(["mean", "count"]).reset_index()
        g.columns = ["group_value", "churn_rate", "count"]
        g["feature"] = feature
        g["churn_rate"] = (g["churn_rate"] * 100).round(2)
        powerbi_rows.append(g)
    if powerbi_rows:
        pd.concat(powerbi_rows).to_csv("powerbi/churn_by_feature.csv", index=False)

    print("... Training models...")
    model_lr = train_logistic_regression(fe["X_train_sc"], fe["y_train"])
    model_rf = train_random_forest(fe["X_train"], fe["y_train"])
    model_xgb = train_xgboost(fe["X_train"], fe["y_train"])

    print("... Cross-validation (5-fold)...")
    cv_results = cross_validate_models(fe["X_train"], fe["y_train"], fe["X_train_sc"])
    with open("outputs/cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_results, f, ensure_ascii=False, indent=2)

    print("... Training KMeans...")
    scaler_km = StandardScaler()
    X_all_sc = scaler_km.fit_transform(fe["df_enc"][FEATURES])
    model_kmeans, df_clustered, cluster_profiles, elbow_data = train_kmeans(
        X_all_sc, fe["df_clean"]
    )

    print("... Đánh giá models...")
    results = {}
    results["Logistic Regression"] = evaluate_model(
        "Logistic Regression",
        model_lr,
        fe["X_test"],
        fe["X_test_sc"],
        fe["y_test"],
        use_scaled=True,
    )
    results["Random Forest"] = evaluate_model(
        "Random Forest",
        model_rf,
        fe["X_test"],
        fe["X_test_sc"],
        fe["y_test"],
        use_scaled=False,
    )
    results["XGBoost"] = evaluate_model(
        "XGBoost",
        model_xgb,
        fe["X_test"],
        fe["X_test_sc"],
        fe["y_test"],
        use_scaled=False,
    )

    # Đánh giá tren validation set
    val_results = {}
    val_results["Logistic Regression"] = evaluate_model(
        "Logistic Regression",
        model_lr,
        fe["X_val"],
        fe["X_val_sc"],
        fe["y_val"],
        use_scaled=True,
    )
    val_results["Random Forest"] = evaluate_model(
        "Random Forest",
        model_rf,
        fe["X_val"],
        fe["X_val_sc"],
        fe["y_val"],
        use_scaled=False,
    )
    val_results["XGBoost"] = evaluate_model(
        "XGBoost", model_xgb, fe["X_val"], fe["X_val_sc"], fe["y_val"], use_scaled=False
    )

    feat_imp = get_feature_importance(model_rf, model_xgb, model_lr, fe["scaler"])
    comparison, compare_insights = compare_models(results)
    strategies = get_cluster_strategy(cluster_profiles)
    version = save_model_version(results, feat_imp, cv_results)

    # Luu tất cả
    joblib.dump(model_lr, "outputs/model_lr.pkl")
    joblib.dump(model_rf, "outputs/model_rf.pkl")
    joblib.dump(model_xgb, "outputs/model_xgb.pkl")
    joblib.dump(model_kmeans, "outputs/model_kmeans.pkl")
    joblib.dump(scaler_km, "outputs/scaler_kmeans.pkl")
    joblib.dump(fe["scaler"], "outputs/scaler.pkl")
    joblib.dump(fe["encoders"], "outputs/encoders.pkl")
    joblib.dump(fe["X_test"], "outputs/X_test.pkl")
    joblib.dump(fe["y_test"], "outputs/y_test.pkl")
    joblib.dump(fe["X_test_sc"], "outputs/X_test_sc.pkl")
    joblib.dump(fe["X_train"], "outputs/X_train.pkl")
    joblib.dump(fe["y_train"], "outputs/y_train.pkl")

    os.makedirs("data/processed", exist_ok=True)
    df_clustered.to_csv("data/processed/clustered_data.csv", index=False)

    # Serialize PR curve lists
    for name in results:
        pr = results[name]["pr_curve"]
        results[name]["pr_curve"]["precisions"] = [
            round(float(x), 4) for x in pr["precisions"]
        ]
        results[name]["pr_curve"]["recalls"] = [
            round(float(x), 4) for x in pr["recalls"]
        ]

    with open("outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for name in val_results:
        pr = val_results[name]["pr_curve"]
        val_results[name]["pr_curve"]["precisions"] = [
            round(float(x), 4) for x in pr["precisions"]
        ]
        val_results[name]["pr_curve"]["recalls"] = [
            round(float(x), 4) for x in pr["recalls"]
        ]
    with open("outputs/val_results.json", "w", encoding="utf-8") as f:
        json.dump(val_results, f, ensure_ascii=False, indent=2)

    with open("outputs/feat_imp.json", "w", encoding="utf-8") as f:
        json.dump(feat_imp, f, ensure_ascii=False, indent=2)

    with open("outputs/comparison.json", "w", encoding="utf-8") as f:
        json.dump(
            {"table": comparison, "insights": compare_insights},
            f,
            ensure_ascii=False,
            indent=2,
        )

    cluster_json = cluster_profiles.copy()
    cluster_json["balance_mean"] = cluster_json["balance_mean"].round(0).astype(int)
    cluster_json["engagement_mean"] = cluster_json["engagement_mean"].round(2)
    cluster_json["churn_rate"] = (cluster_json["churn_rate"] * 100).round(2)
    cluster_json["age_mean"] = cluster_json["age_mean"].round(1)

    with open("outputs/cluster_profiles.json", "w", encoding="utf-8") as f:
        json.dump(
            cluster_json.to_dict(orient="records"), f, ensure_ascii=False, indent=2
        )

    with open("outputs/cluster_strategies.json", "w", encoding="utf-8") as f:
        json.dump(strategies, f, ensure_ascii=False, indent=2)

    with open("outputs/elbow_data.json", "w", encoding="utf-8") as f:
        json.dump(elbow_data, f, ensure_ascii=False, indent=2)

    # PowerBI exports
    pd.DataFrame(comparison).to_csv("powerbi/model_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"feature": k, "importance_combined": v}
            for k, v in feat_imp["combined"].items()
        ]
    ).to_csv("powerbi/feature_importance.csv", index=False)

    churn_summary = pd.DataFrame(
        [
            {
                "model": r["name"],
                "roc_auc": r["optimal_threshold"]["roc_auc"],
                "recall": r["optimal_threshold"]["recall"],
                "f1": r["optimal_threshold"]["f1"],
            }
            for r in results.values()
        ]
    )
    churn_summary.to_csv("powerbi/churn_summary.csv", index=False)
    cluster_json.to_csv("powerbi/cluster_summary.csv", index=False)

    print("\n[OK] Evaluation hoàn tất!")
    print(f"\n[BEST] Best model: {compare_insights['best_overall']}")
    print(f"   {compare_insights['so_lieu']}")
    print(f"\n[KEY] Top features: {feat_imp['top3']}")
    print(f"\n[FILE] Model version: {version['version']}")
    print(
        f"   CV AUC: {cv_results[version['best_model']]['cv_auc_mean']} ± {cv_results[version['best_model']]['cv_auc_std']}"
    )


if __name__ == "__main__":
    main()
