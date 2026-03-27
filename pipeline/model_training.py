import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pipeline.feature_engineering import run_feature_engineering, FEATURES

# ══════════════════════════════════════════════════════════════════════════════
# HUẤN LUYỆN MÔ HÌNH
# Nguyên nhân: Cần 3 mô hình với bản chất khác nhau để so sánh toàn diện:
#   - Logistic Regression: tuyến tính, dễ giải thích, baseline tốt
#   - Random Forest: ensemble, xử lý phi tuyến, ít overfit
#   - XGBoost: boosting, thường cho kết quả tốt nhất trên tabular data
# Lý do tại sao: Không có mô hình nào tốt nhất tuyệt đối -> cần so sánh
# trên cùng dữ liệu để chọn mô hình phù hợp nhất cho bài toán này.
# ══════════════════════════════════════════════════════════════════════════════


def cross_validate_models(X_train, y_train, X_train_sc):
    """
    5-fold cross-validation trên tập train để đánh giá ổn định của mô hình.
    Tránh đánh giá sai do may mắn của một lần split.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            X_train_sc,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
            X_train,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            ),
            X_train,
        ),
    }

    for name, (model, X) in models.items():
        scores = cross_val_score(model, X, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = {
            "cv_auc_mean": round(float(scores.mean()), 4),
            "cv_auc_std": round(float(scores.std()), 4),
            "cv_auc_scores": [round(float(s), 4) for s in scores],
        }
        print(f"  {name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


def train_logistic_regression(X_train_sc, y_train):
    """
    Nguyên nhân: Logistic Regression là baseline quan trọng.
    Lý do tại sao: Nếu mô hình phức tạp không vượt trội đáng kể so với
    LR thì không cần dùng mô hình phức tạp (Occam's Razor).
    class_weight='balanced': tự động điều chỉnh trọng số theo tỉ lệ class.
    """
    model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced", C=1.0
    )
    model.fit(X_train_sc, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Nguyên nhân: Random Forest xử lý tốt dữ liệu phi tuyến và
    tương tác giữa các features.
    Lý do tại sao: Ensemble nhiều cây quyết định -> giảm variance,
    ít overfit hơn single decision tree.
    n_estimators=100: đủ cây để ổn định kết quả mà không quá chậm.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Nguyên nhân: XGBoost thường cho kết quả tốt nhất trên tabular data.
    Lý do tại sao: Gradient boosting học tuần tự, mỗi cây sửa lỗi của
    cây trước -> tối ưu hóa liên tục.
    scale_pos_weight: xử lý imbalance bằng cách tăng trọng số class dương.
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale = round(neg / pos, 2)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_kmeans(X_scaled, df_clean, n_clusters=3):
    """
    Nguyên nhân: Phân cụm khách hàng giúp cá nhân hóa chiến lược
    thay vì áp dụng một chiến lược chung cho tất cả.
    Lý do tại sao: KMeans với n_clusters=3 dựa trên Silhouette Score lớn nhất
    và ý nghĩa kinh doanh. Mặc dù Elbow có vẻ giảm ở 4, nhưng Silhouette đạt đỉnh ở 3.
    Tại sao lại như vậy: Dùng dữ liệu đã scale để khoảng cách Euclidean
    có ý nghĩa thực sự, không bị chi phối bởi balance (giá trị lớn).
    """
    # Elbow method - use sample for faster computation
    inertias = []
    sil_scores = []
    K_range = range(2, 8)
    sample_size = min(5000, len(X_scaled))
    X_sample = (
        X_scaled[:sample_size]
        if sample_size == len(X_scaled)
        else X_scaled[np.random.choice(len(X_scaled), sample_size, replace=False)]
    )

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
        km.fit(X_sample)
        inertias.append(float(km.inertia_))
        sil_scores.append(
            float(
                silhouette_score(
                    X_sample, km.labels_, sample_size=min(1000, sample_size)
                )
            )
        )

    # Train final model on all data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
    kmeans.fit(X_scaled)

    df_clean = df_clean.copy()
    df_clean["cluster"] = kmeans.labels_

    # Đặt tên cụm dựa trên đặc trưng thực tế
    cluster_profiles = (
        df_clean.groupby("cluster")
        .agg(
            balance_mean=("balance", "mean"),
            engagement_mean=("engagement_score", "mean"),
            churn_rate=("exit", "mean"),
            age_mean=("age", "mean"),
            count=("exit", "count"),
        )
        .reset_index()
    )

    # Gán tên cụm dựa trên balance + engagement + churn_rate
    def name_cluster(row):
        churn = row["churn_rate"]
        engagement = row["engagement_mean"]
        balance = row["balance_mean"]

        if churn > 0.25:
            return "Khách hàng rủi ro cao - Cần kích hoạt ngay"
        elif balance > 100e6:
            return "Khách hàng VIP - Tài sản lớn"
        elif engagement > 60 and balance > 35e6:
            return "Khách hàng thân thiết - Đặc biệt"
        elif engagement > 45:
            return "Khách hàng tích cực - Đặc quyền"
        elif balance > 35e6:
            return "Khách hàng tiềm năng - Tăng trưởng"
        else:
            return "Khách hàng phổ thông - Cần nuôi dưỡng"

    cluster_profiles["cluster_name"] = cluster_profiles.apply(name_cluster, axis=1)

    elbow_data = {
        "k_range": list(K_range),
        "inertias": inertias,
        "silhouette_scores": [round(s, 4) for s in sil_scores],
    }

    return kmeans, df_clean, cluster_profiles, elbow_data


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    print("... Đang chạy Feature Engineering...")
    fe = run_feature_engineering()

    print("... Training Logistic Regression...")
    model_lr = train_logistic_regression(fe["X_train_sc"], fe["y_train"])

    print("... Training Random Forest...")
    model_rf = train_random_forest(fe["X_train"], fe["y_train"])

    print("... Training XGBoost...")
    model_xgb = train_xgboost(fe["X_train"], fe["y_train"])

    print("... Training KMeans...")
    # Scale toàn bộ data cho KMeans
    from sklearn.preprocessing import StandardScaler

    scaler_km = StandardScaler()
    fe_cols = fe["df_enc"][FEATURES]
    X_all_sc = scaler_km.fit_transform(fe_cols)
    model_kmeans, df_clustered, cluster_profiles, elbow_data = train_kmeans(
        X_all_sc, fe["df_clean"]
    )

    # Lưu models
    joblib.dump(model_lr, "outputs/model_lr.pkl")
    joblib.dump(model_rf, "outputs/model_rf.pkl")
    joblib.dump(model_xgb, "outputs/model_xgb.pkl")
    joblib.dump(model_kmeans, "outputs/model_kmeans.pkl")
    joblib.dump(scaler_km, "outputs/scaler_kmeans.pkl")
    joblib.dump(fe["scaler"], "outputs/scaler.pkl")
    joblib.dump(fe["encoders"], "outputs/encoders.pkl")

    # Lưu cluster data
    df_clustered.to_csv("outputs/clustered_data.csv", index=False)

    cluster_json = cluster_profiles.copy()
    cluster_json["balance_mean"] = cluster_json["balance_mean"].round(0).astype(int)
    cluster_json["engagement_mean"] = cluster_json["engagement_mean"].round(2)
    cluster_json["churn_rate"] = (cluster_json["churn_rate"] * 100).round(2)
    cluster_json["age_mean"] = cluster_json["age_mean"].round(1)

    with open("outputs/cluster_profiles.json", "w", encoding="utf-8") as f:
        json.dump(
            cluster_json.to_dict(orient="records"), f, ensure_ascii=False, indent=2
        )

    with open("outputs/elbow_data.json", "w", encoding="utf-8") as f:
        json.dump(elbow_data, f, ensure_ascii=False, indent=2)

    # Xuất PowerBI
    os.makedirs("powerbi", exist_ok=True)
    cluster_json.to_csv("powerbi/cluster_summary.csv", index=False)

    print("\n[OK] Training hoàn tất!")
    print("   model_lr.pkl | model_rf.pkl | model_xgb.pkl | model_kmeans.pkl")
    print("\n[PLOT] Cluster Profiles:")
    print(
        cluster_profiles[["cluster", "cluster_name", "count", "churn_rate"]].to_string(
            index=False
        )
    )
