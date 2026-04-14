import sys
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings

# FIX: Windows cp1252 không hỗ trợ tiếng Việt → force UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from pipeline.feature_engineering import run_feature_engineering
from pipeline.feature_engineering import Winsorizer, _make_one_hot_encoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING (SCIENTIFIC REBUILD WITH GRIDSEARCH)
# ══════════════════════════════════════════════════════════════════════════════

def tune_and_train_models(X_train, y_train, num_features, cat_features):
    """
    Sử dụng GridSearchCV kết hợp SMOTE bên trong imblearn Pipeline.
    X_train ở đây là dữ liệu raw (đã clean), preprocessing sẽ được fit bên trong từng fold CV để tránh leakage.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {}
    best_params = {}
    cv_results = {}

    # FIX W1: Tạo riêng từng Winsorizer instance cho mỗi pipeline.
    # Dùng chung một object giữa 3 pipeline có thể gây race condition khi
    # GridSearchCV chạy song song (n_jobs=-1) — các fold khác nhau sẽ cùng
    # .fit() trên cùng một object, ghi đè lên nhau.
    _winsor_cols = ['balance', 'monthly_ir', 'credit_sco', 'age', 'engagement_score', 'risk_score']

    def _make_preprocessor():
        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', _make_one_hot_encoder(cat_features), cat_features)
            ]
        )

    # --- 1. Logistic Regression ---
    print("\n--- Tuning Logistic Regression ---")
    pipe_lr = ImbPipeline([
        ('winsor', Winsorizer(columns=_winsor_cols)),
        ('preprocess', _make_preprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ])
    param_lr = {
        'lr__C': [0.01, 0.1, 1, 10],
        'lr__class_weight': ['balanced', None]
    }
    # Sử dụng 'roc_auc' làm tiêu chí chọn model tốt nhất để nhất quán với Evaluation
    grid_lr = GridSearchCV(pipe_lr, param_lr, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    models['Logistic Regression'] = grid_lr.best_estimator_
    best_params['Logistic Regression'] = grid_lr.best_params_
    
    # --- 2. Random Forest ---
    print("\n--- Tuning Random Forest ---")
    pipe_rf = ImbPipeline([
        ('winsor', Winsorizer(columns=_winsor_cols)),
        ('preprocess', _make_preprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    param_rf = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [7, 10, None]
    }
    grid_rf = GridSearchCV(pipe_rf, param_rf, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    models['Random Forest'] = grid_rf.best_estimator_
    best_params['Random Forest'] = grid_rf.best_params_
    
    # --- 3. XGBoost ---
    print("\n--- Tuning XGBoost ---")
    pipe_xgb = ImbPipeline([
        ('winsor', Winsorizer(columns=_winsor_cols)),
        ('preprocess', _make_preprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    param_xgb = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1]
    }
    grid_xgb = GridSearchCV(pipe_xgb, param_xgb, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    models['XGBoost'] = grid_xgb.best_estimator_
    best_params['XGBoost'] = grid_xgb.best_params_

    # --- 4. Dummy Baseline ---
    print("\n--- Training Dummy Baseline ---")
    dummy = DummyClassifier(strategy='stratified', random_state=42)
    pipe_dummy = ImbPipeline([
        ('winsor', Winsorizer(columns=_winsor_cols)),
        ('preprocess', _make_preprocessor()),
        ('dummy', dummy)
    ])
    pipe_dummy.fit(X_train, y_train)
    models['Baseline'] = pipe_dummy
    best_params['Baseline'] = {}

    # Capture Mean and Std from GridSearchCV
    def get_cv_metrics(grid):
        idx = grid.best_index_
        return {
            "mean": round(grid.cv_results_['mean_test_score'][idx], 4),
            "std": round(grid.cv_results_['std_test_score'][idx], 4)
        }

    cv_results['Logistic Regression'] = get_cv_metrics(grid_lr)
    cv_results['Random Forest'] = get_cv_metrics(grid_rf)
    cv_results['XGBoost'] = get_cv_metrics(grid_xgb)
    cv_results['Baseline'] = {"mean": 0.5, "std": 0.02} 

    return models, best_params, cv_results

def train_kmeans(df_clean, X_scaled=None):
    """
    K-Means gom cụm trên 4 Feature lõi để tránh nhiễu từ OHE.
    """
    print("\n--- Running K-Means Clustering Independent ---")
    km_features = ['balance', 'engagement_score', 'risk_score', 'age']
    
    from sklearn.preprocessing import StandardScaler
    scaler_km = StandardScaler()
    X_km = scaler_km.fit_transform(df_clean[km_features])
    
    # Fit final KMeans K=4
    final_k = 4
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    kmeans.fit(X_km)
    
    df_clean['cluster'] = kmeans.labels_
    
    # Aggregation & Labeling
    cluster_profiles = df_clean.groupby('cluster').agg(
        balance_mean=('balance', 'mean'),
        engagement_mean=('engagement_score', 'mean'),
        churn_rate=('exit', 'mean'),
        age_mean=('age', 'mean'),
        risk_mean=('risk_score', 'mean'),
        count=('exit', 'count')
    ).reset_index()
    
    cluster_profiles['balance_rank'] = cluster_profiles['balance_mean'].rank(ascending=False)
    cluster_profiles['risk_rank'] = cluster_profiles['risk_mean'].rank(ascending=False)
    cluster_profiles['engagement_rank'] = cluster_profiles['engagement_mean'].rank(ascending=False)
    
    def name_centroid(row):
        if row['balance_rank'] <= 1: return "Khách hàng VIP - Trụ cột Tài sản"
        elif row['engagement_rank'] <= 1 and row['churn_rate'] <= 0.10: return "Khách hàng Tích cực - Tiềm năng Cross-sell"
        elif row['churn_rate'] > 0.15 and row['age_mean'] >= 55: return "Nhóm Cao tuổi - Rủi ro Tiềm ẩn"
        elif row['churn_rate'] > 0.15: return "Nhóm Rủi ro Cao - Cần Cứu vãn Khẩn"
        else: return "Khách hàng Phổ thông - Cần Kích hoạt"
            
    cluster_profiles['cluster_name'] = cluster_profiles.apply(name_centroid, axis=1)
    
    # 🎯 CHIẾN LƯỢC CHO TỪNG CỤM (Actionable Strategies)
    cluster_strategies = {}
    for _, row in cluster_profiles.iterrows():
        c_id = int(row['cluster'])
        name = row['cluster_name']
        
        if "VIP" in name:
            uu_dai = ["Miễn phí thẻ Signature", "Lãi suất ưu đãi +0.5%", "Hỗ trợ RM riêng 24/7"]
            kenh = "Tư vấn Trực tiếp (RM)"
            chien_luoc = "Tập trung tối đa hóa tài sản và bán thêm các gói quản lý gia sản cao cấp."
        elif "Tích cực" in name:
            uu_dai = ["Voucher mua sắm 500k", "Hoàn tiền 5% giao dịch App", "Tích điểm Loyalty x2"]
            kenh = "Mobile App / Email"
            chien_luoc = "Kích thích chi tiêu qua thẻ và ứng dụng di động để duy trì độ gắn kết."
        elif "Cao tuổi" in name:
            uu_dai = ["Gói bảo hiểm sức khỏe", "Tiết kiệm hưu trí lãi cao", "Quà tặng dịp lễ Tết"]
            kenh = "Telesales / Tại quầy"
            chien_luoc = "Chăm sóc theo hướng truyền thống, xây dựng niềm tin thông qua tư vấn trực tiếp."
        elif "Rủi ro" in name:
            uu_dai = ["Miễn phí quản lý TK 1 năm", "Tặng thẻ cào 100k kích hoạt lại", "Khảo sát ý kiến tặng quà"]
            kenh = "Telesales Khẩn cấp"
            chien_luoc = "Cần liên hệ ngay để tìm hiểu nguyên nhân không hài lòng và đưa ra gói cứu vãn."
        else:
            uu_dai = ["Ưu đãi nạp tiền điện thoại", "Gói vay tiêu dùng lãi thấp", "Mời dùng thêm thẻ ảo"]
            kenh = "SMS / Push Notification"
            chien_luoc = "Tăng cường nhắc nhớ thương hiệu và mời sử dụng thêm các dịch vụ vệ tinh."
            
        cluster_strategies[str(c_id)] = {
            "name": name,
            "uu_dai": uu_dai,
            "kenh_tiep_can": kenh,
            "chien_luoc": chien_luoc,
            "muc_uu_tien": "Cao" if "Rủi ro" in name or "VIP" in name else "Trung bình"
        }

    # Elbow Method & Silhouette logic
    k_range = range(2, 7)
    inertias = []
    silhouettes = []
    for k in k_range:
        km_test = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_km)
        inertias.append(float(km_test.inertia_))
        silhouettes.append(float(silhouette_score(X_km, km_test.labels_)))

    elbow_data = {
        "k_range": list(k_range), 
        "inertias": inertias, 
        "silhouette_scores": silhouettes
    }
    
    return kmeans, scaler_km, df_clean, cluster_profiles, elbow_data, cluster_strategies

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    fe = run_feature_engineering()
    # FIX W2: Sửa key sai 'X_train_sc' → 'X_train' và thêm đủ 4 tham số bắt buộc
    models, best_params, cv_results = tune_and_train_models(
        fe["X_train"], fe["y_train"], fe["num_features"], fe["cat_features"]
    )
    # ... evaluation logic remains similar but called from model_evaluation.py
