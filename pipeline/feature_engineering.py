import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pipeline.data_cleaning import load_data, clean_pipeline

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# Nguyên nhân: Dữ liệu thô chứa chuỗi, boolean, giá trị thô chưa chuẩn hóa
# -> mô hình không thể học trực tiếp.
# Lý do tại sao: Mỗi thuật toán có yêu cầu đầu vào khác nhau:
#   - Logistic Regression cần scaling (nhạy cảm với magnitude)
#   - Random Forest & XGBoost chịu được không scale nhưng cần encode
#   - KMeans cần scaling (dùng khoảng cách Euclidean)
# Hướng xử lý: Encode -> Scale -> SMOTE -> Split
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = [
    "credit_sco",
    "gender",
    "age",
    "balance",
    "monthly_ir",
    "tenure_ye",
    "married",
    "nums_card",
    "nums_service",
    "active_member",
    "engagement_score",
    "risk_score",
    "customer_segment",
    "loyalty_level",
    "digital_behavior",
]
TARGET = "exit"

CAT_COLS = ["gender", "customer_segment", "loyalty_level", "digital_behavior"]


def encode_features(df):
    """
    VẤN ĐỀ: Sklearn chỉ nhận input dạng số, không xử lý được string.
    TạI SAO LABELENCODER thay vì OneHotEncoder:
    - OneHot tạo N-1 cột mới cho mỗi categorical -> curse of dimensionality
    - customer_segment có 5 giá trị -> OneHot tạo 4 cột mới, không cần thiết
    - LabelEncoder phù hợp vì:
      + loyalty_level có thứ tự rõ ràng: Bronze < Silver < Gold < Platinum
      + gender là binary (2 giá trị)
      + Tree-based models (RF, XGB) không bị ảnh hưởng bởi ordinal encoding
    LưU Ý: LabelEncoder fit trên toàn bộ data trước khi split -> đảm bảo
    các giá trị mới trong production được xử lý đúng.
    """
    encoders = {}
    df_enc = df.copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


def split_data(df_enc):
    """
    Nguyên nhân: Cần tập val để chọn model, tập test để đánh giá cuối cùng.
    Lý do tại sao: 60/20/20 - train đủ lớn, val để tune threshold,
    test hoàn toàn độc lập để báo cáo kết quả thực.
    """
    X = df_enc[FEATURES]
    y = df_enc[TARGET]
    # Tách test 20% trước
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Tách val 20% từ phần còn lại (= 20% tổng)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train):
    """
    VẤN ĐỀ: Dataset churn thường mất cân bằng nặng (tỉ lệ churn ~20-30%).
    Nếu không xử lý, mô hình học cách dự đoán "không churn" cho tất cả
    vẫn đạt accuracy ~75% -> vô dụng trong thực tế.

    TẠI SAO SMOTE thay vì các phương pháp khác:
    - RandomOverSampler: chỉ copy lại mẫu cũ -> mô hình overfit trên
      đúng những điểm đó, không học được pattern mới.
    - Undersampling: xóa mẫu majority -> mất thông tin quý giá,
      đặc biệt nguy hiểm khi dataset không quá lớn.
    - ADASYN: tạo nhiều mẫu hơn ở vùng khó phân loại -> dễ tạo noise.
    - SMOTE: nội suy tuyến tính giữa k láng giềng gần nhất của class
      thiểu số -> tạo mẫu tổng hợp đa dạng, hợp lý về mặt thống kê.

    CƠ CHẾ SMOTE:
    1. Với mỗi mẫu churn, tìm k=5 láng giềng gần nhất cùng class
    2. Chọn ngẫu nhiên 1 láng giềng
    3. Tạo điểm mới = mẫu gốc + random(0,1) × (láng giềng - mẫu gốc)
    -> Điểm mới nằm trên đoạn thẳng nối 2 mẫu thực -> hợp lý về phân phối

    RỦI RO & CÁCH GIẢM THIỂU:
    - SMOTE có thể tạo mẫu ở vùng chồng lấp 2 class -> nhiễu
    - Giảm thiểu: chỉ apply trên TRAIN, không apply val/test
      -> đánh giá trên phân phối thực, tránh data leakage
    - Kết hợp class_weight trong model để double-check
    """
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    before_0 = int((y_train == 0).sum())
    before_1 = int((y_train == 1).sum())
    after_0 = int((y_res == 0).sum())
    after_1 = int((y_res == 1).sum())

    print(
        f"   SMOTE: {before_1:,} -> {after_1:,} mau churn "
        f"(tao them {after_1 - before_1:,} mau tong hop)"
    )
    print(f"   Ti le sau SMOTE: {after_1 / after_0 * 100:.1f}% churn")

    return X_res, y_res


def scale_features(X_train, X_test):
    """
    VẤN ĐỀ: Các feature có scale rất khác nhau:
    - balance: 0 – 500,000,000 VNĐ
    - risk_score: 0.0 – 1.0
    - age: 18 – 100
    Hậu quả nếu không scale:
    - Logistic Regression: gradient descent bị chi phối bởi balance
      -> hội tụ chậm hoặc không hội tụ
    - KMeans: khoảng cách Euclidean bị balance áp đảo hoàn toàn
    - RF/XGB: không cần scale (dùng split point, không dùng khoảng cách)
    TạI SAO STANDARDSCALER thay vì MinMaxScaler:
    - MinMax nhạy cảm với outlier (outlier ở balance sẽ ép các giá trị khác về 0)
    - StandardScaler (z-score) bền vững hơn, phù hợp sau khi đã winsorize
    QUY TẮc: Fit trên TRAIN, transform cả train và test/val
    -> tránh data leakage (không được dùng thông tin test để scale)
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


def run_feature_engineering():
    df_raw = load_data()
    df, _ = clean_pipeline(df_raw)

    df_enc, encoders = encode_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_enc)

    before = y_train.value_counts().to_dict()
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    after = pd.Series(y_train_res).value_counts().to_dict()

    X_train_sc, X_test_sc, scaler = scale_features(X_train_res, X_test)
    X_val_sc = scaler.transform(X_val)

    report = {
        "features": FEATURES,
        "n_train": len(X_train_res),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "split": "60/20/20",
        "smote": {
            "method": "SMOTE k_neighbors=5",
            "applied_on": "train only (tránh data leakage)",
            "before": {str(k): int(v) for k, v in before.items()},
            "after": {str(k): int(v) for k, v in after.items()},
            "synthetic_samples_added": int(after.get(1, 0) - before.get(1, 0)),
            "final_ratio": round(after.get(1, 0) / after.get(0, 1), 4),
        },
    }

    return {
        "X_train": X_train_res,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_sc": X_train_sc,
        "X_val_sc": X_val_sc,
        "X_test_sc": X_test_sc,
        "y_train": y_train_res,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "df_enc": df_enc,
        "df_clean": df,
        "report": report,
    }


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    result = run_feature_engineering()

    joblib.dump(result["scaler"], "outputs/scaler.pkl")
    joblib.dump(result["encoders"], "outputs/encoders.pkl")
    result["df_enc"].to_csv("data/processed/processed_data.csv", index=False)

    with open("outputs/fe_report.json", "w", encoding="utf-8") as f:
        json.dump(result["report"], f, ensure_ascii=False, indent=2)

    print("[OK] Feature Engineering hoàn tất")
    print(
        f"   Train: {result['report']['n_train']:,} | Test: {result['report']['n_test']:,}"
    )
    smote = result["report"]["smote"]
    print(f"   SMOTE trước: {smote['before']} -> sau: {smote['after']}")
