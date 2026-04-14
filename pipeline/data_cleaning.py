import pandas as pd
import numpy as np
import json
import os
from pipeline.data_loading import load_data

# ══════════════════════════════════════════════════════════════════════════════
# LÀM SẠCH DỮ LIỆU
# Nguyên nhân: Dữ liệu thực tế luôn tồn tại lỗi - sai kiểu, giá trị bất thường,
# thiếu dữ liệu - nếu không xử lý sẽ làm lệch kết quả mô hình.
# Lý do tại sao: Mô hình học máy nhạy cảm với nhiễu và outlier, đặc biệt
# Logistic Regression và KMeans. Dữ liệu bẩn -> mô hình học sai pattern.
# ══════════════════════════════════════════════════════════════════════════════

def fix_dtypes(df):
    """
    VẤN ĐỀ: CSV lưu boolean dưới dạng string 'True'/'False'.
    Hậu quả nếu không sửa:
    - df['exit'] == 1 sẽ không match với 'True' -> target sai hoàn toàn
    - sklearn không nhận string làm target -> lỗi runtime
    - pd.to_datetime cần để tính tenure, days_since_active sau này
    Tại sao dùng map thay vì astype(bool):
    - astype(bool) chuyển mọi string khác rỗng thành True, kể cả 'False'
    - map xử lý đúng cả string lẫn bool gốc
    """
    bool_cols = ['married', 'active_member', 'exit']
    for col in bool_cols:
        if df[col].dtype == object:
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    df['exit'] = df['exit'].astype(int)
    for col in ['last_active_date', 'created_date']:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    return df

def handle_missing(df):
    """
    VẤN ĐỀ: Missing values lan truyền qua phép tính -> NaN ở output.
    TạI SAO MEDIAN cho số, MODE cho categorical:
    - Mean bị kéo lệch bởi outlier (balance có thể rất cao)
    - Median = giá trị giữa -> bền vững hơn với phân phối lệch
    - Mode cho categorical = giá trị xuất hiện nhiều nhất -> ít sai lệch nhất
    TạI SAO KHÔNG XÓA DÒNG:
    - Dataset 80k dòng, xóa dòng lãng phí thông tin
    - Nếu missing có pattern (VD: khách hàng churn hay bỏ trống) ->
      xóa dòng gây selection bias
    """
    report = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in num_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            # FIX P2: Thay inplace=True (deprecated) bằng assignment
            df[col] = df[col].fillna(median_val)
            report[col] = {"strategy": "median", "filled": int(n_missing), "value": round(float(median_val), 2)}
    for col in cat_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            # FIX P2: Thay inplace=True (deprecated) bằng assignment
            df[col] = df[col].fillna(mode_val)
            report[col] = {"strategy": "mode", "filled": int(n_missing), "value": str(mode_val)}
    return df, report

def remove_duplicates(df):
    """
    VẤN ĐỀ: Dòng trùng lặp xảy ra do lỗi ETL, import nhiều lần, hoặc join sai.
    Hậu quả:
    - Mô hình thấy mẫu đó nhiều lần -> overfit, tăng trọng số quá mức
    - Metrics đánh giá bị phồng (cùng mẫu xuất hiện cả train lẫn test)
    TạI SAO KHÔNG dùng subset: xét toàn bộ cột để tránh xóa nhầm
    khách hàng khác nhau có cùng một số đặc trưng.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"   Đã xóa {n_removed:,} dòng trùng lặp ({n_removed/n_before*100:.2f}%)")
    return df, n_removed

def handle_outliers(df):
    """
    VẤN ĐỀ: Outlier cực đoan (VD: balance = 10 tỷ, age = 150) làm
    phân phối bị kéo lệch, ảnh hưởng đặc biệt đến:
    - Logistic Regression: gradient bị đẩy bởi giá trị cực lớn
    - KMeans: khoảng cách Euclidean bị chi phối bởi outlier
    - StandardScaler: mean/std bị lệch -> scale sai

    TạI SAO WINSORIZE (cap) thay vì xóa:
    - Xóa dòng: mất thông tin, giảm dataset, có thể xóa nhầm KH thực sự
    - Log transform: khó giải thích, không phù hợp với mọi feature
    - Winsorize 1%-99%: giữ lại dòng, chỉ giới hạn giá trị cực đoan
      về ngưỡng hợp lý -> bảo toàn thông tin, giảm nhiễu

    TạI SAO 1%-99% thay vì IQR (25%-75%):
    - IQR cắt quá nhiều (50% dữ liệu nằm ngoài) -> mất thông tin
    - 1%-99% chỉ loại bỏ 2% cực đoan thực sự bất thường
    """
    outlier_cols = ['balance', 'monthly_ir', 'credit_sco', 'age',
                    'engagement_score', 'risk_score']
    report = {}
    winsorize_thresholds = {}
    for col in outlier_cols:
        if col not in df.columns:
            continue
        q01 = float(df[col].quantile(0.01))
        q99 = float(df[col].quantile(0.99))
        n_outliers = int(((df[col] < q01) | (df[col] > q99)).sum())
        df[col] = df[col].clip(lower=q01, upper=q99)
        report[col] = {
            "q01": round(q01, 2),
            "q99": round(q99, 2),
            "n_capped": n_outliers,
            "pct_capped": round(n_outliers / len(df) * 100, 2)
        }
        winsorize_thresholds[col] = {"lower": q01, "upper": q99}
        
    import os
    import joblib
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(winsorize_thresholds, "outputs/winsorize_thresholds.pkl")
    
    return df, report

def check_imbalance(df):
    """
    VẤN ĐỀ: Mất cân bằng là đặc điểm cốt lõi của bài toán churn.
    Trong thực tế ngân hàng, tỉ lệ churn thường 15-30% -> ratio 3:1 đến 6:1.

    Hậu quả nếu không xử lý:
    - Mô hình dự đoán "không churn" cho tất cả -> accuracy cao nhưng vô dụng
    - Recall của class churn gần 0 -> bỏ sót hầu hết KH sắp rời bỏ
    - Chi phí thực tế: mỗi KH churn mất ~5-10x chi phí giữ chân

    Hướng xử lý (sẽ thực hiện ở Feature Engineering):
    - SMOTE: tạo mẫu tổng hợp cho class thiểu số
    - class_weight='balanced': tăng trọng số cho class churn trong loss function
    - Optimal threshold: hạ ngưỡng quyết định xuống dưới 0.5
    """
    counts = df['exit'].value_counts()
    # FIX P3: Dùng .get() explicit thay vì counts[0]/counts[1] — tránh KeyError
    # khi value_counts() trả về thứ tự khác hoặc thiếu class
    no_churn = int(counts.get(0, 0))
    churn    = int(counts.get(1, 0))
    ratio    = round(no_churn / churn, 2) if churn > 0 else 0
    return {
        "no_churn":  no_churn,
        "churn":     churn,
        "ratio":     ratio,
        "imbalanced": ratio > 3,
        "severity":  "cao" if ratio > 5 else "trung bình" if ratio > 3 else "thấp"
    }

def clean_pipeline(df):
    cleaning_report = {}

    df = fix_dtypes(df)
    cleaning_report["dtype_fix"] = "boolean & date columns converted"

    df, missing_report = handle_missing(df)
    cleaning_report["missing"] = missing_report

    df, n_dup = remove_duplicates(df)
    cleaning_report["duplicates_removed"] = n_dup

    # Chuyển Winsorization sang Feature Engineering Pipeline để tránh Data Leakage
    # (Chỉ thực hiện ở đây nếu muốn quan sát EDA, nhưng mô hình sẽ dùng Pipeline riêng)
    # df, outlier_report = handle_outliers(df)
    # cleaning_report["outliers"] = outlier_report
    cleaning_report["outliers"] = "Moved to FE Pipeline to prevent Data Leakage"

    imbalance = check_imbalance(df)
    cleaning_report["imbalance"] = imbalance

    cleaning_report["final_shape"] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1])
    }

    return df, cleaning_report

def print_cleaning_report(report):
    print("=" * 60)
    print("[CLEAN] BÁO CÁO LÀM SẠCH DỮ LIỆU")
    print("=" * 60)

    print(f"\n[OK] Kiểu dữ liệu: {report.get('dtype_fix', '—')}")
    print(f"[OK] Duplicates đã xóa: {report.get('duplicates_removed', 0)}")

    print("\n[NOTE] MISSING VALUES ĐÃ XỬ LÝ:")
    missing = report.get("missing") or {}
    if missing:
        for col, info in missing.items():
            print(f"  {col}: {info['filled']} giá trị -> dùng {info['strategy']}")
    else:
        print("  Không có missing values")

    print("\n[NOTE] OUTLIERS:")
    outliers = report.get("outliers")
    if isinstance(outliers, dict) and outliers:
        for col, info in outliers.items():
            print(f"  {col}: {info['n_capped']} giá trị capped [{info['q01']} - {info['q99']}]")
    else:
        print(f"  {outliers}")

    print("\n[SCALE]️  KIỂM TRA MẤT CÂN BẰNG:")
    imb = report.get("imbalance") or {}
    if imb:
        print(f"  Không churn : {imb.get('no_churn', 0):,}")
        print(f"  Churn       : {imb.get('churn', 0):,}")
        print(f"  Tỉ lệ       : {imb.get('ratio', '—')}:1")
    if imb.get("imbalanced"):
        print("  [!]️  Dữ liệu MẤT CÂN BẰNG -> sẽ xử lý bằng SMOTE ở bước tiếp theo")
    else:
        print("  [OK] Dữ liệu tương đối cân bằng")

    final_shape = report.get("final_shape") or {}
    print(f"\n[OK] Shape sau làm sạch: {final_shape.get('rows', 0):,} × {final_shape.get('cols', 0)}")

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    df = load_data()
    df_clean, report = clean_pipeline(df)
    print_cleaning_report(report)

    df_clean.to_csv("data/processed/cleaned_data.csv", index=False)

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super(NpEncoder, self).default(obj)

    with open("outputs/cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NpEncoder)

    print("\n[OK] Đã lưu cleaned_data.csv & cleaning_report.json")
