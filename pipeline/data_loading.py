import pandas as pd
import numpy as np
import json
import os

# ══════════════════════════════════════════════════════════════════════════════
# ĐỌC & KIỂM TRA DỮ LIỆU THÔ
# Mục tiêu: Hiểu cấu trúc, kiểu dữ liệu, quy mô và chất lượng ban đầu
# của dataset trước khi bước vào bất kỳ xử lý nào.
# Lý do tại sao: Bỏ qua bước này dễ dẫn đến "garbage in, garbage out" -
# mô hình học trên dữ liệu sai -> kết quả sai hoàn toàn dù thuật toán tốt.
# ══════════════════════════════════════════════════════════════════════════════

FILE_PATH = "data/raw/bank_churn_dataset_80k.csv"

def load_data(path=FILE_PATH):
    """
    Đọc file CSV vào DataFrame.
    Dùng pandas vì tối ưu cho tabular data, hỗ trợ encoding UTF-8
    để xử lý tên tiếng Việt trong dataset.
    """
    df = pd.read_csv(path, encoding='utf-8')
    return df

def inspect_data(df):
    """
    Kiểm tra toàn diện chất lượng dữ liệu thô.
    Tại sao cần: Mỗi vấn đề dữ liệu ảnh hưởng khác nhau đến mô hình:
    - Sai kiểu -> phép tính số học bị lỗi
    - Missing -> NaN lan truyền qua toàn bộ pipeline
    - Duplicate -> mô hình overfit trên mẫu đó
    - Imbalance -> mô hình thiên về class đa số
    """
    report = {}

    # ── 1. Quy mô dataset ─────────────────────────────────────────────────────
    report["n_rows"]    = int(df.shape[0])
    report["n_cols"]    = int(df.shape[1])
    report["columns"]   = df.columns.tolist()

    # ── 2. Kiểu dữ liệu từng cột ──────────────────────────────────────────────
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # ── 3. Thống kê mô tả (numeric) ───────────────────────────────────────────
    desc = df.describe(include="all").fillna("").astype(str)
    report["describe"] = desc.to_dict()

    # ── 4. Missing values ─────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }

    # ── 5. Duplicates ─────────────────────────────────────────────────────────
    report["duplicates"] = int(df.duplicated().sum())

    # ── 6. Phân phối target (exit) ────────────────────────────────────────────
    target_counts = df["exit"].value_counts().to_dict()
    report["target_distribution"] = {str(k): int(v) for k, v in target_counts.items()}

    # ── 7. Sample 5 dòng đầu ─────────────────────────────────────────────────
    report["sample"] = df.head(5).fillna("").astype(str).to_dict(orient="records")

    # ── 8. Unique values cho categorical ─────────────────────────────────────
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    report["unique_values"] = {
        col: df[col].nunique() for col in cat_cols
    }

    return report

def print_summary(report):
    print("=" * 60)
    print("[DATA] RAW DATA OVERVIEW")
    print("=" * 60)
    print(f"  Rows      : {report['n_rows']:,}")
    print(f"  Cols      : {report['n_cols']}")
    print(f"  Duplicates: {report['duplicates']}")
    print()

    print("[NOTE] MISSING VALUES:")
    if report["missing"]:
        for col, info in report["missing"].items():
            print(f"  {col}: {info['count']} ({info['pct']}%)")
    else:
        print("  No missing values")
    print()

    print("TARGET DISTRIBUTION (exit):")
    for k, v in report["target_distribution"].items():
        print(f"  {k}: {v:,}")
    print()

    print("UNIQUE VALUES (categorical):")
    for col, n in report["unique_values"].items():
        print(f"  {col}: {n} values")
    print()

    print("[PLOT] DATA TYPES:")
    for col, dtype in report["dtypes"].items():
        print(f"  {col}: {dtype}")

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    df = load_data()
    report = inspect_data(df)
    print_summary(report)

    # Lưu report để các bước sau dùng
    with open("outputs/loading_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[OK] Đã lưu loading_report.json")
    print(f"[OK] Dataset gốc: {report['n_rows']:,} dòng × {report['n_cols']} cột")
