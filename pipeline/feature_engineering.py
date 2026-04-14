import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from pipeline.data_cleaning import load_data, clean_pipeline

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM TRANSFORMERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_one_hot_encoder(cat_features):
    try:
        return OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for Winsorization (Capping Outliers).
    Fits thresholds only on Training data to prevent Data Leakage.
    """
    def __init__(self, columns=None, lower_quantile=0.01, upper_quantile=0.99):
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.thresholds_ = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in self.columns:
            if col in X.columns:
                self.thresholds_[col] = {
                    'lower': X[col].quantile(self.lower_quantile),
                    'upper': X[col].quantile(self.upper_quantile)
                }
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, limits in self.thresholds_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(lower=limits['lower'], upper=limits['upper'])
        return X_copy

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (SCIENTIFIC REBUILD)
# ══════════════════════════════════════════════════════════════════════════════

def fe_pipeline():
    print("[1] Loading & Cleaning data...")
    df_raw = load_data()
    df_clean, _ = clean_pipeline(df_raw)
    
    # 1. ORDINAL ENCODING (Cho biến có thứ bậc) - Làm trước khi split vì mapping cố định
    loyalty_map = {'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
    if 'loyalty_level' in df_clean.columns:
        df_clean['loyalty_level'] = df_clean['loyalty_level'].map(loyalty_map).fillna(0)
        
    TARGET = "exit"
    
    # Các biến phân loại cần One-Hot Encoding
    cat_features = ["gender", "customer_segment", "digital_behavior"]
    # Các biến số cần Winsorize và Scaling
    num_features = [
        "credit_sco", "age", "balance", "monthly_ir", "tenure_ye", "married", 
        "nums_card", "nums_service", "active_member", "engagement_score", 
        "risk_score", "loyalty_level"
    ]
    
    # 2. SPLIT TRAIN-TEST (80-20)
    # TÁCH TRƯỚC KHI XỬ LÝ THỐNG KÊ ĐỂ CHỐNG DATA LEAKAGE
    print(f"[2] Splitting Data into Train/Test (80-20)...")
    X = df_clean[num_features + cat_features]
    y = df_clean[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Lưu trữ Artifacts
    os.makedirs("outputs", exist_ok=True)
    df_clean.to_csv("outputs/processed_data.csv", index=False)
    
    print("[OK] Feature Engineering completed.")
    return {
        "df_clean": df_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_features": num_features,
        "cat_features": cat_features,
    }

def run_feature_engineering():
    return fe_pipeline()

if __name__ == "__main__":
    fe_pipeline()
