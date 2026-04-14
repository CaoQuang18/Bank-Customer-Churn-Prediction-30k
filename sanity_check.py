import json
import os
import sys
from typing import Any, Dict, List

# FIX: Windows cp1252 không hỗ trợ tiếng Việt → force UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import joblib
import pandas as pd



def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _require_files(paths: List[str]) -> None:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        _fail("Missing files: " + ", ".join(missing))


def _is_pipeline_with_preprocess(model: Any) -> bool:
    return hasattr(model, "named_steps") and getattr(model, "named_steps", {}).get("preprocess") is not None


def main() -> None:
    required = [
        "outputs/results.json",
        "outputs/comparison.json",
        "outputs/model_lr.pkl",
        "outputs/model_rf.pkl",
        "outputs/model_xgb.pkl",
        "outputs/model_kmeans.pkl",
        "outputs/scaler_kmeans.pkl",
        "outputs/cluster_profiles.json",
        "outputs/elbow_data.json",
    ]
    _require_files(required)

    results: Dict[str, Any] = _read_json("outputs/results.json")
    for m in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if m not in results:
            _fail(f"results.json missing model '{m}'")
        opt = results[m].get("optimal_threshold") or {}
        for k in ["threshold", "roc_auc", "recall", "precision", "f1", "accuracy"]:
            if k not in opt:
                _fail(f"results.json missing {m}.optimal_threshold.{k}")
    _ok("results.json schema looks valid")

    comparison = _read_json("outputs/comparison.json")
    if not isinstance(comparison, dict) or "table" not in comparison:
        _fail("comparison.json missing 'table'")
    if not isinstance(comparison["table"], list) or not comparison["table"]:
        _fail("comparison.json 'table' is empty")
    _ok("comparison.json schema looks valid")

    cluster_profiles = _read_json("outputs/cluster_profiles.json")
    if not isinstance(cluster_profiles, list) or not cluster_profiles:
        _fail("cluster_profiles.json empty")
    for p in cluster_profiles[:3]:
        for k in ["cluster", "cluster_name", "count", "churn_rate"]:
            if k not in p:
                _fail(f"cluster_profiles.json missing key '{k}'")
    _ok("cluster_profiles.json schema looks valid")

    elbow = _read_json("outputs/elbow_data.json")
    if not isinstance(elbow, dict) or "k_range" not in elbow or "inertias" not in elbow:
        _fail("elbow_data.json schema invalid")
    _ok("elbow_data.json schema looks valid")

    model_lr = joblib.load("outputs/model_lr.pkl")
    model_rf = joblib.load("outputs/model_rf.pkl")
    model_xgb = joblib.load("outputs/model_xgb.pkl")

    sample = pd.DataFrame(
        [
            {
                "credit_sco": 650.0,
                "gender": "female",
                "age": 40.0,
                "balance": 20000000.0,
                "monthly_ir": 25000000.0,
                "tenure_ye": 3.0,
                "married": 1,
                "nums_card": 2,
                "nums_service": 2,
                "active_member": 1,
                "engagement_score": 50.0,
                "risk_score": 0.2,
                "customer_segment": "Mass",
                # FIX C1: Dùng string "Bronze" thay vì int 0 — phản ánh đúng real-world
                # input flow: app.py nhận string từ API, convert qua LOYALTY_MAP → int
                # Nhưng pipeline expect int nên pre-convert ở đây để test direct pipeline call
                "loyalty_level": 0,  # 0 = Bronze (sau khi đã map)
                "digital_behavior": "mobile",
            }
        ]
    )

    for name, model in [("LR", model_lr), ("RF", model_rf), ("XGB", model_xgb)]:
        if not hasattr(model, "predict_proba"):
            _fail(f"{name} model has no predict_proba")
        if _is_pipeline_with_preprocess(model):
            proba = model.predict_proba(sample)[0][1]
        else:
            pre = joblib.load("outputs/preprocessor.pkl") if os.path.exists("outputs/preprocessor.pkl") else None
            win = joblib.load("outputs/winsorize_thresholds.pkl") if os.path.exists("outputs/winsorize_thresholds.pkl") else None
            X = sample.copy()
            if isinstance(win, dict):
                for col, lim in win.items():
                    if col in X.columns and isinstance(lim, dict) and "lower" in lim and "upper" in lim:
                        X[col] = X[col].clip(lower=lim["lower"], upper=lim["upper"])
            if pre is None:
                _warn(f"{name} model is not end-to-end pipeline and preprocessor.pkl missing; skip proba check")
                continue
            arr = pre.transform(X)
            proba = model.predict_proba(arr)[0][1]
        if not (0.0 <= float(proba) <= 1.0):
            _fail(f"{name} predict_proba out of range: {proba}")
    _ok("model predict_proba sanity check passed")

    # FIX C2: Thêm kiểm tra K-Means clustering — trước đây không được verify
    if os.path.exists("outputs/model_kmeans.pkl") and os.path.exists("outputs/scaler_kmeans.pkl"):
        kmeans = joblib.load("outputs/model_kmeans.pkl")
        scaler_km = joblib.load("outputs/scaler_kmeans.pkl")
        km_input = sample[["balance", "engagement_score", "risk_score", "age"]]
        km_sc = scaler_km.transform(km_input)
        cluster_pred = kmeans.predict(km_sc)
        if not (0 <= int(cluster_pred[0]) <= 10):
            _fail(f"K-Means cluster prediction out of range: {cluster_pred[0]}")
        _ok(f"K-Means predict sanity check passed (cluster={cluster_pred[0]})")
    else:
        _warn("K-Means pkl not found; skip cluster sanity check")

    _ok("All sanity checks passed")


if __name__ == "__main__":
    main()
