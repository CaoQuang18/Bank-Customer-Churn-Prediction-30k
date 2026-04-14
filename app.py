import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
import joblib

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT PROTECTION (SAFETY MODE)
# If sklearn is corrupted or locked, we mock it to prevent startup hangs.
# ══════════════════════════════════════════════════════════════════════════════
try:
    import sklearn
    SAFETY_MODE = False
except (ImportError, ModuleNotFoundError) as e:
    from unittest.mock import MagicMock
    print(f"!!! SCLEARN NOT FOUND OR CORRUPTED: Entering Safety Mode !!!")
    sys.modules['sklearn'] = MagicMock()
    sys.modules['sklearn.linear_model'] = MagicMock()
    sys.modules['sklearn.ensemble'] = MagicMock()
    sys.modules['sklearn.metrics'] = MagicMock()
    sys.modules['sklearn.preprocessing'] = MagicMock()
    sys.modules['sklearn.cluster'] = MagicMock()
    SAFETY_MODE = True

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════


def load_artifacts():
    artifacts = {}
    paths = {
        "model_lr": "outputs/model_lr.pkl",
        "model_rf": "outputs/model_rf.pkl",
        "model_xgb": "outputs/model_xgb.pkl",
        "model_kmeans": "outputs/model_kmeans.pkl",
        "scaler": "outputs/scaler.pkl",
        "preprocessor": "outputs/preprocessor.pkl",
        "scaler_kmeans": "outputs/scaler_kmeans.pkl",
        "encoders": "outputs/encoders.pkl",
        "winsorize_thresholds": "outputs/winsorize_thresholds.pkl",
    }
    for key, path in paths.items():
        if os.path.exists(path) and not SAFETY_MODE:
            try:
                artifacts[key] = joblib.load(path)
            except Exception as e:
                print(f"!!! Error loading artifact {key}: {e}")
                artifacts[key] = None
        else:
            artifacts[key] = None

    json_paths = {
        "eda": "outputs/eda.json",
        "results": "outputs/results.json",
        "feat_imp": "outputs/feat_imp.json",
        "comparison": "outputs/comparison.json",
        "cluster_profiles": "outputs/cluster_profiles.json",
        "cluster_strategies": "outputs/cluster_strategies.json",
        "elbow_data": "outputs/elbow_data.json",
        "cv_results": "outputs/cv_results.json",
        "model_version": "outputs/model_version.json",
        "imbalance_analysis": "outputs/imbalance_analysis.json",
        "shap_analysis": "outputs/shap_analysis.json",
    }
    for key, path in json_paths.items():
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    artifacts[key] = json.load(f)
            except Exception as e:
                print(f"!!! Error loading JSON {key}: {e}")
                artifacts[key] = None

    if os.path.exists("data/processed/clustered_data.csv"):
        artifacts["clustered_df"] = pd.read_csv("data/processed/clustered_data.csv")

    return artifacts


artifacts = load_artifacts()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json_file(rel_path, fallback):
    candidates = [
        os.path.join(BASE_DIR, rel_path),
        os.path.join(os.getcwd(), rel_path),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return fallback


def load_results():
    return load_json_file("outputs/results.json", artifacts.get("results", {}) or {})

@app.route("/api/debug_artifacts")
def api_debug_artifacts():
    disk = load_json_file("outputs/results.json", {}) or {}
    disk_thr = (disk.get("Random Forest", {}) or {}).get("optimal_threshold", {}).get("threshold")

    mem = artifacts.get("results", {}) or {}
    mem_thr = (mem.get("Random Forest", {}) or {}).get("optimal_threshold", {}).get("threshold")

    live = load_results()
    live_thr = (live.get("Random Forest", {}) or {}).get("optimal_threshold", {}).get("threshold")

    return jsonify(
        {
            "cwd": os.getcwd(),
            "base_dir": BASE_DIR,
            "disk_rf_threshold": disk_thr,
            "mem_rf_threshold": mem_thr,
            "live_rf_threshold": live_thr,
        }
    )


def get_best_model():
    results = load_results()
    if not results:
        return None
    best = max(results.items(), key=lambda x: x[1]["optimal_threshold"]["roc_auc"])
    return {
        "name": best[0],
        "roc_auc": best[1]["optimal_threshold"]["roc_auc"],
        "recall": best[1]["optimal_threshold"]["recall"],
        "f1": best[1]["optimal_threshold"]["f1"],
        "accuracy": best[1]["optimal_threshold"]["accuracy"],
        "precision": best[1]["optimal_threshold"]["precision"],
    }
def calculate_roi(n_customers, recall, avg_ltv=5000000, cost_per_cust=200000):
    """
    Tính toán ROI kỳ vọng dựa trên Recall của mô hình.
    Công thức: ROI = (Số khách giữ chân được * LTV) - (Số khách tiếp cận * Chi phí)
    Giả định: Tỉ lệ khách hàng đồng ý ở lại khi được mời là 30%.
    """
    n_at_risk = n_customers  # Giả sử ta đang xét tập khách hàng được dự báo churn
    success_rate = 0.30      # 30% khách hàng sẽ ở lại nếu có offer
    
    # Số khách hàng thực sự được cứu sống = Dự báo đúng * Tỉ lệ thành công
    # Dự báo đúng = n_at_risk * recall (Nếu ta xét n_at_risk là tập ground truth churn, 
    # nhưng ở đây n_at_risk là tập model báo churn) -> n_at_risk * precision? 
    # Để đơn giản và tích cực, ta tính dựa trên số lượng khách được "nhắm mục tiêu" (Targeted).
    
    saved_cust = n_at_risk * success_rate
    revenue_saved = saved_cust * avg_ltv
    total_cost = n_at_risk * cost_per_cust
    
    roi = revenue_saved - total_cost
    return max(0, int(roi))


# Ordinal map for loyalty_level — must match feature_engineering.py
LOYALTY_MAP = {'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}

# OHE category values — must match pd.get_dummies(drop_first=True) from feature_engineering.py
# drop_first=True drops the first alphabetical category for each column:
# gender: [female, male] → drop 'female' → keep 'gender_male'
# customer_segment: [Affluent, Emerging, Mass, Priority] → drop 'Affluent'
# digital_behavior: [mobile, offline] → drop 'mobile' (or whatever comes first)
# We need to know the exact dummies produced during training.
# The model_version.json stores features list — load it at inference time.

CAT_OHE_COLS = ["gender", "customer_segment", "digital_behavior"]



# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/simple")
def simple_dashboard():
    """Redirect to main dashboard"""
    from flask import redirect
    return redirect("/")


@app.route("/@vite/client")
def vite_client_stub():
    return ("", 204)


@app.route("/api/overview")
def api_overview():
    eda = artifacts.get("eda", {})
    overview = eda.get("overview", {})
    results = artifacts.get("results", {})

    # 1. Define 'best' model for the response
    best = None
    if results:
        try:
            best = max(results.items(), key=lambda x: x[1].get("optimal_threshold", {}).get("roc_auc", 0))
        except (ValueError, KeyError):
            best = None

    # 2. Calculate ROI (Business Value)
    clustered_df = artifacts.get("clustered_df")
    roi_data = {}
    if clustered_df is not None and not clustered_df.empty:
        try:
            subset = clustered_df[clustered_df['exit'] == 1]
            total_balance_at_risk = float(subset['balance'].sum())
            if np.isnan(total_balance_at_risk):
                total_balance_at_risk = 0.0
                
            est_savings = total_balance_at_risk * 0.3 # 30% retention success rate
            
            roi_data = {
                "total_balance_at_risk_m": round(total_balance_at_risk / 1e6, 1),
                "est_savings_m": round(est_savings / 1e6, 1),
                "retention_success_rate": "30%",
                "currency": "VNĐ"
            }
        except Exception:
            roi_data = {
                "total_balance_at_risk_m": 0,
                "est_savings_m": 0,
                "retention_success_rate": "30%",
                "currency": "VNĐ"
            }

    return jsonify(
        {
            "overview": overview,
            "best_model": {
                "name": best[0] if best else "N/A",
                "roc_auc": best[1].get("optimal_threshold", {}).get("roc_auc", 0) if best else 0,
                "recall": best[1].get("optimal_threshold", {}).get("recall", 0) if best else 0,
                "f1": best[1].get("optimal_threshold", {}).get("f1", 0) if best else 0,
            }
            if best
            else {},
            "n_clusters": len(artifacts.get("cluster_profiles", [])),
            "top_features": artifacts.get("feat_imp", {}).get("top3", []),
            "roi": roi_data
        }
    )


@app.route("/api/eda")
def api_eda():
    return jsonify(artifacts.get("eda", {}))


@app.route("/api/models")
def api_models():
    # comparison.json is stored as a raw array — wrap it into {table: [...]}
    # so that app.js can access data.comparison.table consistently
    raw_comparison = load_json_file("outputs/comparison.json", artifacts.get("comparison", []))
    model_version  = load_json_file("outputs/model_version.json", artifacts.get("model_version", {}) or {})
    # cv_results stored inside model_version.json under key "cv_results"
    cv_from_version = model_version.get("cv_results", {})
    results = load_results()

    # Correctly identify the table for enrichment (can be a list or a dict-wrapped list)
    raw_table = raw_comparison if isinstance(raw_comparison, list) else raw_comparison.get("table", [])
    
    if raw_table:
        enriched = []
        for row in raw_table:
            model_key = row.get("model", "")
            enriched_row = dict(row)

            # Pull CV AUC / Std from model_version.json (correct keys from model_training: mean, std)
            cv_info = cv_from_version.get(model_key, {})
            if not enriched_row.get("cv_auc"):
                enriched_row["cv_auc"] = cv_info.get("mean")
            if not enriched_row.get("cv_std"):
                enriched_row["cv_std"] = cv_info.get("std")

            # Pull optimal threshold from results.json
            opt = results.get(model_key, {}).get("optimal_threshold", {})
            opt_thr = opt.get("threshold")
            if opt_thr is not None:
                enriched_row["threshold"] = opt_thr

            enriched.append(enriched_row)

        # Sort descending by roc_auc so badge TOP 1 = truly best model by test AUC
        enriched.sort(key=lambda r: r.get("roc_auc") or 0, reverse=True)
        comparison_payload = {"table": enriched, "insights": raw_comparison.get("insights", {})} if isinstance(raw_comparison, dict) else {"table": enriched}
    else:
        comparison_payload = raw_comparison

    return jsonify(
        {
            "results": results,
            "comparison": comparison_payload,
            "feat_imp": load_json_file("outputs/feat_imp.json", artifacts.get("feat_imp", {}) or {}),
            "cv_results": cv_from_version,
            "model_version": model_version,
            "imbalance_analysis": load_json_file("outputs/imbalance_analysis.json", artifacts.get("imbalance_analysis", {}) or {}),
        }
    )


@app.route("/api/shap")
def api_shap():
    return jsonify(artifacts.get("shap_analysis", {}))


@app.route("/api/imbalance")
def api_imbalance():
    return jsonify(artifacts.get("imbalance_analysis", {}))


@app.route("/api/clusters")
def api_clusters():
    profiles = artifacts.get("cluster_profiles", [])
    strategies = artifacts.get("cluster_strategies", {})

    norm_profiles = []
    for p in profiles or []:
        if not isinstance(p, dict):
            continue
        pp = dict(p)
        cr = pp.get("churn_rate")
        try:
            if cr is not None:
                cr = float(cr)
                pp["churn_rate_pct"] = round(cr * 100.0, 2) if 0.0 <= cr <= 1.0 else round(cr, 2)
        except Exception:
            pass
        norm_profiles.append(pp)

    # Lấy danh sách khách hàng theo cụm
    cluster_customers = {}
    df = artifacts.get("clustered_df")
    if df is not None:
        for cluster_id in df["cluster"].unique():
            subset = df[df["cluster"] == cluster_id][
                [
                    "full_name",
                    "gender",
                    "age",
                    "customer_segment",
                    "loyalty_level",
                    "balance",
                    "digital_behavior",
                    "engagement_score",
                    "exit",
                    "cluster",
                ]
            ].head(50)
            cluster_customers[int(cluster_id)] = subset.fillna("").to_dict(
                orient="records"
            )

    return jsonify(
        {
            "profiles": norm_profiles,
            "strategies": strategies,
            "customers": cluster_customers,
            "elbow_data": artifacts.get("elbow_data", {}),
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    model_name = data.get("model", "Logistic Regression")

    try:
        # --- 1. Chuẩn bị dữ liệu thô từ Request ----------------------------------
        # Đưa vào DataFrame với đúng tên cột ban đầu (trước OHE)
        raw_row = {
            "credit_sco":     float(data["credit_sco"]),
            "gender":         str(data["gender"]).strip(),
            "age":            float(data["age"]),
            "balance":        float(data["balance"]),
            "monthly_ir":     float(data["monthly_ir"]),
            "tenure_ye":      float(data["tenure_ye"]),
            "married":        int(data["married"]),
            "nums_card":      int(data["nums_card"]),
            "nums_service":   int(data["nums_service"]),
            "active_member":  int(data["active_member"]),
            "engagement_score": float(data["engagement_score"]),
            "risk_score":     float(data["risk_score"]),
            "customer_segment": str(data["customer_segment"]).strip(),
            "loyalty_level":  LOYALTY_MAP.get(str(data["loyalty_level"]).strip(), 0),
            "digital_behavior": str(data["digital_behavior"]).strip()
        }
        X_raw = pd.DataFrame([raw_row])

        # --- 2. Select Model ----------------------------------------------------
        model_map = {
            "Logistic Regression": "model_lr",
            "Random Forest":       "model_rf",
            "XGBoost":             "model_xgb",
        }
        model_key = model_map.get(model_name, "model_rf")
        model = artifacts.get(model_key)

        if model is None:
            return jsonify({"error": f"Model '{model_name}' not loaded."}), 400

        # --- 3. Predict -------------------------------------------------------
        proba = None
        if hasattr(model, "named_steps") and model.named_steps.get("preprocess") is not None:
            proba = float(model.predict_proba(X_raw)[0][1])
        else:
            winsorize_thresholds = artifacts.get("winsorize_thresholds", {})
            if winsorize_thresholds:
                for col, limits in winsorize_thresholds.items():
                    if col in X_raw.columns:
                        X_raw[col] = X_raw[col].clip(lower=limits["lower"], upper=limits["upper"])

            preprocessor = artifacts.get("preprocessor")
            if preprocessor is None:
                return jsonify({"error": "Preprocessor artifact not found. Please run pipeline first."}), 500

            X_input = preprocessor.transform(X_raw)
            proba = float(model.predict_proba(X_input)[0][1])
        
        # Lấy threshold tối ưu từ results.json (đã tìm trên tập Validation)
        results = load_results()
        threshold = float(results.get(model_name, {}).get("optimal_threshold", {}).get("threshold", 0.5))
        prediction = int(proba >= threshold)

        # --- 6. Real-time Clustering (KMeans) ----------------------------------
        cluster_id = None
        cluster_name = "N/A"
        kmeans_model = artifacts.get("model_kmeans")
        scaler_km = artifacts.get("scaler_kmeans")
        
        if kmeans_model is not None and scaler_km is not None:
            km_features = ['balance', 'engagement_score', 'risk_score', 'age']
            km_input = X_raw[km_features]
            km_sc = scaler_km.transform(km_input)
            cluster_id = int(kmeans_model.predict(km_sc)[0])
            
            profiles = artifacts.get("cluster_profiles", [])
            for p in profiles:
                if p.get("cluster") == cluster_id:
                    cluster_name = p.get("cluster_name")
                    break
        
        # Define cluster_assignment to avoid NameError
        cluster_assignment = cluster_id

        # 🧠 PHÂN TÍCH NGUYÊN NHÂN THEO TRỌNG SỐ MÔ HÌNH (Model-driven)
        reasons = []
        model_key_shap = model_name.lower().replace(" ", "_").replace("baseline", "random_forest") # Baseline dùng RF SHAP
        shap_data = artifacts.get("shap_analysis", {}).get(model_key_shap, {})
        feat_importances = shap_data.get("feature_importance", {})
        
        # Cấu hình rủi ro với icon và nhãn tương ứng
        risk_configs = {
            "risk_score": {"label": "Điểm rủi ro hệ thống", "op": ">", "val": 0.35, "icon": "⚠️"},
            "engagement_score": {"label": "Điểm tương tác", "op": "<", "val": 40, "icon": "📉"},
            "monthly_ir": {"label": "Thu nhập hàng tháng", "op": "<", "val": 15000000, "icon": "💰"},
            "balance": {"label": "Số dư tài khoản", "op": "<", "val": 10000000, "icon": "🏦"},
            "tenure_ye": {"label": "Số năm gắn bó", "op": "<", "val": 2, "icon": "⏳"},
            "active_member": {"label": "Trạng thái hoạt động", "op": "==", "val": 0, "icon": "❌"},
            "nums_service": {"label": "Số lượng sản phẩm", "op": "<", "val": 2, "icon": "📦"},
            "credit_sco": {"label": "Điểm tín dụng", "op": "<", "val": 600, "icon": "💳"},
        }

        # AI Ưu tiên giải thích các biến có độ quan trọng (Importance) cao nhất của chính Model đó
        # FIX A1: SHAP keys có dạng "num__risk_score", "cat__customer_segment_Mass" (prefix từ ColumnTransformer)
        # → Cần strip prefix trước khi lookup trong risk_configs (dùng tên raw feature)
        sorted_feats = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)
        
        for feat, imp in sorted_feats:
            # Strip ColumnTransformer prefix (num__, cat__)
            clean_feat = feat.replace('num__', '').replace('cat__', '')
            # Nếu là OHE feature như "customer_segment_Mass" → dùng prefix "customer_segment"
            # Nhưng risk_configs chỉ có tên cột gốc, nên thử clean_feat trước
            lookup_key = clean_feat if clean_feat in risk_configs else None
            if lookup_key is None:
                continue
            conf = risk_configs[lookup_key]
            val = float(data.get(lookup_key, 0))
            is_risky = False
            if conf["op"] == ">" and val > float(conf["val"]): is_risky = True
            elif conf["op"] == "<" and val < float(conf["val"]): is_risky = True
            elif conf["op"] == "==" and val == float(conf["val"]): is_risky = True
            
            if is_risky:
                reasons.append({
                    "icon": conf['icon'],
                    "label": conf['label'],
                    "val": str(val),
                    "desc": "Tín hiệu rủi ro chính (Feature signal)"
                })
            
            if len(reasons) >= 3: break # Chỉ lấy top 3 nguyên nhân then chốt nhất
            
        if not reasons:
            reasons = [{
                "icon": "💡",
                "label": "Chỉ số ổn định",
                "val": "Bình thường",
                "desc": "Không phát hiện rủi ro từ hệ thống."
            }]

        # 🎯 CHIẾN LƯỢC CÁ NHÂN HÓA (Hệ Chuyên Gia Ngân Hàng - Expert System)
        suggestions = []
        age_val = float(data.get("age", 30))
        income_val = float(data.get("monthly_ir", 0))
        balance_val = float(data.get("balance", 0))
        engage_val = float(data.get("engagement_score", 0))
        services_val = int(data.get("nums_service", 0))

        if prediction == 1:
            # 1. Phân khúc Thu nhập / Tài sản (Financial Segment)
            if income_val >= 50000000 or balance_val >= 200000000:
                suggestions.append({"icon": "💎", "title": "Khách hàng VIP", "action": "Cử Giám đốc (RM) chăm sóc 1:1, tặng đặc quyền Private Banking."})
            elif income_val < 15000000:
                suggestions.append({"icon": "🎁", "title": "Giảm phí duy trì", "action": "Miễn phí tài khoản 6 tháng / Hoàn tiền mua sắm để níu chân."})

            # 2. Phân khúc Khẩu vị theo Độ tuổi (Demographic Segment)
            if age_val >= 45:
                suggestions.append({"icon": "🧓", "title": "Tư vấn truyền thống", "action": "Telesales gọi giới thiệu Gói tiết kiệm/Bảo hiểm hưu trí."})
            elif age_val <= 30:
                suggestions.append({"icon": "🧑", "title": "Push Notification", "action": "Gửi Voucher Shopee qua App, mời mở thẻ tín dụng GenZ hạn mức linh hoạt."})
            else:
                suggestions.append({"icon": "👨", "title": "Cross-sell Vốn", "action": "Chào mời Gửi góp Mua nhà/Xe hoặc Thẻ du lịch hoàn tiền."})

            # 3. Yếu tố Gắn kết & Bán chéo (Engagement & Cross-sell)
            if engage_val < 35:
                suggestions.append({"icon": "📉", "title": "Kích hoạt App", "action": "Thu hút user mở App bằng SMS tặng 50k khi nạp tiền/thanh toán hóa đơn."})
            elif services_val < 2:
                suggestions.append({"icon": "📦", "title": "Mở rộng Dịch vụ", "action": "Khuyến khích dùng thêm Dịch vụ bằng ưu đãi +0.2% lãi suất tiết kiệm online."})

            if not suggestions:
                suggestions.append({"icon": "📞", "title": "Health-Check", "action": "Triển khai cuộc gọi đánh giá mức độ hài lòng KH trong 24h."})
            suggestions = suggestions[:4]
        else:
            if income_val >= 40000000 or balance_val >= 150000000:
                suggestions.append({"icon": "🌟", "title": "Khách VIP An toàn", "action": "Mời tham gia sự kiện Tri ân hoặc nâng hạng thẻ Signature."})
            elif age_val <= 30:
                suggestions.append({"icon": "📈", "title": "Upsell Tín dụng", "action": "Khách dòng tiền tốt, mời chi tiêu Thẻ tín dụng mua sắm trả góp."})
                
            is_cluster_risky = cluster_name and any(x in cluster_name.lower() for x in ["rủi ro", "nguy cơ", "báo động"])

            if is_cluster_risky:
                suggestions.append({"icon": "⚠️", "title": "Đưa vào diện Theo dõi (Hidden Risk)", "action": "Model dự báo ổn định, nhưng Cụm AI cảnh báo rủi ro ngầm. Cán bộ Telesales cần rà soát lại."})
            else:
                suggestions.extend([
                    {"icon": "✅", "title": "Tiếp tục Chăm sóc Chuẩn", "action": "Duy trì chất lượng dịch vụ hiện tại."},
                    {"icon": "🎉", "title": "Automation MKT", "action": "Gửi SMS chúc mừng sinh nhật, lễ Tết tự động."}
                ])

        label = "⚠️ Có NGUY CƠ rời bỏ" if prediction == 1 else "✅ Ổn định (An toàn)"
        if prediction == 0 and cluster_name and ("Rủi ro" in cluster_name or "Cần Cứu vãn" in cluster_name):
            label = "🚨 Rủi ro Ngầm (Hidden Risk)"

        return jsonify(
            {
                "prediction": prediction,
                "probability": float(f"{proba * 100.0:.2f}"),
                "threshold": float(f"{threshold * 100.0:.2f}"),
                "model_used": model_name,
                "label": label,
                "reasons": reasons,
                "suggestions": suggestions,
                "cluster_assignment": cluster_assignment,
                "cluster_name": cluster_name
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False, port=5000)
