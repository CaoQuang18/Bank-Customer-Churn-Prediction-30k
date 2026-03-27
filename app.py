from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os

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
        "scaler_kmeans": "outputs/scaler_kmeans.pkl",
        "encoders": "outputs/encoders.pkl",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)

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
            with open(path, encoding="utf-8") as f:
                artifacts[key] = json.load(f)

    if os.path.exists("data/processed/clustered_data.csv"):
        artifacts["clustered_df"] = pd.read_csv("data/processed/clustered_data.csv")

    return artifacts


artifacts = load_artifacts()


def get_best_model():
    results = artifacts.get("results", {})
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

CAT_COLS = ["gender", "customer_segment", "loyalty_level", "digital_behavior"]

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


@app.route("/api/overview")
def api_overview():
    eda = artifacts.get("eda", {})
    overview = eda.get("overview", {})
    results = artifacts.get("results", {})

    # Lấy best model
    best = (
        max(results.items(), key=lambda x: x[1]["optimal_threshold"]["roc_auc"])
        if results
        else None
    )

    return jsonify(
        {
            "overview": overview,
            "best_model": {
                "name": best[0] if best else "N/A",
                "roc_auc": best[1]["optimal_threshold"]["roc_auc"] if best else 0,
                "recall": best[1]["optimal_threshold"]["recall"] if best else 0,
                "f1": best[1]["optimal_threshold"]["f1"] if best else 0,
            }
            if best
            else {},
            "n_clusters": len(artifacts.get("cluster_profiles", [])),
            "top_features": artifacts.get("feat_imp", {}).get("top3", []),
        }
    )


@app.route("/api/eda")
def api_eda():
    return jsonify(artifacts.get("eda", {}))


@app.route("/api/models")
def api_models():
    return jsonify(
        {
            "results": artifacts.get("results", {}),
            "comparison": artifacts.get("comparison", {}),
            "feat_imp": artifacts.get("feat_imp", {}),
            "cv_results": artifacts.get("cv_results", {}),
            "model_version": artifacts.get("model_version", {}),
            "imbalance_analysis": artifacts.get("imbalance_analysis", {}),
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
            "profiles": profiles,
            "strategies": strategies,
            "customers": cluster_customers,
            "elbow_data": artifacts.get("elbow_data", {}),
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    model_name = data.get("model", "XGBoost")

    try:
        input_df = pd.DataFrame(
            [
                {
                    "credit_sco": float(data["credit_sco"]),
                    "gender": data["gender"],
                    "age": float(data["age"]),
                    "balance": float(data["balance"]),
                    "monthly_ir": float(data["monthly_ir"]),
                    "tenure_ye": float(data["tenure_ye"]),
                    "married": int(data["married"]),
                    "nums_card": int(data["nums_card"]),
                    "nums_service": int(data["nums_service"]),
                    "active_member": int(data["active_member"]),
                    "engagement_score": float(data["engagement_score"]),
                    "risk_score": float(data["risk_score"]),
                    "customer_segment": data["customer_segment"],
                    "loyalty_level": data["loyalty_level"],
                    "digital_behavior": data["digital_behavior"],
                }
            ]
        )

        # Encode
        encoders = artifacts.get("encoders", {})
        for col in CAT_COLS:
            if col in encoders:
                le = encoders[col]
                val = input_df[col].astype(str).values[0]
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    input_df[col] = 0

        X = input_df[FEATURES]

        # Chọn model
        model_map = {
            "Logistic Regression": ("model_lr", True),
            "Random Forest": ("model_rf", False),
            "XGBoost": ("model_xgb", False),
        }
        model_key, use_scaled = model_map.get(model_name, ("model_xgb", False))
        model = artifacts.get(model_key)

        if use_scaled:
            scaler = artifacts.get("scaler")
            if scaler is not None:
                X_input = scaler.transform(X)
            else:
                X_input = X.values
        else:
            X_input = X.values

        if model is None:
            return jsonify({"error": f"Model {model_name} not loaded"}), 400

        proba = float(model.predict_proba(X_input)[0][1])
        threshold = float(
            artifacts.get("results", {})
            .get(model_name, {})
            .get("optimal_threshold", {})
            .get("threshold", 0.5)
        )
        prediction = int(proba >= threshold)

        # 🧠 PHÂN TÍCH NGUYÊN NHÂN THEO TRỌNG SỐ MÔ HÌNH (Model-driven)
        reasons = []
        model_key_shap = model_name.lower().replace(" ", "_")
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
        sorted_feats = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)
        
        for feat, imp in sorted_feats:
            if feat in risk_configs:
                conf = risk_configs[feat]
                val = float(data.get(feat, 0))
                is_risky = False
                if conf["op"] == ">" and val > float(conf["val"]): is_risky = True
                elif conf["op"] == "<" and val < float(conf["val"]): is_risky = True
                elif conf["op"] == "==" and val == float(conf["val"]): is_risky = True
                
                if is_risky:
                    reasons.append(f"{conf['icon']} {conf['label']} bất lợi ({val}) - AI xác định đây là yếu tố rủi ro then chốt.")
            
            if len(reasons) >= 3: break # Chỉ lấy top 3 nguyên nhân then chốt nhất
            
        if not reasons:
            reasons = ["💡 Các chỉ số cơ bản khá ổn định, rủi ro có thể nằm ở các yếu tố phi định lượng khác."]

        # 🎯 CHIẾN LƯỢC CÁ NHÂN HÓA (Personalized Recommendations)
        suggestions = []
        if prediction == 1:
            if float(data.get("monthly_ir", 0)) > 50000000:
                suggestions.append("💎 Ưu tiên gói giữ chân khách hàng VIP (Phòng chờ/Sân Golf).")
            if float(data.get("engagement_score", 0)) < 30:
                suggestions.append("📱 Tăng cường Push Notification và ưu đãi App để cải thiện tương tác.")
            if int(data.get("nums_service", 0)) < 2:
                suggestions.append("📦 Chiến dịch Bán chéo (Cross-selling): Tặng tiền gửi khi dùng thêm thẻ/vay.")
            
            if len(suggestions) < 3:
                suggestions.extend([
                    "📞 RM cấp quản lý liên hệ trực tiếp trong 24h.",
                    "🎁 Gửi voucher tri ân giảm phí giao dịch năm tới."
                ])
        else:
            suggestions = [
                "✅ Tiếp tục duy trì chuẩn chăm sóc hiện tại.",
                "📈 Khai thác Upsell gói vay/đầu tư ưu đãi.",
                "🌟 Ghi nhận và mời khách hàng vào CLB Khách hàng thân thiết."
            ]

        return jsonify(
            {
                "prediction": prediction,
                "probability": float(f"{proba * 100.0:.2f}"),
                "threshold": float(f"{threshold * 100.0:.2f}"),
                "model_used": model_name,
                "label": "⚠️ Có NGUY CƠ rời bỏ" if prediction == 1 else "✅ Khách hàng ỔN ĐỊNH",
                "reasons": reasons,
                "suggestions": suggestions,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
