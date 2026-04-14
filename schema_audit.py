import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# FIX: Windows cp1252 không hỗ trợ tiếng Việt → force UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import requests


STRICT = False


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    if STRICT:
        _fail(msg)
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _require(paths: List[str]) -> None:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        _fail("Missing required files: " + ", ".join(missing))


def _approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def validate_results(results: Dict[str, Any]) -> None:
    for m in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if m not in results:
            _fail(f"outputs/results.json missing model '{m}'")
        opt = results[m].get("optimal_threshold") or {}
        for k in ["threshold", "roc_auc", "recall", "precision", "f1", "accuracy", "confusion_matrix"]:
            if k not in opt:
                _fail(f"outputs/results.json missing {m}.optimal_threshold.{k}")
        thr = _as_float(opt["threshold"])
        if thr is None or not (0.0 <= thr <= 1.0):
            _fail(f"{m} optimal threshold out of range: {opt.get('threshold')}")
    _ok("outputs/results.json schema + ranges look valid")


def validate_comparison(comparison: Dict[str, Any], results: Dict[str, Any]) -> None:
    table = comparison.get("table")
    if not isinstance(table, list) or not table:
        _fail("outputs/comparison.json missing non-empty 'table'")
    for row in table:
        for k in ["model", "roc_auc", "recall", "precision", "f1", "accuracy", "threshold"]:
            if k not in row:
                _fail(f"comparison.table row missing '{k}': {row}")

        m = row["model"]
        if m in results:
            thr_row = _as_float(row.get("threshold"))
            thr_res = _as_float(results[m].get("optimal_threshold", {}).get("threshold"))
            if thr_row is not None and thr_res is not None and not _approx(thr_row, thr_res, tol=5e-3):
                _warn(f"threshold mismatch {m}: comparison={thr_row} vs results={thr_res} (rounding?)")

    insights = comparison.get("insights") or {}
    best_overall = insights.get("best_overall")
    if best_overall:
        best_by_auc = max(table, key=lambda x: float(x.get("roc_auc") or 0)).get("model")
        if best_overall != best_by_auc:
            _warn(f"insights.best_overall='{best_overall}' but best AUC is '{best_by_auc}'")
    _ok("outputs/comparison.json schema looks valid")


def validate_clusters_artifacts(cluster_profiles: Any, elbow: Any) -> None:
    if not isinstance(cluster_profiles, list) or not cluster_profiles:
        _fail("outputs/cluster_profiles.json empty or invalid")
    for p in cluster_profiles:
        for k in ["cluster", "cluster_name", "count", "churn_rate", "balance_mean", "engagement_mean", "age_mean", "risk_mean"]:
            if k not in p:
                _fail(f"cluster profile missing '{k}'")
        cr = _as_float(p.get("churn_rate"))
        if cr is None or not (0.0 <= cr <= 1.0):
            _warn(f"cluster churn_rate not in [0,1]: {p.get('churn_rate')} (expected fraction)")
    _ok("outputs/cluster_profiles.json schema looks valid")

    if not isinstance(elbow, dict):
        _fail("outputs/elbow_data.json invalid")
    k_range = elbow.get("k_range")
    inertias = elbow.get("inertias")
    silhouettes = elbow.get("silhouette_scores")
    if not (isinstance(k_range, list) and isinstance(inertias, list) and isinstance(silhouettes, list)):
        _fail("elbow_data.json missing k_range/inertias/silhouette_scores lists")
    if not (len(k_range) == len(inertias) == len(silhouettes)):
        _fail("elbow_data.json list lengths mismatch")
    _ok("outputs/elbow_data.json schema looks valid")


def validate_ui_contract() -> None:
    app_js = "static/app.js"
    if not os.path.exists(app_js):
        _warn("static/app.js not found; skip UI contract checks")
        return
    with open(app_js, encoding="utf-8") as f:
        s = f.read()
    if "churn_rate_pct" not in s:
        _warn("UI does not reference churn_rate_pct; clusters may rely on churn_rate unit handling")
    if "thr-lr" not in s or "thr-rf" not in s or "thr-xgb" not in s:
        _warn("Predict guide threshold placeholders not found (thr-lr/thr-rf/thr-xgb)")
    _ok("UI contract scan completed")


def validate_ui_no_hardcoded_metrics() -> None:
    targets: List[str] = []
    for root in ["templates", "static"]:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".html") or fn.endswith(".js"):
                    targets.append(os.path.join(dirpath, fn))

    metric_re = re.compile(
        r"(?i)\b(roc[- ]?auc|auc|recall|precision|f1(?:-score)?|accuracy|threshold|ap[_ -]?score)\b[^\n]{0,60}?(\d+(?:\.\d+)?)(%)?"
    )
    range_re = re.compile(r"\b\d+\s*-\s*\d+%\b")

    findings: List[str] = []
    for path in targets:
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            continue

        for i, line in enumerate(lines, start=1):
            scan_line = line
            if path.lower().endswith(".html"):
                scan_line = re.sub(r"(?i)\sstyle\s*=\s*\"[^\"]*\"", "", scan_line)
                scan_line = re.sub(r"(?i)\sstyle\s*=\s*'[^']*'", "", scan_line)
                scan_line = re.sub(r"<[^>]+>", "", scan_line)
            if range_re.search(scan_line):
                continue
            if path.lower().endswith(".js") and not any(q in line for q in ["'", '"', "`"]):
                continue
            m = metric_re.search(scan_line)
            if not m:
                continue
            key = (m.group(1) or "").lower()
            num = (m.group(2) or "").strip()
            has_pct = bool(m.group(3))
            if path.lower().endswith(".js") and (not has_pct) and not (num.startswith("0.") or num.startswith("1.")):
                continue
            if not has_pct and num in {"0", "0.0", "0.00", "1", "1.0", "1.00"}:
                continue
            if num in {"100", "100.0", "100.00"}:
                continue
            if key == "threshold" and num in {"0.5", "0.50", "50", "50.0"}:
                continue
            if key in {"roc", "auc", "roc-auc", "roc auc"} and num in {"0.5", "0.50"}:
                continue
            snippet = line.strip()
            findings.append(f"{path}:{i}: {snippet}")

    if findings:
        msg = "Potential hard-coded metrics found in UI:\n" + "\n".join(findings[:50])
        _warn(msg)
    else:
        _ok("UI hard-coded metric scan: no suspicious matches")


def _get_json(url: str, timeout: int = 30) -> Any:
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        _fail(f"GET {url} -> {r.status_code}")
    return r.json()


def validate_api(base_url: str, results_local: Dict[str, Any], comparison_local: Dict[str, Any]) -> None:
    models = _get_json(base_url.rstrip("/") + "/api/models")
    if not isinstance(models, dict) or "results" not in models or "comparison" not in models:
        _fail("/api/models schema invalid")
    _ok("/api/models schema looks valid")

    results_api = models.get("results") or {}
    for m in ["Logistic Regression", "Random Forest", "XGBoost"]:
        opt_local = (results_local.get(m) or {}).get("optimal_threshold") or {}
        opt_api = (results_api.get(m) or {}).get("optimal_threshold") or {}
        for k in ["threshold", "roc_auc", "recall", "precision", "f1", "accuracy"]:
            lv = _as_float(opt_local.get(k))
            av = _as_float(opt_api.get(k))
            if lv is None or av is None:
                continue
            if not _approx(lv, av, tol=5e-3):
                _warn(f"/api/models results mismatch {m}.{k}: api={av} vs local={lv}")

    cmp_api = models.get("comparison") or {}
    table_api = cmp_api.get("table") or []
    table_local = (comparison_local.get("table") or [])
    by_local = {r.get("model"): r for r in table_local if isinstance(r, dict)}
    for r in table_api:
        if not isinstance(r, dict):
            continue
        name = r.get("model")
        if not name or name not in by_local:
            continue
        for k in ["threshold", "roc_auc", "recall", "precision", "f1", "accuracy"]:
            lv = _as_float(by_local[name].get(k))
            av = _as_float(r.get(k))
            if lv is None or av is None:
                continue
            if not _approx(lv, av, tol=5e-3):
                _warn(f"/api/models comparison mismatch {name}.{k}: api={av} vs local={lv}")

    clusters = _get_json(base_url.rstrip("/") + "/api/clusters")
    profiles = clusters.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        _fail("/api/clusters profiles invalid")
    p0 = profiles[0]
    if "churn_rate" not in p0 or "churn_rate_pct" not in p0:
        _warn("/api/clusters profiles should include churn_rate and churn_rate_pct")
    cr = _as_float(p0.get("churn_rate"))
    crp = _as_float(p0.get("churn_rate_pct"))
    if cr is not None and crp is not None and not _approx(crp, cr * 100.0, tol=1.0):
        _warn(f"/api/clusters churn_rate_pct not ~ churn_rate*100: {crp} vs {cr}")
    _ok("/api/clusters schema looks valid")

    for model_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        payload = {
            "age": 48,
            "gender": "female",
            "tenure_ye": 1,
            "married": 1,
            "credit_sco": 580,
            "risk_score": 0.25,
            "balance": 0,
            "monthly_ir": 5000000,
            "engagement_score": 15,
            "nums_card": 1,
            "nums_service": 1,
            "active_member": 0,
            "customer_segment": "Mass",
            "loyalty_level": "Bronze",
            "digital_behavior": "offline",
            "model": model_name,
        }
        r = requests.post(base_url.rstrip("/") + "/api/predict", json=payload, timeout=30)
        if r.status_code != 200:
            _fail(f"POST /api/predict -> {r.status_code}: {r.text[:200]}")
        pred = r.json()
        for k in ["model_used", "probability", "threshold", "prediction", "reasons", "suggestions"]:
            if k not in pred:
                _fail(f"/api/predict missing '{k}'")

        prob = _as_float(pred.get("probability"))
        thr = _as_float(pred.get("threshold"))
        if prob is not None and not (0.0 <= prob <= 100.0):
            _warn(f"/api/predict probability expected percent 0..100, got {prob}")
        if thr is not None and not (0.0 <= thr <= 100.0):
            _warn(f"/api/predict threshold expected percent 0..100, got {thr}")
        if pred.get("prediction") not in [0, 1]:
            _warn(f"/api/predict prediction expected 0/1, got {pred.get('prediction')}")

        expected_thr = _as_float((results_local.get(model_name) or {}).get("optimal_threshold", {}).get("threshold"))
        if thr is not None and expected_thr is not None:
            exp_pct = expected_thr * 100.0
            if not _approx(thr, exp_pct, tol=1.0):
                _warn(f"/api/predict threshold mismatch {model_name}: api={thr}% vs expected={exp_pct:.2f}%")

    _ok("/api/predict schema + threshold consistency looks valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None, help="If provided, validates live API responses (e.g. http://127.0.0.1:5000)")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings (enforce consistency checks)")
    args = parser.parse_args()

    global STRICT
    STRICT = bool(args.strict)

    _require(
        [
            "outputs/results.json",
            "outputs/comparison.json",
            "outputs/cluster_profiles.json",
            "outputs/elbow_data.json",
        ]
    )

    results = _read_json("outputs/results.json")
    comparison = _read_json("outputs/comparison.json")
    cluster_profiles = _read_json("outputs/cluster_profiles.json")
    elbow = _read_json("outputs/elbow_data.json")

    validate_results(results)
    validate_comparison(comparison, results)
    validate_clusters_artifacts(cluster_profiles, elbow)
    validate_ui_contract()
    validate_ui_no_hardcoded_metrics()

    if args.base_url:
        validate_api(args.base_url, results, comparison)

    _ok("Schema audit completed")


if __name__ == "__main__":
    main()
