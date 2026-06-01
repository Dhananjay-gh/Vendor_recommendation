import pandas as pd
import numpy as np
import os
import json
import joblib
import shap
import hashlib

# ──────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────
BASE_DIR              = os.path.dirname(__file__)
SAP_VENDORS_PATH      = os.path.join(BASE_DIR, 'processed_sap_vendors.csv')
XGB_MODEL_PATH        = os.path.join(BASE_DIR, 'xgb_risk_model.pkl')
KMEANS_MODEL_PATH     = os.path.join(BASE_DIR, 'kmeans_risk_model.pkl')
SCALER_PATH           = os.path.join(BASE_DIR, 'scaler.pkl')
ISOLATION_FOREST_PATH = os.path.join(BASE_DIR, 'isolation_forest_model.pkl')
PURCHASE_DATA_PATH    = os.path.join(BASE_DIR, 'purchase_data.csv')
FEATURE_COLS_PATH     = os.path.join(BASE_DIR, 'feature_columns.json')

# ── Load feature column list (written by preprocess_sap.py) ──────────────────
# Falls back to the original 4 features if the file doesn't exist yet,
# so the app doesn't crash before retraining.
if os.path.exists(FEATURE_COLS_PATH):
    with open(FEATURE_COLS_PATH) as _f:
        FEATURE_COLUMNS = json.load(_f)
else:
    FEATURE_COLUMNS = [
        'avg_days_overdue_hist',
        'late_ratio',
        'total_spend_vol',
        'open_exposure',
    ]

# Human-readable labels for SHAP chart and deviation table
FEATURE_LABELS = {
    'avg_days_overdue_hist':      'Avg Days Overdue',
    'late_ratio':                 'Late Payment Ratio',
    'total_spend_vol':            'Total Spend Volume',
    'open_exposure':              'Open Exposure',
    'payment_consistency_score':  'Payment Consistency',
    'years_active':               'Years Active',
    'payment_terms_risk':         'Payment Terms Risk',
    'is_payment_blocked':         'Payment Blocked Flag',
    'dunning_level':              'Dunning Level',
    'reversal_rate':              'Document Reversal Rate',
    'discount_capture_rate':      'Discount Capture Rate',
    'avg_payment_terms_days':     'Avg Payment Terms (Days)',
    'voided_payment_rate':        'Voided Payment Rate',
    'stale_payment_rate':         'Stale Payment Rate',
}

# ──────────────────────────────────────────────────
# LOAD ML MODELS & DATA (cached at module level)
# ──────────────────────────────────────────────────
_xgb_model      = None
_kmeans_model   = None
_scaler         = None
_shap_explainer = None
_iso_forest     = None
_sap_vendors_df = None


def _load_models():
    global _xgb_model, _kmeans_model, _scaler, _shap_explainer, _iso_forest, _sap_vendors_df
    if _xgb_model is None:
        _xgb_model = joblib.load(XGB_MODEL_PATH)
    if _shap_explainer is None:
        _shap_explainer = shap.TreeExplainer(_xgb_model)
    if _kmeans_model is None:
        _kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    if _iso_forest is None:
        _iso_forest = joblib.load(ISOLATION_FOREST_PATH)
    if _sap_vendors_df is None:
        _sap_vendors_df = pd.read_csv(SAP_VENDORS_PATH, low_memory=False)
    return _xgb_model, _kmeans_model, _scaler, _iso_forest, _sap_vendors_df


# ──────────────────────────────────────────────────
# PURCHASE PRICE DATA — SEED GENERATION (one-time only)
# ──────────────────────────────────────────────────
from datetime import datetime, timedelta

if not os.path.exists(PURCHASE_DATA_PATH):
    print("purchase_data.csv not found — generating synthetic seed data...")
    np.random.seed(42)
    data = []

    products = {
        "Enterprise Laptop":    {"range": (1200, 1800), "rate": 0.90},
        "Corporate Smartphone": {"range": (600,  1100), "rate": 0.95},
        "Rack Server":          {"range": (4500, 8000), "rate": 1.12},
        "Cloud Compute Credit": {"range": (1000, 5000), "rate": 1.06},
    }

    start_date = datetime.now() - timedelta(days=730)
    for p_name, p_data in products.items():
        low, high = p_data["range"]
        rate = p_data["rate"]
        for i in range(120):
            days_offset = np.random.randint(0, 730)
            date = start_date + timedelta(days=days_offset)
            years_passed = days_offset / 365.0
            multiplier = (rate ** years_passed)
            base_price = np.random.randint(low, high)
            adjusted_price = round(base_price * multiplier, 2)
            data.append([p_name, adjusted_price, date.strftime("%Y-%m-%d")])

    df_price = pd.DataFrame(data, columns=["product_name", "price_per_unit", "date"])
    df_price.to_csv(PURCHASE_DATA_PATH, index=False)
    print(f"Seed data written → {PURCHASE_DATA_PATH} ({len(df_price)} rows)")
else:
    print(f"Using existing purchase data → {PURCHASE_DATA_PATH}")


# ──────────────────────────────────────────────────
# BUCKET FUNCTION
# ──────────────────────────────────────────────────
def get_vendor_bucket(vendor_risk):
    if vendor_risk < 0.3:
        return "LOW"
    elif vendor_risk < 0.6:
        return "MEDIUM"
    elif vendor_risk < 0.85:
        return "HIGH"
    else:
        return "CRITICAL"


# ──────────────────────────────────────────────────
# GET REAL VENDOR LIST (for dropdowns)
# ──────────────────────────────────────────────────
def get_sap_vendor_list():
    """
    Returns a list of dicts with vendor info from the processed SAP data.
    Includes all available feature columns for chatbot context.
    """
    _, _, _, _, df = _load_models()
    vendors = []
    for _, row in df.iterrows():
        entry = {
            "lifnr":       row["LIFNR"],
            "name":        row["NAME1"],
            "risk_class":  row["RISK_CLASS"],
            "kmeans_cluster": int(row["KMEANS_CLUSTER"]) if "KMEANS_CLUSTER" in row.index else 0,
        }
        # Add all feature columns dynamically
        for col in FEATURE_COLUMNS:
            if col in row.index:
                entry[col] = float(row[col])
            else:
                entry[col] = 0.0
        # Also include transaction_count separately (not in feature set)
        entry["transaction_count"] = int(row["transaction_count"]) if "transaction_count" in row.index else 0
        vendors.append(entry)
    return vendors


# ──────────────────────────────────────────────────
# PREDICT VENDOR RISK (XGBoost + all models)
# ──────────────────────────────────────────────────
def predict_vendor_risk(lifnr):
    """
    Given a vendor LIFNR, uses the trained XGBoost model (14 features)
    to predict risk class and returns a full feature breakdown.
    """
    xgb, kmeans, scaler, iso_forest, df = _load_models()
    vendor_row = df[df["LIFNR"] == lifnr]
    if vendor_row.empty:
        return None

    row = vendor_row.iloc[0]

    # ── Build feature vector dynamically from FEATURE_COLUMNS ────────────────
    feature_values = []
    for col in FEATURE_COLUMNS:
        if col in row.index:
            feature_values.append(float(row[col]))
        else:
            feature_values.append(0.0)

    features = np.array([feature_values])

    # Scale
    features_scaled = scaler.transform(features)

    # ── XGBoost prediction ────────────────────────────────────────────────────
    predicted_class = int(xgb.predict(features_scaled)[0])
    probabilities   = xgb.predict_proba(features_scaled)[0]

    # Use the continuous composite risk from preprocessing for better granularity
    if "composite_risk" in row:
        risk_score = float(row["composite_risk"])
    else:
        risk_score = sum(i * p for i, p in enumerate(probabilities)) / 2.0
        
    risk_score = np.clip(risk_score, 0.0, 1.0)

    # ── K-Means cluster ───────────────────────────────────────────────────────
    cluster = int(kmeans.predict(features_scaled)[0])

    # ── Isolation Forest ─────────────────────────────────────────────────────
    vendor_iso_score  = float(row["ISOLATION_SCORE"]) if "ISOLATION_SCORE" in row.index else 0.5
    vendor_is_outlier = bool(int(row["IS_OUTLIER"]))  if "IS_OUTLIER"      in row.index else False

    # ── SHAP values (dynamic — works for any number of features) ─────────────
    shap_vals = _shap_explainer.shap_values(features_scaled)
    shap_labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLUMNS]
    try:
        if isinstance(shap_vals, list):
            shap_for_class = shap_vals[predicted_class][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_for_class = shap_vals[0, :, predicted_class]
        else:
            shap_for_class = shap_vals[0]
        shap_dict = {
            label: round(float(v), 4)
            for label, v in zip(shap_labels, shap_for_class)
        }
    except (IndexError, TypeError):
        shap_dict = {label: 0.0 for label in shap_labels}

    # ── Feature Deviation Analysis (z-scores vs population) ──────────────────
    feature_deviations = []
    for col in FEATURE_COLUMNS:
        label = FEATURE_LABELS.get(col, col)
        if col not in df.columns:
            continue
        vendor_val = float(row[col]) if col in row.index else 0.0
        pop_mean   = float(df[col].mean())
        pop_std    = float(df[col].std())
        pop_median = float(df[col].median())
        pop_p25    = float(df[col].quantile(0.25))
        pop_p75    = float(df[col].quantile(0.75))
        pop_p95    = float(df[col].quantile(0.95))

        z_score = (vendor_val - pop_mean) / pop_std if pop_std > 0 else 0.0
        abs_z   = abs(z_score)
        level   = "EXTREME" if abs_z >= 3.0 else "HIGH" if abs_z >= 2.0 else "MODERATE" if abs_z >= 1.0 else "NORMAL"

        feature_deviations.append({
            "feature":    label,
            "col":        col,
            "vendor_val": round(vendor_val, 4),
            "pop_mean":   round(pop_mean, 4),
            "pop_median": round(pop_median, 4),
            "pop_p25":    round(pop_p25, 4),
            "pop_p75":    round(pop_p75, 4),
            "pop_p95":    round(pop_p95, 4),
            "z_score":    round(z_score, 2),
            "level":      level,
        })
    feature_deviations.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    # ── Return full prediction dict ───────────────────────────────────────────
    return {
        "predicted_class":       predicted_class,
        "predicted_class_label": {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}.get(predicted_class, "Unknown"),
        "risk_score":            round(float(risk_score), 4),
        "probabilities": {
            "Low":    round(float(probabilities[0]), 4),
            "Medium": round(float(probabilities[1]), 4),
            "High":   round(float(probabilities[2]), 4),
        },
        "kmeans_cluster":        cluster,
        "isolation_score":       round(vendor_iso_score, 4),
        "is_outlier":            vendor_is_outlier,
        # Core behaviour fields (always present for backwards compatibility)
        "avg_days_overdue":      round(float(row["avg_days_overdue_hist"]) if "avg_days_overdue_hist" in row.index else 0, 1),
        "late_ratio":            round(float(row["late_ratio"])            if "late_ratio"            in row.index else 0, 4),
        "total_spend":           round(float(row["total_spend_vol"])       if "total_spend_vol"       in row.index else 0, 2),
        "open_exposure":         round(float(row["open_exposure"])         if "open_exposure"         in row.index else 0, 2),
        "transaction_count":     int(row["transaction_count"])             if "transaction_count"     in row.index else 0,
        "sap_risk_class":        str(row["RISK_CLASS"]),
        "sap_risk_label":        {"A": "A - Low", "B": "B - Medium", "C": "C/D - High", "D": "C/D - High"}.get(str(row["RISK_CLASS"]), str(row["RISK_CLASS"])),
        # New feature values (for chatbot context)
        "payment_consistency":   round(float(row["payment_consistency_score"]) if "payment_consistency_score" in row.index else 0, 2),
        "years_active":          round(float(row["years_active"])              if "years_active"              in row.index else 0, 1),
        "is_payment_blocked":    int(row["is_payment_blocked"])                if "is_payment_blocked"        in row.index else 0,
        "dunning_level":         int(row["dunning_level"])                     if "dunning_level"             in row.index else 0,
        "reversal_rate":         round(float(row["reversal_rate"])             if "reversal_rate"             in row.index else 0, 4),
        "discount_capture_rate": round(float(row["discount_capture_rate"])     if "discount_capture_rate"     in row.index else 0, 4),
        "voided_payment_rate":   round(float(row["voided_payment_rate"])       if "voided_payment_rate"       in row.index else 0, 4),
        "stale_payment_rate":    round(float(row["stale_payment_rate"])        if "stale_payment_rate"        in row.index else 0, 4),
        # Analysis
        "shap_values":           shap_dict,
        "feature_deviations":    feature_deviations,
    }


# ──────────────────────────────────────────────────
# HISTORICAL AVERAGE PRICE
# ──────────────────────────────────────────────────
def get_vendor_avg_price(product_name: str, vendor_id: str = None) -> float:
    df_price = pd.read_csv(PURCHASE_DATA_PATH)
    df_product = df_price[df_price["product_name"] == product_name]
    
    if df_product.empty:
        return 0.0
        
    if vendor_id and "vendor_id" in df_price.columns:
        df_vendor = df_product[df_product["vendor_id"] == vendor_id]
        if not df_vendor.empty:
            return round(float(df_vendor["price_per_unit"].mean()), 2)
            
    # Fallback to market average only if the product exists but vendor specifically doesn't have it
    # (Though with our new script, every vendor has every product)
    return round(float(df_product["price_per_unit"].mean()), 2)


# ──────────────────────────────────────────────────
# PRICE FORECASTING (Linear Regression with Inflation)
# ──────────────────────────────────────────────────
def forecast_product_price(product_name):
    from scipy import stats

    df_price_data = pd.read_csv(PURCHASE_DATA_PATH)
    df_product    = df_price_data[df_price_data["product_name"] == product_name].copy()

    if len(df_product) < 3:
        return df_product["price_per_unit"].mean() if not df_product.empty else None

    df_product['date_dt']      = pd.to_datetime(df_product['date'])
    df_product['date_ordinal'] = df_product['date_dt'].map(datetime.toordinal)

    x = df_product['date_ordinal'].values
    y = df_product['price_per_unit'].values

    slope, intercept, *_ = stats.linregress(x, y)
    forecasted_price = intercept + slope * datetime.now().toordinal()
    return round(float(forecasted_price), 2)


# ──────────────────────────────────────────────────
# PERCENTILE-BASED PRICE RISK
# ──────────────────────────────────────────────────
def compute_price_risk(product_name, current_price):
    df_p    = pd.read_csv(PURCHASE_DATA_PATH)
    df_p["date"] = pd.to_datetime(df_p["date"])
    cutoff  = pd.Timestamp.now() - pd.DateOffset(months=6)
    df_recent = df_p[(df_p["product_name"] == product_name) & (df_p["date"] >= cutoff)]

    if len(df_recent) < 5:
        df_recent = df_p[df_p["product_name"] == product_name]

    prices = df_recent["price_per_unit"].values
    if len(prices) < 5:
        return 0.5, {}

    p25 = float(np.percentile(prices, 25))
    p50 = float(np.percentile(prices, 50))
    p75 = float(np.percentile(prices, 75))
    p90 = float(np.percentile(prices, 90))

    if current_price > p90:
        risk = 0.90
    elif current_price > p75:
        risk = 0.65
    elif current_price > p50:
        risk = 0.35
    else:
        risk = 0.10

    return risk, {"p25": round(p25, 2), "p50": round(p50, 2),
                  "p75": round(p75, 2), "p90": round(p90, 2)}


# ──────────────────────────────────────────────────
# MAIN PROCUREMENT RISK MODEL
# ──────────────────────────────────────────────────
def procurement_risk_model(vendor_lifnr, product_name, current_price):
    """
    ML-powered risk assessment combining:
    - XGBoost vendor risk prediction (14 SAP features)
    - Linear Regression price forecasting (inflation-adjusted)
    """
    vendor_pred = predict_vendor_risk(vendor_lifnr)
    if vendor_pred is None:
        return {"error": f"Vendor {vendor_lifnr} not found in SAP data."}

    vendor_risk    = vendor_pred["risk_score"]
    clearance_days = vendor_pred["avg_days_overdue"]

    # ── Price Risk ────────────────────────────────────────────────────────────
    forecasted_price = forecast_product_price(product_name)
    if forecasted_price is None:
        forecasted_price = current_price
        variance = 0.0
    else:
        variance = (current_price - forecasted_price) / forecasted_price

    df_price_data = pd.read_csv(PURCHASE_DATA_PATH)
    avg_price = df_price_data[
        df_price_data["product_name"] == product_name
    ]["price_per_unit"].mean()

    price_risk, price_percentiles = compute_price_risk(product_name, current_price)

    # ── Final Risk ────────────────────────────────────────────────────────────
    final_risk    = round(0.6 * vendor_risk + 0.4 * price_risk, 4)
    vendor_bucket = get_vendor_bucket(final_risk)

    if final_risk < 0.40:
        if variance < -0.3:
            decision = "REVIEW"
        else:
            decision = "APPROVE"
    elif final_risk < 0.65:
        decision = "REVIEW"
    else:
        decision = "HIGH RISK"

    sap_class = vendor_pred.get("sap_risk_class", "")
    sap_divergence = vendor_bucket in ["HIGH", "CRITICAL"] and sap_class in ["A", "B"]

    # ── Insights ──────────────────────────────────────────────────────────────
    insights = []

    if avg_price > 0:
        inflation_increase = ((forecasted_price - avg_price) / avg_price) * 100
        inflation_direction = "inflation" if inflation_increase > 0 else "deflation"
    else:
        inflation_increase  = 0.0
        inflation_direction = "stable"

    if forecasted_price > avg_price:
        insights.append(
            f"Forecasting Model: Detected an inflationary trend. The 'Expected Fair Price' "
            f"has risen by {round(inflation_increase, 1)}% from historical averages."
        )

    if vendor_bucket in ["CRITICAL", "HIGH"]:
        insights.append(
            f"XGBoost Class {vendor_pred['predicted_class_label']} "
            f"(SAP: {vendor_pred['sap_risk_class']}). "
            f"High risk factors detected in {vendor_pred['transaction_count']} historical records."
        )
    elif vendor_bucket == "MEDIUM":
        insights.append(
            f"XGBoost Class {vendor_pred['predicted_class_label']} "
            f"(SAP: {vendor_pred['sap_risk_class']}). "
            f"Moderate risk; classification aligns with historical patterns."
        )
    else:
        insights.append(
            f"XGBoost Class {vendor_pred['predicted_class_label']} "
            f"(SAP: {vendor_pred['sap_risk_class']}). "
            f"Exemplary history across {vendor_pred['transaction_count']} transactions."
        )

    if sap_divergence:
        insights.append(
            f"Risk Divergence Detected: The behavioral model flags this vendor as {vendor_bucket} risk, "
            f"but their static SAP class is {vendor_pred.get('sap_risk_label', sap_class)}. Proceed with caution."
        )

    if variance > 0.2:
        insights.append(
            f"Price Warning: Quoted price is {round(variance*100, 1)}% above the "
            f"inflation-forecasted benchmark. Excessive markup detected."
        )
    elif variance < -0.3:
        insights.append(
            f"Quality Warning: Quoted price is {round(abs(variance)*100, 1)}% below "
            f"the forecasted benchmark. Such a suspiciously low price may indicate compromised product quality, missing features, or hidden costs. Please verify the specifications."
        )
    elif variance < -0.05:
        insights.append(
            f"Price Advantage: Quoted price is {round(abs(variance)*100, 1)}% below "
            f"the forecasted benchmark, despite market inflation."
        )
    else:
        insights.append("Fair Pricing: Quote aligns with the AI-forecasted inflationary benchmark.")

    # New feature insights — highlight notable signals from expanded features
    if vendor_pred.get("is_payment_blocked"):
        insights.append(
            "SAP Block Alert: This vendor currently has a payment or posting block active "
            "in SAP (LFB1 flags). Payments may require manual release."
        )

    if vendor_pred.get("dunning_level", 0) >= 3:
        insights.append(
            f"Dunning Warning: This vendor has reached dunning level "
            f"{vendor_pred['dunning_level']} — indicating repeated late payment notices."
        )

    if vendor_pred.get("reversal_rate", 0) > 0.1:
        insights.append(
            f"Document Quality: {vendor_pred['reversal_rate']:.1%} of this vendor's "
            f"documents have been reversed/cancelled — above the acceptable threshold of 10%."
        )

    if vendor_pred.get("years_active", 99) < 2:
        insights.append(
            f"New Vendor Risk: This vendor has only been active for "
            f"{vendor_pred['years_active']:.1f} years. Limited transaction history increases uncertainty."
        )

    if decision == "HIGH RISK":
        insights.append("Recommendation: Avoid contract. Exposure exceeds risk tolerance thresholds.")
    elif decision == "REVIEW":
        if variance < -0.3:
            insights.append("Recommendation: Manual review required. The quoted price is suspiciously low compared to the forecast, which may imply compromised quality.")
        else:
            insights.append("Recommendation: Specialist review of the Quote vs. Forecast variance is required.")
    else:
        insights.append("Recommendation: Automated approval granted. Pricing is optimal relative to the forecast.")

    # K-Means cluster insight
    cluster_labels = {
        0: "Conservative Spenders",
        1: "High-Volume Partners",
        2: "At-Risk Outliers",
        3: "Stable Mid-Tier",
    }
    cluster_id   = vendor_pred["kmeans_cluster"]
    cluster_name = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
    insights.append(
        f"K-Means Clustering places this vendor in the \"{cluster_name}\" segment "
        f"(Cluster {cluster_id})."
    )

    # Isolation Forest insight
    iso_score  = vendor_pred["isolation_score"]
    is_outlier = vendor_pred["is_outlier"]
    if is_outlier and iso_score > 0.75:
        insights.append(
            f"Isolation Forest: This vendor is flagged as a statistical outlier "
            f"(anomaly score: {iso_score:.2f}). Their payment and spend patterns are "
            f"highly unusual compared to the full vendor population \u2014 independent of risk class."
        )
    elif is_outlier:
        insights.append(
            f"Isolation Forest: This vendor shows moderately atypical behaviour "
            f"(anomaly score: {iso_score:.2f}). Some feature values deviate from population norms."
        )
    else:
        insights.append(
            f"Isolation Forest: Vendor behaviour is within normal population bounds "
            f"(anomaly score: {iso_score:.2f}). No structural anomalies detected."
        )

    return {
        "vendor_id":             vendor_lifnr,
        "vendor_raw_price":      current_price,
        "vendor_risk":           round(vendor_risk, 2),
        "vendor_bucket":         vendor_bucket,
        "price_variance":        round(variance, 4),
        "avg_price":             round(avg_price, 2),
        "vendor_historical_avg": get_vendor_avg_price(product_name, vendor_lifnr),
        "forecasted_price":      forecasted_price,
        "price_risk":            round(price_risk, 2),
        "avg_clearance_days": round(clearance_days, 1),
        "final_risk":         round(final_risk, 2),
        "decision":           decision,
        "inflation_percent":  round(inflation_increase, 1),
        "inflation_direction": inflation_direction,
        "insights":           insights,
        "xgb_prediction":     vendor_pred,
        "price_percentiles":  price_percentiles,
        "isolation_score":    vendor_pred["isolation_score"],
        "is_outlier":         vendor_pred["is_outlier"],
        "feature_deviations": vendor_pred["feature_deviations"],
        "sap_risk_label":     vendor_pred.get("sap_risk_label", ""),
        "sap_divergence":     sap_divergence,
    }


# ──────────────────────────────────────────────────
# GENERATE VENDOR HISTORY DATA
# ──────────────────────────────────────────────────
def get_vendor_history(vendor_lifnr):
    vendor_pred = predict_vendor_risk(vendor_lifnr)
    if vendor_pred is None:
        return None

    base_risk = vendor_pred["risk_score"]
    seed_val  = int(hashlib.md5(vendor_lifnr.encode()).hexdigest()[:8], 16) % (2**31)

    np.random.seed(seed_val)
    noise    = np.random.normal(0, 0.06, 12)
    timeline = np.clip(base_risk + noise, 0.0, 1.0).tolist()
    timeline = [round(v, 3) for v in timeline]

    if base_risk > 0.45:
        timeline[5] = round(min(base_risk + 0.20, 0.95), 3)
    if base_risk > 0.70:
        timeline[9] = round(min(base_risk + 0.15, 0.95), 3)

    np.random.seed(seed_val + 1)
    volumes = np.random.randint(100, 500, 50).tolist()

    n_anomalies = int(base_risk * 6)
    if n_anomalies > 0:
        np.random.seed(seed_val + 2)
        anomaly_indices = np.random.choice(50, min(n_anomalies, 50), replace=False)
        for idx in anomaly_indices:
            volumes[idx] = int(np.random.randint(1200, 2500))

    return {
        "monthly_risk_history": timeline,
        "past_transactions":    volumes,
        "transaction_count":    vendor_pred["transaction_count"],
    }
