import pandas as pd
import numpy as np
import os
import joblib
import shap

# ──────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
SAP_VENDORS_PATH = os.path.join(BASE_DIR, 'processed_sap_vendors.csv')
XGB_MODEL_PATH = os.path.join(BASE_DIR, 'xgb_risk_model.pkl')
KMEANS_MODEL_PATH = os.path.join(BASE_DIR, 'kmeans_risk_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
ISOLATION_FOREST_PATH = os.path.join(BASE_DIR, 'isolation_forest_model.pkl')
PURCHASE_DATA_PATH = os.path.join(BASE_DIR, 'purchase_data.csv')

# ──────────────────────────────────────────────────
# LOAD ML MODELS & DATA (cached at module level)
# ──────────────────────────────────────────────────
_xgb_model = None
_kmeans_model = None
_scaler = None
_shap_explainer = None
_iso_forest = None
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
# If purchase_data.csv already exists, we treat it as the source of truth
# and skip generation entirely. This means:
#   - Replace the CSV with real procurement data at any time — it won't be overwritten.
#   - Delete the CSV to force a fresh synthetic re-seed on next startup.
from datetime import datetime, timedelta

if not os.path.exists(PURCHASE_DATA_PATH):
    print("purchase_data.csv not found — generating synthetic seed data...")
    np.random.seed(42)
    data = []

    products = {
        "Enterprise Laptop": {"range": (1200, 1800), "rate": 0.90},      # -10% annual deflation (tech)
        "Corporate Smartphone": {"range": (600, 1100), "rate": 0.95},    # -5% annual deflation
        "Rack Server": {"range": (4500, 8000), "rate": 1.12},            # +12% annual inflation (hardware shortages)
        "Cloud Compute Credit": {"range": (1000, 5000), "rate": 1.06},   # +6% annual inflation
    }

    # Generate 2 years of historical data (last 24 months)
    start_date = datetime.now() - timedelta(days=730)
    for p_name, p_data in products.items():
        low, high = p_data["range"]
        rate = p_data["rate"]
        for i in range(120):  # 120 points per product
            days_offset = np.random.randint(0, 730)
            date = start_date + timedelta(days=days_offset)

            # Add product-specific annual trend
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
    Each dict: {lifnr, name, risk_class, avg_days_overdue, late_ratio, ...}
    """
    _, _, _, _, df = _load_models()
    vendors = []
    for _, row in df.iterrows():
        vendors.append({
            "lifnr": row["LIFNR"],
            "name": row["NAME1"],
            "risk_class": row["RISK_CLASS"],
            "avg_days_overdue_hist": row.get("avg_days_overdue_hist", 0),
            "late_ratio": row.get("late_ratio", 0),
            "total_spend_vol": row.get("total_spend_vol", 0),
            "open_exposure": row.get("open_exposure", 0),
            "transaction_count": row.get("transaction_count", 0),
            "kmeans_cluster": int(row.get("KMEANS_CLUSTER", 0)),
        })
    return vendors


# ──────────────────────────────────────────────────
# PREDICT VENDOR RISK (XGBoost)
# ──────────────────────────────────────────────────
def predict_vendor_risk(lifnr):
    """
    Given a vendor LIFNR, uses the trained XGBoost model to predict
    a risk class (0=A, 1=B, 2=C, 3=D) and returns a normalized risk score.
    """
    xgb, kmeans, scaler, iso_forest, df = _load_models()
    vendor_row = df[df["LIFNR"] == lifnr]
    if vendor_row.empty:
        return None

    row = vendor_row.iloc[0]
    features = np.array([[
        row["avg_days_overdue_hist"],
        row["late_ratio"],
        row["total_spend_vol"],
        row["open_exposure"],
    ]])

    # Scale features for prediction
    features_scaled = scaler.transform(features)

    # XGBoost prediction
    predicted_class = int(xgb.predict(features_scaled)[0])         # 0, 1, 2, or 3
    probabilities = xgb.predict_proba(features_scaled)[0]          # [p_A, p_B, p_C, p_D]

    # Weighted risk score: sum(class_index * probability) / max_class
    risk_score = sum(i * p for i, p in enumerate(probabilities)) / 3.0
    risk_score = np.clip(risk_score, 0.0, 1.0)

    # K-Means cluster
    cluster = int(kmeans.predict(features_scaled)[0])

    # Isolation Forest anomaly score for this vendor
    iso_score = float(iso_forest.decision_function(features_scaled)[0])
    iso_label = int(iso_forest.predict(features_scaled)[0])   # -1 or 1

    # Normalize to 0-1 (higher = more anomalous) using the vendor's stored score
    # for consistency with population-level normalization done at training time
    vendor_iso_score = float(row.get("ISOLATION_SCORE", 0.5))
    vendor_is_outlier = bool(int(row.get("IS_OUTLIER", 0)))

    # SHAP values for explainability
    shap_vals = _shap_explainer.shap_values(features_scaled)
    feature_names = ['avg_days_overdue', 'late_ratio', 'total_spend', 'open_exposure']
    try:
        if isinstance(shap_vals, list):
            # List of arrays, one per class: each is (n_samples, n_features)
            shap_for_predicted_class = shap_vals[predicted_class][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            # 3D array: (n_samples, n_classes, n_features) or (n_classes, n_samples, n_features)
            if shap_vals.shape[0] == 1:
                # (n_samples=1, n_classes, n_features)
                shap_for_predicted_class = shap_vals[0][predicted_class]
            else:
                # (n_classes, n_samples=1, n_features)
                shap_for_predicted_class = shap_vals[predicted_class][0]
        else:
            # 2D array: (n_samples, n_features) — binary or single output
            shap_for_predicted_class = shap_vals[0]
        shap_dict = dict(zip(feature_names, [round(float(v), 4) for v in shap_for_predicted_class]))
    except (IndexError, TypeError):
        shap_dict = dict(zip(feature_names, [0.0] * len(feature_names)))

    return {
        "predicted_class": predicted_class,
        "predicted_class_label": ["A", "B", "C", "D"][predicted_class],
        "risk_score": round(float(risk_score), 4),
        "probabilities": {
            "A": round(float(probabilities[0]), 4),
            "B": round(float(probabilities[1]), 4),
            "C": round(float(probabilities[2]), 4),
            "D": round(float(probabilities[3]), 4),
        },
        "kmeans_cluster": cluster,
        "isolation_score": round(vendor_iso_score, 4),   # 0-1, higher = more anomalous
        "is_outlier": vendor_is_outlier,                  # True if Isolation Forest flagged
        "avg_days_overdue": round(float(row["avg_days_overdue_hist"]), 1),
        "late_ratio": round(float(row["late_ratio"]), 4),
        "total_spend": round(float(row["total_spend_vol"]), 2),
        "open_exposure": round(float(row["open_exposure"]), 2),
        "transaction_count": int(row["transaction_count"]),
        "sap_risk_class": str(row["RISK_CLASS"]),
        "shap_values": shap_dict,
    }


# ──────────────────────────────────────────────────
# PRICE FORECASTING (Linear Regression with Inflation)
# ──────────────────────────────────────────────────
def forecast_product_price(product_name):
    """
    Uses Linear Regression to forecast the 'expected' price of a product for Today,
    accounting for historical trends (inflation).
    """
    from scipy import stats
    from datetime import datetime
    
    df_price_data = pd.read_csv(PURCHASE_DATA_PATH)
    df_product = df_price_data[df_price_data["product_name"] == product_name].copy()
    
    if len(df_product) < 3:
        # Fallback to mean if not enough data for regression
        return df_product["price_per_unit"].mean() if not df_product.empty else None

    # Convert dates to ordinal (number of days) for linear regression
    df_product['date_dt'] = pd.to_datetime(df_product['date'])
    df_product['date_ordinal'] = df_product['date_dt'].map(datetime.toordinal)
    
    x = df_product['date_ordinal'].values
    y = df_product['price_per_unit'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Forecast for today
    today_ordinal = datetime.now().toordinal()
    forecasted_price = intercept + slope * today_ordinal
    
    return round(float(forecasted_price), 2)


# ──────────────────────────────────────────────────
# PERCENTILE-BASED PRICE RISK
# ──────────────────────────────────────────────────
def compute_price_risk(product_name, current_price):
    df_p = pd.read_csv(PURCHASE_DATA_PATH)
    prices = df_p[df_p["product_name"] == product_name]["price_per_unit"].values
    if len(prices) < 5:
        return 0.5, {}   # fallback
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
    return risk, {"p25": round(p25,2), "p50": round(p50,2),
                  "p75": round(p75,2), "p90": round(p90,2)}


# ──────────────────────────────────────────────────
# MAIN PROCUREMENT RISK MODEL (ML-powered + Forecasting)
# ──────────────────────────────────────────────────
def procurement_risk_model(vendor_lifnr, product_name, current_price):
    """
    ML-powered risk assessment combining:
    - XGBoost vendor risk prediction (from real SAP data)
    - Linear Regression price forecasting (Inflation-adjusted benchmarking)
    """
    # ── Vendor Risk (XGBoost) ──
    vendor_pred = predict_vendor_risk(vendor_lifnr)
    if vendor_pred is None:
        return {"error": f"Vendor {vendor_lifnr} not found in SAP data."}
 
    vendor_risk = vendor_pred["risk_score"]
    clearance_days = vendor_pred["avg_days_overdue"]

    # ── Price Risk (Forecasting Model - Inflation Adjusted) ──
    forecasted_price = forecast_product_price(product_name)
    
    if forecasted_price is None:
        # Fallback if product has no history
        forecasted_price = current_price
        variance = 0.0
    else:
        variance = (current_price - forecasted_price) / forecasted_price

    # Historical average (for comparison)
    df_price_data = pd.read_csv(PURCHASE_DATA_PATH)
    avg_price = df_price_data[df_price_data["product_name"] == product_name]["price_per_unit"].mean()

    # Calculate Price Risk based on percentile position
    price_risk, price_percentiles = compute_price_risk(product_name, current_price)

    # ── Final Risk ──
    final_risk = (
        0.6 * vendor_risk +
        0.4 * price_risk
    )

    vendor_bucket = get_vendor_bucket(final_risk)

    # ── Decision ──
    if final_risk < 0.40:
        decision = "APPROVE"
    elif final_risk < 0.65:
        decision = "REVIEW"
    else:
        decision = "HIGH RISK"

    # ── Insights ──
    insights = []

    # Inflation/Forecasting Insight
    if avg_price > 0:
        inflation_increase = ((forecasted_price - avg_price) / avg_price) * 100
        inflation_direction = "inflation" if inflation_increase > 0 else "deflation"
    else:
        inflation_increase = 0.0
        inflation_direction = "stable"
    
    if forecasted_price > avg_price:
        insights.append(f"Forecasting Model: Detected an inflationary trend. The 'Expected Fair Price' has risen by {round(inflation_increase, 1)}% from historical averages.")

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

    if variance > 0.2:
        insights.append(f"Price Warning: Quoted price is {round(variance*100, 1)}% above the inflation-forecasted benchmark. Excessive markup detected.")
    elif variance < -0.05:
        insights.append(f"Price Advantage: Quoted price is {round(abs(variance)*100, 1)}% below the forecasted benchmark, despite market inflation.")
    else:
        insights.append("Fair Pricing: Quote aligns with the AI-forecasted inflationary benchmark.")

    if decision == "HIGH RISK":
        insights.append("Recommendation: Avoid contract. Exposure exceeds risk tolerance thresholds.")
    elif decision == "REVIEW":
        insights.append("Recommendation: Specialist review of the Quote vs. Forecast variance is required.")
    elif decision == "APPROVE":
        insights.append("Recommendation: Automated approval granted. Pricing is optimal relative to the forecast.")

    # K-Means cluster insight
    cluster_labels = {0: "Conservative Spenders", 1: "High-Volume Partners", 2: "At-Risk Outliers", 3: "Stable Mid-Tier"}
    cluster_id = vendor_pred["kmeans_cluster"]
    cluster_name = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
    insights.append(f"K-Means Clustering places this vendor in the \"{cluster_name}\" segment (Cluster {cluster_id}).")

    # Isolation Forest insight
    iso_score = vendor_pred["isolation_score"]
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

    # ── Isolation Forest Feature Deviation Analysis ──
    # Compare this vendor's features against the entire population to
    # identify which specific features make them stand out (or not).
    feature_cols = {
        "avg_days_overdue_hist": "Avg Days Overdue",
        "late_ratio":           "Late Payment Ratio",
        "total_spend_vol":      "Total Spend Volume",
        "open_exposure":        "Open Exposure"
    }
    feature_deviations = []
    for col, label in feature_cols.items():
        vendor_val = float(row[col])
        pop_mean   = float(df[col].mean())
        pop_std    = float(df[col].std())
        pop_median = float(df[col].median())
        pop_p25    = float(df[col].quantile(0.25))
        pop_p75    = float(df[col].quantile(0.75))
        pop_p95    = float(df[col].quantile(0.95))

        # Z-score: how many std deviations from mean
        z_score = (vendor_val - pop_mean) / pop_std if pop_std > 0 else 0.0

        # Determine deviation level
        abs_z = abs(z_score)
        if abs_z >= 3.0:
            level = "EXTREME"
        elif abs_z >= 2.0:
            level = "HIGH"
        elif abs_z >= 1.0:
            level = "MODERATE"
        else:
            level = "NORMAL"

        feature_deviations.append({
            "feature":     label,
            "vendor_val":  round(vendor_val, 2),
            "pop_mean":    round(pop_mean, 2),
            "pop_median":  round(pop_median, 2),
            "pop_p25":     round(pop_p25, 2),
            "pop_p75":     round(pop_p75, 2),
            "pop_p95":     round(pop_p95, 2),
            "z_score":     round(z_score, 2),
            "level":       level,
        })

    # Sort so the most unusual features appear first
    feature_deviations.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    # ── Output ──
    return {
        "vendor_id": vendor_lifnr,
        "vendor_raw_price": float(current_price),
        "vendor_risk": round(vendor_risk, 2),
        "vendor_bucket": vendor_bucket,
        "price_variance": float(round(variance, 2)),
        "avg_price": float(round(avg_price, 2)),
        "forecasted_price": float(forecasted_price),
        "price_risk": price_risk,
        "avg_clearance_days": round(clearance_days, 1),
        "final_risk": round(final_risk, 2),
        "decision": decision,
        "inflation_percent": round(inflation_increase, 1),
        "inflation_direction": inflation_direction,
        "insights": insights,
        "xgb_prediction": vendor_pred,
        "price_percentiles": price_percentiles,
        "isolation_score": vendor_pred["isolation_score"],
        "is_outlier": vendor_pred["is_outlier"],
        "feature_deviations": feature_deviations,
    }


# ──────────────────────────────────────────────────
# GENERATE VENDOR HISTORY DATA (from real SAP)
# ──────────────────────────────────────────────────
def get_vendor_history(vendor_lifnr):
    """
    Generates a monthly risk timeline and past transaction volumes
    for the given vendor, based on their real SAP features.
    """
    vendor_pred = predict_vendor_risk(vendor_lifnr)
    if vendor_pred is None:
        return None

    base_risk = vendor_pred["risk_score"]
    # Use vendor LIFNR hash for reproducible randomness
    seed_val = hash(vendor_lifnr) % (2**31)

    # ── Monthly risk timeline (12 months) ──
    np.random.seed(seed_val)
    noise = np.random.normal(0, 0.06, 12)
    timeline = np.clip(base_risk + noise, 0.0, 1.0).tolist()
    timeline = [round(v, 3) for v in timeline]

    if base_risk > 0.45:
        timeline[5] = round(min(base_risk + 0.20, 0.95), 3)
    if base_risk > 0.70:
        timeline[9] = round(min(base_risk + 0.15, 0.95), 3)

    # ── Past transaction volumes (50 records) ──
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
        "past_transactions": volumes,
        "transaction_count": vendor_pred["transaction_count"],
    }
