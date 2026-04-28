# 🌲 Isolation Forest — Vendor Anomaly Detection Feature

---

## CRITICAL WORKING RULES

You are adding **one specific new feature** to an existing, fully working Streamlit
procurement risk application. Do not touch anything that is not explicitly mentioned
in this prompt. Do not refactor, rename, or reorganize existing code.

---

## WHAT ALREADY EXISTS — DO NOT REPEAT OR TOUCH

### `preprocess_sap.py`
- Already loads and merges `LFA1`, `BSAK`, `BSIK` SAP tables
- Already has a stratified 80/20 train/test split
- Already fits a `StandardScaler` on `X_train`, saves it as `scaler.pkl`
- Already trains `XGBClassifier` on scaled features, saves `xgb_risk_model.pkl`
- Already trains `KMeans(n_clusters=4)` on full scaled `X`, saves `kmeans_risk_model.pkl`
- Already generates `kmeans_elbow_plot.png`
- Already saves `model_metrics.json` with real held-out test accuracy
- Already saves `processed_sap_vendors.csv`
- Features used: `avg_days_overdue_hist`, `late_ratio`, `total_spend_vol`, `open_exposure`

### `model.py`
- Already loads XGBoost, K-Means, StandardScaler, and SAP vendors CSV via `_load_models()`
- Already applies `scaler.transform(features)` before XGBoost and K-Means prediction
- `predict_vendor_risk(lifnr)` already returns: `predicted_class`, `predicted_class_label`,
  `risk_score`, `probabilities`, `kmeans_cluster`, `shap_values`, `avg_days_overdue`,
  `late_ratio`, `total_spend`, `open_exposure`, `transaction_count`, `sap_risk_class`
- `procurement_risk_model()` already returns: `vendor_id`, `vendor_raw_price`,
  `vendor_risk`, `vendor_bucket`, `price_variance`, `avg_price`, `forecasted_price`,
  `price_risk`, `price_percentiles`, `avg_clearance_days`, `final_risk`, `decision`,
  `inflation_percent`, `inflation_direction`, `insights`, `xgb_prediction`
- K-Means cluster insight is already appended to the `insights` list inside
  `procurement_risk_model()` — do not add another cluster insight

### `app.py`
- Already has a Risk Analysis tab with these sections in this exact order:
  1. Core Risk Metrics (4 cards)
  2. 24-Month Market Trend Alert
  3. Price Distribution Context card
  4. Invoice Aging
  5. Risk Visualizations (gauge + donut)
  6. Deep Analytics (bullet gauges)
  7. Executive Summary (`st.info()` per insight)
  8. XGBoost Feature Impact (SHAP horizontal bar chart + summary)
  9. Vendor Behavior Analyzer (12-month line chart + summary)
  10. Vendor Pattern Anomaly Detection (scatter plot + summary)
- Already has a Vendor Comparison tab — do not touch it
- Already has `setup_analytics_styles()` with `.section-header`, `.section-sep`,
  `.analytic-card`, `.aging-card` CSS classes — reuse these, do not add new CSS classes

---

## THE FEATURE: Isolation Forest Vendor Outlier Detection

### What it does and why it's different from K-Means

K-Means groups all vendors into clusters — it forces every vendor into one of 4
segments. It cannot say "this vendor is anomalous" because every vendor must belong
somewhere. Isolation Forest specifically answers the question:
**"Is this vendor behaving unusually compared to the entire vendor population?"**
It assigns an anomaly score — the more isolated a vendor's feature values are from
the rest of the population, the higher the anomaly score. This catches vendors that
K-Means would silently absorb into the nearest cluster.

---

## IMPLEMENTATION — 3 FILES TO EDIT

---

### STEP 1 — Train and Save Isolation Forest in `preprocess_sap.py`

Add the following **after** the K-Means training block and **before** the
"Saving Models and Processed Data" print statement.

```python
from sklearn.ensemble import IsolationForest

print("Training Isolation Forest Anomaly Detector...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # assume ~5% of vendors are genuine outliers
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_scaled)   # use the already-computed X_scaled from StandardScaler

# Score all vendors — lower (more negative) = more anomalous
# decision_function returns raw anomaly scores; convert to 0-1 range for usability
raw_scores = iso_forest.decision_function(X_scaled)          # negative = anomalous
anomaly_labels = iso_forest.predict(X_scaled)                # -1 = outlier, 1 = normal

# Normalize scores to 0-1 where 1.0 = most anomalous
min_s, max_s = raw_scores.min(), raw_scores.max()
normalized_scores = 1 - ((raw_scores - min_s) / (max_s - min_s))
normalized_scores = np.clip(normalized_scores, 0.0, 1.0)

df_master['ISOLATION_SCORE'] = np.round(normalized_scores, 4)
df_master['IS_OUTLIER'] = (anomaly_labels == -1).astype(int)

n_outliers = (anomaly_labels == -1).sum()
print(f"Isolation Forest complete. Outliers detected: {n_outliers} / {len(df_master)}")
```

Then save the model — add this line to the existing saving block:
```python
joblib.dump(iso_forest, os.path.join(SAVE_DIR, 'isolation_forest_model.pkl'))
```

And update the print statement listing saved files to include:
```
 - isolation_forest_model.pkl
```

The `df_master.to_csv(...)` call already saves `processed_sap_vendors.csv` — since
`ISOLATION_SCORE` and `IS_OUTLIER` are now columns in `df_master`, they will be
saved automatically. No additional CSV changes needed.

---

### STEP 2 — Load and Use Isolation Forest in `model.py`

**2a — Add path constant** (add alongside existing path constants at the top):
```python
ISOLATION_FOREST_PATH = os.path.join(BASE_DIR, 'isolation_forest_model.pkl')
```

**2b — Add module-level global** (add alongside `_xgb_model`, `_kmeans_model` etc.):
```python
_iso_forest = None
```

**2c — Load inside `_load_models()`** (add after the K-Means load block):
```python
if _iso_forest is None:
    _iso_forest = joblib.load(ISOLATION_FOREST_PATH)
```
Return `_iso_forest` alongside the other models. Update the return statement and all
callers of `_load_models()` accordingly — currently only `predict_vendor_risk()` and
`get_sap_vendor_list()` call it.

**2d — Add anomaly data to `predict_vendor_risk()`:**

Inside `predict_vendor_risk()`, after the K-Means prediction line, add:
```python
# Isolation Forest anomaly score for this vendor
iso_score = float(_iso_forest.decision_function(features_scaled)[0])
iso_label = int(_iso_forest.predict(features_scaled)[0])   # -1 or 1

# Normalize to 0-1 (higher = more anomalous) using the vendor's stored score
# for consistency with population-level normalization done at training time
vendor_iso_score = float(row.get("ISOLATION_SCORE", 0.5))
vendor_is_outlier = bool(int(row.get("IS_OUTLIER", 0)))
```

Add these keys to the `predict_vendor_risk()` return dict:
```python
"isolation_score": round(vendor_iso_score, 4),   # 0-1, higher = more anomalous
"is_outlier": vendor_is_outlier,                  # True if Isolation Forest flagged
```

**2e — Add Isolation Forest insight to `procurement_risk_model()`:**

Inside the insights block of `procurement_risk_model()`, after the K-Means cluster
insight line and before the return statement, add:
```python
# Isolation Forest insight
iso_score = vendor_pred["isolation_score"]
is_outlier = vendor_pred["is_outlier"]

if is_outlier and iso_score > 0.75:
    insights.append(
        f"Isolation Forest: This vendor is flagged as a statistical outlier "
        f"(anomaly score: {iso_score:.2f}). Their payment and spend patterns are "
        f"highly unusual compared to the full vendor population — independent of risk class."
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
```

Also add `"isolation_score"` and `"is_outlier"` to the `procurement_risk_model()` return
dict by reading them from `vendor_pred`:
```python
"isolation_score": vendor_pred["isolation_score"],
"is_outlier": vendor_pred["is_outlier"],
```

---

### STEP 3 — Display Isolation Forest Results in `app.py`

**Add a new section inside `render_risk_analysis_tab()` only.**

Place it **between section 8 (XGBoost Feature Impact / SHAP) and
section 9 (Vendor Behavior Analyzer)**. Do not move or touch either of those sections.

**3a — Section separator and header:**
```python
st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Isolation Forest — Population Outlier Analysis</div>',
            unsafe_allow_html=True)
```

**3b — Read the data:**
```python
iso_score = result.get("isolation_score", 0.5)
is_outlier = result.get("is_outlier", False)

if iso_score > 0.75:
    iso_color = "#f85149"
    iso_label = "ANOMALOUS"
    iso_desc  = "This vendor's behaviour is highly atypical relative to the full population."
elif iso_score > 0.5:
    iso_color = "#f0b840"
    iso_label = "BORDERLINE"
    iso_desc  = "Some behavioural features deviate moderately from population norms."
else:
    iso_color = "#14f0a0"
    iso_label = "NORMAL"
    iso_desc  = "Vendor behaviour is consistent with the general vendor population."
```

**3c — Display a two-column layout:**

Left column — anomaly score card (reuse `.analytic-card` class):
```python
col_iso1, col_iso2 = st.columns(2, gap="medium")

with col_iso1:
    st.markdown(f"""
    <div class="analytic-card">
        <div class="card-label">Isolation Forest<br>Anomaly Score</div>
        <div class="card-value" style="color:{iso_color};">{iso_score:.2f}</div>
        <div style="margin-top:12px; font-family:'Roboto Mono',monospace;
                    font-size:0.68rem; letter-spacing:0.1em; color:{iso_color};">
            {iso_label}
        </div>
    </div>
    """, unsafe_allow_html=True)
```

Right column — a Plotly gauge showing the anomaly score from 0 to 1, styled
consistently with the existing Final Risk Score gauge (same `paper_bgcolor`,
`plot_bgcolor`, font settings). Color the gauge bar with `iso_color`.
Title: `"Population Outlier Score"`. Height: 260.

**3d — Below the two columns, add the `// CHART INTELLIGENCE SUMMARY` card**
using the same style as the existing behavior and anomaly detection summaries:
```python
st.html(f"""
<div style="background:{'rgba(248,81,73,0.03)' if is_outlier else 'rgba(20,240,160,0.03)'};
            border-left:3px solid {iso_color}; padding:16px 20px; margin-top:16px;
            margin-bottom:16px; border-radius:0 8px 8px 0;">
    <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem;
                color:#8a9ab8; letter-spacing:0.1em; margin-bottom:8px;">
        // CHART INTELLIGENCE SUMMARY
    </div>
    <div style="font-family:'Inter',sans-serif; font-size:0.85rem;
                color:#c8d8f0; line-height:1.5;">
        {iso_desc} Isolation Forest evaluates vendors independently of their SAP risk class —
        a vendor can be K-Means "Stable Mid-Tier" but still be a population outlier if their
        specific combination of spend volume, overdue days, and late payment ratio is unusual.
        This score complements, not replaces, the XGBoost classification.
    </div>
</div>
""")
```

**3e — Also update the Vendor Comparison tab cards in `render_vendor_comparison_tab()`:**

The comparison vendors' scores are stored in `st.session_state['all_vendor_scores']`.
Each score dict already has `isolation_score` and `is_outlier` because
`procurement_risk_model()` now returns them (Step 2e).

In the vendor card HTML loop, add a small anomaly badge below the existing
price risk metric in the center metrics section:
```python
iso_s = s.get("isolation_score", 0.5)
iso_badge_color = "#f85149" if iso_s > 0.75 else "#f0b840" if iso_s > 0.5 else "#14f0a0"
iso_badge_label = "ANOMALOUS" if iso_s > 0.75 else "BORDERLINE" if iso_s > 0.5 else "NORMAL"
```

Then inside the center metrics `<div>`, add a 4th metric after Price Risk:
```html
<div style="text-align:center;">
    <div style="font-family:'Roboto Mono',monospace; font-size:0.58rem; color:#94a3b8;
                letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">
        Outlier Score
    </div>
    <div style="font-family:'Inter',sans-serif; font-size:1.1rem; font-weight:700;
                color:{iso_badge_color};">{iso_s:.2f}</div>
</div>
```

---

## FILES TO REGENERATE AFTER IMPLEMENTATION

After all three steps are implemented:

1. **Re-run `preprocess_sap.py`** — this retrains all models including the new
   Isolation Forest and regenerates `processed_sap_vendors.csv` with the new
   `ISOLATION_SCORE` and `IS_OUTLIER` columns
2. **Restart the Streamlit app** — `model.py` loads models at module level so a
   restart is required to pick up `isolation_forest_model.pkl`

---

## DO NOT TOUCH

- The vendor comparison tab alert banner logic (red/amber/green banners)
- The SELECTED and RECOMMENDED tag logic
- The rank sorting logic in `render_landing_page()`
- Any CSS class definitions in `setup_analytics_styles()` or `inject_landing_styles()`
- The landing page form, hero section, status pills, or metric strip
- `forecast_product_price()`, `compute_price_risk()`, `get_vendor_history()`,
  `get_sap_vendor_list()`, or `get_vendor_bucket()` in `model.py`
- The existing K-Means cluster label insight already in `procurement_risk_model()`
- Any existing chart or section in the Risk Analysis tab
