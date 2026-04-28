# 🧠 AI Procurement Risk Analyzer — Targeted Improvements Prompt

---

## CRITICAL WORKING RULES

You are improving an existing, fully working Streamlit application.

- **Do NOT implement everything at once.** One task at a time only.
- After each task, show the result and **ask "Task N complete — shall I proceed to Task N+1?"**
- **Only proceed when I explicitly say yes or proceed.**
- **Do NOT rewrite, refactor, or touch anything that is not part of the current task.**
- **Read the "WHAT ALREADY EXISTS" section in full before writing a single line of code.**
  Every item listed there is already implemented and working. Do not add it again.

---

## FILES IN THIS PROJECT

| File | Purpose |
|---|---|
| `preprocess_sap.py` | Trains XGBoost + K-Means, saves `.pkl` files and `processed_sap_vendors.csv` |
| `model.py` | Inference engine — loads models, runs predictions, exposes functions to app |
| `app.py` | Streamlit UI — landing page + analytics page |
| `purchase_data.csv` | Synthetic price dataset generated at module load in `model.py` |
| `processed_sap_vendors.csv` | Output of `preprocess_sap.py` — 6,000 real SAP vendors |
| `xgb_risk_model.pkl` | Trained XGBoost classifier |
| `kmeans_risk_model.pkl` | Trained K-Means segmentation model |

---

## WHAT ALREADY EXISTS — DO NOT TOUCH OR REPEAT

### `preprocess_sap.py`
- Loads `LFA1`, `BSAK`, `BSIK` CSVs from `./files/`
- Aggregates `BSAK` into: `avg_days_overdue_hist`, `late_ratio`, `total_spend_vol`, `transaction_count`
- Aggregates `BSIK` into: `open_exposure`, `open_count`
- Merges into `df_master`, maps `RISK_CLASS` A/B/C/D to integers 0/1/2/3
- Trains `XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)`
- Trains `KMeans(n_clusters=4, n_init=10)` on raw unscaled features
- Saves `xgb_risk_model.pkl`, `kmeans_risk_model.pkl`, `processed_sap_vendors.csv`
- Prints training accuracy measured on the training set (not a held-out test set)

### `model.py`
- Module-level caching of XGBoost, K-Means, and SAP vendors CSV via `_load_models()`
- `get_sap_vendor_list()` — returns all vendors as list of dicts for UI dropdowns
- `predict_vendor_risk(lifnr)` — runs XGBoost prediction, returns predicted class A–D,
  probabilities for all 4 classes, weighted risk score 0–1, K-Means cluster,
  `avg_days_overdue`, `late_ratio`, `total_spend`, `open_exposure`, `transaction_count`, `sap_risk_class`
- `forecast_product_price(product_name)` — Linear Regression via `scipy.stats.linregress`
  on `date_ordinal` vs `price_per_unit`, returns today's forecasted price
- Synthetic price dataset with per-product inflation/deflation rates baked in:
  Enterprise Laptop −10%/yr, Corporate Smartphone −5%/yr,
  Rack Server +12%/yr, Cloud Compute Credit +6%/yr — 120 points per product over 24 months
- `procurement_risk_model(vendor_lifnr, product_name, current_price)`:
  - `final_risk = 0.6 × vendor_risk + 0.4 × price_risk`
  - Price risk: >0.4 variance → 0.95, >0.15 → 0.7, >0.05 → 0.4, else → 0.1 (hardcoded)
  - Decision: APPROVE (<0.3), REVIEW (0.3–0.7), HIGH RISK (>0.7)
  - Returns: `vendor_id`, `vendor_raw_price`, `vendor_risk`, `vendor_bucket`,
    `price_variance`, `avg_price`, `forecasted_price`, `price_risk`, `avg_clearance_days`,
    `final_risk`, `decision`, `inflation_percent`, `inflation_direction`, `insights`, `xgb_prediction`
  - Insights cover: inflation trend, XGBoost class vs SAP ground truth, price variance,
    system recommendation, K-Means cluster label
- `get_vendor_bucket(final_risk)` → LOW / MEDIUM / HIGH / CRITICAL thresholds 0.3, 0.6, 0.85
- `get_vendor_history(lifnr)` — 12-month timeline + 50 transaction volumes seeded
  from `hash(lifnr)`, anomaly count = `int(base_risk * 6)` injected at random indices

### `app.py` — Landing Page (`render_landing_page`)
- Dark UI: `#050810` bg, teal grid texture 48×48px, Inter + Roboto Mono fonts
- Animated status pills: SAP ERP LINKED (green pulse), AI ENGINE ONLINE (blue pulse)
- Hero: eyebrow text, gradient title, subtitle, accent line
- Metric strip: Model Accuracy hardcoded `98.4%`, Vendors Tracked `1,240`, Spend Analysed `$2.1B`
- Glass morphism form card with top-edge teal glow
- Real SAP vendor dropdown from `get_sap_vendor_list()`
- Product category dropdown (4 products)
- Quoted price number input
- On submit: runs `procurement_risk_model()` for selected vendor, then samples 9 comparison
  vendors with MD5-seeded deterministic price simulation (multiplier 0.8–2.2 × market avg),
  sorts all 10 by `final_risk`, assigns `rank`, stores everything in `st.session_state`

### `app.py` — Analytics Page Structure
- Ghost "← Back" button top-left, centered vendor title, glowing decision badge top-right
- Decision badge colors: APPROVE=`#14f0a0`, REVIEW=`#f0b840`, HIGH RISK=`#f85149`
- `glowPulse` CSS animation on decision badge
- Two tabs: **Risk Analysis** and **Vendor Comparison**

### `app.py` — Risk Analysis Tab (`render_risk_analysis_tab`)
- 4 metric cards: Vendor Reliability Score, Risk Classification Tier,
  Variance vs AI Forecast (inflation-adjusted %), AI Forecasted Price
- 24-Month Market Trend Alert card with inflation/deflation/stable state and colored border
- Invoice Aging card with `avg_clearance_days`, colored left border, large number display
- Gauge chart: Final Risk Score (color-coded bar + threshold line + colored steps)
- Donut chart: Risk Contribution Breakdown (60% vendor / 40% price, annotated center)
- Bullet gauge: Vendor Reliability Risk
- Bullet gauge: Market Price Variance
- Executive Summary: `st.info()` cards for each item in `result["insights"]`
- Vendor Behavior Analyzer: 12-month area line chart, 0.7 threshold line,
  rule-based `// CHART INTELLIGENCE SUMMARY` card below (trend direction + avg score)
- Vendor Pattern Anomaly Detection: scatter plot with red anomaly dots (>1000 volume),
  threshold line, rule-based summary card below (anomaly count + percentage)

### `app.py` — Vendor Comparison Tab (`render_vendor_comparison_tab`)
- Three-state alert banner: red (high risk, names top 2 alternatives),
  amber (sub-optimal, `final_risk >= 0.4` or `rank > 2`), green (optimal, rank ≤ 2)
- 10 ranked vendor cards showing: rank badge, vendor name + LIFNR, avg clearance days,
  quoted price, overall risk, vendor risk, price risk, bucket badge, progress bar
- SELECTED tag on the user's chosen vendor
- RECOMMENDED tag on top-2 vendors when selected vendor is high risk

---

## TASKS TO IMPLEMENT — ONE AT A TIME

---

### ✅ TASK 1 — Fix Train/Test Split in `preprocess_sap.py`

**The problem:** Training accuracy is currently measured on the same data the model
was trained on — this inflates the reported number and the 98.4% on the landing page
is almost certainly overstated.

**Edit `preprocess_sap.py` only. Exact change location:**
The block starting at `xgb.fit(X, y)` down to `print(f"XGBoost Training Accuracy: {acc:.2%}")`

**Steps:**
1. Before `xgb.fit()`, add a stratified 80/20 train/test split:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```
2. Train XGBoost on `X_train`, `y_train` only — not the full `X`
3. Evaluate on `X_test`, `y_test`:
   ```python
   from sklearn.metrics import classification_report
   test_preds = xgb.predict(X_test)
   real_acc = np.mean(test_preds == y_test)
   print(f"XGBoost Real Test Accuracy: {real_acc:.2%}")
   print(classification_report(y_test, test_preds, target_names=['A','B','C','D']))
   ```
4. Save the real accuracy to `model_metrics.json` in `SAVE_DIR`:
   ```python
   import json
   with open(os.path.join(SAVE_DIR, 'model_metrics.json'), 'w') as f:
       json.dump({"xgb_test_accuracy": round(real_acc * 100, 1)}, f)
   ```
5. K-Means must still be fit on the **full** `X` — clustering is unsupervised,
   no train/test split needed for it

**Do not touch:** K-Means training, data loading, merging, feature engineering,
file saving, `model.py`, or `app.py`

---

### ✅ TASK 2 — Add StandardScaler to K-Means in `preprocess_sap.py`

**The problem:** K-Means uses Euclidean distance. `total_spend_vol` is in the millions
while `late_ratio` is between 0 and 1 — K-Means is currently dominated by spend volume
and the behavioral features contribute almost nothing to cluster assignment.

**Edit `preprocess_sap.py` only:**

1. After the train/test split from Task 1, fit a `StandardScaler` on `X_train`:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   X_scaled = scaler.transform(X)   # full dataset for K-Means
   ```
2. Pass `X_train_scaled`, `y_train` to XGBoost (XGBoost is tree-based and doesn't
   need scaling, but keeping features consistent simplifies inference)
3. Pass `X_scaled` to K-Means
4. Save the scaler: `joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))`
5. Print cluster sizes after fitting:
   ```python
   unique, counts = np.unique(df_master['KMEANS_CLUSTER'], return_counts=True)
   print("K-Means cluster sizes:", dict(zip(unique, counts)))
   ```

**Then update `model.py`:**

6. Add `SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')` to the paths block
7. Add `_scaler = None` to the module-level globals
8. Inside `_load_models()`, load the scaler:
   ```python
   if _scaler is None:
       _scaler = joblib.load(SCALER_PATH)
   ```
   Return it alongside the other models
9. In `predict_vendor_risk()`, apply scaling before prediction:
   ```python
   features_scaled = _scaler.transform(features)
   ```
   Use `features_scaled` for both `xgb.predict()` and `kmeans.predict()`

**Do not touch:** Any UI in `app.py`, data loading/merging,
or any `model.py` function other than `_load_models()` and `predict_vendor_risk()`

---

### ✅ TASK 3 — Display Real Model Accuracy from `model_metrics.json` in `app.py`

**The problem:** The landing page metric strip shows a hardcoded `"98.4%"`.
After Task 1 we have a real held-out test accuracy in `model_metrics.json`.

**Edit `app.py` only, inside `render_landing_page()`:**

1. At the top of `render_landing_page()`, before any rendering, load the real accuracy:
   ```python
   import json
   try:
       with open("model_metrics.json", "r") as f:
           metrics = json.load(f)
       real_acc = f"{metrics['xgb_test_accuracy']}%"
   except Exception:
       real_acc = "N/A"
   ```
2. In the metric strip HTML block, replace the hardcoded `98.4%` string
   with `{real_acc}` using an f-string

**Do not touch:** The other two metrics (Vendors Tracked, Spend Analysed),
any other part of the landing page, or any part of the analytics page

---

### ✅ TASK 4 — Add SHAP Feature Explainability

**What this adds:** Currently the dashboard tells a procurement manager *that* a vendor
is risky. SHAP values tell them *why* — exactly which of the 4 features drove that score
up or down for this specific vendor. This is the single highest-value explainability
upgrade possible given the existing XGBoost model.

**Edit `model.py`:**

1. Add `import shap` at the top
2. Add `_shap_explainer = None` to module-level globals
3. Inside `_load_models()`, after loading `_xgb_model`, initialize:
   ```python
   _shap_explainer = shap.TreeExplainer(_xgb_model)
   ```
4. In `predict_vendor_risk()`, after computing `probabilities`, add:
   ```python
   shap_vals = _shap_explainer.shap_values(features_scaled)
   # shap_vals is shape (n_classes, n_samples, n_features) for multiclass
   shap_for_predicted_class = shap_vals[predicted_class][0]
   feature_names = ['avg_days_overdue', 'late_ratio', 'total_spend', 'open_exposure']
   shap_dict = dict(zip(feature_names, [round(float(v), 4) for v in shap_for_predicted_class]))
   ```
5. Add `"shap_values": shap_dict` to the return dict of `predict_vendor_risk()`
   (it is already returned inside `result["xgb_prediction"]` in `procurement_risk_model()`,
   so no change needed to `procurement_risk_model()` itself)

**Edit `app.py` — add a new section inside `render_risk_analysis_tab()`:**

6. Add this section **between the Executive Summary section and the
   Vendor Behavior Analyzer section** (do not move or touch either of those)
7. Add a `section-sep` div, then a `section-header` div labelled "XGBoost Feature Impact"
8. Read SHAP values: `shap_vals = result["xgb_prediction"]["shap_values"]`
9. Build a Plotly horizontal bar chart:
   - Y-axis: human-readable feature labels:
     `avg_days_overdue` → "Avg Days Overdue", `late_ratio` → "Late Payment Ratio",
     `total_spend` → "Total Spend Volume", `open_exposure` → "Open Exposure"
   - X-axis: SHAP value (positive = pushes risk up, negative = pushes risk down)
   - Bar color: `#f85149` if positive (increases risk), `#14f0a0` if negative (reduces risk)
   - Match chart styling: `paper_bgcolor='rgba(0,0,0,0)'`, `plot_bgcolor='rgba(10,14,26,0.6)'`,
     `font color '#94a3b8'`, height=220, minimal margins
10. Below the chart, add a rule-based text summary card using the same
    `// CHART INTELLIGENCE SUMMARY` style already used in the behavior/anomaly sections:
    - Primary driver: feature with highest positive SHAP value
    - Risk mitigator: feature with most negative SHAP value (skip if all are positive)
    - One sentence verdict based on which feature dominates

**Do not touch:** Any existing section in the analytics tab, the vendor comparison tab,
the landing page, or any `model.py` function other than `_load_models()` and `predict_vendor_risk()`

---

### ✅ TASK 5 — Add Elbow Plot Validation for K-Means in `preprocess_sap.py`

**The problem:** `n_clusters=4` was chosen arbitrarily. This task generates a validation
plot so you can visually confirm whether 4 is the right number of clusters.

**Edit `preprocess_sap.py` only:**

1. Before fitting the final K-Means model, loop k=2 to k=10 on `X_scaled`:
   ```python
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt

   inertias = []
   k_range = range(2, 11)
   for k in k_range:
       km = KMeans(n_clusters=k, random_state=42, n_init=10)
       km.fit(X_scaled)
       inertias.append(km.inertia_)
   ```
2. Save the elbow plot as `kmeans_elbow_plot.png` in `SAVE_DIR`:
   ```python
   fig, ax = plt.subplots(figsize=(8, 4), facecolor='#050810')
   ax.set_facecolor('#0a0e1a')
   ax.plot(list(k_range), inertias, color='#14f0a0', linewidth=2, marker='o',
           markerfacecolor='#14f0a0', markersize=6)
   ax.axvline(x=4, color='#f0b840', linestyle='--', alpha=0.6, label='Current k=4')
   ax.set_xlabel('Number of Clusters (k)', color='#94a3b8')
   ax.set_ylabel('Inertia', color='#94a3b8')
   ax.set_title('K-Means Elbow Plot — Validate Cluster Count', color='#f0f4ff')
   ax.tick_params(colors='#94a3b8')
   ax.legend(facecolor='#0a0e1a', labelcolor='#94a3b8')
   for spine in ax.spines.values():
       spine.set_edgecolor('#2a3a5a')
   plt.tight_layout()
   plt.savefig(os.path.join(SAVE_DIR, 'kmeans_elbow_plot.png'), dpi=120, facecolor='#050810')
   plt.close()
   print("Elbow plot saved → kmeans_elbow_plot.png. Review to validate k=4.")
   ```
3. Then continue with the existing `KMeans(n_clusters=4)` fit unchanged

**Do not touch:** The final K-Means model, `model.py`, or `app.py`

---

### ✅ TASK 6 — Replace Hardcoded Price Risk Thresholds with Percentile-Based Thresholds

**The problem:** Price risk thresholds (0.4, 0.15, 0.05) are hardcoded guesses.
They should be derived from the actual distribution of prices in `purchase_data.csv`.

**Edit `model.py` only:**

1. Create a new function `compute_price_risk(product_name, current_price)`:
   ```python
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
   ```
2. Inside `procurement_risk_model()`, replace the existing inline threshold block:
   ```python
   # OLD (remove this):
   if variance > 0.4:
       price_risk = 0.95
   elif variance > 0.15:
       ...
   # NEW (replace with):
   price_risk, price_percentiles = compute_price_risk(product_name, current_price)
   ```
3. Add `"price_percentiles": price_percentiles` to the `procurement_risk_model()` return dict

**Do not touch:** `forecast_product_price()`, `predict_vendor_risk()`,
any other function, or `app.py`

---

### ✅ TASK 7 — Show Price Percentile Context in the Analytics UI

**What this adds:** With real percentile data from Task 6, the UI can now show
*where* the quoted price sits in the market distribution, not just a variance percentage.

**Edit `app.py` only, inside `render_risk_analysis_tab()`:**

1. Read percentiles: `pp = result.get("price_percentiles", {})`
2. In the **3rd metric card** (currently "Variance vs AI Forecast"), keep the card
   but change its label to `"Price Position"` and its value to the percentile band:
   - `current_price > pp["p90"]` → show `"> P90"` in red (`#f85149`)
   - `current_price > pp["p75"]` → show `"P75–P90"` in red
   - `current_price > pp["p50"]` → show `"P50–P75"` in amber (`#f0b840`)
   - else → show `"≤ P50"` in green (`#14f0a0`)
   - `current_price` = `result["vendor_raw_price"]`
3. Immediately after the 24-Month Market Trend Alert card (before the section-sep),
   add a compact **Price Distribution Context** card:
   - Show a horizontal 5-point reference bar using inline divs:
     P25 | P50 | P75 | P90 — with the quoted price marked by a small teal triangle above
   - Show the actual dollar values at each percentile below the bar
   - Use the same `aging-card` CSS class for the card container
   - Keep it compact — max 80px height for the bar itself

**Do not touch:** Any other metric card, any chart, any other section,
the vendor comparison tab, or any file other than `app.py`

---

### ✅ TASK 8 — Final Integrity Check

After all tasks are complete, verify the following — fix anything that is wrong,
leave everything else exactly as it is:

1. **`preprocess_sap.py`**: train/test split exists using `stratify=y`,
   scaler is fit on `X_train` and saved as `scaler.pkl`,
   XGBoost is trained on `X_train_scaled`,
   K-Means is trained on full `X_scaled`,
   `model_metrics.json` is written with `xgb_test_accuracy`,
   elbow plot is generated and saved

2. **`model.py`**: `SCALER_PATH` constant exists, scaler is loaded in `_load_models()`,
   `features_scaled` is used in `predict_vendor_risk()` for both XGBoost and K-Means,
   `_shap_explainer` is initialized after XGBoost loads,
   `shap_values` dict is in the `predict_vendor_risk()` return dict,
   `compute_price_risk()` function exists and is called from `procurement_risk_model()`,
   `price_percentiles` is in the `procurement_risk_model()` return dict

3. **`app.py`**: `model_metrics.json` is loaded at the top of `render_landing_page()`,
   no hardcoded `"98.4%"` remains in the metric strip,
   SHAP section exists in Risk Analysis tab between Executive Summary
   and Vendor Behavior Analyzer,
   3rd metric card shows price position band not raw variance,
   price distribution context card exists after the Market Trend Alert

4. Confirm `get_sap_vendor_list()` is called at most twice across the full app
   (once in `render_landing_page()`, once in `render_vendor_comparison_tab()`).
   If Task 4–7 introduced a third call, cache the result in `st.session_state`

5. Confirm no existing feature from the "WHAT ALREADY EXISTS" section was
   accidentally removed or duplicated

---

## REMINDER

**Start with Task 1 only.**
Show me the modified `preprocess_sap.py`.
Then ask: *"Task 1 complete — shall I proceed to Task 2?"*
Wait for my approval before continuing.
