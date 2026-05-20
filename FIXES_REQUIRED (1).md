# Procurement Risk Analyzer — Must-Fix Issues

Five bugs and model reliability problems that must be addressed before this system
is used in any real procurement decision-making context.

---

## Fix 1 — `row.get()` Does Not Work on a Pandas Series

**Files affected:** `model.py`
**Severity:** Hard crash if CSV columns are missing (e.g. running against an older
`processed_sap_vendors.csv` before retraining)

Pandas `Series` does not have a `.get()` method that returns a default on missing keys
the way a Python `dict` does. Calling `row.get("KMEANS_CLUSTER", 0)` raises a
`KeyError` rather than falling back to `0`. This affects three reads:
`KMEANS_CLUSTER` in `get_sap_vendor_list()`, and `ISOLATION_SCORE` / `IS_OUTLIER`
in `predict_vendor_risk()`.

**Current code (broken):**

```python
# get_sap_vendor_list()
"kmeans_cluster": int(row.get("KMEANS_CLUSTER", 0)),

# predict_vendor_risk()
vendor_iso_score = float(row.get("ISOLATION_SCORE", 0.5))
vendor_is_outlier = bool(int(row.get("IS_OUTLIER", 0)))
```

**Fixed code:**

```python
# get_sap_vendor_list()
"kmeans_cluster": int(row["KMEANS_CLUSTER"]) if "KMEANS_CLUSTER" in row.index else 0,

# predict_vendor_risk()
vendor_iso_score = float(row["ISOLATION_SCORE"]) if "ISOLATION_SCORE" in row.index else 0.5
vendor_is_outlier = bool(int(row["IS_OUTLIER"])) if "IS_OUTLIER" in row.index else False
```

---

## Fix 2 — Vendor Comparison Tab Has No Guard Against Missing Selected Vendor

**Files affected:** `app.py` → `render_vendor_comparison_tab()`
**Severity:** Silent logic failure — shows misleading "Optimal Vendor Selection" banner
for a vendor that errored out during scoring

`selected_score` is correctly guarded with `if selected_score else 0` for the rank
and final risk reads, but if `selected_score` is `None` (vendor not found in
`all_scores` due to a model error), both values default to `0`. This causes the
comparison tab to render the green "Optimal Vendor Selection" banner silently instead
of surfacing the actual error.

**Current code (missing guard):**

```python
selected_score = next((s for s in all_scores if s["vendor_id"] == selected_lifnr), None)
selected_rank = selected_score["rank"] if selected_score else 0
selected_final_risk = selected_score["final_risk"] if selected_score else 0
# --- no early return here, rendering continues with misleading defaults ---
```

**Fixed code — add an early return immediately after the None check:**

```python
selected_score = next((s for s in all_scores if s["vendor_id"] == selected_lifnr), None)

if not selected_score:
    st.error("Selected vendor data not found in comparison results. Please re-run the analysis.")
    return

selected_rank = selected_score["rank"]
selected_final_risk = selected_score["final_risk"]
```

---

## Fix 3 — `hash()` Is Not Stable Across Python Processes

**Files affected:** `model.py` → `get_vendor_history()`
**Severity:** The "deterministic" 12-month risk history chart shows different data
every time the Streamlit server restarts, breaking reproducibility

Python randomizes `hash()` output per-process by default (controlled by
`PYTHONHASHSEED`). The seed `hash(vendor_lifnr) % (2**31)` produces a different
integer on every server restart, meaning the same vendor shows a different historical
risk trend each session. This makes the Vendor Behavior Analyzer useless for
monitoring trends over time.

Note that `app.py` already handles this correctly in the vendor scoring loop using
`hashlib.md5` — `model.py` just missed it.

**Current code (unstable):**

```python
# model.py — get_vendor_history()
seed_val = hash(vendor_lifnr) % (2**31)
```

**Fixed code:**

```python
import hashlib

# model.py — get_vendor_history()
seed_val = int(hashlib.md5(vendor_lifnr.encode()).hexdigest()[:8], 16) % (2**31)
```

No other changes needed — `np.random.seed(seed_val)` calls below this line remain
as-is.

---

## Fix 4 — XGBoost Model Accuracy Is 40.2% on a 4-Class Problem

**Files affected:** `preprocess_sap.py`, then retrain all models
**Severity:** Core model reliability — every downstream output (vendor risk score,
APPROVE/REVIEW/HIGH RISK decision, insights, final risk) inherits this weakness

On a 4-class classification problem, random guessing achieves 25%. At 40.2% the model
is learning signal but is not reliable enough to drive procurement decisions. The most
likely cause is class imbalance — SAP vendor populations are typically heavily skewed
toward risk class A/B, so the model learns to predict the majority class and struggles
with C and D vendors (which are exactly the ones that matter most).

**Step 1 — Add sample weighting to the XGBoost training call in `preprocess_sap.py`:**

```python
from sklearn.utils.class_weight import compute_sample_weight

# Add this after the train/test split, before xgb.fit()
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42
)
xgb.fit(X_train_scaled, y_train, sample_weight=sample_weights)
```

**Step 2 — After retraining, inspect the classification report:**

```
              precision    recall  f1-score
           A       ...
           B       ...
           C       ...       ← check these two specifically
           D       ...       ← should improve most with balancing
```

If accuracy is still below ~55% after balancing, the 4 features alone may lack enough
signal to separate all 4 classes cleanly. In that case, consider merging classes C and D
into a single HIGH class, making this a 3-class problem (LOW / MEDIUM / HIGH). This
also aligns better with how the app actually uses the output — `vendor_bucket` already
collapses the 4 XGBoost classes into LOW / MEDIUM / HIGH / CRITICAL anyway.

**Step 3 — Retrain all models and restart Streamlit:**

```bash
python preprocess_sap.py
# then restart the Streamlit server to reload the new .pkl files
```

---

## Fix 5 — Price Percentile Buckets Are Calculated Against the Full 24-Month Distribution

**Files affected:** `model.py` → `compute_price_risk()`
**Severity:** Logic flaw — inflation-adjusted fair quotes are incorrectly penalized
as expensive because percentiles are dominated by older, cheaper historical prices

`compute_price_risk()` calculates P25/P50/P75/P90 against all 120 historical price
points spanning 24 months. For a product that has experienced inflation, the majority
of those data points were recorded when prices were lower. This means a quote that
the regression correctly identifies as fair for today can still land above P75 or P90
of the historical distribution, generating an inflated price risk score — directly
contradicting the inflation-adjusted forecast.

**Concrete example:**
A Rack Server has been inflating at +12%/year. Historical prices cluster around
$5,500–$6,500. The regression forecasts today's fair price at $7,050. A vendor
quotes $7,100 — only 0.7% above the forecast, essentially fair. But against the
full 24-month distribution, $7,100 sits above P90, triggering `price_risk = 0.90`.
The price risk component (40% of final risk) then unfairly drags down the vendor's
score despite their quote being reasonable.

**Current code (full distribution):**

```python
def compute_price_risk(product_name, current_price):
    df_p = pd.read_csv(PURCHASE_DATA_PATH)
    prices = df_p[df_p["product_name"] == product_name]["price_per_unit"].values
    if len(prices) < 5:
        return 0.5, {}
    p25 = float(np.percentile(prices, 25))
    p50 = float(np.percentile(prices, 50))
    p75 = float(np.percentile(prices, 75))
    p90 = float(np.percentile(prices, 90))
    ...
```

**Fixed code — filter to the last 6 months before calculating percentiles:**

```python
def compute_price_risk(product_name, current_price):
    df_p = pd.read_csv(PURCHASE_DATA_PATH)
    df_p["date"] = pd.to_datetime(df_p["date"])
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=6)
    df_recent = df_p[(df_p["product_name"] == product_name) & (df_p["date"] >= cutoff)]

    # Fall back to full history if recent window has too few points
    if len(df_recent) < 5:
        df_recent = df_p[df_p["product_name"] == product_name]

    prices = df_recent["price_per_unit"].values
    if len(prices) < 5:
        return 0.5, {}
    p25 = float(np.percentile(prices, 25))
    p50 = float(np.percentile(prices, 50))
    p75 = float(np.percentile(prices, 75))
    p90 = float(np.percentile(prices, 90))
    ...
```

The 6-month window ensures percentile thresholds reflect current market conditions
rather than a distribution anchored to old prices. The fallback to full history
prevents failures when a product has sparse recent data.

---

## Summary

| # | Location | Type | Impact |
|---|----------|------|--------|
| 1 | `model.py` — `get_sap_vendor_list()`, `predict_vendor_risk()` | Hard crash on missing CSV columns | App unusable against older CSV |
| 2 | `app.py` — `render_vendor_comparison_tab()` | Silent wrong output | Misleading green banner on errored vendor |
| 3 | `model.py` — `get_vendor_history()` | Non-deterministic output | History chart changes on every server restart |
| 4 | `preprocess_sap.py` — XGBoost training | Model reliability | All risk scores and decisions are unreliable at 40.2% accuracy |
| 5 | `model.py` — `compute_price_risk()` | Logic flaw | Inflation-adjusted fair quotes penalized by stale price distribution |

Fixes 1–3 are one-line or two-line changes that can be applied immediately. Fix 5
is also a small code change but has meaningful impact on price fairness scoring for
any product with an inflation or deflation trend. Fix 4 requires a retrain cycle
but is the most important one for production use.
