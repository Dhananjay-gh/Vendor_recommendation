# AI Procurement Risk Analyzer — Project README

> **Living document.** Update this file whenever a feature is added, a model is retrained, or a file is renamed.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [SAP Data Sources](#3-sap-data-sources)
4. [Feature Engineering](#4-feature-engineering)
5. [ML Pipeline](#5-ml-pipeline)
6. [Model Files](#6-model-files)
7. [Application Architecture](#7-application-architecture)
8. [UI & Theming](#8-ui--theming)
9. [AI Chatbot](#9-ai-chatbot)
10. [Pending Changes (Approved, Not Yet Implemented)](#10-pending-changes-approved-not-yet-implemented)
11. [Change Rules](#11-change-rules)
12. [Run Instructions](#12-run-instructions)

---

## 1. Project Overview

A Streamlit web application that ingests SAP procurement data and produces a real-time, ML-powered vendor risk assessment. For any given vendor + product + quoted price, the system outputs:

- A **Final Risk Score** (0–1) combining vendor behavioral risk and price risk
- An **Approve / Review / High Risk** procurement decision
- **Explainability** via SHAP values, K-Means cluster assignment, and Isolation Forest anomaly scoring
- A **Vendor Comparison** tab that auto-scores all vendors against the same product at market price
- An **AI Risk Analyst chatbot** (OpenRouter LLM) that answers questions grounded in the live session data

---

## 2. Repository Structure

```
project-root/
│
├── app.py                        # Streamlit UI — all pages and tabs
├── model.py                      # ML inference, price forecasting, chatbot context builder
├── preprocess_sap.py             # One-time data pipeline: feature engineering + model training
│
├── processed_sap_vendors.csv     # Output of preprocess_sap.py — vendor feature table
├── purchase_data.csv             # Synthetic or real historical price data (auto-generated if missing)
├── feature_columns.json          # Ordered list of the 14 feature columns used by all models
│
├── xgb_risk_model.pkl            # Trained XGBoost classifier
├── kmeans_risk_model.pkl         # Trained K-Means (k=4) clustering model
├── scaler.pkl                    # StandardScaler fitted on training data
├── isolation_forest_model.pkl    # Trained Isolation Forest anomaly detector
├── model_metrics.json            # Held-out test accuracy written by preprocess_sap.py
├── kmeans_elbow_plot.png         # Elbow plot for validating k=4 choice
│
├── files/                        # Raw SAP table CSVs (never modified by the app)
│   ├── LFA1_Vendor_Master_General.csv
│   ├── LFB1_Vendor_Master_CompCode.csv
│   ├── BSAK_Cleared_Items.csv
│   ├── BSIK_Open_Items.csv
│   ├── BKPF_Document_Header.csv
│   ├── BSEG_Document_Segment.csv
│   └── PAYR_Payment_Medium.csv
│
└── README.md                     # This file
```

---

## 3. SAP Data Sources

| Table | Columns Used | Purpose |
|---|---|---|
| `LFA1` | `LIFNR`, `NAME1`, `RISK_CLASS` | Vendor identity + SAP static risk class |
| `LFB1` | `LIFNR`, `ZTERM`, `ZAHLS`, `SPERR`, `DUNNLEVEL`, `ERDAT` | Payment terms, block flags, dunning, vendor age |
| `BSAK` | `LIFNR`, `WRBTR`, `DAYS_OVERDUE_AT_CLEAR`, `CLEARED_LATE`, `BELNR` | Historical cleared invoice behavior |
| `BSIK` | `LIFNR`, `WRBTR`, `BELNR` | Open (unpaid) invoice exposure |
| `BKPF` | `BELNR`, `BUKRS`, `GJAHR`, `STBLG`, `BSTAT`, `BLART` | Document reversal detection |
| `BSEG` | `LIFNR`, `BELNR`, `BUKRS`, `GJAHR`, `ZTERM`, `ZBD1T`, `ZBD1P`, `AUGDT`, `FAEDT`, `WRBTR`, `KOART` | Discount capture, payment terms days (vendor lines only: `KOART = 'K'`) |
| `PAYR` | `LIFNR`, `BELNR`, `XVOIDED`, `STALE`, `RWBTR` | Voided and stale payment detection |

**Rule:** Raw files in `files/` are read-only inputs. Never write to them.

---

## 4. Feature Engineering

All features are computed in `preprocess_sap.py` and stored in `processed_sap_vendors.csv`. The ordered list is also saved to `feature_columns.json` so `model.py` loads it dynamically — **no hardcoded feature lists in model.py**.

### The 14 Training Features

| # | Column | Source | What it measures |
|---|---|---|---|
| 1 | `avg_days_overdue_hist` | BSAK | Mean days late at payment clearance |
| 2 | `late_ratio` | BSAK | Fraction of invoices cleared late |
| 3 | `total_spend_vol` | BSAK | Cumulative spend volume (absolute) |
| 4 | `open_exposure` | BSIK | Current outstanding unpaid balance |
| 5 | `payment_consistency_score` | BSAK | Std deviation of overdue days (erratic = risky) |
| 6 | `years_active` | LFB1 | Years since vendor creation date in SAP |
| 7 | `payment_terms_risk` | LFB1 | Risk score of agreed payment terms (NET90 = 0.75, NET15 = 0.10) |
| 8 | `is_payment_blocked` | LFB1 | 1 if blocked in any company code via ZAHLS or SPERR |
| 9 | `dunning_level` | LFB1 | Highest dunning notice level (0–4) across company codes |
| 10 | `reversal_rate` | BKPF + BSEG | Fraction of vendor documents that were reversed |
| 11 | `discount_capture_rate` | BSEG | Fraction of discount-eligible invoices paid on time |
| 12 | `avg_payment_terms_days` | BSEG | Mean net days (ZBD1T) across all invoices |
| 13 | `voided_payment_rate` | PAYR | Fraction of payments voided |
| 14 | `stale_payment_rate` | PAYR | Fraction of payments gone stale |

### Non-Training Columns in `processed_sap_vendors.csv`

These are stored alongside the features but are **not** passed to any ML model:

| Column | Source | Purpose |
|---|---|---|
| `LIFNR` | LFA1 | Vendor ID — primary key |
| `NAME1` | LFA1 | Vendor name for display |
| `RISK_CLASS` | LFA1 | SAP's static vendor class (A/B/C/D) — shown in UI as a reference indicator only |
| `transaction_count` | BSAK | Number of cleared invoices (display + chatbot context) |
| `KMEANS_CLUSTER` | Computed | K-Means cluster assignment (0–3) |
| `ISOLATION_SCORE` | Computed | Normalized anomaly score (0=normal, 1=most anomalous) |
| `IS_OUTLIER` | Computed | 1 if Isolation Forest flagged this vendor as an outlier |

---

## 5. ML Pipeline

### 5a. Label Engineering — **CRITICAL: Read Before Retraining**

> **Current status: the label change below is APPROVED but NOT YET implemented.** The model currently trains on SAP's `RISK_CLASS` (A/B/C/D mapped to 0/1/2), which explains the ~33% test accuracy — SAP's static label does not correlate well with the behavioral features.

**Approved fix — composite behavioral label:**

Replace the `risk_map` block in `preprocess_sap.py` with:

```python
df_master['composite_risk'] = (
    df_master['late_ratio'].clip(0, 1)                              * 0.30 +
    (df_master['avg_days_overdue_hist'].clip(0, 60) / 60)           * 0.25 +
    (df_master['dunning_level'] / 4).clip(0, 1)                     * 0.15 +
    df_master['reversal_rate'].clip(0, 1)                           * 0.10 +
    df_master['voided_payment_rate'].clip(0, 1)                     * 0.10 +
    df_master['is_payment_blocked'].clip(0, 1)                      * 0.10
)
df_master['RISK_CLASS_LABEL'] = pd.cut(
    df_master['composite_risk'],
    bins=[0, 0.25, 0.55, 1.01],
    labels=[0, 1, 2],   # 0=Low, 1=Medium, 2=High
    include_lowest=True
).astype(int)
```

`RISK_CLASS` from LFA1 must remain in `df_master` as a reference column and must flow through to `processed_sap_vendors.csv` — it is displayed in the UI alongside the behavioral model output.

**After this label change:**
- Update `classification_report` target names: `['Low', 'Medium', 'High']`
- Update `predict_vendor_risk()` return dict: `predicted_class_label` should use `{0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}`
- Add `sap_divergence` flag in `procurement_risk_model()`: fires when behavioral model says HIGH/CRITICAL but SAP class is A or B
- Add `sap_risk_label` (human-readable SAP class) to both `predict_vendor_risk()` and `procurement_risk_model()` return dicts
- Update chatbot system prompt in `build_chatbot_context()` to include `sap_divergence` and `sap_risk_label`
- Update Risk Classification card in `render_risk_analysis_tab()` to show both labels + divergence badge
- Update Vendor Comparison cards to show SAP class badge

### 5b. Train/Test Split

- 80/20 stratified split, `random_state=42`
- Scaler fitted on `X_train` only; `X_test` and full `X` are transformed (not fitted)

### 5c. XGBoost Classifier

```python
XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
)
```
Trained with `compute_sample_weight(class_weight='balanced')` to handle class imbalance.

### 5d. K-Means Clustering

- `KMeans(n_clusters=4, random_state=42, n_init=10)` fitted on full scaled `X`
- Cluster labels used in UI and chatbot: `{0: "Conservative Spenders", 1: "High-Volume Partners", 2: "At-Risk Outliers", 3: "Stable Mid-Tier"}`
- Elbow plot (k=2–10) saved as `kmeans_elbow_plot.png`

### 5e. Isolation Forest

- `IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)` fitted on full scaled `X`
- Raw `decision_function` scores normalized to 0–1 (1 = most anomalous)
- Population-level scores stored in `processed_sap_vendors.csv` as `ISOLATION_SCORE` and `IS_OUTLIER`
- At inference, `model.py` reads the stored score for consistency with training-time normalization

### 5f. Price Forecasting

- `scipy.stats.linregress` on historical `purchase_data.csv` prices (date ordinal vs. price)
- Forecasted price = regression value at current date
- Price risk = percentile of quoted price within last 6 months of data

---

## 6. Model Files

| File | Created by | Used by | Notes |
|---|---|---|---|
| `xgb_risk_model.pkl` | `preprocess_sap.py` | `model.py` | Retrain if features change |
| `kmeans_risk_model.pkl` | `preprocess_sap.py` | `model.py` | Retrain alongside XGBoost |
| `scaler.pkl` | `preprocess_sap.py` | `model.py` | Must match training feature order |
| `isolation_forest_model.pkl` | `preprocess_sap.py` | `model.py` | Retrain if feature set changes |
| `feature_columns.json` | `preprocess_sap.py` | `model.py` | Source of truth for feature order |
| `model_metrics.json` | `preprocess_sap.py` | `app.py` (landing page) | Displayed as accuracy pill |

**Rule:** Whenever `features` list changes in `preprocess_sap.py`, ALL five `.pkl` files and `feature_columns.json` must be regenerated together. A scaler from one feature set is incompatible with models from another.

---

## 7. Application Architecture

### 7a. Pages

| Page | Function | Trigger |
|---|---|---|
| Landing / Input | `render_landing_page()` | `st.session_state['page'] == 'input'` |
| Analytics | `render_analytics_page()` | `st.session_state['page'] == 'analytics'` |

### 7b. Tabs (Analytics page)

| Tab | Function |
|---|---|
| Risk Analysis | `render_risk_analysis_tab()` |
| Vendor Comparison | `render_vendor_comparison_tab()` |
| AI Risk Analyst (chat) | `render_chatbot_tab()` |

### 7c. Risk Analysis Tab — Section Order

Do not reorder or remove these sections without updating this document:

1. Core Risk Metrics (4 cards: Final Risk Score, Vendor Risk, Price Risk, Risk Classification)
2. 24-Month Market Trend Alert
3. Price Distribution Context card
4. Invoice Aging
5. Risk Visualizations (Final Risk gauge + class probability donut)
6. Deep Analytics (bullet gauges for key features)
7. Executive Summary (`st.info()` per insight string from `result["insights"]`)
8. XGBoost Feature Impact (SHAP horizontal bar + summary)
9. **Isolation Forest — Population Outlier Analysis** (score card + gauge + summary)
10. Vendor Behavior Analyzer (12-month line chart + summary)
11. Vendor Pattern Anomaly Detection (scatter plot + summary)

### 7d. Key Session State Variables

| Key | Type | Set by | Used by |
|---|---|---|---|
| `page` | `str` | `render_landing_page()` | `main()` |
| `result` | `dict` | `render_landing_page()` | All analytics functions |
| `selected_vendor` | `str` | `render_landing_page()` | Tabs, chatbot |
| `selected_product` | `str` | `render_landing_page()` | Tabs, chatbot |
| `all_vendor_scores` | `list[dict]` | `render_landing_page()` | Comparison tab, chatbot |
| `chat_history` | `list[dict]` | `render_chatbot_tab()` | Chatbot |
| `chat_prefill` | `str` | Chip buttons, vendor list | Chatbot input |
| `theme` | `str` | Theme toggle | `setup_analytics_styles()`, `inject_landing_styles()` |
| `plot_bg` | `str` | `setup_analytics_styles()` | Plotly `plot_bgcolor` |
| `paper_bg` | `str` | `setup_analytics_styles()` | Plotly `paper_bgcolor` |
| `text_main` | `str` | `setup_analytics_styles()` | Plotly font color |

### 7e. `procurement_risk_model()` Return Dict

All keys currently returned (app.py must not assume keys beyond this list):

```
vendor_id, vendor_raw_price, vendor_risk, vendor_bucket, price_variance,
avg_price, forecasted_price, price_risk, avg_clearance_days, final_risk,
decision, inflation_percent, inflation_direction, insights, xgb_prediction,
price_percentiles, isolation_score, is_outlier, feature_deviations
```

When the label-change feature is implemented, these keys will be added:
```
sap_divergence, sap_risk_label
```

### 7f. `predict_vendor_risk()` Return Dict

```
predicted_class, predicted_class_label, risk_score, probabilities,
kmeans_cluster, isolation_score, is_outlier,
avg_days_overdue, late_ratio, total_spend, open_exposure, transaction_count,
sap_risk_class, payment_consistency, years_active, is_payment_blocked,
dunning_level, reversal_rate, discount_capture_rate, voided_payment_rate,
stale_payment_rate, shap_values, feature_deviations
```

---

## 8. UI & Theming

### CSS Variables (defined in `setup_analytics_styles()` and `inject_landing_styles()`)

| Variable | Light value | Dark value |
|---|---|---|
| `--bg-app` | `#f8fafc` | `#050810` |
| `--bg-card` | `#ffffff` | `#0a0e1a` |
| `--text-main` | `#334155` | `#c8d8f0` |
| `--text-head` | `#0f172a` | `#f0f4ff` |
| `--text-mute` | `#64748b` | `#94a3b8` |
| `--border-light` | `rgba(59,130,246,0.2)` | `rgba(255,255,255,0.06)` |
| `--border-med` | `rgba(59,130,246,0.5)` | `rgba(255,255,255,0.15)` |
| `--bg-hover` | `rgba(0,0,0,0.03)` | `rgba(255,255,255,0.03)` |
| `--shadow-str` | `rgba(0,0,0,0.1)` | `rgba(0,0,0,0.5)` |

### Theming Rules

- All HTML strings inside `st.markdown()` / `st.html()` must use CSS variables, never hardcoded hex colors
- All Plotly charts use `st.session_state['plot_bg']`, `st.session_state['paper_bg']`, and `st.session_state['text_main']` — never hardcoded
- Accent colors (`#14f0a0` green, `#50a0ff` blue, `#f0b840` amber, `#f85149` red) are semantic and fixed — do not theme-vary these
- Theme toggle appears on both landing page (`key="landing_theme"`) and analytics page — both rerun on change
- Default theme: `light`

### CSS Classes (defined in `setup_analytics_styles()` — do not add new ones without updating this list)

| Class | Purpose |
|---|---|
| `.analytic-card` | Metric and content cards throughout analytics page |
| `.card-label` | Label inside `.analytic-card` |
| `.card-value` | Large value number inside `.analytic-card` |
| `.section-header` | Monospace uppercase section title with green left border |
| `.section-sep` | Thin horizontal rule between sections |
| `.aging-card` | Invoice aging display card |
| `.decision-badge` | Approve/Review/High Risk badge with glow |

---

## 9. AI Chatbot

### LLM Provider

OpenRouter API (`https://openrouter.ai/api/v1/chat/completions`) with a three-model fallback chain:
1. `meta-llama/llama-3.3-70b-instruct:free` (primary)
2. `nvidia/nemotron-3-super-120b-a12b:free` (fallback)
3. `google/gemma-4-31b-it:free` (fallback)

API key sourced from `st.secrets["OPENROUTER_API_KEY"]` or environment variable. Streaming enabled.

### Context Passed to LLM (`build_chatbot_context()`)

The system prompt includes all of the following — **when new data fields are added to `result` or `xgb_prediction`, they must also be added here:**

- Selected vendor name, ID, product, quoted price
- Final risk score, vendor risk, price risk, decision, vendor bucket
- XGBoost class label + SAP risk class
- Isolation Forest score + outlier flag
- K-Means cluster name + ID
- All 14 feature values with population context (years active, dunning, blocked flag, reversal rate, discount capture, voided/stale rates, payment consistency)
- Full SHAP values dict (all 14 features)
- Full feature deviation table (all 14 features, z-scores + level)
- Price analysis: forecast, historical avg, quoted price, variance, trend, percentiles (P25/P50/P75/P90)
- Invoice clearance days
- Comparison table of all vendors with rank, final risk, vendor risk, decision, isolation score

**Fields to add to chatbot context when label-change is implemented:**
- `sap_risk_label` (human-readable SAP class)
- `sap_divergence` (True/False with explanation)

### Chatbot Quick-Action Chips

Four fixed chips rendered below the chat input:
1. "Why is this vendor high risk?"
2. "Compare with best alternative"
3. "Should I approve this purchase?"
4. "Explain the SHAP values"

These set `st.session_state['chat_prefill']` and trigger a rerun.

---

## 10. Pending Changes (Approved, Not Yet Implemented)

These changes have been discussed and approved. Implement them one at a time, in order, confirming with the user before starting each.

### Task A — Composite Behavioral Label (accuracy fix)
**Priority: High**
- Replace SAP `RISK_CLASS` as training target with composite behavioral score in `preprocess_sap.py`
- Keep `RISK_CLASS` as a reference column throughout
- Update `predict_vendor_risk()` class label strings
- Add `sap_risk_label` and `sap_divergence` to model return dicts
- Add divergence insight to `insights` list
- Update chatbot context to include new fields
- Update Risk Classification card in UI (show both labels + divergence badge)
- Update Vendor Comparison cards (add SAP class badge)
- Retrain all models after implementation
- Full light/dark theme compatibility required

### Task B — Selected Vendor Auto-Price from History
**Priority: Medium**
- Add `get_vendor_avg_price(product_name: str) -> float` to `model.py`
- Pre-fill the price `number_input` in `render_landing_page()` with `value=get_vendor_avg_price(selected_product)`
- Input must remain editable (analyst can override with real quote)
- This makes the selected vendor's price baseline consistent with comparison vendors

---

## 11. Change Rules

1. **One task at a time.** Confirm with the user before starting each task in Section 10.
2. **No breaking changes.** If a return dict key is renamed, update every consumer in `app.py`, `model.py`, and `build_chatbot_context()`.
3. **No UI structure changes** unless explicitly requested. Do not reorder sections, rename tabs, or remove cards.
4. **All new HTML must use CSS variables.** No hardcoded hex colors in `st.markdown()` or `st.html()`.
5. **All new Plotly figures must use session state** for `plot_bgcolor`, `paper_bgcolor`, and font color.
6. **Chatbot context is always updated** when new data fields are added to any return dict.
7. **Retrain trigger:** Any change to the 14 features list requires re-running `preprocess_sap.py` and restarting the Streamlit app.
8. **Do not edit files in `files/`.** Raw SAP CSVs are read-only.
9. **feature_columns.json is the single source of truth** for feature order. Never hardcode the feature list in `model.py`.

---

## 12. Run Instructions

### First-Time Setup

```bash
pip install streamlit xgboost scikit-learn shap pandas numpy scipy joblib \
            plotly streamlit-lottie matplotlib
```

### Train Models (required once, and after any feature/label change)

```bash
python preprocess_sap.py
```

This writes: `xgb_risk_model.pkl`, `kmeans_risk_model.pkl`, `scaler.pkl`, `isolation_forest_model.pkl`, `feature_columns.json`, `processed_sap_vendors.csv`, `model_metrics.json`, `kmeans_elbow_plot.png`

### Run the App

```bash
streamlit run app.py
```

### Configure OpenRouter API Key

In `.streamlit/secrets.toml`:
```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
```

Or as an environment variable:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

---

*Last updated: May 2026 — reflects current codebase state including 14-feature expansion, Isolation Forest integration, and light/dark theme system. Pending: composite label retraining (Task A) and selected vendor auto-price (Task B).*
