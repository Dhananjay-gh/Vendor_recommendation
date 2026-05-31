# Feature Expansion Update — Procurement Risk Analyzer
### Prepared for: Antigravity Development Team

---

## What This Update Is

The ML model has been expanded from **4 features to 14 features** using additional
SAP tables that were already available in the project's `files/` folder but previously
unused. This significantly improves XGBoost classification accuracy (baseline: 35.1%).

No new external data sources are needed. Everything comes from the existing SAP export.

---

## New SAP Tables Being Used

All 4 new CSV files are already present in the **`files/` folder inside the project
directory**. No uploading or downloading required.

| File | Was It Used Before? | What It Adds |
|---|---|---|
| `LFA1_Vendor_Master_General.csv` | ✅ Yes | Unchanged |
| `BSAK_Cleared_Items.csv` | ✅ Yes | Unchanged + new consistency score |
| `BSIK_Open_Items.csv` | ✅ Yes | Unchanged |
| `LFB1_Vendor_Master_CompCode.csv` | ❌ No — now used | Payment terms, block flags, dunning level, vendor age |
| `BKPF_Document_Header.csv` | ❌ No — now used | Document reversal rate |
| `BSEG_Document_Segment.csv` | ❌ No — now used | Discount capture rate, payment terms days |
| `PAYR_Payment_Medium.csv` | ❌ No — now used | Voided and stale payment rates |

---

## New Features Added (10 new, total 14)

| # | Feature | Source Table | What It Measures |
|---|---|---|---|
| 1 | `avg_days_overdue_hist` | BSAK | ✅ Existing |
| 2 | `late_ratio` | BSAK | ✅ Existing |
| 3 | `total_spend_vol` | BSAK | ✅ Existing |
| 4 | `open_exposure` | BSIK | ✅ Existing |
| 5 | `payment_consistency_score` | BSAK | **NEW** — std dev of overdue days; erratic vendors = higher risk |
| 6 | `years_active` | LFB1 `ERDAT` | **NEW** — newer vendors = less track record = higher risk |
| 7 | `payment_terms_risk` | LFB1 `ZTERM` | **NEW** — numeric risk score of agreed payment terms |
| 8 | `is_payment_blocked` | LFB1 `ZAHLS/SPERR` | **NEW** — vendor currently blocked in SAP (0/1) |
| 9 | `dunning_level` | LFB1 `DUNNLEVEL` | **NEW** — dunning notices sent (0–4); higher = more overdue chasing |
| 10 | `reversal_rate` | BKPF `STBLG` via BSEG | **NEW** — fraction of documents reversed/cancelled |
| 11 | `discount_capture_rate` | BSEG `ZBD1P/AUGDT/FAEDT` | **NEW** — does vendor take early payment discounts offered? |
| 12 | `avg_payment_terms_days` | BSEG `ZBD1T` | **NEW** — average net days agreed per invoice |
| 13 | `voided_payment_rate` | PAYR `XVOIDED` | **NEW** — fraction of payments voided |
| 14 | `stale_payment_rate` | PAYR `STALE` | **NEW** — fraction of payments gone stale |

---

## Files to Replace

Two files need to be replaced with the new versions provided:

| File | What Changed |
|---|---|
| `preprocess_sap.py` | Loads 4 new SAP tables, engineers 10 new features, trains all models on 14 features, saves `feature_columns.json` |
| `model.py` | Loads `feature_columns.json` dynamically, builds 14-feature vectors, returns all new fields in prediction dicts, includes new insight messages for blocked vendors / high dunning / high reversal rate / new vendors |

**Do not modify `app.py`** — the dashboard and chatbot both consume data from
`model.py`'s return dicts, which now automatically include all 14 features.
The chatbot context builder (`build_chatbot_context()`) will pick up the new
fields with zero changes needed.

---

## New File Generated at Training Time

`preprocess_sap.py` now saves one additional file:

```
feature_columns.json
```

This file contains the ordered list of feature column names used during training.
`model.py` reads this file at startup to build feature vectors in the exact same
order the scaler and model expect. **This file must be committed to the repo**
alongside the `.pkl` files — without it, `model.py` falls back to the original
4 features only.

---

## Tasks — Do One at a Time, Wait for Client Approval Before Next

---

### Task 1 — Replace `preprocess_sap.py`

Replace the existing `preprocess_sap.py` in the project root with the new version
provided. Do not modify any other file yet.

Verify the new file loads these columns:
- From `LFB1`: `LIFNR, ZTERM, ZAHLS, SPERR, DUNNLEVEL, ERDAT`
- From `BKPF`: `BELNR, BUKRS, GJAHR, STBLG, BSTAT, BLART`
- From `BSEG`: `LIFNR, BELNR, BUKRS, GJAHR, ZTERM, ZBD1T, ZBD1P, AUGDT, FAEDT, WRBTR, KOART`
- From `PAYR`: `LIFNR, BELNR, BUDAT, RWBTR, WAERS, XVOIDED, STALE`

**Stop and wait for client approval.**

---

### Task 2 — Run `preprocess_sap.py`

From the project root directory, run:
```bash
python preprocess_sap.py
```

This will:
1. Load all 7 SAP CSV files from the `files/` folder
2. Engineer all 14 features
3. Retrain XGBoost, K-Means, and Isolation Forest
4. Save updated `.pkl` files + new `feature_columns.json`
5. Print the new test accuracy in the terminal

The terminal output will end with something like:
```
Final accuracy: 54.3%  (baseline was 35.1%)
```

Share the full terminal output with the client before proceeding.

**Stop and wait for client approval.**

---

### Task 3 — Replace `model.py`

Replace the existing `model.py` with the new version provided.

Key things to verify after replacement:
- `FEATURE_COLS_PATH` points to `feature_columns.json` in the project root
- `FEATURE_LABELS` dict contains all 14 feature human-readable names
- `predict_vendor_risk()` builds the feature vector using `FEATURE_COLUMNS` list dynamically
- The return dict of `predict_vendor_risk()` includes these new keys:
  `payment_consistency`, `years_active`, `is_payment_blocked`, `dunning_level`,
  `reversal_rate`, `discount_capture_rate`, `voided_payment_rate`, `stale_payment_rate`
- `procurement_risk_model()` includes new insight messages for: payment block,
  high dunning level, high reversal rate, and new vendors

**Stop and wait for client approval.**

---

### Task 4 — Restart Streamlit and Verify

Restart the Streamlit app. On startup, `model.py` will log:
```
Using existing purchase data → purchase_data.csv
```
This confirms the new `.pkl` files and `feature_columns.json` were loaded.

Verify the following in the live app:
1. Select any vendor — the Risk Analysis tab should load without errors
2. The SHAP chart should now show up to 14 bars (one per feature)
3. The Feature Deviation section should show all 14 features with z-scores
4. The Executive Summary should include new insights where relevant
   (e.g. SAP Block Alert, Dunning Warning, Document Quality)
5. The AI Risk Analyst chatbot — ask "why is this vendor high risk?" and confirm
   the response references the new features (dunning level, years active, etc.)

**Stop and wait for client approval.**

---

## Do Not Touch

- `app.py` — no changes needed anywhere
- Any existing CSS classes or styling
- `purchase_data.csv` — price forecasting data, unrelated to this update
- `.streamlit/secrets.toml` — API key, unrelated to this update
- Any `.pkl` files manually — they are regenerated by running `preprocess_sap.py`

---

## Expected Accuracy Improvement

| Scenario | Expected Accuracy |
|---|---|
| Original (4 features) | 35.1% |
| After this update (14 features) | 50–63% |

Actual result depends on data quality in the new SAP tables. The terminal output
from Task 2 will show the exact number.

---

*End of specification.*
