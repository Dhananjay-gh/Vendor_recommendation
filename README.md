# AI Vendor Risk Engine

An enterprise-grade procurement intelligence platform that uses machine learning to assess vendor risk and optimize purchasing decisions. Built with real SAP ERP data, the system combines multiple AI models to deliver actionable risk insights for procurement teams.

---

## Overview

The platform analyzes vendor reliability and market pricing dynamics to generate a composite risk score for any procurement transaction. It answers two critical questions:

1. **Is this vendor trustworthy?** — Using historical payment behavior from SAP ERP records  
2. **Is the quoted price fair?** — Using AI-forecasted market benchmarks with inflation adjustment

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                    │
│                       (app.py)                           │
│  ┌─────────────────┐  ┌──────────────────────────────┐   │
│  │  Landing Page    │  │     Analytics Dashboard      │   │
│  │  - Vendor Select │  │  - Risk Gauges & Charts      │   │
│  │  - Product Select│  │  - SHAP Explainability       │   │
│  │  - Price Input   │  │  - Vendor Comparison (10)    │   │
│  └────────┬────────┘  │  - Anomaly Detection         │   │
│           │           │  - Behavior Analysis          │   │
│           ▼           └──────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              ML Inference Engine (model.py)          │ │
│  │  ┌──────────┐ ┌────────┐ ┌───────┐ ┌────────────┐  │ │
│  │  │ XGBoost  │ │K-Means │ │ SHAP  │ │ Isolation  │  │ │
│  │  │Classifier│ │Cluster │ │Explain│ │  Forest    │  │ │
│  │  └──────────┘ └────────┘ └───────┘ └────────────┘  │ │
│  │  ┌──────────────────┐ ┌─────────────────────────┐  │ │
│  │  │ Linear Regression│ │ Percentile Price Risk   │  │ │
│  │  │ Price Forecasting│ │   (P25/P50/P75/P90)     │  │ │
│  │  └──────────────────┘ └─────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌─────────────────────┐
│  SAP Vendor Data │         │  Purchase Price Data │
│ (6,000+ vendors) │         │  (Historical Market) │
└──────────────────┘         └─────────────────────┘
```

---

## ML Models

| Model | Purpose | Input Features |
|---|---|---|
| **XGBoost Classifier** | Predicts vendor risk class (A/B/C/D) from payment behavior | `avg_days_overdue`, `late_ratio`, `total_spend_vol`, `open_exposure` |
| **K-Means Clustering** | Segments vendors into behavioral groups | Same 4 features (StandardScaler normalized) |
| **Isolation Forest** | Detects statistical outliers in the vendor population | Same 4 features |
| **SHAP TreeExplainer** | Explains which features drove each XGBoost prediction | XGBoost model internals |
| **Linear Regression** | Forecasts expected fair price for today using historical trends | `date_ordinal` vs `price_per_unit` |
| **Percentile Engine** | Calculates P25/P50/P75/P90 price bands per product | Historical price distribution |

---

## Risk Scoring

The final procurement risk score combines two independent assessments:

```
Final Risk = 0.6 × Vendor Risk + 0.4 × Price Risk
```

- **Vendor Risk (60%):** XGBoost-predicted reliability score (0–1) based on SAP payment history  
- **Price Risk (40%):** Percentile-based position of the quoted price against historical market data

| Final Risk | Decision | Action |
|---|---|---|
| < 0.40 | APPROVE | Automated approval — pricing is optimal |
| 0.40 – 0.65 | REVIEW | Specialist review required |
| ≥ 0.65 | HIGH RISK | Avoid contract — exceeds risk tolerance |

---

## Project Structure

```
├── app.py                      # Streamlit UI — landing page + analytics dashboard
├── model.py                    # ML inference engine — predictions, forecasting, explainability
├── preprocess_sap.py           # Training pipeline — XGBoost, K-Means, Isolation Forest, Scaler
├── requirements.txt            # Python dependencies
│
├── files/                      # Raw SAP ERP data (not tracked in git)
│   ├── LFA1_Vendor_Master_General.csv
│   ├── BSAK_Cleared_Items.csv
│   └── BSIK_Open_Items.csv
│
├── processed_sap_vendors.csv   # Preprocessed vendor features (output of preprocess_sap.py)
├── purchase_data.csv           # Historical purchase price data (seed-generated or real)
├── model_metrics.json          # Held-out test accuracy from training
├── kmeans_elbow_plot.png       # K-Means cluster validation plot
│
├── xgb_risk_model.pkl          # Trained XGBoost classifier
├── kmeans_risk_model.pkl       # Trained K-Means segmentation model
├── isolation_forest_model.pkl  # Trained Isolation Forest outlier detector
└── scaler.pkl                  # StandardScaler fitted on training features
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Raw SAP data files in `./files/` directory (see Project Structure)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Models (first time only)

```bash
python preprocess_sap.py
```

This will:
- Load and aggregate SAP tables (LFA1, BSAK, BSIK)
- Train XGBoost, K-Means, Isolation Forest models
- Save `.pkl` model files, `scaler.pkl`, and `processed_sap_vendors.csv`
- Generate `model_metrics.json` with held-out test accuracy
- Produce `kmeans_elbow_plot.png` for cluster validation

### Launch the Dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Dashboard Features

### Landing Page
- Real-time model accuracy loaded from `model_metrics.json`
- Dynamic vendor count and total spend calculated from SAP data
- SAP vendor dropdown with 6,000+ real vendors
- Product selection and quoted price input

### Analytics Dashboard — Risk Analysis Tab
- **Core Risk Metrics:** Vendor reliability score, risk tier, price position (percentile band), AI forecasted price
- **Market Trend Alert:** Inflation/deflation detection with percentage change
- **Price Distribution Context:** Visual percentile bar showing where the quoted price sits
- **Risk Gauges:** Final risk score gauge + risk contribution donut chart (60/40 split)
- **Deep Analytics:** Vendor reliability and market price variance bullet gauges
- **Executive Summary:** AI-generated insights covering XGBoost classification, price analysis, and recommendations
- **SHAP Feature Impact:** Horizontal bar chart showing which features drove the prediction up or down
- **Isolation Forest Analysis:** Population outlier detection with anomaly score gauge
- **Vendor Behavior Analyzer:** 12-month historical risk timeline with trend analysis
- **Anomaly Detection:** Transaction volume scatter plot with anomaly threshold detection

### Analytics Dashboard — Vendor Comparison Tab
- Smart alert banner (recommends alternatives if selected vendor is high risk)
- 10 ranked vendor cards with full risk breakdown
- Two-stage ranking: quick pre-score funnel → full ML scoring on top candidates

---

## Data Pipeline

```
SAP ERP Tables (LFA1, BSAK, BSIK)
        │
        ▼
  preprocess_sap.py
        │
        ├──→ processed_sap_vendors.csv  (6,000+ vendors with engineered features)
        ├──→ xgb_risk_model.pkl         (XGBoost classifier)
        ├──→ kmeans_risk_model.pkl       (K-Means segmentation)
        ├──→ isolation_forest_model.pkl  (Outlier detector)
        ├──→ scaler.pkl                  (StandardScaler)
        └──→ model_metrics.json          (Test accuracy)
```

---

## Key Design Decisions

- **Percentile-based price risk** instead of hardcoded thresholds — adapts to actual price distribution per product
- **One-time seed generation** for purchase data — `purchase_data.csv` is generated only if it doesn't exist, making it safe to replace with real procurement records
- **Two-stage vendor comparison** — pre-scores all vendors cheaply using raw features, then runs full ML pipeline only on top 9 candidates for efficiency
- **SHAP explainability** — provides transparency into XGBoost predictions, critical for enterprise procurement decisions
- **StandardScaler** on K-Means input — prevents high-magnitude features (spend volume) from dominating cluster assignments

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, Custom CSS |
| ML Models | XGBoost, Scikit-learn (K-Means, Isolation Forest, StandardScaler) |
| Explainability | SHAP (TreeExplainer) |
| Forecasting | SciPy (Linear Regression) |
| Data | Pandas, NumPy |
| Data Source | SAP ERP (LFA1, BSAK, BSIK tables) |
