# Dashboard Visualization Update — Procurement Risk Analyzer
### Prepared for: Antigravity Development Team

---

## Overview

Three UI changes to the Risk Analysis tab in `app.py`:

1. **Replace** the "Risk Contribution Breakdown" donut chart with a **Payment Discipline Radar Chart**
2. **Add** a **Vendor Health Scorecard** row (4 new cards) as a new section
3. **Add** `years_active` as a 5th card to the existing **Core Risk Metrics** row at the top

All data for these visualizations already exists in the result dict returned by
`procurement_risk_model()` after the feature expansion update. No changes to
`model.py` or `preprocess_sap.py` needed.

**Only `app.py` is modified in this update.**

---

## Change 1 — Replace Donut Chart with Payment Discipline Radar Chart

### Why
The "Risk Contribution Breakdown" donut shows 60% Vendor / 40% Price — these
weights never change regardless of vendor, so the chart conveys zero information.
The radar chart replaces it with 5 vendor-specific payment behaviour dimensions
that actually differ between vendors.

### Where in `app.py`
Inside `render_risk_analysis_tab()`, in the **Risk Visualizations** section.
Currently this section uses `st.columns(2)` — left column has the gauge, right
column has the donut. **Replace only the right column content.** Do not touch
the left column gauge or the section header.

### The 5 Radar Axes and Their Data Sources

| Axis Label | Data Field | Source | Direction |
|---|---|---|---|
| Late Payment Ratio | `result['xgb_prediction']['late_ratio']` | model | Higher = worse |
| Payment Consistency | `result['xgb_prediction']['payment_consistency']` | model | Higher = worse (normalize 0–1, cap at 30 days = 1.0) |
| Discount Capture | `result['xgb_prediction']['discount_capture_rate']` | model | **Invert** — lower capture = higher risk (use `1 - value`) |
| Voided Payments | `result['xgb_prediction']['voided_payment_rate']` | model | Higher = worse |
| Reversal Rate | `result['xgb_prediction']['reversal_rate']` | model | Higher = worse |

All values must be normalized to **0–1** before plotting. The radar shows how
bad each dimension is — 1.0 = worst possible, 0.0 = best possible.

### Plotly Radar Chart Spec

```python
import plotly.graph_objects as go

# Pull values from result dict
xgb = result['xgb_prediction']

axes = ['Late Payment\nRatio', 'Payment\nConsistency', 'Discount\nCapture Risk',
        'Voided\nPayments', 'Reversal\nRate']

values = [
    float(xgb.get('late_ratio', 0)),
    min(float(xgb.get('payment_consistency', 0)) / 30.0, 1.0),  # normalize: 30 days = max
    1.0 - float(xgb.get('discount_capture_rate', 0.5)),          # invert: low capture = high risk
    float(xgb.get('voided_payment_rate', 0)),
    float(xgb.get('reversal_rate', 0)),
]
values_closed = values + [values[0]]   # close the polygon
axes_closed   = axes  + [axes[0]]

# Color based on vendor_bucket
bucket = result.get('vendor_bucket', 'MEDIUM')
radar_color = {
    'LOW':      '#14f0a0',
    'MEDIUM':   '#f0b840',
    'HIGH':     '#f85149',
    'CRITICAL': '#ff0000',
}.get(bucket, '#f0b840')

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=values_closed,
    theta=axes_closed,
    fill='toself',
    fillcolor=radar_color.replace(')', ', 0.15)').replace('rgb', 'rgba'),
    line=dict(color=radar_color, width=2),
    name='Payment Discipline',
))
fig_radar.update_layout(
    polar=dict(
        bgcolor='rgba(10,14,26,0.0)',
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickfont=dict(size=9, color='#94a3b8'),
            gridcolor='rgba(255,255,255,0.08)',
            linecolor='rgba(255,255,255,0.08)',
        ),
        angularaxis=dict(
            tickfont=dict(size=10, color='#c8d8f0'),
            gridcolor='rgba(255,255,255,0.08)',
            linecolor='rgba(255,255,255,0.12)',
        ),
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=40, r=40, t=40, b=40),
    height=320,
    showlegend=False,
    title=dict(
        text='Payment Discipline Profile',
        font=dict(size=12, color='#94a3b8',
                  family="'Roboto Mono', monospace"),
        x=0.5,
    ),
)
st.plotly_chart(fig_radar, use_container_width=True)
```

Add the `// CHART INTELLIGENCE SUMMARY` block below the radar chart (same style
as existing summaries in the tab):

```python
st.html(f"""
<div style="border-left:3px solid {radar_color}; padding:14px 18px;
            margin-top:12px; border-radius:0 8px 8px 0;
            background:rgba(255,255,255,0.02);">
    <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem;
                color:#8a9ab8; letter-spacing:0.1em; margin-bottom:6px;">
        // CHART INTELLIGENCE SUMMARY
    </div>
    <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
                color:#c8d8f0; line-height:1.5;">
        Each axis shows a distinct dimension of payment behaviour (0 = best,
        1 = worst). A wide polygon indicates systemic payment problems across
        multiple dimensions. A narrow polygon concentrated on one axis suggests
        a single specific issue rather than general unreliability.
    </div>
</div>
""")
```

---

## Change 2 — Vendor Health Scorecard (New Section)

### Where in `app.py`
Add this as a **new section** inside `render_risk_analysis_tab()`, placed
**between the existing "Invoice Aging" section and the "Risk Visualizations"
section**. Use the existing `.section-sep` and `.section-header` classes.

### The 4 Cards

| Card | Data Field | Color Logic |
|---|---|---|
| Years Active | `result['xgb_prediction']['years_active']` | Green ≥5yr, Amber 2–5yr, Red <2yr |
| Dunning Level | `result['xgb_prediction']['dunning_level']` | Green=0, Amber=1–2, Red=3–4 |
| SAP Block Status | `result['xgb_prediction']['is_payment_blocked']` | Green=CLEAR, Red=BLOCKED |
| Reversal Rate | `result['xgb_prediction']['reversal_rate']` | Green<5%, Amber 5–15%, Red>15% |

### Implementation

```python
st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Vendor Health Signals</div>',
            unsafe_allow_html=True)

xgb_pred = result.get('xgb_prediction', {})

years_active   = float(xgb_pred.get('years_active', 0))
dunning_level  = int(xgb_pred.get('dunning_level', 0))
is_blocked     = int(xgb_pred.get('is_payment_blocked', 0))
reversal_rate  = float(xgb_pred.get('reversal_rate', 0))

# Color logic
years_color   = '#14f0a0' if years_active >= 5 else '#f0b840' if years_active >= 2 else '#f85149'
dunning_color = '#14f0a0' if dunning_level == 0 else '#f0b840' if dunning_level <= 2 else '#f85149'
block_color   = '#f85149' if is_blocked else '#14f0a0'
reversal_color= '#14f0a0' if reversal_rate < 0.05 else '#f0b840' if reversal_rate < 0.15 else '#f85149'

block_label   = 'BLOCKED' if is_blocked else 'CLEAR'
dunning_label = f'{dunning_level} / 4'

col_h1, col_h2, col_h3, col_h4 = st.columns(4, gap="small")

for col, label, value, color, sublabel in [
    (col_h1, 'Years Active',    f'{years_active:.1f} yrs', years_color,
     'NEW VENDOR' if years_active < 2 else 'ESTABLISHED' if years_active >= 5 else 'GROWING'),
    (col_h2, 'Dunning Level',   dunning_label,              dunning_color,
     'NO NOTICES' if dunning_level == 0 else 'ESCALATED' if dunning_level >= 3 else 'WARNED'),
    (col_h3, 'SAP Block Status',block_label,                block_color,
     'PAYMENT BLOCKED' if is_blocked else 'NO BLOCKS'),
    (col_h4, 'Reversal Rate',   f'{reversal_rate:.1%}',     reversal_color,
     'HIGH REVERSALS' if reversal_rate > 0.15 else 'ACCEPTABLE' if reversal_rate > 0.05 else 'CLEAN'),
]:
    with col:
        st.markdown(f"""
        <div class="analytic-card">
            <div class="card-label">{label}</div>
            <div class="card-value" style="color:{color};">{value}</div>
            <div style="margin-top:10px; font-family:'Roboto Mono',monospace;
                        font-size:0.65rem; letter-spacing:0.1em; color:{color};">
                {sublabel}
            </div>
        </div>
        """, unsafe_allow_html=True)
```

---

## Change 3 — Add Years Active to Core Risk Metrics Row

### Where in `app.py`
The existing **Core Risk Metrics** section at the top of `render_risk_analysis_tab()`
currently renders 4 cards in `st.columns(4)`. Change this to `st.columns(5)` and
add a 5th card for Years Active.

### What to Change

Find the existing `st.columns(4, gap="small")` call in the Core Risk Metrics section
and change it to:
```python
col1, col2, col3, col4, col5 = st.columns(5, gap="small")
```

Then add a 5th card after the existing 4, inside `col5`:
```python
years_active = float(result.get('xgb_prediction', {}).get('years_active', 0))
years_color  = '#14f0a0' if years_active >= 5 else '#f0b840' if years_active >= 2 else '#f85149'
years_tag    = 'ESTABLISHED' if years_active >= 5 else 'GROWING' if years_active >= 2 else 'NEW VENDOR'

with col5:
    st.markdown(f"""
    <div class="analytic-card">
        <div class="card-label">Vendor<br>Tenure</div>
        <div class="card-value" style="color:{years_color};">{years_active:.1f}<span style="font-size:1rem;"> yrs</span></div>
        <div style="margin-top:10px; font-family:'Roboto Mono',monospace;
                    font-size:0.65rem; letter-spacing:0.1em; color:{years_color};">
            {years_tag}
        </div>
    </div>
    """, unsafe_allow_html=True)
```

---

## Tasks — Do One at a Time, Wait for Client Approval Before Next

### Task 1 — Add Years Active card to Core Risk Metrics row (Change 3)

This is the smallest change. Find the Core Risk Metrics `st.columns(4)` call,
expand to `st.columns(5)`, add the 5th card as specified above.

Verify: top card row shows 5 cards, last one shows vendor tenure in years with
correct color (green ≥5, amber 2–5, red <2).

**Stop and wait for client approval.**

---

### Task 2 — Add Vendor Health Scorecard section (Change 2)

Add the new section between Invoice Aging and Risk Visualizations as specified.
Use `st.columns(4)` with the `.analytic-card` class. No new CSS.

Verify: 4 new cards appear showing Years Active, Dunning Level, SAP Block Status,
Reversal Rate. Colors match the thresholds above. Section header reads
"Vendor Health Signals".

Note: Years Active will now appear in two places (top row + health scorecard).
This is intentional — the top row gives a quick number, the health scorecard
gives the labeled status badge. Do not remove either.

**Stop and wait for client approval.**

---

### Task 3 — Replace Donut Chart with Radar Chart (Change 1)

In the Risk Visualizations section, locate the right column (currently rendering
the donut chart). Delete the donut chart code entirely and replace with the
Plotly radar chart code specified above, followed by the intelligence summary div.

Do not touch the left column gauge at all.

Verify: left side still shows the Final Risk Score gauge unchanged. Right side
now shows the Payment Discipline radar with 5 axes. Chart color matches the
vendor's risk bucket color. Intelligence summary appears below the chart.

**Stop and wait for client approval.**

---

## Do Not Touch

- The left column gauge in Risk Visualizations
- Any other section in `render_risk_analysis_tab()`
- The Vendor Comparison tab
- The AI Risk Analyst chatbot tab
- `setup_analytics_styles()` — no new CSS classes needed; all styling is inline
- `model.py` and `preprocess_sap.py`

---

## Data Availability Note

All fields used in these visualizations (`years_active`, `dunning_level`,
`is_payment_blocked`, `reversal_rate`, `payment_consistency`, `discount_capture_rate`,
`voided_payment_rate`) are returned by `predict_vendor_risk()` inside the
`xgb_prediction` sub-dict. They are only available **after** the feature expansion
update (`preprocess_sap.py` and `model.py`) has been applied and `preprocess_sap.py`
has been re-run to retrain the models.

**This update must be applied after the Feature Expansion update, not before.**

---

*End of specification.*
