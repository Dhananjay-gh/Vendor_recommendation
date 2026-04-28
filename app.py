import streamlit as st
import plotly.graph_objects as go
import json
import requests
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from model import procurement_risk_model, get_sap_vendor_list, get_vendor_history

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALYTICS PAGE — PREMIUM STYLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def setup_analytics_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Roboto+Mono:wght@400;500&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="collapsedControl"] { display: none; }

    .block-container {
        padding-top: 1rem !important;
        max-width: 100% !important;
    }

    /* ── Full page background with grid texture ── */
    .stApp {
        background: #050810;
        background-image:
            linear-gradient(rgba(20, 240, 200, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(20, 240, 200, 0.025) 1px, transparent 1px);
        background-size: 48px 48px;
        min-height: 100vh;
        color: #c8d8f0;
    }

    /* ── Typography ── */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: #f0f4ff !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* ── Section header (monospace uppercase with left-border accent) ── */
    .section-header {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #94a3b8;
        border-left: 3px solid #14f0a0;
        padding-left: 14px;
        margin-bottom: 24px;
        margin-top: 8px;
        animation: fadeSlideUp 0.6s ease-out both;
    }

    /* ── Section separator (replaces hr / ---) ── */
    .section-sep {
        border: none;
        border-top: 1px solid rgba(20, 240, 160, 0.08);
        margin: 48px 0;
    }

    /* ── Animations ── */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 20px var(--glow-color, rgba(20,240,160,0.15)); }
        50%      { box-shadow: 0 0 35px var(--glow-color, rgba(20,240,160,0.25)); }
    }

    /* ── Metric card (analytics) ── */
    .analytic-card {
        background: rgba(10, 14, 26, 0.85);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 28px 24px;
        position: relative;
        overflow: hidden;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        animation: fadeSlideUp 0.7s ease-out both;
    }
    .analytic-card:nth-child(1) { animation-delay: 0.05s; }
    .analytic-card:nth-child(2) { animation-delay: 0.15s; }
    .analytic-card:nth-child(3) { animation-delay: 0.25s; }
    .analytic-card:nth-child(4) { animation-delay: 0.35s; }
    .analytic-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(20, 240, 160, 0.25), transparent);
    }
    .analytic-card:hover {
        transform: translateY(-4px);
        border-color: rgba(20, 240, 160, 0.25);
        box-shadow: 0 12px 32px rgba(0,0,0,0.5);
    }
    .analytic-card .card-label {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 16px;
        line-height: 1.5;
    }
    .analytic-card .card-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1;
        margin: 0;
    }

    /* ── Decision badge with glow ── */
    .decision-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 12px 20px;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border: 1px solid;
        height: 100%;
        min-height: 52px;
        animation: fadeSlideUp 0.5s ease-out both, glowPulse 3s ease-in-out infinite;
    }

    /* ── Ghost back button ── */
    .ghost-back .stButton > button {
        background: #0a0e1a !important;
        background-color: #0a0e1a !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #ffffff !important;
        box-shadow: none !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.05em !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .ghost-back .stButton > button:hover {
        border-color: rgba(255, 255, 255, 0.5) !important;
        color: #ffffff !important;
        transform: none !important;
        background: #0a0e1a !important;
        background-color: #0a0e1a !important;
        box-shadow: inset 0 0 8px rgba(255,255,255,0.15), 0 0 8px rgba(255,255,255,0.15) !important;
    }

    /* ── Invoice aging card ── */
    .aging-card {
        background: rgba(10, 14, 26, 0.85);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 28px 32px;
        border-radius: 14px;
        position: relative;
        overflow: hidden;
        animation: fadeSlideUp 0.7s ease-out both;
    }
    .aging-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 3px;
    }

    /* ── Insight alert overrides ── */
    .stAlert > div {
        background: rgba(10, 14, 26, 0.85) !important;
        color: #c8d8f0 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-left: 3px solid #50a0ff !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        animation: fadeSlideUp 0.6s ease-out both !important;
    }

    /* ── Chart container entrance animation ── */
    [data-testid="stPlotlyChart"] {
        animation: fadeSlideUp 0.8s ease-out both;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94a3b8;
        padding: 12px 24px;
        border-bottom: 2px solid transparent;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #14f0a0 !important;
        border-bottom-color: #14f0a0 !important;
        background: transparent !important;
    }

    /* hide default hr */
    hr { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LANDING PAGE — PREMIUM STYLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def inject_landing_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Roboto+Mono:wght@400;500&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ── Full page background ── */
    .stApp {
        background: #050810;
        background-image:
            linear-gradient(rgba(20, 240, 200, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(20, 240, 200, 0.03) 1px, transparent 1px);
        background-size: 48px 48px;
        min-height: 100vh;
    }

    /* ── Status pill ── */
    .status-bar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 10px;
        padding: 18px 32px 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 14px;
        border-radius: 20px;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        border: 1px solid;
    }
    .pill-green {
        background: rgba(20, 240, 160, 0.06);
        border-color: rgba(20, 240, 160, 0.25);
        color: #14f0a0;
    }
    .pill-blue {
        background: rgba(80, 160, 255, 0.06);
        border-color: rgba(80, 160, 255, 0.25);
        color: #50a0ff;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    .dot-green { background: #14f0a0; box-shadow: 0 0 6px #14f0a0; }
    .dot-blue  { background: #50a0ff; box-shadow: 0 0 6px #50a0ff; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.5; }
    }

    /* ── Hero section ── */
    .hero-wrap {
        text-align: center;
        padding: 48px 20px 16px;
    }
    .hero-eyebrow {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.22em;
        color: #14f0a0;
        text-transform: uppercase;
        margin-bottom: 16px;
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4rem);
        font-weight: 800;
        color: #f0f4ff;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin: 0 0 8px;
    }
    .hero-title span {
        background: linear-gradient(90deg, #14f0a0, #50a0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 300;
        letter-spacing: 0.01em;
        margin: 0;
    }

    /* ── Divider accent ── */
    .accent-line {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, #14f0a0, #50a0ff);
        border-radius: 2px;
        margin: 18px auto 36px;
    }

    /* ── Form card ── */
    .form-card {
        background: rgba(12, 18, 36, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 36px 40px 40px;
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }
    .form-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(20, 240, 160, 0.4), transparent);
    }
    .form-card-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 28px;
        text-align: center;
    }

    /* ── Field labels ── */
    .field-label {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        color: #50a0ff;
        text-transform: uppercase;
        margin-bottom: 6px;
        margin-top: 0;
    }

    /* ── Override Streamlit selectbox & number_input ── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: rgba(6, 10, 24, 0.9) !important;
        border: 1px solid rgba(80, 160, 255, 0.18) !important;
        border-radius: 8px !important;
        color: #c8d8f8 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        transition: border-color 0.2s !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: rgba(80, 160, 255, 0.45) !important;
    }
    div[data-baseweb="select"] svg { color: #50a0ff !important; }
    div[data-baseweb="popover"], ul[data-baseweb="menu"] {
        background-color: #0a1020 !important;
        border: 1px solid rgba(80, 160, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    li[role="option"] { 
        color: #000000 !important; 
        font-size: 0.88rem !important; 
        background-color: transparent !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] { 
        background-color: rgba(80, 160, 255, 0.2) !important; 
        color: #000000 !important; 
    }
    input[type="number"] { color: #c8d8f8 !important; }

    /* ── Primary submit button ── */
    div.stForm [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #0f9e6e 0%, #0a6eb8 100%) !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 32px !important;
        cursor: pointer !important;
        position: relative;
        overflow: hidden;
        transition: opacity 0.2s, transform 0.1s !important;
    }
    div.stForm [data-testid="stFormSubmitButton"] > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
    }
    div.stForm [data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Metric strip ── */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 32px;
    }
    .metric-cell {
        background: rgba(10, 16, 32, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #f0f4ff;
        line-height: 1;
        margin-bottom: 4px;
    }
    .metric-value.green { color: #14f0a0; }
    .metric-value.blue  { color: #50a0ff; }
    .metric-value.amber { color: #f0b840; }
    .metric-label {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        color: #94a3b8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* ── Field spacer ── */
    .field-gap { margin-top: 20px; }
    .submit-gap { margin-top: 28px; }

    /* ── Footer ── */
    .footer-note {
        text-align: center;
        padding: 24px 0 32px;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        color: #6a7c97;
        letter-spacing: 0.08em;
    }

    /* hide default hr */
    hr { display: none !important; }
    </style>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LANDING PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_landing_page():
    inject_landing_styles()

    # Load real model accuracy from model_metrics.json
    try:
        with open("model_metrics.json", "r") as f:
            metrics = json.load(f)
        real_acc = f"{metrics['xgb_test_accuracy']}%"
    except Exception:
        real_acc = "N/A"

    # ── Status bar ──
    st.markdown("""
    <div class="status-bar">
        <div class="status-pill pill-green">
            <span class="status-dot dot-green"></span>SAP ERP LINKED
        </div>
        <div class="status-pill pill-blue">
            <span class="status-dot dot-blue"></span>AI ENGINE ONLINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero ──
    st.markdown("""
    <div class="hero-wrap">
        <p class="hero-eyebrow">// Procurement Intelligence Platform</p>
        <h1 class="hero-title">AI Vendor<br><span>Risk Engine</span></h1>
        <p class="hero-sub">Enterprise intelligence for procurement and supply chain optimization</p>
        <div class="accent-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Centered form column ──
    _, col, _ = st.columns([1, 2.4, 1])

    with col:
        # Load real SAP vendors (used for metrics + dropdown below)
        sap_vendors = get_sap_vendor_list()
        vendor_count = f"{len(sap_vendors):,}"
        total_spend = sum(v.get('total_spend_vol', 0) for v in sap_vendors)
        if total_spend >= 1_000_000_000:
            spend_display = f"${total_spend / 1_000_000_000:.1f}B"
        elif total_spend >= 1_000_000:
            spend_display = f"${total_spend / 1_000_000:.1f}M"
        else:
            spend_display = f"${total_spend:,.0f}"

        # Metric strip
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-cell">
                <div class="metric-value green">{real_acc}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            <div class="metric-cell">
                <div class="metric-value blue">{vendor_count}</div>
                <div class="metric-label">Vendors Tracked</div>
            </div>
            <div class="metric-cell">
                <div class="metric-value amber">{spend_display}</div>
                <div class="metric-label">Spend Analysed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Form card
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<p class="form-card-title">Target Transaction Configuration</p>', unsafe_allow_html=True)

        # sap_vendors already loaded above for metrics
        vendor_display_map = {f"{v['name']} ({v['lifnr']})": v['lifnr'] for v in sap_vendors}
        vendor_display_names = list(vendor_display_map.keys())

        with st.form("engine_params", border=False):
            st.markdown('<p class="field-label">Vendor Profile</p>', unsafe_allow_html=True)
            selected_vendor_display = st.selectbox(
                label="vendor",
                options=vendor_display_names,
                label_visibility="collapsed",
                help="Real SAP Vendor — powered by XGBoost ML",
            )

            st.markdown('<div class="field-gap"></div><p class="field-label">Product Category</p>', unsafe_allow_html=True)
            product_options = [
                "Enterprise Laptop",
                "Corporate Smartphone",
                "Rack Server",
                "Cloud Compute Credit",
            ]
            selected_product = st.selectbox(
                label="product",
                options=product_options,
                label_visibility="collapsed",
                help="Hardware or service being procured",
            )

            st.markdown('<div class="field-gap"></div><p class="field-label">Quoted Price Target (USD)</p>', unsafe_allow_html=True)
            item_price = st.number_input(
                label="price",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                label_visibility="collapsed",
                format="%.2f",
            )

            st.markdown('<div class="submit-gap"></div>', unsafe_allow_html=True)
            submit_button = st.form_submit_button(
                "Launch AI Analytics Phase",
                use_container_width=True,
            )

            if submit_button:
                selected_lifnr = vendor_display_map[selected_vendor_display]
                selected_vendor_name = selected_vendor_display.split(" (")[0]

                # Run model for the selected vendor
                result = procurement_risk_model(
                    vendor_lifnr=selected_lifnr,
                    product_name=selected_product,
                    current_price=item_price,
                )

                if "error" in result:
                    st.error(result["error"])
                else:
                    # Run model using two-stage candidate generation + ranking
                    import hashlib
                    import random

                    df_price = pd.read_csv("purchase_data.csv")
                    df_product = df_price[df_price["product_name"] == selected_product.strip()]
                    market_avg = df_product["price_per_unit"].mean() if len(df_product) > 0 else item_price

                    # ── Step 1: Calculate a quick pre-score for every vendor ──
                    # We don't run the full model yet — just use the raw SAP features
                    # to cheaply rank all vendors before doing expensive ML calls
                    candidate_scores = []
                    for v in sap_vendors:
                        if v['lifnr'] == selected_lifnr:
                            continue  # skip the selected vendor, add them separately

                        # Quick behaviour score from raw features (no ML needed yet)
                        late_ratio     = v.get('late_ratio', 0.5)
                        days_overdue   = v.get('avg_days_overdue_hist', 30)
                        open_exposure  = v.get('open_exposure', 0)

                        # Normalise each feature to 0-1 roughly
                        behaviour_pre = (
                            0.5 * min(late_ratio, 1.0) +
                            0.3 * min(days_overdue / 90, 1.0) +
                            0.2 * min(open_exposure / 1_000_000, 1.0)
                        )
                        candidate_scores.append((v, behaviour_pre))

                    # ── Step 2: Sort all vendors by pre-score, take top 9 ──
                    candidate_scores.sort(key=lambda x: x[1])          # ascending = best first
                    top_9_vendors = [v for v, _ in candidate_scores[:9]]

                    # ── Step 3: Now run the full risk model on those top 9 ──
                    all_scores = [result]  # selected vendor already scored

                    for v in top_9_vendors:
                        seed_val = int(hashlib.md5(v['lifnr'].encode()).hexdigest()[:8], 16)
                        v_rng = random.Random(seed_val)
                        variance_multiplier = v_rng.uniform(0.8, 1.3)   # tighter range — good vendors price fairly
                        vendor_price = round(market_avg * variance_multiplier, 2)

                        v_result = procurement_risk_model(
                            vendor_lifnr=v['lifnr'],
                            product_name=selected_product,
                            current_price=vendor_price,
                        )
                        if "error" not in v_result:
                            all_scores.append(v_result)

                    # ── Step 4: Final sort by full risk score ──
                    all_scores.sort(key=lambda x: (x["final_risk"], x["vendor_risk"]))
                    for rank_idx, s in enumerate(all_scores):
                        s["rank"] = rank_idx + 1

                    st.session_state['analysis_result'] = result
                    st.session_state['selected_vendor'] = selected_vendor_name
                    st.session_state['selected_lifnr'] = selected_lifnr
                    st.session_state['selected_product'] = selected_product
                    st.session_state['all_vendor_scores'] = all_scores
                    st.session_state['page'] = 'analytics'
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div class="footer-note">
        SECURED · ENTERPRISE GRADE · SOC-2 COMPLIANT
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALYTICS PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_analytics_page():
    setup_analytics_styles()
    result = st.session_state.get('analysis_result', None)
    selected_vendor = st.session_state.get('selected_vendor', 'Vendor')
    selected_product = st.session_state.get('selected_product', 'Product')

    if not result or "error" in result:
        st.error("Invalid state. Returning to search.")
        st.button("Back", on_click=lambda: st.session_state.update(page='input'))
        return

    # ── Top Navigation Row ──
    col_back, col_spacer_left, col_title, col_spacer_right, col_status = st.columns([1.5, 2.0, 5.0, 0.1, 3.4])

    with col_back:
        st.markdown('<div class="ghost-back">', unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            st.session_state['page'] = 'input'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_title:
        st.markdown(f"""
        <h2 style="text-align:center; margin:0; padding:0; font-size:1.6rem;">
            Intelligence Report: <span style="color:#14f0a0;">{selected_vendor}</span>
        </h2>
        """, unsafe_allow_html=True)

    with col_status:
        decision = result["decision"]
        if decision == "APPROVE":
            d_color = "#14f0a0"
        elif decision == "REVIEW":
            d_color = "#f0b840"
        else:
            d_color = "#f85149"
            
        variance = result.get('price_variance', 0.0)
        warning_html = ""
        # If the quoted price is heavily inflated (e.g., >80% over AI baseline)
        if variance > 0.8:
            w_color = "#f85149" # Red
            warning_html = f'<div class="decision-badge" style="background: rgba({int(w_color[1:3],16)},{int(w_color[3:5],16)},{int(w_color[5:7],16)},0.08); border-color: {w_color}; color: {w_color}; margin-right: 12px; box-shadow: 0 0 24px rgba({int(w_color[1:3],16)},{int(w_color[3:5],16)},{int(w_color[5:7],16)},0.25);">PRICE ANOMALY</div>'

        st.markdown(f'<div style="display: flex; flex-direction: row; justify-content: flex-end; align-items: center;">{warning_html}<div class="decision-badge" style="background: rgba({int(d_color[1:3],16)},{int(d_color[3:5],16)},{int(d_color[5:7],16)},0.08); border-color: {d_color}; color: {d_color}; box-shadow: 0 0 24px rgba({int(d_color[1:3],16)},{int(d_color[3:5],16)},{int(d_color[5:7],16)},0.25);">{decision}</div></div>', unsafe_allow_html=True)

    # ── Section separator ──
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── TABS: Risk Analysis + Vendor Comparison ──
    tab_analysis, tab_comparison = st.tabs(["Risk Analysis", "Vendor Comparison"])

    with tab_analysis:
        render_risk_analysis_tab(result, selected_vendor, selected_product)

    with tab_comparison:
        render_vendor_comparison_tab(selected_vendor, selected_product)


def render_risk_analysis_tab(result, selected_vendor, selected_product):
    """Content of the Risk Analysis tab (existing analytics content)."""

    # ── SECTION: Core Metrics ──
    st.markdown('<div class="section-header">Core Risk Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        vr = result['vendor_risk']
        vr_color = "#14f0a0" if vr < 0.3 else "#f0b840" if vr < 0.6 else "#f85149"
        st.markdown(f"""
        <div class="analytic-card">
            <div class="card-label">Vendor Reliability Score<br>{selected_vendor}</div>
            <div class="card-value" style="color:{vr_color};">{vr:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        bucket = result['vendor_bucket']
        b_color = "#14f0a0" if bucket == "LOW" else "#f0b840" if bucket == "MEDIUM" else "#f85149"
        st.markdown(f"""
        <div class="analytic-card">
            <div class="card-label">Risk Classification<br>Tier</div>
            <div class="card-value" style="color:{b_color};">{bucket}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        pp = result.get("price_percentiles", {})
        current_price = result["vendor_raw_price"]
        if pp:
            if current_price > pp["p90"]:
                pos_label = "> P90"
                pos_color = "#f85149"
            elif current_price > pp["p75"]:
                pos_label = "P75\u2013P90"
                pos_color = "#f85149"
            elif current_price > pp["p50"]:
                pos_label = "P50\u2013P75"
                pos_color = "#f0b840"
            else:
                pos_label = "\u2264 P50"
                pos_color = "#14f0a0"
        else:
            pos_label = "N/A"
            pos_color = "#94a3b8"
        pv = result['price_variance']
        pv_pct = f"{(pv * 100):+.1f}%"
        pv_color = "#f85149" if pv > 0.15 else "#14f0a0" if pv < -0.05 else "#94a3b8"
        st.markdown(f"""
        <div class="analytic-card">
            <div class="card-label">Price Position<br>(Market Percentile)</div>
            <div class="card-value" style="color:{pos_color};">{pos_label}</div>
            <div style="font-family:'Roboto Mono',monospace; font-size:0.7rem; color:{pv_color}; margin-top:4px; letter-spacing:0.03em;">{pv_pct} vs AI Forecast</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="analytic-card">
            <div class="card-label">AI Forecasted Price<br>For {selected_product}</div>
            <div class="card-value" style="color:#50a0ff;">${result['forecasted_price']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION: Market Trend Alert ──
    direction = result.get('inflation_direction', 'stable')
    pct = result.get('inflation_percent', 0.0)
    
    if direction == "inflation":
        trend_color = "#f85149" if pct > 10 else "#f0b840"
        trend_icon = "↗"
        trend_text = f"This product has faced <b>{pct}% inflation</b> over the last 24 months based on historical purchasing data. Forecasted AI baseline prices have been adjusted accordingly to prevent unfair vendor penalization."
    elif direction == "deflation":
        trend_color = "#14f0a0"
        trend_icon = "↘"
        trend_text = f"This product has experienced <b>{abs(pct)}% deflation</b> over the last 24 months as market costs have decreased. The expected baseline price has been lowered."
    else:
        trend_color = "#50a0ff"
        trend_icon = "→"
        trend_text = "This product's price has remained <b>stable</b> over the last 24 months with no significant inflationary or deflationary trends."

    st.markdown(f"""
    <div style="background: rgba(10, 14, 26, 0.85); border: 1px solid rgba(255,255,255,0.06); border-left: 3px solid {trend_color}; border-radius: 10px; padding: 20px 24px; margin-top: 32px; animation: fadeSlideUp 0.6s ease-out both;">
        <div style="font-family:'Roboto Mono',monospace; font-size:0.75rem; font-weight:600; color:{trend_color}; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;">
            <span style="font-size:1.1rem; margin-right:6px;">{trend_icon}</span> 24-Month Market Trend Analysis
        </div>
        <div style="font-family:'Inter',sans-serif; font-size:0.95rem; color:#c8d8f0; line-height:1.5;">
            {trend_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Price Distribution Context Card ──
    pp = result.get("price_percentiles", {})
    if pp:
        current_price = result["vendor_raw_price"]
        p25, p50, p75, p90 = pp["p25"], pp["p50"], pp["p75"], pp["p90"]
        price_min = p25 * 0.85
        price_max = p90 * 1.15
        price_range = price_max - price_min
        if price_range <= 0:
            price_range = 1  # avoid division by zero

        def pct_pos(val):
            return max(0, min(100, ((val - price_min) / price_range) * 100))

        pos_p25 = pct_pos(p25)
        pos_p50 = pct_pos(p50)
        pos_p75 = pct_pos(p75)
        pos_p90 = pct_pos(p90)
        pos_current = pct_pos(current_price)

        st.markdown(f"""
        <div class="aging-card" style="margin-top:24px; border-left:3px solid #50a0ff;">
            <div style="font-family:'Roboto Mono',monospace; font-size:0.75rem; font-weight:600; color:#50a0ff; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:16px;">
                Price Distribution Context
            </div>
            <div style="position:relative; height:50px; margin-bottom:8px;">
                <!-- Quoted price triangle marker -->
                <div style="position:absolute; left:{pos_current}%; transform:translateX(-50%); top:-6px; text-align:center;">
                    <div style="font-family:'Roboto Mono',monospace; font-size:0.55rem; color:#14f0a0; margin-bottom:2px; line-height:1.2;">QUOTED<br><span style="font-size:0.65rem; font-family:'Inter',sans-serif; font-weight:700;">${current_price:,.0f}</span></div>
                    <div style="width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-top:8px solid #14f0a0; margin:0 auto;"></div>
                </div>
                <!-- Bar track -->
                <div style="position:absolute; bottom:0; left:0; right:0; height:6px; background:rgba(255,255,255,0.05); border-radius:3px;"></div>
                <!-- Percentile markers -->
                <div style="position:absolute; bottom:-2px; left:{pos_p25}%; width:2px; height:10px; background:#94a3b8; transform:translateX(-50%);"></div>
                <div style="position:absolute; bottom:-2px; left:{pos_p50}%; width:2px; height:10px; background:#94a3b8; transform:translateX(-50%);"></div>
                <div style="position:absolute; bottom:-2px; left:{pos_p75}%; width:2px; height:10px; background:#94a3b8; transform:translateX(-50%);"></div>
                <div style="position:absolute; bottom:-2px; left:{pos_p90}%; width:2px; height:10px; background:#94a3b8; transform:translateX(-50%);"></div>
            </div>
            <!-- Labels row -->
            <div style="position:relative; height:28px; font-family:'Roboto Mono',monospace; font-size:0.65rem; color:#94a3b8;">
                <div style="position:absolute; left:{pos_p25}%; transform:translateX(-50%); text-align:center;">
                    <div>P25</div>
                    <div style="color:#c8d8f0;">${p25:,.0f}</div>
                </div>
                <div style="position:absolute; left:{pos_p50}%; transform:translateX(-50%); text-align:center;">
                    <div>P50</div>
                    <div style="color:#c8d8f0;">${p50:,.0f}</div>
                </div>
                <div style="position:absolute; left:{pos_p75}%; transform:translateX(-50%); text-align:center;">
                    <div>P75</div>
                    <div style="color:#c8d8f0;">${p75:,.0f}</div>
                </div>
                <div style="position:absolute; left:{pos_p90}%; transform:translateX(-50%); text-align:center;">
                    <div>P90</div>
                    <div style="color:#c8d8f0;">${p90:,.0f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Section separator ──
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SECTION: Invoice Aging ──
    st.markdown('<div class="section-header">Invoice Aging & Operational Timeline</div>', unsafe_allow_html=True)

    aging_color = "#14f0a0" if result['avg_clearance_days'] < 30 else "#f0b840" if result['avg_clearance_days'] < 60 else "#f85149"

    st.markdown(f"""
    <div class="aging-card" style="border-left: 3px solid {aging_color};">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
            <div style="flex:1; min-width:300px;">
                <p style="margin:0 0 6px 0; color:#94a3b8; font-family:'Roboto Mono',monospace; font-size:0.75rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;">Clearance Timeline</p>
                <p style="margin:0; color:#c8d8f0; font-size:0.95rem; font-family:'Inter',sans-serif;">
                    Supplier profiling indicates that <span style="color:#f0f4ff; font-weight:600;">{selected_product}</span> orders from
                    <span style="color:#f0f4ff; font-weight:600;">{selected_vendor}</span> usually take
                    <span style="color:{aging_color}; font-weight:700;">{result['avg_clearance_days']} days</span> to clear.
                </p>
            </div>
            <div style="text-align:right;">
                <span style="font-family:'Inter',sans-serif; font-size:2.8rem; font-weight:800; color:{aging_color}; line-height:1;">{result['avg_clearance_days']}</span>
                <span style="font-family:'Roboto Mono',monospace; font-size:0.9rem; color:{aging_color}; margin-left:4px;">d</span>
                <div style="color:#94a3b8; font-family:'Roboto Mono',monospace; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; margin-top:4px;">Avg Clearance</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section separator ──
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SECTION: Risk Visualizations ──
    st.markdown('<div class="section-header">Risk Visualizations</div>', unsafe_allow_html=True)

    viz_col1, viz_col2 = st.columns(2, gap="medium")

    with viz_col1:
        # Determine gauge bar color based on risk level
        fr = result['final_risk']
        if fr < 0.3:
            gauge_color = '#14f0a0'  # green — low risk
        elif fr < 0.7:
            gauge_color = '#f0b840'  # amber — medium risk
        else:
            gauge_color = '#f85149'  # red — high/critical risk

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fr,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Final Risk Score", 'font': {'color': '#94a3b8', 'family': 'Roboto Mono', 'size': 13}},
            number={'font': {'color': '#f0f4ff', 'family': 'Inter', 'size': 40}},
            gauge={
                'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': 'rgba(255,255,255,0.1)'},
                'bar': {'color': gauge_color, 'thickness': 0.35},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 1,
                'bordercolor': 'rgba(255,255,255,0.08)',
                'steps': [
                    {'range': [0, 0.3], 'color': 'rgba(20, 240, 160, 0.25)'},
                    {'range': [0.3, 0.7], 'color': 'rgba(240, 184, 64, 0.25)'},
                    {'range': [0.7, 1.0], 'color': 'rgba(248, 81, 73, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': gauge_color, 'width': 4},
                    'thickness': 0.85,
                    'value': fr
                }
            }
        ))
        fig_gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,26,0.6)',
            font={'family': "Inter, sans-serif", 'color': '#c8d8f0'},
            transition={'duration': 1200, 'easing': 'cubic-in-out'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with viz_col2:
        labels = ['Vendor Reliability Risk', 'Market Price Risk']
        values = [result['vendor_risk'] * 0.6, result['price_risk'] * 0.4]
        colors_donut = ['#50a0ff', '#14f0a0']

        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.65,
            marker=dict(colors=colors_donut, line=dict(color='#050810', width=3)),
            textinfo='label+percent',
            textposition='outside',
            hoverinfo='label+percent+value',
            textfont=dict(color='#c8d8f0', family='Inter', size=11)
        )])

        fig_donut.update_layout(
            title={'text': "Risk Contribution Breakdown", 'font': {'color': '#94a3b8', 'family': 'Roboto Mono', 'size': 13}},
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,26,0.6)',
            showlegend=False,
            font={'family': "Inter, sans-serif", 'color': '#c8d8f0'}
        )

        fig_donut.add_annotation(
            text="Weights<br>60% / 40%",
            showarrow=False,
            font=dict(size=13, color="#94a3b8", family="Roboto Mono")
        )

        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Section separator ──
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SECTION: Deep Analytics ──
    st.markdown('<div class="section-header">Deep Analytics & Risk Factors</div>', unsafe_allow_html=True)

    insight_col1, insight_col2 = st.columns(2, gap="medium")

    with insight_col1:
        v_color = "#14f0a0" if result['vendor_risk'] < 0.3 else "#f0b840" if result['vendor_risk'] < 0.6 else "#f85149"
        fig_v = go.Figure(go.Indicator(
            mode="number+gauge",
            value=result['vendor_risk'],
            domain={'x': [0.1, 1], 'y': [0, 1]},
            number={'font': {'color': '#f0f4ff', 'family': 'Inter'}},
            title={'text': "<b>Vendor Reliability Risk</b><br><span style='font-size:0.8em;color:#94a3b8'>Based on internal transaction history</span>"},
            gauge={
                'shape': "bullet",
                'axis': {'range': [min(0.0, result['vendor_risk'] - 0.1), 1]},
                'threshold': {'line': {'color': v_color, 'width': 4}, 'thickness': 0.75, 'value': result['vendor_risk']},
                'bar': {'color': v_color},
                'bgcolor': 'rgba(255,255,255,0.03)',
            }
        ))
        fig_v.update_layout(height=150, margin=dict(t=30, b=20, l=200, r=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "#c8d8f0", 'family': 'Inter'})
        st.plotly_chart(fig_v, use_container_width=True)

    with insight_col2:
        variance = result['price_variance']
        p_color = "#14f0a0" if variance < 0.3 else "#f85149"
        fig_p = go.Figure(go.Indicator(
            mode="number+gauge",
            value=variance,
            domain={'x': [0.1, 1], 'y': [0, 1]},
            number={'font': {'color': '#f0f4ff', 'family': 'Inter'}},
            title={'text': "<b>Market Price Variance</b><br><span style='font-size:0.8em;color:#94a3b8'>Deviation from average</span>"},
            gauge={
                'shape': "bullet",
                'axis': {'range': [min(0.0, variance - 0.1), max(1.0, variance + 0.2)]},
                'threshold': {'line': {'color': p_color, 'width': 4}, 'thickness': 0.75, 'value': variance},
                'bar': {'color': p_color},
                'bgcolor': 'rgba(255,255,255,0.03)',
            }
        ))
        fig_p.update_layout(height=150, margin=dict(t=30, b=20, l=200, r=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "#c8d8f0", 'family': 'Inter'})
        st.plotly_chart(fig_p, use_container_width=True)

    # ── Executive Summary ──
    st.markdown("""
    <div style="margin-top:24px;">
        <div class="section-header">Executive Summary</div>
    </div>
    """, unsafe_allow_html=True)
    for insight in result["insights"]:
        st.info(f"{insight}")

    # ── Section separator ──
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SECTION: XGBoost Feature Impact (SHAP) ──
    st.markdown('<div class="section-header">XGBoost Feature Impact</div>', unsafe_allow_html=True)

    shap_vals = result["xgb_prediction"]["shap_values"]
    feature_label_map = {
        "avg_days_overdue": "Avg Days Overdue",
        "late_ratio": "Late Payment Ratio",
        "total_spend": "Total Spend Volume",
        "open_exposure": "Open Exposure",
    }
    shap_labels = [feature_label_map[k] for k in shap_vals.keys()]
    shap_values_list = list(shap_vals.values())
    shap_colors = ['#f85149' if v > 0 else '#14f0a0' for v in shap_values_list]

    fig_shap = go.Figure(go.Bar(
        x=shap_values_list,
        y=shap_labels,
        orientation='h',
        marker_color=shap_colors,
        text=[f"{v:+.4f}" for v in shap_values_list],
        textposition='outside',
        textfont=dict(color='#94a3b8', family='Roboto Mono', size=11),
    ))
    fig_shap.update_layout(
        height=220,
        margin=dict(l=20, r=60, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,14,26,0.6)',
        font=dict(color='#94a3b8', family='Inter'),
        xaxis=dict(
            title="SHAP Value (impact on prediction)",
            gridcolor='rgba(255,255,255,0.03)',
            zerolinecolor='rgba(255,255,255,0.1)',
            title_font=dict(size=11, color='#94a3b8', family='Roboto Mono'),
        ),
        yaxis=dict(gridcolor='rgba(255,255,255,0.03)'),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # ── SHAP Intelligence Summary ──
    positive_shap = {k: v for k, v in shap_vals.items() if v > 0}
    negative_shap = {k: v for k, v in shap_vals.items() if v < 0}

    primary_driver = max(shap_vals, key=lambda k: shap_vals[k]) if positive_shap else None
    risk_mitigator = min(shap_vals, key=lambda k: shap_vals[k]) if negative_shap else None

    driver_text = f"Primary risk driver: <strong style='color:#f0f4ff;'>{feature_label_map.get(primary_driver, primary_driver)}</strong> (SHAP: <span style='color:#f85149; font-weight:700;'>+{shap_vals[primary_driver]:.4f}</span>)." if primary_driver else "No features are actively increasing risk for this vendor."
    mitigator_text = f" Risk mitigator: <strong style='color:#f0f4ff;'>{feature_label_map.get(risk_mitigator, risk_mitigator)}</strong> (SHAP: <span style='color:#14f0a0; font-weight:700;'>{shap_vals[risk_mitigator]:.4f}</span>)." if risk_mitigator else ""

    if primary_driver and abs(shap_vals.get(primary_driver, 0)) > abs(shap_vals.get(risk_mitigator, 0) if risk_mitigator else 0):
        verdict = f"The model's classification is primarily driven by <strong style='color:#f0f4ff;'>{feature_label_map.get(primary_driver, primary_driver)}</strong>."
    elif risk_mitigator:
        verdict = f"<strong style='color:#f0f4ff;'>{feature_label_map.get(risk_mitigator, risk_mitigator)}</strong> is the strongest factor pulling risk down for this vendor."
    else:
        verdict = "Feature contributions are balanced with no single dominant factor."

    if positive_shap:
        s_bg, s_border = "rgba(248,81,73,0.03)", "#f85149"
    elif negative_shap:
        s_bg, s_border = "rgba(20,240,160,0.03)", "#14f0a0"
    else:
        s_bg, s_border = "rgba(80,160,255,0.03)", "#50a0ff"

    st.html(f"""
<div style="background:{s_bg}; border-left:3px solid {s_border}; padding:16px 20px; margin-top:16px; margin-bottom:16px; border-radius:0 8px 8px 0;">
    <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem; color:#8a9ab8; letter-spacing:0.1em; margin-bottom:8px;">
        // CHART INTELLIGENCE SUMMARY
    </div>
    <div style="font-family:'Inter',sans-serif; font-size:0.85rem; color:#c8d8f0; line-height:1.5;">
        {driver_text}{mitigator_text} {verdict}
    </div>
</div>
""")

    # ── Section separator ──
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SECTION: Isolation Forest — Population Outlier Analysis ──
    st.markdown('<div class="section-header">Isolation Forest — Population Outlier Analysis</div>',
                unsafe_allow_html=True)

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

    with col_iso2:
        fig_iso = go.Figure(go.Indicator(
            mode="gauge+number",
            value=iso_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Population Outlier Score", 'font': {'color': '#94a3b8', 'family': 'Roboto Mono', 'size': 13}},
            number={'font': {'color': '#f0f4ff', 'family': 'Inter', 'size': 40}},
            gauge={
                'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': 'rgba(255,255,255,0.1)'},
                'bar': {'color': iso_color, 'thickness': 0.35},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 1,
                'bordercolor': 'rgba(255,255,255,0.08)',
                'steps': [
                    {'range': [0, 0.5], 'color': 'rgba(20, 240, 160, 0.25)'},
                    {'range': [0.5, 0.75], 'color': 'rgba(240, 184, 64, 0.25)'},
                    {'range': [0.75, 1.0], 'color': 'rgba(248, 81, 73, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': iso_color, 'width': 4},
                    'thickness': 0.85,
                    'value': iso_score
                }
            }
        ))
        fig_iso.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,26,0.6)',
            font={'family': "Inter, sans-serif", 'color': '#c8d8f0'},
            transition={'duration': 1200, 'easing': 'cubic-in-out'}
        )
        st.plotly_chart(fig_iso, use_container_width=True)

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

    # ── Section separator ──
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SECTION: Vendor Behavior Analyzer ──
    st.markdown('<div class="section-header">Vendor Behavior Analyzer</div>', unsafe_allow_html=True)

    # Use real SAP history for selected vendor
    selected_lifnr = st.session_state.get('selected_lifnr', '')
    vendor_history_data = get_vendor_history(selected_lifnr)

    if vendor_history_data is None:
        st.warning("No historical data available for this vendor.")
        return

    past_dealings_count = vendor_history_data.get("transaction_count", 64)
    st.markdown(f"""
    <p style="color:#8a9ab8; font-size:0.82rem; font-family:'Roboto Mono',monospace; letter-spacing:0.05em; margin-bottom:16px;">
        Historical Risk Fluctuation (Past 12 Months) — Calculated from <span style="color:#c8d8f0;">{past_dealings_count}</span> past dealings
    </p>
    """, unsafe_allow_html=True)

    months = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='ME').strftime('%b %Y')
    historical_risk = vendor_history_data["monthly_risk_history"]

    fig_history = go.Figure()
    fig_history.add_trace(go.Scatter(
        x=months,
        y=historical_risk,
        mode='lines+markers',
        name='Historical Risk Score',
        line=dict(color='#50a0ff', width=2.5, shape='spline'),
        marker=dict(size=7, color='#50a0ff', symbol='diamond', line=dict(color='#050810', width=1.5)),
        fill='tozeroy',
        fillcolor='rgba(80,160,255,0.05)',
    ))

    fig_history.add_hline(y=0.7, line_dash="dash", line_color="rgba(248, 81, 73, 0.6)",
                          annotation_text="High Risk Threshold",
                          annotation_position="top right",
                          annotation_font=dict(color="#f85149", family="Roboto Mono", size=10))

    fig_history.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Risk Score (0 to 1)",
        yaxis=dict(range=[0, 1], gridcolor='rgba(255,255,255,0.03)', zerolinecolor='rgba(255,255,255,0.03)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.03)'),
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,14,26,0.6)',
        font={'color': '#94a3b8', 'family': 'Inter'}
    )

    st.plotly_chart(fig_history, use_container_width=True)

    # ── History Insight Summary ──
    start_risk = historical_risk[0]
    end_risk = historical_risk[-1]
    avg_risk = sum(historical_risk) / len(historical_risk)
    trend = "decreasing" if end_risk < start_risk else "increasing"
    
    if avg_risk < 0.4:
        bg_col, border_col = "rgba(20,240,160,0.03)", "#14f0a0"
    elif avg_risk < 0.7:
        bg_col, border_col = "rgba(240,184,64,0.03)", "#f0b840"
    else:
        bg_col, border_col = "rgba(248,81,73,0.03)", "#f85149"
        
    trend_text = "The vendor's risk profile is stabilizing/improving." if trend == "decreasing" else "Close monitoring is advised as historical risk exposure is climbing."

    st.html(f"""
<div style="background:{bg_col}; border-left:3px solid {border_col}; padding:16px 20px; margin-top:16px; margin-bottom:16px; border-radius:0 8px 8px 0;">
    <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem; color:#8a9ab8; letter-spacing:0.1em; margin-bottom:8px;">
        // CHART INTELLIGENCE SUMMARY
    </div>
    <div style="font-family:'Inter',sans-serif; font-size:0.85rem; color:#c8d8f0; line-height:1.5;">
        Analysis of the past 12 months indicates an <strong style="color:#f0f4ff;">{trend}</strong> risk trend, 
        with an average risk score of <span style="color:{border_col}; font-weight:700;">{avg_risk:.2f}</span>. 
        {trend_text}
    </div>
</div>
""")

    # ── Section separator ──
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SECTION: Anomaly Detection ──
    st.markdown('<div class="section-header">Vendor Pattern Anomaly Detection</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#8a9ab8; font-size:0.82rem; font-family:'Roboto Mono',monospace; letter-spacing:0.05em; margin-bottom:16px;">
        Order Volume Pattern Analysis — Detecting abnormal quotation quantities
    </p>
    """, unsafe_allow_html=True)

    transactions = list(range(1, 51))
    order_sizes = vendor_history_data["past_transactions"]

    colors_scatter = ['#f85149' if size > 1000 else '#50a0ff' for size in order_sizes]
    sizes_scatter = [14 if size > 1000 else 7 for size in order_sizes]

    fig_patterns = go.Figure()
    fig_patterns.add_trace(go.Scatter(
        x=transactions,
        y=order_sizes,
        mode='markers',
        name='Quotation Volume',
        marker=dict(
            color=colors_scatter,
            size=sizes_scatter,
            line=dict(color='rgba(255,255,255,0.15)', width=1),
            opacity=0.9,
        ),
        text=[f"Transaction {t}: {s} units" + (" (ANOMALY DETECTED)" if s > 1000 else "") for t, s in zip(transactions, order_sizes)],
        hoverinfo="text"
    ))

    fig_patterns.add_hline(y=1000, line_dash="dash", line_color="rgba(240, 184, 64, 0.5)",
                           annotation_text="Anomaly Threshold",
                           annotation_position="top left",
                           annotation_font=dict(color="#f0b840", family="Roboto Mono", size=10))

    fig_patterns.update_layout(
        xaxis_title="Recent Transactions (Last 50)",
        yaxis_title="Volume / Quantity Quoted",
        yaxis=dict(gridcolor='rgba(255,255,255,0.03)', zerolinecolor='rgba(255,255,255,0.03)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.03)'),
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,14,26,0.6)',
        font={'color': '#94a3b8', 'family': 'Inter'}
    )

    st.plotly_chart(fig_patterns, use_container_width=True)

    # ── Anomaly Insight Summary ──
    anomalies_count = sum(1 for size in order_sizes if size > 1000)
    anomaly_pct = (anomalies_count / len(order_sizes)) * 100
    
    if anomalies_count > 5:
        a_bg, a_border, a_text = "rgba(248,81,73,0.03)", "#f85149", "Volume patterns are highly irregular and require immediate investigation."
    elif anomalies_count > 0:
        a_bg, a_border, a_text = "rgba(240,184,64,0.03)", "#f0b840", "Volume patterns are generally stable with minor high-volume deviations."
    else:
        a_bg, a_border, a_text = "rgba(20,240,160,0.03)", "#14f0a0", "Perfect consistency in order volumes. No operational alerts."

    st.html(f"""
<div style="background:{a_bg}; border-left:3px solid {a_border}; padding:16px 20px; margin-top:16px; margin-bottom:32px; border-radius:0 8px 8px 0;">
    <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem; color:#8a9ab8; letter-spacing:0.1em; margin-bottom:8px;">
        // CHART INTELLIGENCE SUMMARY
    </div>
    <div style="font-family:'Inter',sans-serif; font-size:0.85rem; color:#c8d8f0; line-height:1.5;">
        Detected <span style="color:{a_border}; font-weight:700;">{anomalies_count} anomalous transactions</span> 
        out of the last 50 orders ({anomaly_pct:.1f}% frequency). {a_text}
    </div>
</div>
""")


def render_vendor_comparison_tab(selected_vendor, selected_product):
    """Content of the Vendor Comparison tab."""
    all_scores = st.session_state.get('all_vendor_scores', [])

    if not all_scores:
        st.warning("No vendor comparison data available. Please re-run the analysis.")
        return

    # Build a LIFNR → Name lookup from SAP data
    sap_vendors = get_sap_vendor_list()
    vendor_name_map = {v['lifnr']: v['name'] for v in sap_vendors}

    # Find selected vendor's rank and score
    selected_lifnr = st.session_state.get('selected_lifnr', '')
    selected_score = next((s for s in all_scores if s["vendor_id"] == selected_lifnr), None)
    selected_rank = selected_score["rank"] if selected_score else 0
    selected_final_risk = selected_score["final_risk"] if selected_score else 0

    # ── Alert Banner ──
    if selected_final_risk >= 0.7:
        # High risk — recommend better alternatives
        top_2 = [s for s in all_scores if s["rank"] <= 2]
        top_names = ", ".join([vendor_name_map.get(s["vendor_id"], s["vendor_id"]) for s in top_2])
        st.html(f"""
<div style="background:rgba(248,81,73,0.08); border:1px solid rgba(248,81,73,0.3); border-radius:12px;
            padding:20px 28px; margin-bottom:32px; animation: fadeSlideUp 0.5s ease-out both;">
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:1.5rem;">⚠</span>
        <div>
            <p style="color:#f85149; font-family:'Inter',sans-serif; font-weight:700; font-size:1rem; margin:0 0 6px 0;">
                Better Alternatives Exist for This Transaction
            </p>
            <p style="color:#c8d8f0; font-family:'Inter',sans-serif; font-size:0.88rem; margin:0;">
                <span style="color:#f0f4ff; font-weight:600;">{selected_vendor}</span> carries elevated risk.
                Consider <span style="color:#14f0a0; font-weight:600;">{top_names}</span> for lower risk exposure.
            </p>
        </div>
    </div>
</div>
""")
    elif selected_final_risk >= 0.4 or selected_rank > 2:
        # Medium risk or sub-optimal rank
        top_2 = [s for s in all_scores if s["rank"] <= 2]
        top_names = ", ".join([vendor_name_map.get(s["vendor_id"], s["vendor_id"]) for s in top_2])
        st.html(f"""
<div style="background:rgba(240,184,64,0.08); border:1px solid rgba(240,184,64,0.3); border-radius:12px;
            padding:20px 28px; margin-bottom:32px; animation: fadeSlideUp 0.5s ease-out both;">
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:1.5rem;">⚠</span>
        <div>
            <p style="color:#f0b840; font-family:'Inter',sans-serif; font-weight:700; font-size:1rem; margin:0 0 6px 0;">
                Sub-optimal Vendor Selection
            </p>
            <p style="color:#c8d8f0; font-family:'Inter',sans-serif; font-size:0.88rem; margin:0;">
                <span style="color:#f0f4ff; font-weight:600;">{selected_vendor}</span> has a moderate risk profile or is outranked.
                Consider <span style="color:#14f0a0; font-weight:600;">{top_names}</span> for better terms.
            </p>
        </div>
    </div>
</div>
""")
    else:
        # Best or second best and low risk
        st.html(f"""
<div style="background:rgba(20,240,160,0.06); border:1px solid rgba(20,240,160,0.25); border-radius:12px;
            padding:20px 28px; margin-bottom:32px; animation: fadeSlideUp 0.5s ease-out both;">
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:1.5rem;">✓</span>
        <div>
            <p style="color:#14f0a0; font-family:'Inter',sans-serif; font-weight:700; font-size:1rem; margin:0 0 6px 0;">
                Optimal Vendor Selection
            </p>
            <p style="color:#c8d8f0; font-family:'Inter',sans-serif; font-size:0.88rem; margin:0;">
                You have selected one of the lowest-risk vendors for this transaction.
            </p>
        </div>
    </div>
</div>
""")

    st.html('<div class="section-header">Vendor Rankings — All Vendors Compared</div>')

    st.html(f"""
<p style="color:#8a9ab8; font-size:0.82rem; font-family:'Roboto Mono',monospace; letter-spacing:0.05em; margin-bottom:24px;">
    Ranked by overall risk score for <span style="color:#c8d8f0;">{selected_product}</span> — lowest risk = best
</p>
""")

    # ── Vendor Cards ──
    for s in all_scores:
        is_selected = s["vendor_id"] == selected_lifnr
        is_recommended = (selected_final_risk >= 0.7 and s["rank"] <= 2)

        # Border color
        if is_selected:
            border_color = "#14f0a0"
            border_width = "2px"
        elif is_recommended:
            border_color = "rgba(20, 240, 160, 0.4)"
            border_width = "1px"
        else:
            border_color = "rgba(255,255,255,0.06)"
            border_width = "1px"

        # Risk badge
        bucket = s["vendor_bucket"]
        if bucket == "LOW":
            badge_color = "#14f0a0"
            badge_bg = "rgba(20,240,160,0.1)"
        elif bucket == "MEDIUM":
            badge_color = "#f0b840"
            badge_bg = "rgba(240,184,64,0.1)"
        else:
            badge_color = "#f85149"
            badge_bg = "rgba(248,81,73,0.1)"

        # Progress bar color
        fr = s["final_risk"]
        if fr < 0.3:
            bar_color = "#14f0a0"
        elif fr < 0.7:
            bar_color = "#f0b840"
        else:
            bar_color = "#f85149"

        # Rank display
        rank_num = s["rank"]
        rank_color = "#14f0a0" if rank_num <= 2 else "#f0b840" if rank_num <= 4 else "#f85149"

        selected_tag = ""
        if is_selected:
            selected_tag = '<span style="background:rgba(20,240,160,0.12); color:#14f0a0; padding:3px 10px; border-radius:6px; font-family:\'Roboto Mono\',monospace; font-size:0.62rem; letter-spacing:0.08em; margin-left:12px;">SELECTED</span>'

        recommended_tag = ""
        if is_recommended and not is_selected:
            recommended_tag = '<span style="background:rgba(20,240,160,0.12); color:#14f0a0; padding:3px 10px; border-radius:6px; font-family:\'Roboto Mono\',monospace; font-size:0.62rem; letter-spacing:0.08em; margin-left:12px;">RECOMMENDED</span>'

        st.html(f"""
<div style="background:rgba(10,14,26,0.85); border:{border_width} solid {border_color}; border-radius:14px;
            padding:24px 28px; margin-bottom:16px; position:relative; overflow:hidden;
            animation: fadeSlideUp 0.6s ease-out both; animation-delay: {rank_num * 0.08}s;">
    <div style="position:absolute; top:0; left:0; right:0; height:1px;
                background:linear-gradient(90deg, transparent, {'rgba(20,240,160,0.3)' if is_selected else 'rgba(255,255,255,0.05)'}, transparent);"></div>

    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
        <!-- Left: Rank + Name -->
        <div style="display:flex; align-items:center; gap:16px; min-width:280px;">
            <div style="width:44px; height:44px; border-radius:10px; background:rgba({int(rank_color[1:3],16)},{int(rank_color[3:5],16)},{int(rank_color[5:7],16)},0.1);
                        display:flex; align-items:center; justify-content:center;
                        font-family:'Inter',sans-serif; font-size:1.2rem; font-weight:800; color:{rank_color};">
                #{rank_num}
            </div>
            <div>
                <div style="font-family:'Inter',sans-serif; font-size:1.05rem; font-weight:700; color:#f0f4ff;">
                    {vendor_name_map.get(s['vendor_id'], s['vendor_id'])}{selected_tag}{recommended_tag}
                </div>
                <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem; color:#94a3b8; letter-spacing:0.08em; margin-top:4px;">
                    {s['vendor_id']} &nbsp;|&nbsp; AVG CLEARANCE: {s['avg_clearance_days']}d &nbsp;|&nbsp; QUOTED: <span style="color:#c8d8f0;">${s['vendor_raw_price']:,.2f}</span>
                </div>
            </div>
        </div>

        <!-- Center: Score Metrics -->
        <div style="display:flex; gap:32px; align-items:center;">
            <div style="text-align:center;">
                <div style="font-family:'Roboto Mono',monospace; font-size:0.58rem; color:#94a3b8; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">Overall Risk</div>
                <div style="font-family:'Inter',sans-serif; font-size:1.4rem; font-weight:800; color:{bar_color};">{fr:.2f}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Roboto Mono',monospace; font-size:0.58rem; color:#94a3b8; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">Vendor Risk</div>
                <div style="font-family:'Inter',sans-serif; font-size:1.1rem; font-weight:700; color:#c8d8f0;">{s['vendor_risk']:.2f}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Roboto Mono',monospace; font-size:0.58rem; color:#94a3b8; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">Price Risk</div>
                <div style="font-family:'Inter',sans-serif; font-size:1.1rem; font-weight:700; color:#c8d8f0;">{s['price_risk']:.1f}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Roboto Mono',monospace; font-size:0.58rem; color:#94a3b8; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">Outlier Score</div>
                <div style="font-family:'Inter',sans-serif; font-size:1.1rem; font-weight:700; color:{'#f85149' if s.get('isolation_score', 0.5) > 0.75 else '#f0b840' if s.get('isolation_score', 0.5) > 0.5 else '#14f0a0'};">{s.get('isolation_score', 0.5):.2f}</div>
            </div>
        </div>

        <!-- Right: Progress Bar + Badge -->
        <div style="min-width:200px; display:flex; flex-direction:column; align-items:flex-end; gap:10px;">
            <div style="background:{badge_bg}; border:1px solid {badge_color}; color:{badge_color};
                        padding:4px 14px; border-radius:8px; font-family:'Inter',sans-serif; font-size:0.72rem;
                        font-weight:700; letter-spacing:0.08em;">
                {bucket}
            </div>
            <div style="width:100%; height:6px; background:rgba(255,255,255,0.05); border-radius:3px; overflow:hidden;">
                <div style="width:{fr * 100}%; height:100%; background:{bar_color}; border-radius:3px;
                            transition: width 1.2s ease;"></div>
            </div>
        </div>
    </div>
</div>
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    st.set_page_config(
        page_title="AI Procurement Risk Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    if 'page' not in st.session_state:
        st.session_state['page'] = 'input'

    if st.session_state['page'] == 'input':
        render_landing_page()
    elif st.session_state['page'] == 'analytics':
        render_analytics_page()

if __name__ == "__main__":
    main()
