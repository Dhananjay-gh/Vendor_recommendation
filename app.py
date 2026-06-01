import streamlit as st
import plotly.graph_objects as go
import json
import requests
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from model import procurement_risk_model, get_sap_vendor_list, get_vendor_history, get_vendor_avg_price
import requests as _requests

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OPENROUTER API — AI CHATBOT (Task 1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os
try:
    import streamlit as st
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
except Exception:
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",   # primary — very reliable
    "nvidia/nemotron-3-super-120b-a12b:free",    # fallback — strong reasoning
    "google/gemma-4-31b-it:free",                # fallback — Google latest
]


def call_openrouter(system_prompt, messages, max_tokens=1200, stream_placeholder=None):
    """
    Call OpenRouter with model fallback chain.
    Returns (text, error) — one will always be None.
    """
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer":  "https://procurement-risk-analyzer.app",
        "X-Title":       "Procurement Risk Analyzer",
    }

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    errors = []
    for model in OPENROUTER_MODELS:
        try:
            resp = _requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model":      model,
                    "max_tokens": max_tokens,
                    "messages":   full_messages,
                    "stream":     True,
                },
                stream=True,
                timeout=60,
            )
            if resp.status_code == 200:
                full_text = ""
                for chunk in resp.iter_lines():
                    if chunk and chunk.startswith(b"data: "):
                        data = chunk[6:]
                        if data == b"[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_text += delta
                                if stream_placeholder:
                                    stream_text = full_text.replace('$', '\\$')
                                    stream_placeholder.markdown(f'''<div class="analytic-card" style="padding: 16px 20px; margin-bottom: 16px;">
<div style="color: #14f0a0; font-family: 'Roboto Mono', monospace; font-size: 0.7rem; margin-bottom: 12px; letter-spacing: 0.1em;">AI RISK ANALYST</div>

{stream_text}▌

</div>''', unsafe_allow_html=True)
                        except Exception as json_e:
                            continue
                if stream_placeholder:
                    stream_text = full_text.replace('$', '\\$')
                    stream_placeholder.markdown(f'''<div class="analytic-card" style="padding: 16px 20px; margin-bottom: 16px;">
<div style="color: #14f0a0; font-family: 'Roboto Mono', monospace; font-size: 0.7rem; margin-bottom: 12px; letter-spacing: 0.1em;">AI RISK ANALYST</div>

{stream_text}

</div>''', unsafe_allow_html=True)
                return full_text.strip(), None
            else:
                errors.append(f"{model} ({resp.status_code})")
        except Exception as e:
            errors.append(f"{model} ({type(e).__name__})")
            continue  # try next model

    return None, f"All models failed. Debug info: {', '.join(errors)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD CHATBOT CONTEXT — SYSTEM PROMPT (Task 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_chatbot_context(result, selected_vendor, selected_product, all_vendor_scores):
    """
    Build the full system prompt for the AI Risk Analyst chatbot.
    Uses all data from the analysis result, XGBoost prediction,
    feature deviations, and all comparison vendor scores.
    """
    xgb = result.get("xgb_prediction", {})
    deviations = result.get("feature_deviations", [])
    pp = result.get("price_percentiles", {})

    # K-Means cluster label mapping
    cluster_labels = {
        0: "Conservative Spenders",
        1: "High-Volume Partners",
        2: "At-Risk Outliers",
        3: "Stable Mid-Tier",
    }
    cluster_id = xgb.get("kmeans_cluster", 0)
    cluster_name = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")

    # Build deviation lines dynamically for ALL features
    deviation_lines = []
    for d in deviations:
        feat_name = d["feature"]
        vendor_val = d.get("vendor_val", 0)
        pop_mean = d.get("pop_mean", 0)
        z_score = d.get("z_score", 0)
        level = d.get("level", "N/A")
        deviation_lines.append(
            f"{feat_name:<28}: vendor={vendor_val:.4f}  pop_mean={pop_mean:.4f}  "
            f"z-score={z_score:+.2f}  level={level}"
        )
    deviation_block = "\n".join(deviation_lines) if deviation_lines else "No deviation data available."

    # Build SHAP lines dynamically for ALL features
    shap = xgb.get("shap_values", {})
    def _shap_dir(val):
        return "increases" if val > 0 else "decreases"

    shap_lines = []
    for feat_name, shap_val in shap.items():
        shap_lines.append(
            f"{feat_name:<28}: {shap_val:+.4f}  ← {_shap_dir(shap_val)} risk"
        )
    shap_block = "\n".join(shap_lines) if shap_lines else "No SHAP data available."

    # Price percentile strings
    p25_str = f"${pp.get('p25', 0):,.2f}" if pp else "N/A"
    p50_str = f"${pp.get('p50', 0):,.2f}" if pp else "N/A"
    p75_str = f"${pp.get('p75', 0):,.2f}" if pp else "N/A"
    p90_str = f"${pp.get('p90', 0):,.2f}" if pp else "N/A"

    # Build vendor comparison table
    sap_vendors = get_sap_vendor_list()
    vendor_name_map = {v['lifnr']: v['name'] for v in sap_vendors}

    comparison_lines = []
    for s in all_vendor_scores:
        v_name = vendor_name_map.get(s["vendor_id"], s["vendor_id"])
        comparison_lines.append(
            f"  {s.get('rank', '-'):>4} | {v_name:<30} | {s['vendor_id']:<12} | "
            f"{s['final_risk']:.2f}       | {s['vendor_risk']:.2f}        | "
            f"{s['decision']:<10} | {s.get('isolation_score', 0.5):.2f}"
        )
    comparison_table = "\n".join(comparison_lines)

    # Number of vendors in the population (for context)
    pop_size = len(sap_vendors)

    # Vendor health signals for chatbot context
    years_active = xgb.get('years_active', 0)
    dunning_level = xgb.get('dunning_level', 0)
    is_blocked = xgb.get('is_payment_blocked', 0)
    reversal_rate = xgb.get('reversal_rate', 0)
    payment_consistency = xgb.get('payment_consistency', 0)
    discount_capture = xgb.get('discount_capture_rate', 0)
    voided_rate = xgb.get('voided_payment_rate', 0)
    stale_rate = xgb.get('stale_payment_rate', 0)

    system_prompt = f"""You are an AI Procurement Risk Analyst assistant. You have full access to the \
dashboard data for the current analysis session. Answer all questions based \
ONLY on this data. Be specific with numbers. Never make up values.

=== SELECTED VENDOR ===
Vendor: {selected_vendor} ({result.get('vendor_id', 'N/A')})
Product Being Procured: {selected_product}
Quoted Price: ${result.get('vendor_raw_price', 0):,.2f}

=== RISK SCORES ===
Final Risk Score    : {result.get('final_risk', 0):.2f} ({result.get('vendor_bucket', 'N/A')}) → {result.get('decision', 'N/A')}
Vendor Risk         : {result.get('vendor_risk', 0):.2f}
Price Risk          : {result.get('price_risk', 0):.2f}
XGBoost Class       : {xgb.get('predicted_class_label', 'N/A')}
SAP Class           : {xgb.get('sap_risk_label', 'N/A')} (Divergence: {'YES' if result.get('sap_divergence') else 'NO'})
Isolation Forest    : {'OUTLIER' if result.get('is_outlier', False) else 'NORMAL'} (score: {result.get('isolation_score', 0.5):.4f})
K-Means Cluster     : {cluster_name} (Cluster {cluster_id})

=== VENDOR BEHAVIOUR — CORE METRICS (vs population of {pop_size} vendors) ===
Avg Days Overdue    : {xgb.get('avg_days_overdue', 0):.1f} days
Late Payment Ratio  : {xgb.get('late_ratio', 0):.4f}
Total Spend Volume  : ${xgb.get('total_spend', 0):,.2f}
Open Exposure       : ${xgb.get('open_exposure', 0):,.2f}
Transaction Count   : {xgb.get('transaction_count', 0)}

=== VENDOR HEALTH SIGNALS (from expanded SAP features) ===
Years Active        : {years_active:.1f} years  ({'ESTABLISHED' if years_active >= 5 else 'GROWING' if years_active >= 2 else 'NEW VENDOR'})
Payment Consistency : {payment_consistency:.2f} days std dev  (higher = more erratic payment behaviour)
Payment Blocked     : {'YES — vendor has active payment/posting block in SAP' if is_blocked else 'NO — no blocks'}
Dunning Level       : {dunning_level} / 4  ({'NO NOTICES' if dunning_level == 0 else 'ESCALATED — multiple dunning notices' if dunning_level >= 3 else 'WARNED — some dunning notices'})
Reversal Rate       : {reversal_rate:.2%}  ({'HIGH — above 10% threshold' if reversal_rate > 0.1 else 'ACCEPTABLE' if reversal_rate > 0.05 else 'CLEAN'})
Discount Capture    : {discount_capture:.2%}  (fraction of early payment discounts taken when offered)
Voided Payment Rate : {voided_rate:.2%}  (fraction of payments voided)
Stale Payment Rate  : {stale_rate:.2%}  (fraction of payments gone stale)

=== FEATURE DEVIATION ANALYSIS (all {len(deviations)} features — z-scores vs population) ===
{deviation_block}

=== SHAP FEATURE IMPACT — ALL {len(shap)} FEATURES (XGBoost explanation) ===
{shap_block}

=== PRICE ANALYSIS ===
AI Forecasted Price : ${result.get('forecasted_price', 0):,.2f}
Market Hist. Average: ${result.get('avg_price', 0):,.2f}
Vendor Hist. Average: ${result.get('vendor_historical_avg', 0):,.2f}
Quoted Price        : ${result.get('vendor_raw_price', 0):,.2f}
Price Variance      : {result.get('price_variance', 0):+.1%} vs forecast
24-Month Trend      : {result.get('inflation_direction', 'stable')} ({result.get('inflation_percent', 0):+.1f}%)
Price Percentiles   : P25={p25_str} | P50={p50_str} | P75={p75_str} | P90={p90_str}

=== INVOICE CLEARANCE ===
Avg Clearance Days  : {result.get('avg_clearance_days', 0)} days

=== ALL VENDORS COMPARED (for {selected_product} at market price) ===
  Rank | Vendor Name                    | Vendor ID    | Final Risk | Vendor Risk | Decision   | Outlier Score
{comparison_table}

Answer in clear, professional language. Use bullet points for lists.
Always cite exact numbers from the data above in your answers.
When explaining SHAP values, reference ALL {len(shap)} features listed above, not just the top 4.
When comparing vendors, reference both vendors' actual values side by side."""

    return system_prompt


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALYTICS PAGE — PREMIUM STYLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def setup_analytics_styles():
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'light':
        css_vars = """
        :root {
            --bg-app: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #334155;
            --text-head: #0f172a;
            --text-mute: #64748b;
            --border-light: rgba(59, 130, 246, 0.2);
            --border-med: rgba(59, 130, 246, 0.5);
            --bg-hover: rgba(0,0,0,0.03);
            --shadow-str: rgba(0,0,0,0.1);
        }
        """
        st.session_state['plot_bg'] = '#ffffff'
        st.session_state['paper_bg'] = '#f8fafc'
        st.session_state['text_main'] = '#334155'
    else:
        css_vars = """
        :root {
            --bg-app: #050810;
            --bg-card: #0a0e1a;
            --text-main: #c8d8f0;
            --text-head: #f0f4ff;
            --text-mute: #94a3b8;
            --border-light: rgba(255,255,255,0.06);
            --border-med: rgba(255,255,255,0.15);
            --bg-hover: rgba(255,255,255,0.03);
            --shadow-str: rgba(0,0,0,0.5);
        }
        """
        st.session_state['plot_bg'] = 'rgba(10,14,26,0.6)'
        st.session_state['paper_bg'] = '#050810'
        st.session_state['text_main'] = '#c8d8f0'
        
    st.markdown(f"<style>{css_vars}</style>", unsafe_allow_html=True)
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
        background: var(--bg-app);
        background-image:
            linear-gradient(rgba(20, 240, 200, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(20, 240, 200, 0.025) 1px, transparent 1px);
        background-size: 48px 48px;
        min-height: 100vh;
        color: var(--text-main);
    }
    
    /* ── Fix Streamlit Widget Labels (e.g. toggle text) ── */
    [data-testid="stWidgetLabel"] p, 
    [data-testid="stToggle"] p {
        color: var(--text-main) !important;
    }

    /* ── Typography ── */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-head) !important;
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
        color: var(--text-mute);
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
        background: var(--bg-card);
        border: 1px solid var(--border-light);
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
        box-shadow: 0 12px 32px var(--shadow-str);
    }
    .analytic-card .card-label {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-mute);
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
    div[data-testid="stButton"] > button,
    .stButton > button {
        background: transparent !important;
        background-color: transparent !important;
        border: 1px solid var(--border-med) !important;
        color: var(--text-main) !important;
        box-shadow: none !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.05em !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stButton"] > button:hover,
    .stButton > button:hover {
        border-color: var(--text-main) !important;
        color: var(--text-main) !important;
        transform: none !important;
        background: var(--bg-hover) !important;
        background-color: var(--bg-hover) !important;
        box-shadow: none !important;
    }

    /* ── Invoice aging card ── */
    .aging-card {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
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
        background: var(--bg-card) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-light) !important;
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
        border-bottom: 1px solid var(--border-light);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-mute);
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
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'light':
        css_vars = """
        :root {
            --bg-app: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #334155;
            --text-head: #0f172a;
            --text-mute: #64748b;
        }
        """
    else:
        css_vars = """
        :root {
            --bg-app: #050810;
            --bg-card: #0a0e1a;
            --text-main: #c8d8f0;
            --text-head: #f0f4ff;
            --text-mute: #94a3b8;
        }
        """

    st.markdown(f"<style>{css_vars}</style>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Roboto+Mono:wght@400;500&display=swap');

    /* ── Fix Streamlit Widget Labels (e.g. toggle text) ── */
    [data-testid="stWidgetLabel"] p, 
    [data-testid="stToggle"] p {
        color: var(--text-main) !important;
    }

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
        background: var(--bg-app);
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
        color: var(--text-head);
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
        color: var(--text-mute);
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

    /* ── Form card (targets the Streamlit container by key) ── */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stForm"]) {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 16px;
        padding: 36px 40px 40px;
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }
    .form-card-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--text-mute);
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
        background: var(--bg-card) !important;
        border: 1px solid rgba(80, 160, 255, 0.18) !important;
        border-radius: 8px !important;
        color: var(--text-main) !important;
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
        background-color: var(--bg-card) !important;
        border: 1px solid rgba(80, 160, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    li[role="option"] { 
        color: #0f172a !important; /* Force dark text since background is white */
        font-size: 0.88rem !important; 
        background-color: transparent !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] { 
        background-color: rgba(80, 160, 255, 0.2) !important; 
        color: #0f172a !important; 
    }
    input[type="number"] { color: var(--text-main) !important; }

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
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 32px;
    }
    .metric-cell {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-head);
        line-height: 1;
        margin-bottom: 4px;
    }
    .metric-value.green { color: #14f0a0; }
    .metric-value.blue  { color: #50a0ff; }
    .metric-value.amber { color: #f0b840; }
    .metric-label {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-mute);
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
    # ── THEME TOGGLE ──
    col_empty, col_toggle = st.columns([8, 1])
    with col_toggle:
        is_light = st.session_state.get('theme', 'light') == 'light'
        if st.toggle("☀ Light Mode", value=is_light, key="landing_theme"):
            if st.session_state.get('theme') != 'light':
                st.session_state['theme'] = 'light'
                st.rerun()
        else:
            if st.session_state.get('theme') != 'dark':
                st.session_state['theme'] = 'dark'
                st.rerun()

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
                value=0.0,
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
                    st.session_state.pop('chat_history', None)
                    st.session_state.pop('chat_prefill', None)
                    st.session_state['page'] = 'analytics'
                    st.rerun()

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
    # ── THEME TOGGLE ──
    col_empty, col_toggle = st.columns([8, 1])
    with col_toggle:
        is_light = st.session_state.get('theme', 'light') == 'light'
        if st.toggle("☀ Light Mode", value=is_light):
            if st.session_state.get('theme') != 'light':
                st.session_state['theme'] = 'light'
                st.rerun()
        else:
            if st.session_state.get('theme') != 'dark':
                st.session_state['theme'] = 'dark'
                st.rerun()

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
        if st.button("← Back", use_container_width=True):
            st.session_state['page'] = 'input'
            st.rerun()

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

    # ── TABS: Risk Analysis + Vendor Comparison + AI Risk Analyst ──
    tab_analysis, tab_comparison, tab_chatbot = st.tabs(["Risk Analysis", "Vendor Comparison", "AI Risk Analyst"])

    with tab_analysis:
        render_risk_analysis_tab(result, selected_vendor, selected_product)

    with tab_comparison:
        render_vendor_comparison_tab(selected_vendor, selected_product)

    with tab_chatbot:
        render_chatbot_tab(result, selected_vendor, selected_product)


def render_risk_analysis_tab(result, selected_vendor, selected_product):
    """Content of the Risk Analysis tab (existing analytics content)."""

    # ── SECTION: Core Metrics ──
    st.markdown('<div class="section-header">Core Risk Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5, gap="small")

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
        
        sap_label = result.get('sap_risk_label', result.get('xgb_prediction', {}).get('sap_risk_class', 'Unknown'))
        divergence = result.get('sap_divergence', False)
        
        if divergence:
            badge_html = f'<div style="margin-top: 8px; font-size: 0.7rem; color: #f0b840; border: 1px solid rgba(240, 184, 64, 0.4); background: rgba(240, 184, 64, 0.1); padding: 4px 6px; border-radius: 4px; display: inline-block;">⚠️ SAP: {sap_label} (Divergence)</div>'
        else:
            badge_html = f'<div style="margin-top: 8px; font-size: 0.7rem; color: var(--text-mute); border: 1px solid var(--border-med); background: transparent; padding: 4px 6px; border-radius: 4px; display: inline-block;">SAP: {sap_label}</div>'

        st.markdown(f"""
        <div class="analytic-card">
            <div class="card-label">Risk Classification<br>Tier</div>
            <div class="card-value" style="color:{b_color};">{bucket}</div>
            {badge_html}
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
            pos_color = "var(--text-mute)"
        pv = result['price_variance']
        pv_pct = f"{(pv * 100):+.1f}%"
        pv_color = "#f85149" if pv > 0.15 else "#14f0a0" if pv < -0.05 else "var(--text-mute)"
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
            <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem; color:var(--text-mute); margin-top:8px; letter-spacing:0.03em;">
                HISTORICAL AVG (VENDOR): <span style="color:#14f0a0;">${result.get('vendor_historical_avg', 0):,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 5th Card: Vendor Tenure (Years Active) ──
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
    <div style="background: var(--bg-card); border: 1px solid var(--border-light); border-left: 3px solid {trend_color}; border-radius: 10px; padding: 20px 24px; margin-top: 32px; animation: fadeSlideUp 0.6s ease-out both;">
        <div style="font-family:'Roboto Mono',monospace; font-size:0.75rem; font-weight:600; color:{trend_color}; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;">
            <span style="font-size:1.1rem; margin-right:6px;">{trend_icon}</span> 24-Month Market Trend Analysis
        </div>
        <div style="font-family:'Inter',sans-serif; font-size:0.95rem; color:var(--text-main); line-height:1.5;">
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
                <div style="position:absolute; bottom:0; left:0; right:0; height:6px; background:var(--border-light); border-radius:3px;"></div>
                <!-- Percentile markers -->
                <div style="position:absolute; bottom:-2px; left:{pos_p25}%; width:2px; height:10px; background:var(--text-mute); transform:translateX(-50%);"></div>
                <div style="position:absolute; bottom:-2px; left:{pos_p50}%; width:2px; height:10px; background:var(--text-mute); transform:translateX(-50%);"></div>
                <div style="position:absolute; bottom:-2px; left:{pos_p75}%; width:2px; height:10px; background:var(--text-mute); transform:translateX(-50%);"></div>
                <div style="position:absolute; bottom:-2px; left:{pos_p90}%; width:2px; height:10px; background:var(--text-mute); transform:translateX(-50%);"></div>
            </div>
            <!-- Labels row -->
            <div style="position:relative; height:28px; font-family:'Roboto Mono',monospace; font-size:0.65rem; color:var(--text-mute);">
                <div style="position:absolute; left:{pos_p25}%; transform:translateX(-50%); text-align:center;">
                    <div>P25</div>
                    <div style="color:var(--text-main);">${p25:,.0f}</div>
                </div>
                <div style="position:absolute; left:{pos_p50}%; transform:translateX(-50%); text-align:center;">
                    <div>P50</div>
                    <div style="color:var(--text-main);">${p50:,.0f}</div>
                </div>
                <div style="position:absolute; left:{pos_p75}%; transform:translateX(-50%); text-align:center;">
                    <div>P75</div>
                    <div style="color:var(--text-main);">${p75:,.0f}</div>
                </div>
                <div style="position:absolute; left:{pos_p90}%; transform:translateX(-50%); text-align:center;">
                    <div>P90</div>
                    <div style="color:var(--text-main);">${p90:,.0f}</div>
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
                <p style="margin:0 0 6px 0; color:var(--text-mute); font-family:'Roboto Mono',monospace; font-size:0.75rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;">Clearance Timeline</p>
                <p style="margin:0; color:var(--text-main); font-size:0.95rem; font-family:'Inter',sans-serif;">
                    Supplier profiling indicates that <span style="color:var(--text-head); font-weight:600;">{selected_product}</span> orders from
                    <span style="color:var(--text-head); font-weight:600;">{selected_vendor}</span> usually take
                    <span style="color:{aging_color}; font-weight:700;">{result['avg_clearance_days']} days</span> to clear.
                </p>
            </div>
            <div style="text-align:right;">
                <span style="font-family:'Inter',sans-serif; font-size:2.8rem; font-weight:800; color:{aging_color}; line-height:1;">{result['avg_clearance_days']}</span>
                <span style="font-family:'Roboto Mono',monospace; font-size:0.9rem; color:{aging_color}; margin-left:4px;">d</span>
                <div style="color:var(--text-mute); font-family:'Roboto Mono',monospace; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; margin-top:4px;">Avg Clearance</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section separator ──
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SECTION: Vendor Health Scorecard ──
    st.markdown('<div class="section-header">Vendor Health Signals</div>',
                unsafe_allow_html=True)

    xgb_pred = result.get('xgb_prediction', {})

    years_active_h  = float(xgb_pred.get('years_active', 0))
    dunning_level   = int(xgb_pred.get('dunning_level', 0))
    is_blocked      = int(xgb_pred.get('is_payment_blocked', 0))
    reversal_rate_h = float(xgb_pred.get('reversal_rate', 0))

    # Color logic
    years_color_h   = '#14f0a0' if years_active_h >= 5 else '#f0b840' if years_active_h >= 2 else '#f85149'
    dunning_color   = '#14f0a0' if dunning_level == 0 else '#f0b840' if dunning_level <= 2 else '#f85149'
    block_color     = '#f85149' if is_blocked else '#14f0a0'
    reversal_color  = '#14f0a0' if reversal_rate_h < 0.05 else '#f0b840' if reversal_rate_h < 0.15 else '#f85149'

    block_label     = 'BLOCKED' if is_blocked else 'CLEAR'
    dunning_label   = f'{dunning_level} / 4'

    col_h1, col_h2, col_h3, col_h4 = st.columns(4, gap="small")

    for col, label, value, color, sublabel in [
        (col_h1, 'Years Active',    f'{years_active_h:.1f} yrs', years_color_h,
         'NEW VENDOR' if years_active_h < 2 else 'ESTABLISHED' if years_active_h >= 5 else 'GROWING'),
        (col_h2, 'Dunning Level',   dunning_label,              dunning_color,
         'NO NOTICES' if dunning_level == 0 else 'ESCALATED' if dunning_level >= 3 else 'WARNED'),
        (col_h3, 'SAP Block Status',block_label,                block_color,
         'PAYMENT BLOCKED' if is_blocked else 'NO BLOCKS'),
        (col_h4, 'Reversal Rate',   f'{reversal_rate_h:.1%}',   reversal_color,
         'HIGH REVERSALS' if reversal_rate_h > 0.15 else 'ACCEPTABLE' if reversal_rate_h > 0.05 else 'CLEAN'),
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

    # ── Section separator ──
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── SECTION: Risk Visualizations ──
    st.markdown('<div class="section-header">Risk Visualizations</div>', unsafe_allow_html=True)

    viz_col1, viz_col2 = st.columns(2, gap="medium")

    is_light = st.session_state.get('theme', 'light') == 'light'
    if is_light:
        radar_grid_color = 'rgba(0,0,0,0.1)'
        radar_line_color = 'rgba(0,0,0,0.1)'
        radar_tick_color = '#334155'
        radar_text_color = '#334155'
        radar_bg_color = 'rgba(0,0,0,0.03)'
    else:
        radar_grid_color = 'rgba(255,255,255,0.25)'  # brighter
        radar_line_color = 'rgba(255,255,255,0.30)'  # brighter
        radar_tick_color = '#c8d8f0'
        radar_text_color = '#c8d8f0'
        radar_bg_color = 'rgba(255,255,255,0.02)'

    with viz_col1:
        # Determine gauge bar color based on risk level
        fr = result['final_risk']
        if fr < 0.3:
            gauge_color = '#14f0a0'  # green — low risk
            gauge_label = "LOW RISK"
        elif fr < 0.7:
            gauge_color = '#f0b840'  # amber — medium risk
            gauge_label = "MEDIUM RISK"
        else:
            gauge_color = '#f85149'  # red — high/critical risk
            gauge_label = "HIGH RISK"

        if is_light:
            bar_color_rgba = 'rgba(0,0,0,0)'
            step1_color = 'rgba(20, 240, 160, 0.85)' if fr < 0.3 else 'rgba(20, 240, 160, 0.12)'
            step2_color = 'rgba(240, 184, 64, 0.85)' if 0.3 <= fr < 0.7 else 'rgba(240, 184, 64, 0.12)'
            step3_color = 'rgba(248, 81, 73, 0.85)' if fr >= 0.7 else 'rgba(248, 81, 73, 0.12)'
            threshold_dict = {}
        else:
            bar_color_rgba = gauge_color
            step1_color = 'rgba(20, 240, 160, 0.25)'
            step2_color = 'rgba(240, 184, 64, 0.25)'
            step3_color = 'rgba(248, 81, 73, 0.25)'
            threshold_dict = {
                'line': {'color': gauge_color, 'width': 4},
                'thickness': 0.85,
                'value': fr
            }

        gauge_config = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': 'var(--border-med)'},
            'bar': {'color': bar_color_rgba, 'thickness': 0.35},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 1,
            'bordercolor': 'var(--border-light)',
            'steps': [
                {'range': [0, 0.3], 'color': step1_color},
                {'range': [0.3, 0.7], 'color': step2_color},
                {'range': [0.7, 1.0], 'color': step3_color}
            ]
        }
        if threshold_dict:
            gauge_config['threshold'] = threshold_dict

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge",
            value=fr,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Final Risk Score", 'font': {'color': 'var(--text-mute)', 'family': 'Roboto Mono', 'size': 13}},
            gauge=gauge_config
        ))
        
        # Explicit annotation prevents text drift on zoom
        fig_gauge.add_annotation(
            text=f"{fr:.2f}",
            showarrow=False,
            font=dict(color='var(--text-head)', family='Inter', size=40),
            x=0.5, y=0.15,
            xanchor='center', yanchor='bottom'
        )
        # Add risk level label
        fig_gauge.add_annotation(
            text=gauge_label,
            showarrow=False,
            font=dict(color=gauge_color, family='Roboto Mono', size=13),
            x=0.5, y=0.10,
            xanchor='center', yanchor='top'
        )

        fig_gauge.update_layout(
            height=320,
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor=st.session_state.get('paper_bg', 'rgba(0,0,0,0)'),
            plot_bgcolor=st.session_state.get('plot_bg', 'rgba(10,14,26,0.6)'),
            font={'family': "Inter, sans-serif", 'color': st.session_state.get('text_main', 'var(--text-main)')},
            transition={'duration': 1200, 'easing': 'cubic-in-out'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Gauge Intelligence Summary ──
        st.html(f"""
<div style="border-left:3px solid {gauge_color}; padding:14px 18px;
            margin-top:12px; border-radius:0 8px 8px 0;
            background:{radar_bg_color};">
    <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem;
                color:#8a9ab8; letter-spacing:0.1em; margin-bottom:6px;">
        // RISK SCORE SUMMARY
    </div>
    <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
                color:{radar_text_color}; line-height:1.5;">
        This combined risk score reflects overall vendor unreliability, aggregating late payments,
        historical defaults, and SAP flags. A high score suggests immediate mitigation is required before further procurement.
    </div>
</div>
""")

    with viz_col2:
        # ── Payment Discipline Radar Chart ──
        xgb_radar = result['xgb_prediction']

        axes = ['Late Payment\nRatio', 'Payment\nConsistency', 'Discount\nCapture Risk',
                'Voided\nPayments', 'Reversal\nRate']

        values_radar = [
            float(xgb_radar.get('late_ratio', 0)),
            min(float(xgb_radar.get('payment_consistency', 0)) / 30.0, 1.0),
            1.0 - float(xgb_radar.get('discount_capture_rate', 0.5)),
            float(xgb_radar.get('voided_payment_rate', 0)),
            float(xgb_radar.get('reversal_rate', 0)),
        ]
        values_closed = values_radar + [values_radar[0]]
        axes_closed   = axes  + [axes[0]]

        # Color based on vendor_bucket
        bucket_radar = result.get('vendor_bucket', 'MEDIUM')
        radar_color = {
            'LOW':      '#14f0a0',
            'MEDIUM':   '#f0b840',
            'HIGH':     '#f85149',
            'CRITICAL': '#ff0000',
        }.get(bucket_radar, '#f0b840')

        # Build rgba fill color from hex
        r_hex = int(radar_color[1:3], 16)
        g_hex = int(radar_color[3:5], 16)
        b_hex = int(radar_color[5:7], 16)
        fill_rgba = f'rgba({r_hex},{g_hex},{b_hex},0.15)'

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=axes_closed,
            fill='toself',
            fillcolor=fill_rgba,
            line=dict(color=radar_color, width=2),
            name='Payment Discipline',
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(10,14,26,0.0)',
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=9, color=radar_tick_color),
                    gridcolor=radar_grid_color,
                    linecolor=radar_line_color,
                ),
                angularaxis=dict(
                    tickfont=dict(size=10, color=radar_tick_color),
                    gridcolor=radar_grid_color,
                    linecolor=radar_line_color,
                ),
            ),
            paper_bgcolor=st.session_state.get('paper_bg', 'rgba(0,0,0,0)'),
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

        # ── Radar Intelligence Summary ──
        st.html(f"""
<div style="border-left:3px solid {radar_color}; padding:14px 18px;
            margin-top:12px; border-radius:0 8px 8px 0;
            background:{radar_bg_color};">
    <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem;
                color:#8a9ab8; letter-spacing:0.1em; margin-bottom:6px;">
        // CHART INTELLIGENCE SUMMARY
    </div>
    <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
                color:{radar_text_color}; line-height:1.5;">
        Each axis shows a distinct dimension of payment behaviour (0 = best,
        1 = worst). A wide polygon indicates systemic payment problems across
        multiple dimensions. A narrow polygon concentrated on one axis suggests
        a single specific issue rather than general unreliability.
    </div>
</div>
""")

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
            number={'font': {'color': 'var(--text-head)', 'family': 'Inter'}},
            title={'text': "<b>Vendor Reliability Risk</b><br><span style='font-size:0.8em;color:var(--text-mute)'>Based on internal transaction history</span>"},
            gauge={
                'shape': "bullet",
                'axis': {'range': [min(0.0, result['vendor_risk'] - 0.1), 1]},
                'threshold': {'line': {'color': v_color, 'width': 4}, 'thickness': 0.75, 'value': result['vendor_risk']},
                'bar': {'color': v_color},
                'bgcolor': 'var(--bg-hover)',
            }
        ))
        fig_v.update_layout(height=150, margin=dict(t=30, b=20, l=200, r=20), paper_bgcolor=st.session_state.get('paper_bg', 'rgba(0,0,0,0)'), font={'color': "var(--text-main)", 'family': 'Inter'})
        st.plotly_chart(fig_v, use_container_width=True)

    with insight_col2:
        variance = result['price_variance']
        p_color = "#14f0a0" if variance < 0.3 else "#f85149"
        fig_p = go.Figure(go.Indicator(
            mode="number+gauge",
            value=variance,
            domain={'x': [0.1, 1], 'y': [0, 1]},
            number={'font': {'color': 'var(--text-head)', 'family': 'Inter'}},
            title={'text': "<b>Market Price Variance</b><br><span style='font-size:0.8em;color:var(--text-mute)'>Deviation from average</span>"},
            gauge={
                'shape': "bullet",
                'axis': {'range': [min(0.0, variance - 0.1), max(1.0, variance + 0.2)]},
                'threshold': {'line': {'color': p_color, 'width': 4}, 'thickness': 0.75, 'value': variance},
                'bar': {'color': p_color},
                'bgcolor': 'var(--bg-hover)',
            }
        ))
        fig_p.update_layout(height=150, margin=dict(t=30, b=20, l=200, r=20), paper_bgcolor=st.session_state.get('paper_bg', 'rgba(0,0,0,0)'), font={'color': "var(--text-main)", 'family': 'Inter'})
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
    
    # Split into positive (risk drivers) and negative (risk mitigators)
    positive_shap = {k: v for k, v in shap_vals.items() if v > 0}
    negative_shap = {k: v for k, v in shap_vals.items() if v < 0}

    # Sort each separately
    pos_sorted = sorted(positive_shap.items(), key=lambda x: x[1], reverse=True)
    neg_sorted = sorted(negative_shap.items(), key=lambda x: x[1])

    # Take top 3 positive and top 2 negative
    top_pos = dict(pos_sorted[:3])
    top_neg = dict(neg_sorted[:2])
    
    # Sum the remaining into "Other Factors"
    other_pos = sum(v for k, v in pos_sorted[3:])
    other_neg = sum(v for k, v in neg_sorted[2:])

    top_shap = {**top_pos, **top_neg}
    if other_pos > 0.0001:
        top_shap["Other Risk Drivers"] = other_pos
    if other_neg < -0.0001:
        top_shap["Other Risk Mitigators"] = other_neg
    
    # Sort final for visual descending order
    top_shap = dict(sorted(top_shap.items(), key=lambda x: x[1], reverse=True))

    # Reverse for plotting (Plotly horizontal bar charts draw from bottom up)
    shap_labels = list(reversed(list(top_shap.keys())))
    shap_values_list = list(reversed(list(top_shap.values())))
    shap_colors = ['#f85149' if v > 0 else '#14f0a0' for v in shap_values_list]

    # Compact chart height since we limit to max 6 bars
    shap_chart_height = 240

    is_light = st.session_state.get('theme', 'light') == 'light'
    shap_grid_color = 'rgba(0,0,0,0.05)' if is_light else 'rgba(255,255,255,0.03)'
    shap_text_color = 'var(--text-main)' if is_light else '#ffffff'

    fig_shap = go.Figure(go.Bar(
        x=shap_values_list,
        y=shap_labels,
        orientation='h',
        marker_color=shap_colors,
        text=[f"{v:+.4f}" for v in shap_values_list],
        textposition='outside',
        textfont=dict(color=shap_text_color, family='Roboto Mono', size=11),
    ))
    fig_shap.update_layout(
        height=shap_chart_height,
        margin=dict(l=20, r=60, t=10, b=10),
        paper_bgcolor=st.session_state.get('paper_bg', 'rgba(0,0,0,0)'),
        plot_bgcolor=st.session_state.get('plot_bg', 'rgba(10,14,26,0.6)'),
        font=dict(color='var(--text-mute)', family='Inter'),
        xaxis=dict(
            title="SHAP Value (impact on prediction)",
            gridcolor=shap_grid_color,
            zerolinecolor='var(--border-med)',
            title_font=dict(size=11, color='var(--text-mute)', family='Roboto Mono'),
        ),
        yaxis=dict(gridcolor=shap_grid_color),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    with st.expander("View All Feature Impacts"):
        sorted_all = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
        all_shap_labels = list(reversed(list(dict(sorted_all).keys())))
        all_shap_values = list(reversed(list(dict(sorted_all).values())))
        all_shap_colors = ['#f85149' if v > 0 else '#14f0a0' for v in all_shap_values]
        
        all_chart_height = max(220, len(all_shap_labels) * 30 + 40)
        
        fig_shap_all = go.Figure(go.Bar(
            x=all_shap_values,
            y=all_shap_labels,
            orientation='h',
            marker_color=all_shap_colors,
            text=[f"{v:+.4f}" for v in all_shap_values],
            textposition='outside',
            textfont=dict(color=shap_text_color, family='Roboto Mono', size=11),
        ))
        fig_shap_all.update_layout(
            height=all_chart_height,
            margin=dict(l=20, r=60, t=10, b=10),
            paper_bgcolor=st.session_state.get('paper_bg', 'rgba(0,0,0,0)'),
            plot_bgcolor=st.session_state.get('plot_bg', 'rgba(10,14,26,0.6)'),
            font=dict(color='var(--text-mute)', family='Inter'),
            xaxis=dict(
                title="SHAP Value (impact on prediction)",
                gridcolor=shap_grid_color,
                zerolinecolor='var(--border-med)',
                title_font=dict(size=11, color='var(--text-mute)', family='Roboto Mono'),
            ),
            yaxis=dict(gridcolor=shap_grid_color),
        )
        st.plotly_chart(fig_shap_all, use_container_width=True)

    # ── SHAP Intelligence Summary ──
    positive_shap = {k: v for k, v in shap_vals.items() if v > 0}
    negative_shap = {k: v for k, v in shap_vals.items() if v < 0}

    primary_driver = max(shap_vals, key=lambda k: shap_vals[k]) if positive_shap else None
    risk_mitigator = min(shap_vals, key=lambda k: shap_vals[k]) if negative_shap else None

    driver_text = f"Primary risk driver: <strong style='color:var(--text-head);'>{primary_driver}</strong> (SHAP: <span style='color:#f85149; font-weight:700;'>+{shap_vals[primary_driver]:.4f}</span>)." if primary_driver else "No features are actively increasing risk for this vendor."
    mitigator_text = f" Risk mitigator: <strong style='color:var(--text-head);'>{risk_mitigator}</strong> (SHAP: <span style='color:#14f0a0; font-weight:700;'>{shap_vals[risk_mitigator]:.4f}</span>)." if risk_mitigator else ""

    if primary_driver and abs(shap_vals.get(primary_driver, 0)) > abs(shap_vals.get(risk_mitigator, 0) if risk_mitigator else 0):
        verdict = f"The model's classification is primarily driven by <strong style='color:var(--text-head);'>{primary_driver}</strong>."
    elif risk_mitigator:
        verdict = f"<strong style='color:var(--text-head);'>{risk_mitigator}</strong> is the strongest factor pulling risk down for this vendor."
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
    <div style="font-family:'Inter',sans-serif; font-size:0.85rem; color:var(--text-main); line-height:1.5;">
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
        is_light = st.session_state.get('theme', 'light') == 'light'
        if is_light:
            iso_bar_color_rgba = 'rgba(0,0,0,0)'
            iso_step1_color = 'rgba(20, 240, 160, 0.85)' if iso_score <= 0.5 else 'rgba(20, 240, 160, 0.12)'
            iso_step2_color = 'rgba(240, 184, 64, 0.85)' if 0.5 < iso_score <= 0.75 else 'rgba(240, 184, 64, 0.12)'
            iso_step3_color = 'rgba(248, 81, 73, 0.85)' if iso_score > 0.75 else 'rgba(248, 81, 73, 0.12)'
            iso_threshold_dict = {}
        else:
            iso_bar_color_rgba = iso_color
            iso_step1_color = 'rgba(20, 240, 160, 0.25)'
            iso_step2_color = 'rgba(240, 184, 64, 0.25)'
            iso_step3_color = 'rgba(248, 81, 73, 0.25)'
            iso_threshold_dict = {
                'line': {'color': iso_color, 'width': 4},
                'thickness': 0.85,
                'value': iso_score
            }

        iso_gauge_config = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': 'var(--border-med)'},
            'bar': {'color': iso_bar_color_rgba, 'thickness': 0.35},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 1,
            'bordercolor': 'var(--border-light)',
            'steps': [
                {'range': [0, 0.5], 'color': iso_step1_color},
                {'range': [0.5, 0.75], 'color': iso_step2_color},
                {'range': [0.75, 1.0], 'color': iso_step3_color}
            ]
        }
        if iso_threshold_dict:
            iso_gauge_config['threshold'] = iso_threshold_dict

        fig_iso = go.Figure(go.Indicator(
            mode="gauge",
            value=iso_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Population Outlier Score", 'font': {'color': 'var(--text-mute)', 'family': 'Roboto Mono', 'size': 13}},
            gauge=iso_gauge_config
        ))
        
        # Explicit annotation prevents text drift on zoom
        fig_iso.add_annotation(
            text=f"{iso_score:.3f}",
            showarrow=False,
            font=dict(color='var(--text-head)', family='Inter', size=40),
            x=0.5, y=0.15,
            xanchor='center', yanchor='bottom'
        )
        # Add isolation level label
        fig_iso.add_annotation(
            text=iso_label,
            showarrow=False,
            font=dict(color=iso_color, family='Roboto Mono', size=13),
            x=0.5, y=0.10,
            xanchor='center', yanchor='top'
        )

        fig_iso.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor=st.session_state.get('paper_bg', 'rgba(0,0,0,0)'),
            plot_bgcolor=st.session_state.get('plot_bg', 'rgba(10,14,26,0.6)'),
            font={'family': "Inter, sans-serif", 'color': st.session_state.get('text_main', 'var(--text-main)')},
            transition={'duration': 1200, 'easing': 'cubic-in-out'}
        )
        st.plotly_chart(fig_iso, use_container_width=True)

    # ── Feature Deviation Breakdown ──
    feature_deviations = result.get("feature_deviations", [])

    if feature_deviations:
        # Sort feature deviations by absolute z-score descending
        feature_deviations = sorted(feature_deviations, key=lambda x: abs(x["z_score"]), reverse=True)
        
        top_n = 3
        top_deviations = feature_deviations[:top_n]
        other_deviations = feature_deviations[top_n:]

        def build_deviation_html(deviations_list):
            html = ""
            for fd in deviations_list:
                level = fd["level"]
                z = fd["z_score"]
                # Color coding per level
                if level == "EXTREME":
                    bar_color = "#f85149"
                    level_color = "#f85149"
                elif level == "HIGH":
                    bar_color = "#f0b840"
                    level_color = "#f0b840"
                elif level == "MODERATE":
                    bar_color = "#50a0ff"
                    level_color = "#50a0ff"
                else:
                    bar_color = "#14f0a0"
                    level_color = "#14f0a0"

                # Compute bar width (clamp z-score to 0-4 range, map to 0-100%)
                bar_pct = min(abs(z) / 4.0, 1.0) * 100
                direction = "above" if z > 0 else "below"

                # Format vendor value nicely
                v = fd["vendor_val"]
                m = fd["pop_mean"]
                if fd["feature"] == "Total Spend Volume" or fd["feature"] == "Open Exposure":
                    val_fmt = f"${v:,.0f}"
                    mean_fmt = f"${m:,.0f}"
                elif fd["feature"] in ("Late Payment Ratio", "Reversal Rate", "Discount Capture Rate", "Voided Payment Rate", "Stale Payment Rate"):
                    val_fmt = f"{v:.1%}"
                    mean_fmt = f"{m:.1%}"
                elif fd["feature"] in ("Payment Consistency Score", "Years Active", "Dunning Level", "Payment Terms Risk", "Is Payment Blocked"):
                    val_fmt = f"{v:.1f}"
                    mean_fmt = f"{m:.1f}"
                else:
                    val_fmt = f"{v:.1f} days"
                    mean_fmt = f"{m:.1f} days"

                html += f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-family:'Inter',sans-serif; font-size:0.8rem; color:var(--text-main);">
                            {fd['feature']}
                        </span>
                        <span style="font-family:'Roboto Mono',monospace; font-size:0.62rem;
                                    letter-spacing:0.08em; color:{level_color}; padding:2px 8px;
                                    background:rgba({','.join(str(int(level_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.12);
                                    border-radius:4px;">
                            {level}
                        </span>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <div style="flex:1; height:6px; background:var(--border-light); border-radius:3px; overflow:hidden;">
                            <div style="width:{bar_pct}%; height:100%; background:{bar_color};
                                        border-radius:3px; transition:width 1.2s cubic-bezier(0.4,0,0.2,1);"></div>
                        </div>
                        <span style="font-family:'Roboto Mono',monospace; font-size:0.68rem; color:#8a9ab8;
                                    min-width:60px; text-align:right;">
                            z = {z:+.1f}
                        </span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:3px;">
                        <span style="font-family:'Roboto Mono',monospace; font-size:0.62rem; color:#6b7b98;">
                            Vendor: <span style="color:var(--text-main);">{val_fmt}</span>
                        </span>
                        <span style="font-family:'Roboto Mono',monospace; font-size:0.62rem; color:#6b7b98;">
                            Pop. Mean: {mean_fmt}
                        </span>
                    </div>
                </div>
                """
            return html

        deviation_html_top = build_deviation_html(top_deviations)
        deviation_html_other = build_deviation_html(other_deviations) if other_deviations else ""

        # Generate natural language summary
        unusual = [fd for fd in feature_deviations if fd["level"] in ("HIGH", "EXTREME")]
        moderate = [fd for fd in feature_deviations if fd["level"] == "MODERATE"]
        normal = [fd for fd in feature_deviations if fd["level"] == "NORMAL"]

        if unusual:
            standout_names = " and ".join([f'<span style="color:{("#f85149" if fd["level"]=="EXTREME" else "#f0b840")}">{fd["feature"]}</span>' for fd in unusual])
            if normal:
                normal_names = ", ".join([fd["feature"] for fd in normal])
                nl_summary = f"Although {normal_names} {'are' if len(normal) > 1 else 'is'} within normal population bounds, {standout_names} {'are' if len(unusual) > 1 else 'is'} a statistically significant outlier — this specific combination is what makes this vendor stand out from the population."
            else:
                nl_summary = f"{standout_names} {'deviate' if len(unusual) > 1 else 'deviates'} significantly from population norms — this is the primary driver behind the anomaly detection flag."
        elif moderate:
            mod_names = " and ".join([fd["feature"] for fd in moderate])
            nl_summary = f"No extreme deviations detected, but {mod_names} {'show' if len(moderate) > 1 else 'shows'} moderate variance from population averages. The vendor is not an outlier, but these features are worth monitoring over time."
        else:
            nl_summary = "All features fall within normal population bounds. This vendor's behaviour is statistically typical — no single feature or combination of features deviates meaningfully from the norm."

        st.html(f"""
<div style="background:{'rgba(248,81,73,0.03)' if is_outlier else 'rgba(20,240,160,0.03)'};
            border-left:3px solid {iso_color}; padding:16px 20px; margin-top:16px;
            margin-bottom:16px; border-radius:0 8px 8px 0;">
    <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem;
                color:#8a9ab8; letter-spacing:0.1em; margin-bottom:12px;">
        // FEATURE DEVIATION ANALYSIS — POPULATION COMPARISON
    </div>
    {deviation_html_top}
</div>
""")
        
        if deviation_html_other:
            with st.expander("View All Feature Deviations"):
                st.html(f"""
                <div style="padding-top: 10px;">
                    {deviation_html_other}
                </div>
                """)

        st.html(f"""
<div style="background:{'rgba(248,81,73,0.03)' if is_outlier else 'rgba(20,240,160,0.03)'};
            border-left:3px solid {iso_color}; padding:16px 20px; margin-top:0px;
            margin-bottom:16px; border-radius:0 8px 8px 0;">
    <div style="padding-top:0px;">
        <div style="font-family:'Roboto Mono',monospace; font-size:0.62rem;
                    color:#6b7b98; letter-spacing:0.08em; margin-bottom:6px;">
            // AI INTERPRETATION
        </div>
        <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
                    color:var(--text-main); line-height:1.6;">
            {nl_summary}
        </div>
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
    <p style="color:#8a9ab8; font-size:0.82rem; font-family:'Roboto Mono',monospace; letter-spacing:0.05em; margin-bottom:6px;">
        Historical Risk Fluctuation (Past 12 Months) — Calculated from <span style="color:var(--text-main);">{past_dealings_count}</span> past dealings
    </p>
    <p style="color:#6b7b98; font-size:0.62rem; font-family:'Roboto Mono',monospace; letter-spacing:0.04em;
              margin-bottom:16px; padding:6px 10px; background:var(--bg-hover); border-radius:4px;
              border-left:2px solid rgba(240,184,64,0.3);">
        ⚠ SIMULATED DATA — This timeline is synthetically generated around the vendor's real XGBoost risk score ({vendor_history_data.get('monthly_risk_history', [0])[0]:.2f} baseline).
        It illustrates plausible risk fluctuation patterns but does not reflect actual month-by-month historical records.
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
        marker=dict(size=7, color='#50a0ff', symbol='diamond', line=dict(color='var(--bg-app)', width=1.5)),
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
        yaxis=dict(range=[0, 1], gridcolor='var(--bg-hover)', zerolinecolor='var(--bg-hover)'),
        xaxis=dict(gridcolor='var(--bg-hover)'),
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=st.session_state.get('paper_bg', 'rgba(0,0,0,0)'),
        plot_bgcolor=st.session_state.get('plot_bg', 'rgba(10,14,26,0.6)'),
        font={'color': 'var(--text-mute)', 'family': 'Inter'}
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
    <div style="font-family:'Inter',sans-serif; font-size:0.85rem; color:var(--text-main); line-height:1.5;">
        Analysis of the past 12 months indicates an <strong style="color:var(--text-head);">{trend}</strong> risk trend, 
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
    <p style="color:#8a9ab8; font-size:0.82rem; font-family:'Roboto Mono',monospace; letter-spacing:0.05em; margin-bottom:6px;">
        Order Volume Pattern Analysis — Detecting abnormal quotation quantities
    </p>
    <p style="color:#6b7b98; font-size:0.62rem; font-family:'Roboto Mono',monospace; letter-spacing:0.04em;
              margin-bottom:16px; padding:6px 10px; background:var(--bg-hover); border-radius:4px;
              border-left:2px solid rgba(240,184,64,0.3);">
        ⚠ SIMULATED DATA — Order volumes are synthetically generated based on the vendor's risk profile.
        Higher-risk vendors produce more volume anomalies (spikes above threshold). This demonstrates the anomaly detection
        concept but does not reflect actual purchase order history.
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
            line=dict(color='var(--border-med)', width=1),
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
        yaxis=dict(gridcolor='var(--bg-hover)', zerolinecolor='var(--bg-hover)'),
        xaxis=dict(gridcolor='var(--bg-hover)'),
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=st.session_state.get('paper_bg', 'rgba(0,0,0,0)'),
        plot_bgcolor=st.session_state.get('plot_bg', 'rgba(10,14,26,0.6)'),
        font={'color': 'var(--text-mute)', 'family': 'Inter'}
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
    <div style="font-family:'Inter',sans-serif; font-size:0.85rem; color:var(--text-main); line-height:1.5;">
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

    if not selected_score:
        st.error("Selected vendor data not found in comparison results. Please re-run the analysis.")
        return

    selected_rank = selected_score["rank"]
    selected_final_risk = selected_score["final_risk"]

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
            <p style="color:var(--text-main); font-family:'Inter',sans-serif; font-size:0.88rem; margin:0;">
                <span style="color:var(--text-head); font-weight:600;">{selected_vendor}</span> carries elevated risk.
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
            <p style="color:var(--text-main); font-family:'Inter',sans-serif; font-size:0.88rem; margin:0;">
                <span style="color:var(--text-head); font-weight:600;">{selected_vendor}</span> has a moderate risk profile or is outranked.
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
            <p style="color:var(--text-main); font-family:'Inter',sans-serif; font-size:0.88rem; margin:0;">
                You have selected one of the lowest-risk vendors for this transaction.
            </p>
        </div>
    </div>
</div>
""")

    st.html('<div class="section-header">Vendor Rankings — All Vendors Compared</div>')

    st.html(f"""
<p style="color:#8a9ab8; font-size:0.82rem; font-family:'Roboto Mono',monospace; letter-spacing:0.05em; margin-bottom:24px;">
    Ranked by overall risk score for <span style="color:var(--text-main);">{selected_product}</span> — lowest risk = best
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
            border_color = "var(--border-light)"
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

        # Right: Progress Bar + Badge
        sap_label = s.get('sap_risk_label', s.get('xgb_prediction', {}).get('sap_risk_class', 'Unknown'))
        sap_divergence = s.get('sap_divergence', False)
        
        if sap_divergence:
            sap_badge = f'<div style="background:rgba(240,184,64,0.1); border:1px solid rgba(240,184,64,0.4); color:#f0b840; padding:3px 10px; border-radius:6px; font-family:\'Inter\',sans-serif; font-size:0.65rem; font-weight:600; letter-spacing:0.05em; margin-top:6px;">⚠️ SAP: {sap_label}</div>'
        else:
            sap_badge = f'<div style="background:transparent; border:1px solid var(--border-light); color:var(--text-mute); padding:3px 10px; border-radius:6px; font-family:\'Inter\',sans-serif; font-size:0.65rem; font-weight:600; letter-spacing:0.05em; margin-top:6px;">SAP: {sap_label}</div>'

        st.html(f"""
<div style="background:var(--bg-card); border:{border_width} solid {border_color}; border-radius:14px;
            padding:24px 28px; margin-bottom:16px; position:relative; overflow:hidden;
            animation: fadeSlideUp 0.6s ease-out both; animation-delay: {rank_num * 0.08}s;">
    <div style="position:absolute; top:0; left:0; right:0; height:1px;
                background:linear-gradient(90deg, transparent, {'rgba(20,240,160,0.3)' if is_selected else 'var(--border-light)'}, transparent);"></div>

    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
        <!-- Left: Rank + Name -->
        <div style="display:flex; align-items:center; gap:16px; min-width:280px;">
            <div style="width:44px; height:44px; border-radius:10px; background:rgba({int(rank_color[1:3],16)},{int(rank_color[3:5],16)},{int(rank_color[5:7],16)},0.1);
                        display:flex; align-items:center; justify-content:center;
                        font-family:'Inter',sans-serif; font-size:1.2rem; font-weight:800; color:{rank_color};">
                #{rank_num}
            </div>
            <div>
                <div style="font-family:'Inter',sans-serif; font-size:1.05rem; font-weight:700; color:var(--text-head);">
                    {vendor_name_map.get(s['vendor_id'], s['vendor_id'])}{selected_tag}{recommended_tag}
                </div>
                <div style="font-family:'Roboto Mono',monospace; font-size:0.65rem; color:var(--text-mute); letter-spacing:0.08em; margin-top:4px;">
                    {s['vendor_id']} &nbsp;|&nbsp; AVG CLEARANCE: {s['avg_clearance_days']}d &nbsp;|&nbsp; QUOTED: <span style="color:var(--text-main);">${s['vendor_raw_price']:,.2f}</span>
                </div>
            </div>
        </div>

        <!-- Center: Score Metrics -->
        <div style="display:flex; gap:32px; align-items:center;">
            <div style="text-align:center;">
                <div style="font-family:'Roboto Mono',monospace; font-size:0.58rem; color:var(--text-mute); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">Overall Risk</div>
                <div style="font-family:'Inter',sans-serif; font-size:1.4rem; font-weight:800; color:{bar_color};">{fr:.2f}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Roboto Mono',monospace; font-size:0.58rem; color:var(--text-mute); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">Vendor Risk</div>
                <div style="font-family:'Inter',sans-serif; font-size:1.1rem; font-weight:700; color:var(--text-main);">{s['vendor_risk']:.2f}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Roboto Mono',monospace; font-size:0.58rem; color:var(--text-mute); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">Price Risk</div>
                <div style="font-family:'Inter',sans-serif; font-size:1.1rem; font-weight:700; color:var(--text-main);">{s['price_risk']:.1f}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Roboto Mono',monospace; font-size:0.58rem; color:var(--text-mute); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">Outlier Score</div>
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
            {sap_badge}
            <div style="width:100%; height:6px; background:var(--border-light); border-radius:3px; overflow:hidden;">
                <div style="width:{fr * 100}%; height:100%; background:{bar_color}; border-radius:3px;
                            transition: width 1.2s ease;"></div>
            </div>
        </div>
    </div>
</div>
""")


def render_chatbot_tab(result, selected_vendor, selected_product):
    st.markdown("""
    <style>
    /* Darken standard Streamlit buttons in the Chatbot tab */
    .stTabs .stButton > button {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-med) !important;
        color: var(--text-main) !important;
        border-radius: 8px !important;
    }
    .stTabs .stButton > button:hover {
        border-color: #14f0a0 !important;
        color: #14f0a0 !important;
        background-color: rgba(20, 240, 160, 0.05) !important;
    }
    
    /* Processing Indicator Animations */
    .ai-processing-ring {
        width: 14px;
        height: 14px;
        border: 2px solid rgba(20, 240, 160, 0.2);
        border-top-color: #14f0a0;
        border-radius: 50%;
        animation: ai-spin 1s linear infinite;
    }
    @keyframes ai-spin { 100% { transform: rotate(360deg); } }
    @keyframes ai-pulse { 50% { opacity: 0.5; } }
    
    /* Risk-based Vendor Button Borders */
    div.element-container:has(.risk-marker-LOW) + div.element-container .stButton > button {
        border-left: 4px solid #14f0a0 !important;
    }
    div.element-container:has(.risk-marker-MEDIUM) + div.element-container .stButton > button {
        border-left: 4px solid #f0b840 !important;
    }
    div.element-container:has(.risk-marker-HIGH) + div.element-container .stButton > button {
        border-left: 4px solid #f85149 !important;
    }
    
    /* Hide the marker containers to prevent empty gaps */
    div.element-container:has(.risk-marker-LOW),
    div.element-container:has(.risk-marker-MEDIUM),
    div.element-container:has(.risk-marker-HIGH) {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    all_scores = st.session_state.get('all_vendor_scores', [])
    
    col_left, col_right = st.columns([1, 2.5], gap="large")
    
    with col_left:
        st.markdown("""
        <div style="font-family:'Roboto Mono',monospace; font-size:0.75rem; font-weight:600; color:var(--text-mute); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:16px;">
            Alternative Vendors
        </div>
        """, unsafe_allow_html=True)
        
        sap_vendors = get_sap_vendor_list()
        vendor_name_map = {v['lifnr']: v['name'] for v in sap_vendors}
        
        with st.container(height=650):
            for s in all_scores:
                v_name = vendor_name_map.get(s["vendor_id"], s["vendor_id"])
                is_selected = s["vendor_id"] == result.get("vendor_id")
                rank = s.get("rank", "-")
                bucket = s.get("vendor_bucket", "N/A")
                score = s.get("final_risk", 0.0)
                
                selected_text = " ✓" if is_selected else ""
                btn_label = f"#{rank} {v_name} | {bucket} | {score:.2f}{selected_text}"
                
                st.markdown(f'<div class="risk-marker-{bucket}"></div>', unsafe_allow_html=True)
                if st.button(btn_label, key=f"chat_vendor_{s['vendor_id']}", use_container_width=True):
                    if is_selected:
                        st.session_state['chat_prefill'] = f"Give me a full breakdown of {v_name}'s risk"
                    else:
                        st.session_state['chat_prefill'] = f"Compare {selected_vendor} with {v_name}"
                    st.rerun()

    with col_right:
        st.markdown('<div class="section-header">Chat with AI Risk Analyst</div>', unsafe_allow_html=True)
        
        if "chat_history" not in st.session_state:
            final_risk = result.get('final_risk', 0)
            decision = result.get('decision', 'N/A')
            greeting = f"Hello! I am your AI Risk Analyst. I have reviewed the data for **{selected_vendor}**. Their final risk score is **{final_risk:.2f}** ({decision}). How can I help you analyze this vendor or compare alternatives?"
            st.session_state["chat_history"] = [{"role": "assistant", "content": greeting}]
            
        with st.container(height=500):
            for msg in st.session_state["chat_history"]:
                if msg["role"] == "user":
                    col1, col2 = st.columns([1, 5])
                    with col2:
                        content_text = msg['content'].replace('$', '\\$').replace(chr(10), '<br>')
                        st.markdown(f'''<div style="background: rgba(80, 160, 255, 0.1); border: 1px solid rgba(80, 160, 255, 0.3); border-radius: 12px; padding: 12px 16px; margin-bottom: 16px; text-align: right; color: var(--text-main); line-height: 1.5; font-family: 'Inter', sans-serif; font-size: 0.9rem;">
{content_text}
</div>''', unsafe_allow_html=True)
                else:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        content_text = msg['content'].replace('$', '\\$')
                        st.markdown(f'''<div class="analytic-card" style="padding: 16px 20px; margin-bottom: 16px;">
<div style="color: #14f0a0; font-family: 'Roboto Mono', monospace; font-size: 0.7rem; margin-bottom: 12px; letter-spacing: 0.1em;">AI RISK ANALYST</div>

{content_text}

</div>''', unsafe_allow_html=True)
            
        prefill = st.session_state.get('chat_prefill')
        user_input = st.chat_input("Type your question here...")

        if prefill or user_input:
            prompt_text = prefill if prefill else user_input
            if prefill:
                st.session_state['chat_prefill'] = None
            
            st.session_state["chat_history"].append({"role": "user", "content": prompt_text})
            
            col1, col2 = st.columns([1, 5])
            with col2:
                prompt_display = prompt_text.replace('$', '\\$').replace(chr(10), '<br>')
                st.markdown(f'''<div style="background: rgba(80, 160, 255, 0.1); border: 1px solid rgba(80, 160, 255, 0.3); border-radius: 12px; padding: 12px 16px; margin-bottom: 16px; text-align: right; color: var(--text-main); line-height: 1.5; font-family: 'Inter', sans-serif; font-size: 0.9rem;">
{prompt_display}
</div>''', unsafe_allow_html=True)
            
            col1, col2 = st.columns([5, 1])
            with col1:
                placeholder = st.empty()
                placeholder.markdown(f'''<div class="analytic-card" style="padding: 16px 20px; margin-bottom: 16px;">
<div style="color: #14f0a0; font-family: 'Roboto Mono', monospace; font-size: 0.7rem; margin-bottom: 12px; letter-spacing: 0.1em;">AI RISK ANALYST</div>
<div style="display: flex; align-items: center; gap: 12px;">
    <div class="ai-processing-ring"></div>
    <div style="color: #14f0a0; font-family: 'Roboto Mono', monospace; font-size: 0.8rem; letter-spacing: 0.1em; animation: ai-pulse 1.5s infinite;">SYNTHESIZING INSIGHTS...</div>
</div>
</div>''', unsafe_allow_html=True)
            
            sys_prompt = build_chatbot_context(result, selected_vendor, selected_product, all_scores)
            full_response, err = call_openrouter(sys_prompt, st.session_state["chat_history"], stream_placeholder=placeholder)
            
            if err:
                st.error(err)
                st.session_state["chat_history"].pop()
            elif full_response:
                st.session_state["chat_history"].append({"role": "assistant", "content": full_response})
                st.rerun()

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
        with chip_col1:
            if st.button("Why is this vendor high risk?", use_container_width=True, key="chip1"):
                st.session_state['chat_prefill'] = "Why is this vendor high risk?"
                st.rerun()
        with chip_col2:
            if st.button("Compare with best alternative", use_container_width=True, key="chip2"):
                st.session_state['chat_prefill'] = "Compare with best alternative"
                st.rerun()
        with chip_col3:
            if st.button("Should I approve this purchase?", use_container_width=True, key="chip3"):
                st.session_state['chat_prefill'] = "Should I approve this purchase?"
                st.rerun()
        with chip_col4:
            if st.button("Explain the SHAP values", use_container_width=True, key="chip4"):
                st.session_state['chat_prefill'] = "Explain the SHAP values"
                st.rerun()


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
    main()# End of application script.
