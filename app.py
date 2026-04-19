import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Ola Driver Churn Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

* { font-family: 'DM Sans', sans-serif; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0D0D0D;
    color: #F0EDE8;
}

[data-testid="stAppViewContainer"] {
    background: #0D0D0D;
}

[data-testid="stHeader"] { background: transparent; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: #F0EDE8;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    line-height: 1.1;
    color: #F0EDE8;
    margin: 0;
    padding: 0;
}

.hero-sub {
    font-size: 1rem;
    color: #888;
    font-weight: 300;
    letter-spacing: 0.04em;
    margin-top: 8px;
}

.accent { color: #FF5733; }

.card {
    background: #161616;
    border: 1px solid #262626;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 16px;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 16px;
}

.result-card {
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    margin-top: 8px;
}

.result-high {
    background: linear-gradient(135deg, #2A0A0A, #1A0505);
    border: 1px solid #FF3B3B44;
}

.result-medium {
    background: linear-gradient(135deg, #1A1200, #120E00);
    border: 1px solid #FF990044;
}

.result-low {
    background: linear-gradient(135deg, #001A0D, #00120A);
    border: 1px solid #00CC6644;
}

.result-prob {
    font-family: 'DM Serif Display', serif;
    font-size: 4.5rem;
    font-weight: 400;
    line-height: 1;
    margin: 0;
}

.result-high .result-prob   { color: #FF4444; }
.result-medium .result-prob { color: #FFAA00; }
.result-low .result-prob    { color: #00CC66; }

.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 10px;
    margin-bottom: 4px;
}

.result-high .result-label   { color: #FF4444; }
.result-medium .result-label { color: #FFAA00; }
.result-low .result-label    { color: #00CC66; }

.result-desc {
    font-size: 0.9rem;
    color: #888;
    font-weight: 300;
    margin-top: 8px;
    line-height: 1.5;
}

.insight-pill {
    display: inline-block;
    background: #1E1E1E;
    border: 1px solid #2E2E2E;
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: #AAA;
    margin: 4px;
}

.metric-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
}

.metric-box {
    flex: 1;
    background: #1A1A1A;
    border: 1px solid #252525;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #F0EDE8;
}

.metric-key {
    font-size: 0.7rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

.divider {
    height: 1px;
    background: #1E1E1E;
    margin: 24px 0;
}

/* Streamlit widget overrides */
[data-testid="stSlider"] > div > div { background: #262626; }
.stSlider [data-baseweb="slider"] { color: #FF5733; }
[data-testid="stSelectbox"] select,
[data-testid="stNumberInput"] input {
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    color: #F0EDE8 !important;
    border-radius: 8px !important;
}
label { color: #AAA !important; font-size: 0.82rem !important; }

.stButton button {
    width: 100%;
    background: #FF5733 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stButton button:hover {
    background: #E84820 !important;
    transform: translateY(-1px) !important;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'ola_churn_model')

@st.cache_resource
def load_artifacts():
    model     = joblib.load(os.path.join(MODEL_DIR, 'ola_churn_xgb.pkl'))
    le        = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    caps      = joblib.load(os.path.join(MODEL_DIR, 'outlier_caps.pkl'))
    feat_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    return model, le, caps, feat_cols

def predict_churn(input_df, model, le, caps, feat_cols):
    df = input_df.copy()
    if df['City_last'].dtype == object:
        known = list(le.classes_)
        df['City_last'] = df['City_last'].apply(
            lambda x: le.transform([x])[0] if x in known else 0
        )
        
    df = df.reindex(columns=feat_cols, fill_value=0)
    # Safely convert all columns to numeric, handling any remaining non-numeric values
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    prob = model.predict_proba(df)[:, 1][0]
    return round(float(prob), 4)

def preprocess_raw_input(raw, caps):
    d = raw.copy()
    tenure_months = max(d['Tenure_Days_last'] / 30, 0.01)
    d['Avg_Business_Per_Month_mean'] = d['Total_Business_Value_sum'] / tenure_months
    d['Income_Grade_Ratio_last']     = d['Income_last'] / max(d['Grade_last'], 1)
    rc = d['Rating_Change_mean']
    d['Rating_Trend_Encoded_mean']   = 1 if rc > 0 else (-1 if rc < 0 else 0)
    for col, cap_val in caps.items():
        if col in d:
            d[col] = min(d[col], cap_val)
    for col in ['Total_Business_Value_sum', 'Tenure_Days_last', 'Avg_Business_Per_Month_mean']:
        if col in d:
            d[col] = float(np.log1p(max(d[col], 0)))
    return d

try:
    model, le, caps, feat_cols = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"Could not load model artifacts: {e}")

# ── HERO ───────────────────────────────────────────────
st.markdown("""
<div style="padding: 40px 0 32px;">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
        <span style="font-size:2rem;">🚗</span>
        <span style="font-size:0.75rem; font-weight:600; letter-spacing:0.14em; 
                     text-transform:uppercase; color:#555;">Ola Intelligence</span>
    </div>
    <p class="hero-title">Driver Churn<br><span class="accent">Predictor</span></p>
    <p class="hero-sub">XGBoost · AUC-ROC 0.93 · 3,000+ driver records</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── LAYOUT ─────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<p class="section-label">Driver Profile</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        city = st.selectbox("City", sorted(le.classes_) if artifacts_loaded else ["C1"])
        grade = st.selectbox("Grade", [1, 2, 3, 4, 5], index=0)
        joining_desig = st.selectbox("Joining Designation", [1, 2, 3, 4, 5], index=0)
    with c2:
        income = st.number_input("Monthly Income (₹)", min_value=0, max_value=200000,
                                  value=20000, step=1000)
        tenure_days = st.number_input("Tenure Days", min_value=1, max_value=5000,
                                       value=180, step=30)
        months_active = st.number_input("Months Active", min_value=1, max_value=100,
                                         value=6, step=1)

    st.markdown('<p class="section-label" style="margin-top:20px;">Performance Metrics</p>',
                unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        rating_mean = st.slider("Avg Quarterly Rating", 1.0, 5.0, 3.5, 0.1)
        rating_min  = st.slider("Min Quarterly Rating", 1.0, 5.0, 2.5, 0.1)
    with c4:
        rating_max    = st.slider("Max Quarterly Rating", 1.0, 5.0, 4.5, 0.1)
        rating_change = st.slider("Rating Change (last quarter)", -4.0, 4.0, 0.0, 0.1)

    c5, c6 = st.columns(2)
    with c5:
        total_bv = st.number_input("Total Business Value (₹)", min_value=0,
                                    max_value=5000000, value=150000, step=5000)
    with c6:
        ever_low = st.selectbox("Ever Had Low Rating?", ["No", "Yes"])

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Churn Probability", use_container_width=True)

with right:
    st.markdown('<p class="section-label">Prediction Result</p>', unsafe_allow_html=True)

    if predict_btn and artifacts_loaded:
        raw = {
            'City_last'                  : city,
            'Income_last'                : income,
            'Joining_Designation_last'   : joining_desig,
            'Grade_last'                 : grade,
            'Total_Business_Value_sum'   : total_bv,
            'Quarterly_Rating_mean'      : rating_mean,
            'Quarterly_Rating_min'       : rating_min,
            'Quarterly_Rating_max'       : rating_max,
            'Tenure_Days_last'           : tenure_days,
            'Rating_Change_mean'         : rating_change,
            'Months_Active'              : months_active,
            'Ever_Had_Low_Rating'        : 1 if ever_low == "Yes" else 0,
        }

        processed = preprocess_raw_input(raw, caps)
        input_df  = pd.DataFrame([processed])
        prob      = predict_churn(input_df, model, le, caps, feat_cols)
        pct       = round(prob * 100, 1)

        if prob >= 0.65:
            risk, css_cls = "High Risk", "result-high"
            desc = "This driver is very likely to churn. Immediate retention action recommended."
        elif prob >= 0.40:
            risk, css_cls = "Medium Risk", "result-medium"
            desc = "This driver shows signs of disengagement. Monitor closely and consider outreach."
        else:
            risk, css_cls = "Low Risk", "result-low"
            desc = "This driver appears stable. Continue standard engagement practices."

        st.markdown(f"""
        <div class="result-card {css_cls}">
            <p class="result-label">{risk}</p>
            <p class="result-prob">{pct}%</p>
            <p class="result-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        gauge_color = "#FF4444" if prob >= 0.65 else ("#FFAA00" if prob >= 0.40 else "#00CC66")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={'suffix': '%', 'font': {'color': gauge_color, 'size': 28,
                                             'family': 'DM Serif Display'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#333',
                         'tickfont': {'color': '#555', 'size': 10}},
                'bar': {'color': gauge_color, 'thickness': 0.25},
                'bgcolor': '#161616',
                'borderwidth': 0,
                'steps': [
                    {'range': [0,  40],  'color': '#0D1A12'},
                    {'range': [40, 65],  'color': '#1A1200'},
                    {'range': [65, 100], 'color': '#1A0505'},
                ],
                'threshold': {
                    'line': {'color': gauge_color, 'width': 3},
                    'thickness': 0.75,
                    'value': pct
                }
            }
        ))
        fig.update_layout(
            height=220,
            margin=dict(t=20, b=0, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#888'}
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Key driver insights
        st.markdown('<p class="section-label" style="margin-top:8px;">Key signals detected</p>',
                    unsafe_allow_html=True)

        signals = []
        if rating_change < 0:
            signals.append("📉 Declining rating trend")
        elif rating_change > 0:
            signals.append("📈 Improving rating trend")
        if tenure_days < 180:
            signals.append("⏱ Early tenure — high risk window")
        if grade == 1:
            signals.append("🔰 Grade 1 — highest churn segment")
        if ever_low == "Yes":
            signals.append("⚠️ Has had low rating before")
        if income / max(grade, 1) < 8000:
            signals.append("💸 Income-grade mismatch detected")
        if rating_mean < 2.5:
            signals.append("⭐ Below average rating")
        if not signals:
            signals.append("✅ No major risk signals detected")

        pills = "".join([f'<span class="insight-pill">{s}</span>' for s in signals])
        st.markdown(f'<div style="margin-top:4px">{pills}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">{pct}%</div>
                <div class="metric-key">Churn Prob</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{risk.split()[0]}</div>
                <div class="metric-key">Risk Level</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">0.93</div>
                <div class="metric-key">Model AUC</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#111; border:1px dashed #2A2A2A; border-radius:20px;
                    padding:60px 32px; text-align:center; margin-top:8px;">
            <div style="font-size:2.5rem; margin-bottom:16px;">🎯</div>
            <p style="color:#444; font-size:0.9rem; margin:0; line-height:1.7;">
                Fill in the driver details<br>and click <strong style="color:#666">Predict</strong> 
                to see the churn probability
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; 
            padding:8px 0 24px; color:#333; font-size:0.75rem;">
    <span>Built with XGBoost · SMOTE · scikit-learn</span>
    <span>Prashant · Ola Churn ML Project</span>
</div>
""", unsafe_allow_html=True)
