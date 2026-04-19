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
html, body, [data-testid="stAppViewContainer"] { background: #0D0D0D; color: #F0EDE8; }
[data-testid="stAppViewContainer"] { background: #0D0D0D; }
[data-testid="stHeader"] { background: transparent; }
h1,h2,h3 { font-family: 'DM Serif Display', serif; color: #F0EDE8; }

.hero-title { font-family:'DM Serif Display',serif; font-size:3rem; line-height:1.1; color:#F0EDE8; margin:0; }
.hero-sub   { font-size:0.95rem; color:#666; font-weight:300; letter-spacing:0.04em; margin-top:8px; }
.accent     { color:#FF5733; }

.section-label {
    font-size:0.7rem; font-weight:600; letter-spacing:0.12em;
    text-transform:uppercase; color:#555; margin-bottom:16px;
}

.result-card { border-radius:16px; padding:32px; text-align:center; margin-bottom:16px; }
.result-high   { background:linear-gradient(135deg,#2A0A0A,#1A0505); border:1px solid #FF3B3B44; }
.result-medium { background:linear-gradient(135deg,#1A1200,#120E00); border:1px solid #FF990044; }
.result-low    { background:linear-gradient(135deg,#001A0D,#00120A); border:1px solid #00CC6644; }

.result-prob { font-family:'DM Serif Display',serif; font-size:4rem; line-height:1; margin:0; }
.result-high   .result-prob { color:#FF4444; }
.result-medium .result-prob { color:#FFAA00; }
.result-low    .result-prob { color:#00CC66; }

.result-label { font-size:0.72rem; font-weight:600; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:4px; }
.result-high   .result-label { color:#FF4444; }
.result-medium .result-label { color:#FFAA00; }
.result-low    .result-label { color:#00CC66; }

.result-desc { font-size:0.88rem; color:#888; font-weight:300; margin-top:8px; line-height:1.5; }

.insight-pill {
    display:inline-block; background:#1E1E1E; border:1px solid #2E2E2E;
    border-radius:100px; padding:5px 12px; font-size:0.76rem; color:#AAA; margin:3px;
}

.metric-row { display:flex; gap:10px; margin-top:14px; }
.metric-box {
    flex:1; background:#1A1A1A; border:1px solid #252525;
    border-radius:12px; padding:14px; text-align:center;
}
.metric-val { font-family:'DM Serif Display',serif; font-size:1.6rem; color:#F0EDE8; }
.metric-key { font-size:0.68rem; color:#555; text-transform:uppercase; letter-spacing:0.1em; margin-top:3px; }

.divider { height:1px; background:#1A1A1A; margin:20px 0; }

.hint-text { font-size:0.75rem; color:#444; margin-top:4px; font-style:italic; }

[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background:#1A1A1A !important; border:1px solid #2A2A2A !important;
    color:#F0EDE8 !important; border-radius:8px !important;
}
label { color:#888 !important; font-size:0.8rem !important; }
.stSlider [data-baseweb="slider"] { color:#FF5733; }

.stButton button {
    width:100%; background:#FF5733 !important; color:white !important;
    border:none !important; border-radius:12px !important; padding:16px !important;
    font-size:1rem !important; font-weight:600 !important;
    letter-spacing:0.04em !important; font-family:'DM Sans',sans-serif !important;
}
.stButton button:hover { background:#E84820 !important; }
footer,#MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = 'ola_churn_model'

@st.cache_resource
def load_artifacts():
    model     = joblib.load(f'{MODEL_DIR}/ola_churn_xgb.pkl')
    le        = joblib.load(f'{MODEL_DIR}/label_encoder.pkl')
    caps      = joblib.load(f'{MODEL_DIR}/outlier_caps.pkl')
    feat_cols = joblib.load(f'{MODEL_DIR}/feature_columns.pkl')
    return model, le, caps, feat_cols

def preprocess_and_predict(raw, model, le, caps, feat_cols):
    d = raw.copy()

    # Step 1 — outlier cap first (exact order as Cell 61)
    for col, cap_val in caps.items():
        if col in d:
            d[col] = min(float(d[col]), float(cap_val))

    # Step 2 — compute derived features AFTER capping
    tenure_months = max(d['Tenure_Days_last'] / 30, 0.01)
    d['Avg_Business_Per_Month_mean'] = d['Total_Business_Value_sum'] / tenure_months
    d['Income_Grade_Ratio_last']     = d['Income_last'] / max(d['Grade_last'], 1)
    rc = d['Rating_Change_mean']
    d['Rating_Trend_Encoded_mean']   = 1.0 if rc > 0 else (-1.0 if rc < 0 else 0.0)
    d['Quarterly_Rating_min']        = max(d['Quarterly_Rating_mean'] - 0.5, 1.0)
    d['Quarterly_Rating_max']        = min(d['Quarterly_Rating_mean'] + 0.5, 4.0)

    # Step 3 — log transform (same 3 cols as notebook)
    for col in ['Total_Business_Value_sum', 'Tenure_Days_last', 'Avg_Business_Per_Month_mean']:
        if col in d:
            d[col] = float(np.log1p(max(d[col], 0)))

    # Step 4 — encode City
    known = list(le.classes_)
    d['City_last'] = le.transform([d['City_last']])[0] if d['City_last'] in known else 0

    # Step 5 — build DataFrame aligned to training columns
    df = pd.DataFrame([d])
    df = df.reindex(columns=feat_cols, fill_value=0)

    # Step 6 — convert each col to float safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # Step 7 — predict using numpy array (bypass XGBoost dtype check)
    prob = model.predict_proba(df.values)[:, 1][0]
    return round(float(prob), 4)

try:
    model, le, caps, feat_cols = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"Could not load model artifacts: {e}")

# ── HERO ──────────────────────────────────────────────
st.markdown("""
<div style="padding:36px 0 28px;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
        <span style="font-size:1.8rem;">🚗</span>
        <span style="font-size:0.72rem;font-weight:600;letter-spacing:0.14em;
              text-transform:uppercase;color:#444;">Ola Intelligence</span>
    </div>
    <p class="hero-title">Driver Churn<br><span class="accent">Predictor</span></p>
    <p class="hero-sub">XGBoost · AUC-ROC 0.93 · 3,000+ driver records</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<p class="section-label">Driver Profile</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        city = st.selectbox(
            "City",
            sorted(le.classes_) if artifacts_loaded else ["C1"],
            help="City where the driver operates"
        )
        grade = st.selectbox(
            "Grade",
            [1, 2, 3, 4, 5],
            index=0,
            help="Driver grade level. Grade 1 = entry level"
        )
        joining_desig = st.selectbox(
            "Joining Designation",
            [1, 2, 3, 4, 5],
            index=0,
            help="Designation at the time of joining"
        )

    with c2:
        # Real range: 10747 to 188418, mean 59334
        income = st.number_input(
            "Monthly Income (₹)",
            min_value=10000, max_value=200000,
            value=55000, step=1000,
            help="Mean: ₹59,334 | Churned avg: ₹55,391"
        )
        # Real range: 1 to 2801, mean 417
        tenure_days = st.number_input(
            "Tenure Days",
            min_value=1, max_value=3000,
            value=173, step=30,
            help="Median: 173 days | Active drivers avg: 566 days"
        )
        # Real range: 1 to 24, mean 8
        months_active = st.number_input(
            "Months Active",
            min_value=1, max_value=24,
            value=5, step=1,
            help="Median: 5 months | Active drivers avg: 11 months"
        )

    st.markdown('<p class="section-label" style="margin-top:20px;">Performance Metrics</p>',
                unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        # Real range: 1.0 to 4.0 (max is 4 not 5!)
        rating_mean = st.slider(
            "Avg Quarterly Rating",
            min_value=1.0, max_value=4.0,
            value=1.0, step=0.1,
            help="Real data range: 1.0–4.0 | Churned avg: 1.38 | Active avg: 1.96"
        )
        # Real range: -1.0 to 1.0 (not -4 to 4!)
        rating_change = st.slider(
            "Rating Change (mean)",
            min_value=-1.0, max_value=1.0,
            value=0.0, step=0.05,
            help="Mean change in rating over driver's history. Churned avg: -0.04"
        )

    with c4:
        # Real range: 0 to 95,331,060 — log scale in model
        total_bv = st.number_input(
            "Total Business Value (₹)",
            min_value=0, max_value=10000000,
            value=817680, step=100000,
            help="Median: ₹817,680 | Churned avg: ₹2.2M | Active avg: ₹9.6M"
        )
        ever_low = st.selectbox(
            "Ever Had Low Rating?",
            ["Yes", "No"],
            help="98% of all drivers have had a low rating — less discriminating"
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Churn Probability", use_container_width=True)

with right:
    st.markdown('<p class="section-label">Prediction Result</p>', unsafe_allow_html=True)

    if predict_btn and artifacts_loaded:
        raw = {
            'City_last'                : city,
            'Income_last'              : float(income),
            'Joining_Designation_last' : float(joining_desig),
            'Grade_last'               : float(grade),
            'Total_Business_Value_sum' : float(total_bv),
            'Quarterly_Rating_mean'    : float(rating_mean),
            'Quarterly_Rating_min'     : max(float(rating_mean) - 0.5, 1.0),
            'Quarterly_Rating_max'     : min(float(rating_mean) + 0.5, 4.0),
            'Tenure_Days_last'         : float(tenure_days),
            'Rating_Change_mean'       : float(rating_change),
            'Months_Active'            : float(months_active),
            'Ever_Had_Low_Rating'      : 1.0 if ever_low == "Yes" else 0.0,
        }

        prob = preprocess_and_predict(raw, model, le, caps, feat_cols)
        pct  = round(prob * 100, 1)

        # Risk thresholds based on real model distribution
        # > 0.65 = High (295/477 test drivers)
        # 0.40-0.65 = Medium (49/477)
        # < 0.40 = Low (133/477)
        if prob >= 0.65:
            risk, css_cls = "High Risk", "result-high"
            desc = "This driver is very likely to churn. Immediate retention action recommended."
        elif prob >= 0.40:
            risk, css_cls = "Medium Risk", "result-medium"
            desc = "Watch closely. Consider targeted outreach and engagement."
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

        # Gauge
        gc = "#FF4444" if prob >= 0.65 else ("#FFAA00" if prob >= 0.40 else "#00CC66")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={'suffix':'%','font':{'color':gc,'size':26,'family':'DM Serif Display'}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#333',
                        'tickfont':{'color':'#444','size':9}},
                'bar':{'color':gc,'thickness':0.25},
                'bgcolor':'#161616','borderwidth':0,
                'steps':[
                    {'range':[0,40],'color':'#0D1A12'},
                    {'range':[40,65],'color':'#1A1200'},
                    {'range':[65,100],'color':'#1A0505'},
                ],
                'threshold':{'line':{'color':gc,'width':3},'thickness':0.75,'value':pct}
            }
        ))
        fig.update_layout(
            height=200, margin=dict(t=16,b=0,l=16,r=16),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color':'#666'}
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

        # Key signals — based on real data thresholds
        st.markdown('<p class="section-label" style="margin-top:4px;">Key signals detected</p>',
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
        if rating_mean <= 1.38:
            signals.append("⭐ Below churned-driver avg rating (1.38)")
        if months_active <= 6:
            signals.append("📅 Low activity months")
        if total_bv < 817680:
            signals.append("💼 Below median business value")
        if income < 55391:
            signals.append("💸 Below churned-driver avg income")
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
                <div class="metric-val">0.9318</div>
                <div class="metric-key">Model AUC</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#0F0F0F;border:1px dashed #222;border-radius:16px;
                    padding:56px 28px;text-align:center;margin-top:8px;">
            <div style="font-size:2.2rem;margin-bottom:14px;">🎯</div>
            <p style="color:#333;font-size:0.88rem;margin:0;line-height:1.8;">
                Fill in the driver details<br>
                and click <strong style="color:#555">Predict</strong>
                to see churn probability
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Show what High Risk looks like
        st.markdown("""
        <div style="margin-top:20px;">
            <p class="section-label">High risk driver profile (reference)</p>
            <div style="background:#1A0505;border:1px solid #FF3B3B22;border-radius:12px;
                        padding:16px;font-size:0.8rem;color:#888;line-height:2;">
                  City: C1, C4, C17, C20 (common)<br>
                  Tenure Days: 100 – 500<br>
                  Total Business Value: ₹0 – ₹100<br>
                  Avg Quarterly Rating: 1.0<br>
                  Months Active: 3 – 7<br>
                  Grade: 1
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:6px 0 20px;color:#2A2A2A;font-size:0.72rem;">
    <span>Built with XGBoost · SMOTE · scikit-learn</span>
    <span>Prashant · Ola Churn ML Project</span>
</div>
""", unsafe_allow_html=True)
