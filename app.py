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
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Sora:wght@400;600&display=swap');

* { font-family: 'Poppins', sans-serif; }
html, body, [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #FAFBFC 0%, #F3F5F7 100%); color: #0F1419; }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #FAFBFC 0%, #F3F5F7 100%); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #FFFFFF; }
h1,h2,h3 { font-family: 'Sora', sans-serif; color: #0F1419; font-weight:700; }

.hero-title { font-family:'Sora',sans-serif; font-size:3.6rem; line-height:1.1; color:#0F1419; margin:0; font-weight:700; }
.accent { background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }

.section-label {
    font-size:0.7rem; font-weight:600; letter-spacing:0.13em;
    text-transform:uppercase; color:#7F8C9A; margin-bottom:16px;
}

.card-section { background:#FFFFFF; border-radius:16px; padding:28px; box-shadow:0 4px 12px rgba(0,0,0,0.08); margin-bottom:20px; }

.input-group-title { font-size:0.85rem; font-weight:600; color:#0F1419; margin-bottom:12px; }

.result-card { border-radius:20px; padding:36px; text-align:center; margin-bottom:20px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.5); }
.result-high   { background: linear-gradient(135deg, rgba(229,57,53,0.1) 0%, rgba(229,57,53,0.05) 100%); border: 1px solid rgba(229,57,53,0.3); }
.result-medium { background: linear-gradient(135deg, rgba(245,124,0,0.1) 0%, rgba(245,124,0,0.05) 100%); border: 1px solid rgba(245,124,0,0.3); }
.result-low    { background: linear-gradient(135deg, rgba(27,140,90,0.1) 0%, rgba(27,140,90,0.05) 100%); border: 1px solid rgba(27,140,90,0.3); }

.result-prob { font-family:'Sora',sans-serif; font-size:4.5rem; line-height:1; margin:0; font-weight:700; }
.result-high   .result-prob { background: linear-gradient(135deg, #E53935 0%, #C62828 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.result-medium .result-prob { background: linear-gradient(135deg, #F57C00 0%, #E65100 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.result-low    .result-prob { background: linear-gradient(135deg, #1B8C5A 0%, #0D5F3F 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }

.result-label { font-size:0.75rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px; }
.result-high   .result-label { color:#E53935; }
.result-medium .result-label { color:#F57C00; }
.result-low    .result-label { color:#1B8C5A; }

.result-desc { font-size:0.9rem; color:#556B7D; font-weight:400; margin-top:12px; line-height:1.6; }

.insight-pill {
    display:inline-block; background:#FFFFFF; border:1.5px solid #E8EEF5;
    border-radius:24px; padding:8px 14px; font-size:0.76rem; color:#556B7D; margin:4px; font-weight:500;
    box-shadow:0 2px 6px rgba(0,0,0,0.06); transition: all 0.2s ease;
}
.insight-pill:hover { box-shadow:0 4px 12px rgba(0,0,0,0.1); }

.metric-row { display:flex; gap:12px; margin-top:20px; }
.metric-box {
    flex:1; background:#FFFFFF; border:1.5px solid #E8EEF5;
    border-radius:14px; padding:20px; text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.05); transition: all 0.3s ease;
}
.metric-box:hover { transform: translateY(-2px); box-shadow:0 6px 16px rgba(0,0,0,0.1); }
.metric-val { font-family:'Sora',sans-serif; font-size:1.8rem; color:#0F1419; font-weight:700; }
.metric-key { font-size:0.68rem; color:#7F8C9A; text-transform:uppercase; letter-spacing:0.11em; margin-top:6px; font-weight:600; }

.divider { height:1.5px; background: linear-gradient(90deg, transparent, #E8EEF5 20%, #E8EEF5 80%, transparent); margin:24px 0; }

[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] button,
[data-testid="stSlider"] {
    background:#FFFFFF !important; 
    border:1.5px solid #E8EEF5 !important;
    color:#0F1419 !important; 
    border-radius:10px !important;
}

label { 
    color:#556B7D !important; 
    font-size:0.82rem !important; 
    font-weight:500 !important;
}

.stButton button {
    width:100% !important; 
    background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important; 
    color:white !important;
    border:none !important; 
    border-radius:12px !important; 
    padding:16px !important;
    font-size:1.02rem !important; 
    font-weight:600 !important;
    letter-spacing:0.05em !important; 
    font-family:'Poppins',sans-serif !important;
    box-shadow:0 6px 20px rgba(255,107,53,0.35) !important;
    transition: all 0.3s ease !important;
}
.stButton button:hover { 
    transform: translateY(-2px) !important;
    box-shadow:0 10px 28px rgba(255,107,53,0.45) !important; 
}

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
<div style="padding:48px 0 36px;">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
        <span style="font-size:2.2rem;">🚗</span>
        <span style="font-size:0.74rem;font-weight:700;letter-spacing:0.15em;
              text-transform:uppercase;color:#7F8C9A;">Ola Intelligence</span>
    </div>
    <h1 class="hero-title">Driver Churn<br><span class="accent">Predictor</span></h1>
    <p style="font-size:1.02rem;color:#556B7D;margin-top:12px;font-weight:400;">
        AI-powered prediction to identify high-risk drivers before they churn
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<p class="section-label">📋 Driver Profile</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-section">', unsafe_allow_html=True)
        
        # Demographics section
        st.markdown('<p class="input-group-title">🏢 Basic Info</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            city = st.selectbox(
                "City",
                sorted(le.classes_) if artifacts_loaded else ["C1"],
                help="City where the driver operates"
            )
        with c2:
            grade = st.selectbox(
                "Grade",
                [1, 2, 3, 4, 5],
                index=0,
                help="Driver grade level (1=entry, 5=senior)"
            )
        
        c1, c2 = st.columns(2)
        with c1:
            joining_desig = st.selectbox(
                "Joining Designation",
                [1, 2, 3, 4, 5],
                index=0,
                help="Designation at joining"
            )
        with c2:
            months_active = st.number_input(
                "Months Active",
                min_value=1, max_value=24,
                value=5, step=1,
                help="Duration in months (median: 5)"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card-section">', unsafe_allow_html=True)
        
        # Financial section
        st.markdown('<p class="input-group-title">💰 Financial Metrics</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input(
                "Monthly Income (₹)",
                min_value=10000, max_value=200000,
                value=55000, step=1000,
                help="Mean: ₹59,334 | Churned avg: ₹55,391"
            )
        with c2:
            total_bv = st.number_input(
                "Total Business Value (₹)",
                min_value=0, max_value=10000000,
                value=817680, step=100000,
                help="Median: ₹817,680 | Churned: ₹2.2M"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card-section">', unsafe_allow_html=True)
        
        # Activity section
        st.markdown('<p class="input-group-title">📊 Activity & Tenure</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            tenure_days = st.number_input(
                "Tenure Days",
                min_value=1, max_value=3000,
                value=173, step=30,
                help="Median: 173 days"
            )
        with c2:
            ever_low = st.selectbox(
                "Ever Had Low Rating?",
                ["Yes", "No"],
                help="98% have had low rating"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card-section">', unsafe_allow_html=True)
        
        # Performance section
        st.markdown('<p class="input-group-title">⭐ Performance Metrics</p>', unsafe_allow_html=True)
        
        st.markdown('<p style="font-size:0.82rem;color:#556B7D;margin-bottom:8px;"><strong>Avg Quarterly Rating</strong></p>', unsafe_allow_html=True)
        rating_mean = st.slider(
            "Rating",
            min_value=1.0, max_value=4.0,
            value=1.0, step=0.1,
            help="Real data range: 1.0–4.0",
            label_visibility="collapsed"
        )
        
        st.markdown('<p style="font-size:0.82rem;color:#556B7D;margin-bottom:8px;margin-top:16px;"><strong>Rating Change (mean)</strong></p>', unsafe_allow_html=True)
        rating_change = st.slider(
            "Change",
            min_value=-1.0, max_value=1.0,
            value=0.0, step=0.05,
            help="Trend in driver ratings",
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🎯  Predict Churn Risk", use_container_width=True)

with right:
    st.markdown('<p class="section-label">📈 Prediction Result</p>', unsafe_allow_html=True)

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

        if prob >= 0.65:
            risk, css_cls = "High Risk", "result-high"
            emoji = "🚨"
            desc = "Strong churn signal. Immediate retention action recommended."
        elif prob >= 0.40:
            risk, css_cls = "Medium Risk", "result-medium"
            emoji = "⚠️"
            desc = "Watch closely. Consider targeted outreach and engagement."
        else:
            risk, css_cls = "Low Risk", "result-low"
            emoji = "✅"
            desc = "Stable driver. Continue standard engagement practices."

        st.markdown(f"""
        <div class="result-card {css_cls}">
            <p style="font-size:2.2rem;margin:0 0 12px 0;">{emoji}</p>
            <p class="result-label">{risk}</p>
            <p class="result-prob">{pct}%</p>
            <p class="result-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge Chart
        gc = "#E53935" if prob >= 0.65 else ("#F57C00" if prob >= 0.40 else "#1B8C5A")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={'suffix':'%','font':{'color':gc,'size':28,'family':'Sora','weight':'bold'}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#E8EEF5',
                        'tickfont':{'color':'#7F8C9A','size':10}},
                'bar':{'color':gc,'thickness':0.28},
                'bgcolor':'#F3F5F7','borderwidth':0,
                'steps':[
                    {'range':[0,40],'color':'#E8F5EA'},
                    {'range':[40,65],'color':'#FFF8E1'},
                    {'range':[65,100],'color':'#FFEBEE'},
                ],
                'threshold':{'line':{'color':gc,'width':4},'thickness':0.8,'value':pct}
            }
        ))
        fig.update_layout(
            height=220, margin=dict(t=20,b=0,l=20,r=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color':'#556B7D','family':'Poppins'}
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

        # Key Signals
        st.markdown('<p class="section-label" style="margin-top:8px;">🔍 Risk Signals</p>', unsafe_allow_html=True)

        signals = []
        if rating_change < 0:
            signals.append("📉 Declining ratings")
        elif rating_change > 0:
            signals.append("📈 Improving ratings")
        if tenure_days < 180:
            signals.append("⏱️ Early tenure")
        if grade == 1:
            signals.append("🔰 Grade 1 driver")
        if ever_low == "Yes":
            signals.append("⚠️ Had low rating")
        if rating_mean <= 1.38:
            signals.append("⭐ Below avg rating")
        if months_active <= 6:
            signals.append("📅 Low activity")
        if total_bv < 817680:
            signals.append("💼 Low business")
        if income < 55391:
            signals.append("💸 Low income")
        if not signals:
            signals.append("✨ No major risks")

        pills = "".join([f'<span class="insight-pill">{s}</span>' for s in signals])
        st.markdown(f'<div style="margin-top:8px">{pills}</div>', unsafe_allow_html=True)

        # Metrics Footer
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">{pct}%</div>
                <div class="metric-key">Churn Risk</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">0.93</div>
                <div class="metric-key">Model AUC</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">87%</div>
                <div class="metric-key">Accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#FFFFFF;border:2px dashed #E8EEF5;border-radius:20px;
                    padding:64px 32px;text-align:center;margin-top:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.04);">
            <div style="font-size:3rem;margin-bottom:16px;animation:pulse 2s infinite;">🎯</div>
            <p style="color:#7F8C9A;font-size:0.95rem;margin:0;line-height:1.8;font-weight:500;">
                Fill in the driver details<br>
                and click <strong style="color:#0F1419;">Predict Churn Risk</strong><br>
                <span style="font-size:0.85rem;color:#AAA;margin-top:8px;display:block;">
                to see detailed risk assessment
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:12px 0 32px;color:#7F8C9A;font-size:0.76rem;">
    <span style="font-weight:500;">🤖 AI-Powered Churn Prediction</span>
    <span>Built with Random Forest · SMOTE · scikit-learn</span>
    <span>Ola Churn ML Project</span>
</div>
""", unsafe_allow_html=True)
