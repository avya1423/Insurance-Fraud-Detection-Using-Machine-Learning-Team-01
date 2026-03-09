"""
=============================================================================
  Insurance Fraud Detection System
  Module: Streamlit Web Application
  Description: Interactive UI for submitting insurance claims and getting
               real-time fraud predictions across all three scenarios.

  Run:
    streamlit run app.py
=============================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark, professional theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  html, body, [data-testid="stAppViewContainer"] {
      background: #0F172A;
      color: #F1F5F9;
  }
  [data-testid="stSidebar"] {
      background: #1E293B;
  }
  .block-container { padding-top: 2rem; }

  /* ── Cards ── */
  .fraud-card {
      background: #1E293B;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1rem;
      border-left: 4px solid #38BDF8;
  }
  .result-fraud {
      background: linear-gradient(135deg, #450a0a, #7f1d1d);
      border: 2px solid #F43F5E;
      border-radius: 12px;
      padding: 2rem;
      text-align: center;
  }
  .result-legit {
      background: linear-gradient(135deg, #052e16, #14532d);
      border: 2px solid #22c55e;
      border-radius: 12px;
      padding: 2rem;
      text-align: center;
  }
  .metric-box {
      background: #0F172A;
      border-radius: 8px;
      padding: 1rem;
      text-align: center;
      border: 1px solid #334155;
  }
  .big-label {
      font-size: 2.5rem;
      font-weight: 900;
      letter-spacing: 0.05em;
  }
  .sub-label {
      font-size: 1rem;
      opacity: 0.8;
      margin-top: 0.3rem;
  }

  /* ── Progress bar ── */
  .stProgress > div > div > div > div {
      background: linear-gradient(90deg, #38BDF8, #F43F5E);
  }

  /* ── Buttons ── */
  .stButton > button {
      background: linear-gradient(135deg, #0EA5E9, #6366F1);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 0.6rem 2rem;
      font-weight: 700;
      font-size: 1rem;
      letter-spacing: 0.05em;
      width: 100%;
  }
  .stButton > button:hover {
      background: linear-gradient(135deg, #38BDF8, #818CF8);
      transform: translateY(-1px);
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
      background: #1E293B;
      border-radius: 8px;
      padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
      color: #94A3B8;
      font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
      background: #0F172A;
      color: #38BDF8;
      border-radius: 6px;
  }

  /* ── Inputs ── */
  .stNumberInput > div > div > input,
  .stSlider > div > div > div > div {
      background: #0F172A;
      color: #F1F5F9;
      border-color: #334155;
  }

  /* ── Divider ── */
  hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def models_trained() -> bool:
    """Check if all three models have been trained and saved."""
    for sc in ["auto", "health", "property"]:
        if not os.path.exists(f"saved_models/{sc}_model.pkl"):
            return False
    return True


@st.cache_resource
def load_predictor(scenario: str):
    """Lazily import and return the predict function for a scenario."""
    if scenario == "auto":
        from scenarios.auto_fraud     import predict_auto_fraud
        return predict_auto_fraud
    elif scenario == "health":
        from scenarios.health_fraud   import predict_health_fraud
        return predict_health_fraud
    else:
        from scenarios.property_fraud import predict_property_fraud
        return predict_property_fraud


def render_result(result: dict):
    """Render the prediction result card."""
    is_fraud  = "FRAUD" in result["label"]
    cls       = "result-fraud" if is_fraud else "result-legit"
    icon      = "🚨" if is_fraud else "✅"
    color_hex = "#F43F5E"   if is_fraud else "#22c55e"
    verb      = "FRAUDULENT" if is_fraud else "LEGITIMATE"

    st.markdown(f"""
    <div class="{cls}">
        <div class="big-label" style="color:{color_hex}">{icon} {verb}</div>
        <div class="sub-label">Fraud Score: <strong>{result['fraud_score']}%</strong>
         &nbsp;|&nbsp; Risk Level: <strong>{result['confidence']}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(result["probability"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-box">
            <div style="font-size:1.8rem;font-weight:900;color:#38BDF8">
                {result['fraud_score']}%
            </div><div style="font-size:0.8rem;color:#94A3B8;margin-top:4px">
                Fraud Probability
            </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-box">
            <div style="font-size:1.8rem;font-weight:900;color:#A78BFA">
                {result['confidence']}
            </div><div style="font-size:0.8rem;color:#94A3B8;margin-top:4px">
                Risk Level
            </div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-box">
            <div style="font-size:1.8rem;font-weight:900;color:{'#F43F5E' if is_fraud else '#22c55e'}">
                {'HIGH' if is_fraud else 'LOW'}
            </div><div style="font-size:0.8rem;color:#94A3B8;margin-top:4px">
                Alert Status
            </div></div>""", unsafe_allow_html=True)


def show_plot_if_exists(path: str, caption: str = ""):
    """Show a saved plot image if file exists."""
    if path and os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0">
        <div style="font-size:2.5rem">🛡️</div>
        <div style="font-size:1.2rem;font-weight:900;color:#38BDF8;
                    letter-spacing:0.08em;margin-top:0.5rem">
            FRAUD GUARD
        </div>
        <div style="font-size:0.75rem;color:#64748B;margin-top:0.2rem">
            Insurance Fraud Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🚀 Quick Start")
    st.info("**Step 1:** Train models using the terminal:\n```\npython train.py\n```\n\n"
            "**Step 2:** Return here and submit a claim for analysis.")

    st.divider()
    st.markdown("### ℹ️ System Info")
    status_ok   = models_trained()
    status_icon = "🟢" if status_ok else "🔴"
    st.markdown(f"{status_icon} Models: {'**Ready**' if status_ok else '**Not Trained**'}")
    st.markdown("🤖 Algorithms: Random Forest, Decision Tree, Logistic Regression")
    st.markdown("📊 Scenarios: Auto · Health · Property")
    st.divider()
    st.markdown("<div style='color:#475569;font-size:0.75rem;text-align:center'>"
                "College Mini-Project Demo<br>Insurance Fraud Detection using ML"
                "</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem">
    <h1 style="color:#38BDF8;font-size:2rem;font-weight:900;
               letter-spacing:0.04em;margin-bottom:0.2rem">
        🛡️ Insurance Fraud Detection System
    </h1>
    <p style="color:#94A3B8;font-size:1rem">
        AI-powered fraud analysis for Automobile, Health, and Property insurance claims.
    </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_auto, tab_health, tab_prop, tab_analytics = st.tabs([
    "🚗  Auto Insurance",
    "🏥  Health Insurance",
    "🏠  Property Insurance",
    "📊  Analytics Dashboard",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – AUTOMOBILE
# ─────────────────────────────────────────────────────────────────────────────
with tab_auto:
    st.markdown('<div class="fraud-card">'
                '<h3 style="color:#38BDF8;margin:0 0 .4rem">🚗 Automobile Insurance Claim</h3>'
                '<p style="color:#94A3B8;margin:0;font-size:.9rem">'
                'Enter vehicle and claim details to check for potential fraud.</p>'
                '</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🚙 Vehicle Details**")
        v_age   = st.number_input("Vehicle Age (years)",   0, 30, 5,  key="a_vage")
        v_price = st.number_input("Vehicle Price ($)",     0, 150000, 20000, 500, key="a_vprice")
        v_dmg   = st.slider("Vehicle Damage Scale (1-5)", 1, 5, 3, key="a_vdmg")
        n_veh   = st.number_input("Number of Vehicles",   1, 6, 1, key="a_nveh")

    with c2:
        st.markdown("**👤 Driver Details**")
        d_age  = st.number_input("Driver Age",             18, 80, 35, key="a_dage")
        d_exp  = st.number_input("Driving Experience (yrs)", 0, 50, 10, key="a_dexp")
        past_c = st.number_input("Past Claims (count)",    0, 15, 0, key="a_pastc")
        d_pol  = st.number_input("Days Since Policy Start",0, 3000, 500, key="a_dpol")

    with c3:
        st.markdown("**📋 Claim Details**")
        c_amt  = st.number_input("Claim Amount ($)",       0, 100000, 5000, 500, key="a_camt")
        i_hr   = st.slider("Incident Hour (0-23)",         0, 23, 14, key="a_ihr")
        wit    = st.number_input("Witnesses",               0, 10, 2, key="a_wit")
        pol_r  = st.selectbox("Police Report Filed?",      ["Yes", "No"], key="a_polr")
        inj    = st.number_input("Injury Claim ($)",        0, 50000, 0, 500, key="a_inj")
        prop_d = st.number_input("Property Damage ($)",    0, 50000, 0, 500, key="a_propd")

    st.markdown("---")
    if st.button("🔍 Analyse Automobile Claim", key="btn_auto"):
        if not models_trained():
            st.error("⚠️ Models not trained yet. Run `python train.py` first.")
        else:
            with st.spinner("Analysing claim..."):
                claim = {
                    "vehicle_age": v_age, "vehicle_price": v_price,
                    "claim_amount": c_amt, "driver_age": d_age,
                    "driver_experience": d_exp, "num_past_claims": past_c,
                    "days_since_policy": d_pol, "incident_hour": i_hr,
                    "witnesses": wit, "police_report": 1 if pol_r=="Yes" else 0,
                    "injury_claim": inj, "property_damage": prop_d,
                    "vehicle_damage_scale": v_dmg, "num_vehicles": n_veh,
                }
                predict_fn = load_predictor("auto")
                result     = predict_fn(claim)
            st.markdown("### 🎯 Prediction Result")
            render_result(result)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – HEALTH
# ─────────────────────────────────────────────────────────────────────────────
with tab_health:
    st.markdown('<div class="fraud-card">'
                '<h3 style="color:#38BDF8;margin:0 0 .4rem">🏥 Health Insurance Claim</h3>'
                '<p style="color:#94A3B8;margin:0;font-size:.9rem">'
                'Detect overbilling, duplicate claims, and billing anomalies.</p>'
                '</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**👤 Patient Details**")
        p_age  = st.number_input("Patient Age",              18, 100, 45, key="h_page")
        chron  = st.selectbox("Chronic Condition?",         ["No", "Yes"], key="h_chron")
        h_stay = st.number_input("Hospital Stay (days)",     0, 60, 3, key="h_stay")

    with c2:
        st.markdown("**🩺 Medical Details**")
        n_diag = st.number_input("Number of Diagnoses",      1, 20, 2, key="h_ndiag")
        n_proc = st.number_input("Number of Procedures",     1, 30, 3, key="h_nproc")
        n_phys = st.number_input("Number of Physicians",     1, 10, 2, key="h_nphys")
        n_lab  = st.number_input("Lab Tests Ordered",        0, 30, 2, key="h_nlab")
        n_meds = st.number_input("Medications Prescribed",   0, 20, 2, key="h_nmeds")

    with c3:
        st.markdown("**💰 Billing Details**")
        bill   = st.number_input("Billing Amount ($)",        0, 100000, 3000, 500, key="h_bill")
        freq   = st.number_input("Claim Frequency (per yr)", 1, 50, 3, key="h_freq")
        auth   = st.selectbox("Prior Auth Obtained?",        ["Yes", "No"], key="h_auth")
        d_bet  = st.number_input("Days Between Claims",       1, 365, 90, key="h_dbet")
        dup    = st.selectbox("Duplicate Claim Flag?",        ["No", "Yes"], key="h_dup")
        ob_r   = st.number_input("Overbilling Ratio",         0.5, 5.0, 1.0, 0.1, key="h_obr")

    st.markdown("---")
    if st.button("🔍 Analyse Health Claim", key="btn_health"):
        if not models_trained():
            st.error("⚠️ Models not trained yet. Run `python train.py` first.")
        else:
            with st.spinner("Analysing claim..."):
                claim = {
                    "patient_age": p_age,
                    "num_diagnoses": n_diag, "num_procedures": n_proc,
                    "billing_amount": bill, "claim_frequency": freq,
                    "hospital_stay_days": h_stay, "num_physicians": n_phys,
                    "is_chronic_condition": 1 if chron=="Yes" else 0,
                    "prior_auth_obtained": 1 if auth=="Yes" else 0,
                    "days_between_claims": d_bet,
                    "lab_tests_ordered": n_lab,
                    "medications_prescribed": n_meds,
                    "duplicate_claim_flag": 1 if dup=="Yes" else 0,
                    "overbilling_ratio": ob_r,
                }
                predict_fn = load_predictor("health")
                result     = predict_fn(claim)
            st.markdown("### 🎯 Prediction Result")
            render_result(result)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – PROPERTY
# ─────────────────────────────────────────────────────────────────────────────
with tab_prop:
    st.markdown('<div class="fraud-card">'
                '<h3 style="color:#38BDF8;margin:0 0 .4rem">🏠 Property Insurance Claim</h3>'
                '<p style="color:#94A3B8;margin:0;font-size:.9rem">'
                'Identify inflated damage reports and suspicious claim timing.</p>'
                '</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🏘️ Property Details**")
        prop_v  = st.number_input("Property Value ($)",      10000, 1000000, 150000, 5000, key="p_pval")
        prop_a  = st.number_input("Property Age (years)",    1, 100, 20, key="p_page")
        d_pol2  = st.number_input("Days Since Policy Start", 1, 5000, 300, key="p_dpol")

    with c2:
        st.markdown("**💥 Damage Details**")
        c_amt2  = st.number_input("Claim Amount ($)",         0, 500000, 10000, 1000, key="p_camt")
        dmg_sev = st.slider("Damage Severity (1-5)",         1, 5, 2, key="p_dsev")
        rep_est = st.number_input("Repair Cost Estimate ($)", 0, 500000, 9000, 500, key="p_rest")
        ctv_r   = st.number_input("Claim-to-Value Ratio",    0.0, 3.0, 0.1, 0.01, key="p_ctvr",
                                  format="%.2f")

    with c3:
        st.markdown("**📋 Documentation**")
        past_c2 = st.number_input("Past Claims (count)",      0, 20, 0, key="p_pastc")
        contr   = st.selectbox("Contractor Verified?",        ["Yes", "No"], key="p_contr")
        photos  = st.selectbox("Photos Submitted?",           ["Yes", "No"], key="p_photo")
        rep_f   = st.selectbox("Police/Fire Report?",         ["Yes", "No"], key="p_repf")
        witn2   = st.number_input("Witnesses",                 0, 10, 1, key="p_wit")
        tpa     = st.selectbox("Third-Party Assessment?",     ["Yes", "No"], key="p_tpa")
        m_last  = st.number_input("Months Since Last Claim",  0, 120, 24, key="p_mlast")

    st.markdown("---")
    if st.button("🔍 Analyse Property Claim", key="btn_prop"):
        if not models_trained():
            st.error("⚠️ Models not trained yet. Run `python train.py` first.")
        else:
            with st.spinner("Analysing claim..."):
                claim = {
                    "property_value": prop_v,
                    "claim_amount": c_amt2, "property_age": prop_a,
                    "days_since_policy": d_pol2, "num_past_claims": past_c2,
                    "damage_severity": dmg_sev, "repair_cost_estimate": rep_est,
                    "contractor_verified": 1 if contr=="Yes" else 0,
                    "photos_submitted": 1 if photos=="Yes" else 0,
                    "police_fire_report": 1 if rep_f=="Yes" else 0,
                    "claim_to_value_ratio": ctv_r, "num_witnesses": witn2,
                    "third_party_assessment": 1 if tpa=="Yes" else 0,
                    "months_since_last_claim": m_last,
                }
                predict_fn = load_predictor("property")
                result     = predict_fn(claim)
            st.markdown("### 🎯 Prediction Result")
            render_result(result)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown("## 📊 Model Analytics Dashboard")

    if not models_trained():
        st.warning("⚠️ Models have not been trained yet. "
                   "Run `python train.py` in your terminal first, then refresh.")
    else:
        scenario_map = {
            "🚗 Automobile": "auto",
            "🏥 Health":     "health",
            "🏠 Property":   "property",
        }
        sel = st.selectbox("Select Scenario", list(scenario_map.keys()))
        sc  = scenario_map[sel]

        # ── Plot panels ───────────────────────────────────────────────────
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("### Confusion Matrices")
            show_plot_if_exists(f"plots/{sc}_confusion.png")

        with c_right:
            st.markdown("### ROC Curves")
            show_plot_if_exists(f"plots/{sc}_roc.png")

        c2_left, c2_right = st.columns(2)

        with c2_left:
            st.markdown("### Model Comparison")
            show_plot_if_exists(f"plots/{sc}_model_comparison.png")

        with c2_right:
            st.markdown("### Feature Importance")
            fi = f"plots/{sc}_feature_importance.png"
            if os.path.exists(fi):
                show_plot_if_exists(fi)
            else:
                st.info("Feature importance is only available for tree-based models.")

        # ── Data Distribution ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📦 Dataset Distribution")
        try:
            df = pd.read_csv(f"data/{sc}_insurance.csv")
            counts = df["fraud_label"].value_counts()

            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            fig.patch.set_facecolor("#0F172A")

            # Pie chart
            axes[0].pie(
                counts, labels=["Legitimate", "Fraud"],
                autopct="%1.1f%%", startangle=90,
                colors=["#38BDF8", "#F43F5E"],
                textprops={"color": "#F1F5F9", "fontsize": 11},
                wedgeprops={"edgecolor": "#0F172A", "linewidth": 2},
            )
            axes[0].set_facecolor("#0F172A")
            axes[0].set_title("Class Distribution", color="#F1F5F9", pad=12)

            # Claim amount histogram (if column exists)
            col = "claim_amount" if "claim_amount" in df.columns else df.columns[2]
            for lbl, color in [(0, "#38BDF8"), (1, "#F43F5E")]:
                subset = df[df["fraud_label"] == lbl][col]
                axes[1].hist(subset, bins=30, alpha=0.7, color=color,
                             label=["Legitimate", "Fraud"][lbl],
                             edgecolor="#0F172A", linewidth=0.5)
            axes[1].set_facecolor("#1E293B")
            axes[1].set_xlabel(col.replace("_", " ").title(), color="#F1F5F9", fontsize=9)
            axes[1].set_ylabel("Count", color="#F1F5F9", fontsize=9)
            axes[1].set_title("Claim Distribution by Label", color="#F1F5F9")
            axes[1].legend(facecolor="#0F172A", edgecolor="#334155",
                           labelcolor="#F1F5F9", fontsize=9)
            axes[1].tick_params(colors="#94A3B8")
            for spine in axes[1].spines.values():
                spine.set_edgecolor("#334155")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Summary stats
            st.markdown("---")
            st.markdown("### 📋 Dataset Summary")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Total Records",   f"{len(df):,}")
            cc2.metric("Fraud Cases",     f"{counts.get(1,0):,}")
            cc3.metric("Legit Cases",     f"{counts.get(0,0):,}")
            cc4.metric("Fraud Rate",      f"{counts.get(1,0)/len(df):.1%}")

        except FileNotFoundError:
            st.info("Dataset not found. Run `python train.py` to generate and train.")
