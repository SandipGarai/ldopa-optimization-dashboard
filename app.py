# =============================================================================
# L-DOPA OPTIMIZATION & DECISION SUPPORT DASHBOARD
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import plotly.graph_objects as go

# =============================================================================
# PAGE THEME
# =============================================================================
theme = st.sidebar.radio(
    "Theme",
    ["Light", "Dark"],
    index=0
)
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #262730;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="L-DOPA Optimization & Decision Support",
    layout="wide"
)

sns.set(style="whitegrid", font_scale=1.1)

# =============================================================================
# ACCESSIBILITY & DARK-MODE VISIBILITY FIXES
# =============================================================================
st.markdown(
    """
    <style>
    /* ---------- GLOBAL TEXT ---------- */
    html, body, [class*="css"] {
        color: #FAFAFA;
    }

    /* ---------- HEADERS ---------- */
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
    }

    /* ---------- TABS ---------- */
    button[data-baseweb="tab"] {
        font-size: 16px;
        color: #FAFAFA !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #2C7BE5;
        font-weight: 600;
    }

    /* ---------- WIDGET LABELS ---------- */
    label, .stSelectbox label, .stSlider label, .stRadio label {
        color: #FAFAFA !important;
        font-weight: 500;
    }

    /* ---------- SELECTBOX / INPUT TEXT ---------- */
    div[data-baseweb="select"] span {
        color: #111111 !important;
    }

    /* ---------- BUTTON ---------- */
    button[kind="primary"] {
        background-color: #2C7BE5 !important;
        color: white !important;
        font-weight: 600;
        border-radius: 6px;
    }

    /* ---------- METRICS ---------- */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px;
    }

    div[data-testid="metric-container"] label {
        color: #D1D5DB !important;
        font-size: 14px;
    }

    div[data-testid="metric-container"] div {
        color: #FAFAFA !important;
        font-size: 22px;
        font-weight: 700;
    }

    /* ---------- DATAFRAME ---------- */
    .stDataFrame {
        background-color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# PATHS
# =============================================================================
MODEL_PATH = "Results/ML_Model_Selection/Final_Model/CatBoost_Final_Optuna.cbm"
BOOTSTRAP_NPY = "Results/Optimization/Uncertainty/bootstrap_preds_final.npy"
BOOTSTRAP_CSV = "Results/Optimization/Uncertainty/Bootstrap_Predictions.csv"

# =============================================================================
# LOAD MODEL & DATA
# =============================================================================


@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_bootstrap():
    preds = np.load(BOOTSTRAP_NPY)
    df = pd.read_csv(BOOTSTRAP_CSV)
    return preds, df


model = load_model()
bootstrap_preds, boot_df = load_bootstrap()

FACTOR_COLS = [
    c for c in boot_df.columns
    if c not in [
        "Predicted_LDOPA", 
        "Mean_Predicted_LDOPA",
        "Std_Predicted_LDOPA",
        "CI_05",
        "CI_95",
        "Robust_Score"
    ]
]

# =============================================================================
# HEADER
# =============================================================================
st.title("🔬 L-DOPA Process Optimization & Decision Support System")

st.markdown("""
This dashboard enables:

• **Prediction under new experimental conditions**  
• **Uncertainty-aware optimization (μ, σ, CI)**  
• **Robust & risk-averse decision making**  
• **Suggested next validation experiments**
""")

tabs = st.tabs([
    "Prediction",
    "Optimization",
    "Trade-off & Surfaces",
    "Validation",
    "Export"
])

# =============================================================================
# TAB 1 — PREDICTION
# =============================================================================
with tabs[0]:
    st.header("Predict L-DOPA under New Experimental Conditions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        concentration = st.selectbox(
            "Concentration", sorted(boot_df["Concentration"].unique()))
    with col2:
        sl_ratio = st.selectbox(
            "S/L ratio", sorted(boot_df["S/L ratio"].unique()))
    with col3:
        pretreat = st.selectbox(
            "Pre-treatment", sorted(boot_df["Pre-treatments"].unique()))
    with col4:
        time = st.selectbox("Time", sorted(boot_df["Time"].unique()))

    if st.button("Predict L-DOPA"):
        X_new = pd.DataFrame([{
            "Concentration": concentration,
            "S/L ratio": sl_ratio,
            "Pre-treatments": pretreat,
            "Time": time
        }])

        mean_pred = model.predict(X_new)[0]

        mask = (
            (boot_df["Concentration"] == concentration) &
            (boot_df["S/L ratio"] == sl_ratio) &
            (boot_df["Pre-treatments"] == pretreat) &
            (boot_df["Time"] == time)
        )

        idx = boot_df[mask].index[0]
        boot_vals = bootstrap_preds[:, idx]
        std_pred = boot_vals.std()
        ci05, ci95 = np.percentile(boot_vals, [5, 95])

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean L-DOPA", f"{mean_pred:.1f}")
        m2.metric("Uncertainty (σ)", f"{std_pred:.1f}")
        m3.metric("90% CI", f"[{ci05:.1f}, {ci95:.1f}]")

        # Interactive uncertainty plot
        fig = go.Figure()
        fig.add_histogram(x=boot_vals, nbinsx=40, name="Bootstrap")
        fig.add_vline(x=mean_pred, line_color="red", name="Mean")
        fig.add_vline(x=ci05, line_dash="dash",
                      line_color="black", name="CI 5%")
        fig.add_vline(x=ci95, line_dash="dash",
                      line_color="black", name="CI 95%")
        fig.update_layout(
            title="Prediction Uncertainty Distribution",
            xaxis_title="Predicted L-DOPA",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2 — OPTIMIZATION
# =============================================================================
with tabs[1]:
    st.header("Optimal & Robust Experimental Conditions")

    lambda_val = st.slider("Risk Aversion (λ)", 0.0, 2.0, 1.0, 0.1)

    opt_df = boot_df.copy()
    opt_df["Robust_Score"] = (
        opt_df["Mean_Predicted_LDOPA"] -
        lambda_val * opt_df["Std_Predicted_LDOPA"]
    )

    best_mean = opt_df.loc[opt_df["Mean_Predicted_LDOPA"].idxmax()]
    best_robust = opt_df.loc[opt_df["Robust_Score"].idxmax()]
    best_safe = opt_df.loc[opt_df["Std_Predicted_LDOPA"].idxmin()]

    st.dataframe(pd.DataFrame([
        {"Strategy": "Max Yield", **best_mean[FACTOR_COLS].to_dict()},
        {"Strategy": "Robust Optimum", **best_robust[FACTOR_COLS].to_dict()},
        {"Strategy": "Low Risk", **best_safe[FACTOR_COLS].to_dict()}
    ]), use_container_width=True)

# =============================================================================
# TAB 3 — TRADE-OFF & SURFACES
# =============================================================================
with tabs[2]:
    st.header("Mean–Risk Trade-off & Response Surfaces")

    # ---- Pareto with color coding ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(opt_df["Std_Predicted_LDOPA"],
               opt_df["Mean_Predicted_LDOPA"],
               alpha=0.4, label="All")

    ax.scatter(best_mean["Std_Predicted_LDOPA"],
               best_mean["Mean_Predicted_LDOPA"],
               c="red", s=120, label="Max Yield")

    ax.scatter(best_robust["Std_Predicted_LDOPA"],
               best_robust["Mean_Predicted_LDOPA"],
               c="green", s=120, label="Robust")

    ax.scatter(best_safe["Std_Predicted_LDOPA"],
               best_safe["Mean_Predicted_LDOPA"],
               c="blue", s=120, label="Low Risk")

    ax.set_xlabel("Risk (σ)")
    ax.set_ylabel("Expected L-DOPA")
    ax.set_title("Mean–Risk Trade-off")
    ax.legend()
    st.pyplot(fig)

    # ---- 3D Response Surface ----
    st.subheader("3D Response Surface")

    f1 = st.selectbox("X-axis factor", FACTOR_COLS, index=0)
    f2 = st.selectbox("Y-axis factor", FACTOR_COLS, index=1)

    if f1 != f2:
        pivot = opt_df.pivot_table(
            index=f1, columns=f2,
            values="Mean_Predicted_LDOPA", aggfunc="mean"
        )

        fig3d = go.Figure(
            data=[go.Surface(
                z=pivot.values,
                x=list(range(len(pivot.columns))),
                y=list(range(len(pivot.index))),
                colorscale="Viridis"
            )]
        )

        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title=f2,
                           tickvals=list(range(len(pivot.columns))),
                           ticktext=pivot.columns),
                yaxis=dict(title=f1,
                           tickvals=list(range(len(pivot.index))),
                           ticktext=pivot.index),
                zaxis=dict(title="Predicted L-DOPA")
            ),
            height=650
        )

        st.plotly_chart(fig3d, use_container_width=True)

# =============================================================================
# TAB 4 — VALIDATION
# =============================================================================
with tabs[3]:
    st.header("Suggested Validation Experiments")

    opt_df["Exploration_Score"] = (
        opt_df["Std_Predicted_LDOPA"] /
        opt_df["Mean_Predicted_LDOPA"]
    )

    st.dataframe(
        opt_df.sort_values("Exploration_Score", ascending=False)
        .head(3)[FACTOR_COLS + ["Mean_Predicted_LDOPA", "Std_Predicted_LDOPA"]],
        use_container_width=True
    )

# =============================================================================
# TAB 5 — EXPORT
# =============================================================================
with tabs[4]:
    st.header("Export Decision Report")

    def generate_pdf(summary):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = [Paragraph("L-DOPA Optimization Report",
                           styles["Title"]), Spacer(1, 12)]

        for k, v in summary.items():
            story.append(Paragraph(f"<b>{k}</b>: {v}", styles["Normal"]))
            story.append(Spacer(1, 8))

        doc.build(story)
        buffer.seek(0)
        return buffer

    if st.button("Download PDF"):
        pdf = generate_pdf({
            "Max Yield": best_mean[FACTOR_COLS].to_dict(),
            "Robust Optimum": best_robust[FACTOR_COLS].to_dict(),
            "Low Risk": best_safe[FACTOR_COLS].to_dict()
        })
        st.download_button("Download PDF", pdf,
                           "LDOPA_Report.pdf", "application/pdf")


