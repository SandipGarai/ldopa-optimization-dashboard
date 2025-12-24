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
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="L-DOPA Optimization & Decision Support",
    layout="wide"
)

# =============================================================================
# SESSION STATE INITIALIZATION (CRITICAL)
# =============================================================================
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
    st.session_state.prediction_results = None

# =============================================================================
# THEME CONFIGURATION WITH ENHANCED VISIBILITY
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
        /* Main background */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* Headers and titles */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
        }
        
        /* Regular text and paragraphs */
        p, span, div, label {
            color: #E0E0E0 !important;
        }
        
        /* Tab labels */
        .stTabs [data-baseweb="tab-list"] button {
            color: #FFFFFF !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #2C7BE5 !important;
        }
        
        /* Metric labels and values */
        [data-testid="stMetricLabel"] {
            color: #E0E0E0 !important;
        }
        
        [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }
        
        /* Selectbox and input labels */
        .stSelectbox label, .stSlider label {
            color: #FFFFFF !important;
        }
        
        /* Selectbox dropdown */
        .stSelectbox > div > div {
            background-color: #262730 !important;
            color: #FFFFFF !important;
        }
        
        /* Selectbox selected value */
        .stSelectbox [data-baseweb="select"] > div {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
        }
        
        /* Dropdown options */
        [role="option"] {
            background-color: #262730 !important;
            color: #FFFFFF !important;
        }
        
        [role="option"]:hover {
            background-color: #2C7BE5 !important;
            color: #FFFFFF !important;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #2C7BE5 !important;
            color: #FFFFFF !important;
            border: none !important;
        }
        
        .stButton > button:hover {
            background-color: #1e5bb8 !important;
            color: #FFFFFF !important;
        }
        
        /* Download button */
        .stDownloadButton > button {
            background-color: #2C7BE5 !important;
            color: #FFFFFF !important;
        }
        
        /* Slider */
        .stSlider [data-baseweb="slider"] {
            background-color: #262730 !important;
        }
        
        /* DataFrame */
        .stDataFrame {
            color: #E0E0E0 !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E !important;
        }
        
        [data-testid="stSidebar"] * {
            color: #E0E0E0 !important;
        }
        
        /* Radio buttons */
        .stRadio label {
            color: #E0E0E0 !important;
        }
        
        /* Markdown text */
        .stMarkdown {
            color: #E0E0E0 !important;
        }
        
        /* Code blocks */
        code {
            color: #FF6B6B !important;
            background-color: #262730 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    plotly_template = "plotly_dark"
    mpl_style = "dark_background"
else:
    st.markdown(
        """
        <style>
        /* Light theme styling */
        .stApp {
            background-color: #FFFFFF;
            color: #262730;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #262730 !important;
        }
        
        p, span, div, label {
            color: #262730 !important;
        }
        
        .stButton > button {
            background-color: #2C7BE5 !important;
            color: #FFFFFF !important;
        }
        
        .stButton > button:hover {
            background-color: #1e5bb8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    plotly_template = "plotly_white"
    mpl_style = "seaborn-v0_8-whitegrid"

# Set matplotlib style based on theme
plt.style.use(mpl_style if theme == "Dark" else "seaborn-v0_8-whitegrid")
sns.set(style="whitegrid", font_scale=1.1)

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

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        concentration = st.selectbox(
            "Concentration", sorted(boot_df["Concentration"].unique()))
    with c2:
        sl_ratio = st.selectbox(
            "S/L ratio", sorted(boot_df["S/L ratio"].unique()))
    with c3:
        pretreat = st.selectbox(
            "Pre-treatments", sorted(boot_df["Pre-treatments"].unique()))
    with c4:
        time = st.selectbox("Time", sorted(boot_df["Time"].unique()))

    # ---------------------------
    # BUTTON
    # ---------------------------
    if st.button("Predict L-DOPA"):
        X_new = pd.DataFrame([{
            "Concentration": concentration,
            "S/L ratio": sl_ratio,
            "Pre-treatments": pretreat,
            "Time": time
        }])

        mask = (
            (boot_df["Concentration"] == concentration) &
            (boot_df["S/L ratio"] == sl_ratio) &
            (boot_df["Pre-treatments"] == pretreat) &
            (boot_df["Time"] == time)
        )

        # ---- Existing combination → bootstrap posterior ----
        if mask.any():
            idx = boot_df[mask].index[0]
            mean_pred = boot_df.loc[idx, "Mean_Predicted_LDOPA"]
            std_pred = boot_df.loc[idx, "Std_Predicted_LDOPA"]
            ci05 = boot_df.loc[idx, "CI_05"]
            ci95 = boot_df.loc[idx, "CI_95"]
            boot_vals = bootstrap_preds[:, idx]

        # ---- Truly new combination ----
        else:
            mean_pred = model.predict(X_new)[0]
            std_pred = boot_df["Std_Predicted_LDOPA"].mean()
            boot_vals = np.random.normal(mean_pred, std_pred, 1000)
            ci05, ci95 = np.percentile(boot_vals, [5, 95])

        st.session_state.prediction_done = True
        st.session_state.prediction_results = {
            "mean": mean_pred,
            "std": std_pred,
            "ci05": ci05,
            "ci95": ci95,
            "boot_vals": boot_vals
        }

    # ---------------------------
    # DISPLAY (ONLY AFTER CLICK)
    # ---------------------------
    if st.session_state.prediction_done:
        res = st.session_state.prediction_results

        m1, m2, m3 = st.columns(3)
        m1.metric("Posterior Mean LDOPA (μ)", f"{res['mean']:.2f}")
        m2.metric("Uncertainty (σ)", f"{res['std']:.2f}")
        m3.metric("90% CI", f"[{res['ci05']:.2f}, {res['ci95']:.2f}]")

        fig = go.Figure()
        fig.add_histogram(x=res["boot_vals"], nbinsx=40)
        fig.add_vline(x=res["mean"], line_color="red", annotation_text="Mean")
        fig.add_vline(x=res["ci05"], line_dash="dash", line_color="orange")
        fig.add_vline(x=res["ci95"], line_dash="dash", line_color="orange")

        fig.update_layout(
            title="Prediction Uncertainty Distribution",
            template=plotly_template,
            height=450
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

    st.subheader("Recommended Experimental Strategies")

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

    # =========================================================
    # INTERACTIVE MEAN–RISK PARETO FRONT (PLOTLY)
    # =========================================================

    hover_text = []
    for _, row in opt_df.iterrows():
        txt = "<br>".join([
            f"<b>Concentration:</b> {row['Concentration']}",
            f"<b>S/L ratio:</b> {row['S/L ratio']}",
            f"<b>Pre-treatment:</b> {row['Pre-treatments']}",
            f"<b>Time:</b> {row['Time']}",
            f"<b>Mean LDOPA (μ):</b> {row['Mean_Predicted_LDOPA']:.2f}",
            f"<b>Uncertainty (σ):</b> {row['Std_Predicted_LDOPA']:.2f}"
        ])
        hover_text.append(txt)

    fig_pareto = go.Figure()

    # ---- All conditions ----
    fig_pareto.add_trace(go.Scatter(
        x=opt_df["Std_Predicted_LDOPA"],
        y=opt_df["Mean_Predicted_LDOPA"],
        mode="markers",
        marker=dict(
            size=8,
            color=opt_df["Mean_Predicted_LDOPA"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Mean LDOPA")
        ),
        text=hover_text,
        hoverinfo="text",
        name="All Conditions"
    ))

    # ---- Highlight key strategies ----
    def add_highlight(row, name, color):
        fig_pareto.add_trace(go.Scatter(
            x=[row["Std_Predicted_LDOPA"]],
            y=[row["Mean_Predicted_LDOPA"]],
            mode="markers",
            marker=dict(size=16, color=color, symbol="star"),
            text=[f"<b>{name}</b>"],
            hoverinfo="text",
            name=name
        ))

    add_highlight(best_mean, "Max Yield", "red")
    add_highlight(best_robust, "Robust Optimum", "green")
    add_highlight(best_safe, "Low Risk", "blue")

    fig_pareto.update_layout(
        title="Interactive Mean–Risk Trade-off (Pareto Front)",
        xaxis_title="Risk (σ)",
        yaxis_title="Expected L-DOPA (μ)",
        template=plotly_template,
        height=600
    )

    st.plotly_chart(fig_pareto, use_container_width=True)

    # =========================================================
    # INTERACTIVE 3D RESPONSE SURFACE (ALREADY PLOTLY)
    # =========================================================
    st.subheader("3D Response Surface")

    col_a, col_b = st.columns(2)
    with col_a:
        f1 = st.selectbox("X-axis factor", FACTOR_COLS, index=0)
    with col_b:
        f2 = st.selectbox("Y-axis factor", FACTOR_COLS, index=1)

    if f1 != f2:
        pivot = opt_df.pivot_table(
            index=f1,
            columns=f2,
            values="Mean_Predicted_LDOPA",
            aggfunc="mean"
        )

        fig3d = go.Figure(
            data=[go.Surface(
                z=pivot.values,
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                colorscale="Viridis",
                colorbar=dict(title="Mean LDOPA")
            )]
        )

        fig3d.update_layout(
            scene=dict(
                xaxis_title=f2,
                yaxis_title=f1,
                zaxis_title="Mean LDOPA"
            ),
            template=plotly_template,
            height=650
        )

        st.plotly_chart(fig3d, use_container_width=True)


# =============================================================================
# TAB 4 — VALIDATION
# =============================================================================
with tabs[3]:
    st.header("Suggested Validation Experiments")

    st.markdown("""
    These experiments have high uncertainty relative to their predicted yield, 
    making them valuable for model improvement and knowledge discovery.
    """)

    opt_df["Exploration_Score"] = (
        opt_df["Std_Predicted_LDOPA"] /
        opt_df["Mean_Predicted_LDOPA"]
    )

    validation_df = opt_df.sort_values(
        "Exploration_Score", ascending=False).head(3)
    display_cols = FACTOR_COLS + \
        ["Mean_Predicted_LDOPA", "Std_Predicted_LDOPA", "Exploration_Score"]

    st.dataframe(
        validation_df[display_cols].reset_index(drop=True),
        use_container_width=True
    )

# =============================================================================
# TAB 5 — EXPORT
# =============================================================================
with tabs[4]:
    st.header("Export Decision Report")

    st.markdown("""
    Generate a comprehensive PDF report summarizing the optimal experimental 
    conditions identified through the optimization process.
    """)

    def generate_pdf(summary):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("L-DOPA Optimization Report", styles["Title"]),
            Spacer(1, 20),
            Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                      styles["Normal"]),
            Spacer(1, 20)
        ]

        for strategy, conditions in summary.items():
            story.append(Paragraph(f"<b>{strategy}</b>", styles["Heading2"]))
            story.append(Spacer(1, 8))

            for key, value in conditions.items():
                story.append(Paragraph(f"• {key}: {value}", styles["Normal"]))

            story.append(Spacer(1, 16))

        doc.build(story)
        buffer.seek(0)
        return buffer

    col_x, col_y = st.columns([1, 3])

    with col_x:
        if st.button("Generate PDF Report", use_container_width=True):
            pdf = generate_pdf({
                "Maximum Yield Strategy": best_mean[FACTOR_COLS].to_dict(),
                "Robust Optimum Strategy": best_robust[FACTOR_COLS].to_dict(),
                "Low Risk Strategy": best_safe[FACTOR_COLS].to_dict()
            })

            st.download_button(
                "📥 Download PDF",
                pdf,
                "LDOPA_Optimization_Report.pdf",
                "application/pdf",
                use_container_width=True
            )

            st.success("✅ Report generated successfully!")
