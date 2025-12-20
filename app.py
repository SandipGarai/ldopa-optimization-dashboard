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

        # Interactive uncertainty plot with theme-aware colors
        fig = go.Figure()
        fig.add_histogram(x=boot_vals, nbinsx=40, name="Bootstrap",
                         marker_color="#2C7BE5")
        fig.add_vline(x=mean_pred, line_color="red", 
                     annotation_text="Mean", annotation_position="top")
        fig.add_vline(x=ci05, line_dash="dash", line_color="orange",
                     annotation_text="CI 5%", annotation_position="top left")
        fig.add_vline(x=ci95, line_dash="dash", line_color="orange",
                     annotation_text="CI 95%", annotation_position="top right")
        
        fig.update_layout(
            title="Prediction Uncertainty Distribution",
            xaxis_title="Predicted L-DOPA",
            yaxis_title="Frequency",
            template=plotly_template,
            showlegend=True,
            height=500
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

    # ---- Pareto with color coding ----
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set background color based on theme
    if theme == "Dark":
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        text_color = '#FFFFFF'
    else:
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        text_color = '#262730'
    
    ax.scatter(opt_df["Std_Predicted_LDOPA"],
               opt_df["Mean_Predicted_LDOPA"],
               alpha=0.4, label="All", c='gray')

    ax.scatter(best_mean["Std_Predicted_LDOPA"],
               best_mean["Mean_Predicted_LDOPA"],
               c="red", s=150, label="Max Yield", edgecolors='white', linewidths=2)

    ax.scatter(best_robust["Std_Predicted_LDOPA"],
               best_robust["Mean_Predicted_LDOPA"],
               c="green", s=150, label="Robust", edgecolors='white', linewidths=2)

    ax.scatter(best_safe["Std_Predicted_LDOPA"],
               best_safe["Mean_Predicted_LDOPA"],
               c="blue", s=150, label="Low Risk", edgecolors='white', linewidths=2)

    ax.set_xlabel("Risk (σ)", color=text_color, fontsize=12)
    ax.set_ylabel("Expected L-DOPA", color=text_color, fontsize=12)
    ax.set_title("Mean–Risk Trade-off", color=text_color, fontsize=14, fontweight='bold')
    ax.tick_params(colors=text_color)
    ax.legend(facecolor=fig.patch.get_facecolor(), edgecolor=text_color, 
             labelcolor=text_color)
    
    # Set spine colors
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)
    
    st.pyplot(fig)

    # ---- 3D Response Surface ----
    st.subheader("3D Response Surface")

    col_a, col_b = st.columns(2)
    with col_a:
        f1 = st.selectbox("X-axis factor", FACTOR_COLS, index=0)
    with col_b:
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
                colorscale="Viridis",
                colorbar=dict(title="L-DOPA")
            )]
        )

        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title=f2,
                           tickvals=list(range(len(pivot.columns))),
                           ticktext=[str(c) for c in pivot.columns]),
                yaxis=dict(title=f1,
                           tickvals=list(range(len(pivot.index))),
                           ticktext=[str(i) for i in pivot.index]),
                zaxis=dict(title="Predicted L-DOPA")
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

    validation_df = opt_df.sort_values("Exploration_Score", ascending=False).head(3)
    display_cols = FACTOR_COLS + ["Mean_Predicted_LDOPA", "Std_Predicted_LDOPA", "Exploration_Score"]
    
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
