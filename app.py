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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors as rl_colors
import plotly.graph_objects as go
import plotly.express as px
import os

# =============================================================================
# PAGE CONFIG  ← must be first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="L-DOPA Optimization Dashboard",
    layout="wide",
    page_icon="🔬"
)

# =============================================================================
# THEME
# =============================================================================
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"], index=0)
if theme == "Dark":
    st.markdown("""<style>.stApp{background-color:#0E1117;color:#FAFAFA;}</style>""",
                unsafe_allow_html=True)

sns.set(style="whitegrid", font_scale=1.1)

# =============================================================================
# PATHS
# =============================================================================
MODEL_PATH    = "Results/ML_Model_Selection/Final_Model/CatBoost_Final_Optuna.cbm"
BOOTSTRAP_NPY = "Results/Optimization/Uncertainty/bootstrap_preds_final.npy"
BOOTSTRAP_CSV = "Results/Optimization/Uncertainty/Bootstrap_Predictions.csv"
MULTI_LAMBDA_CSV = "Results/Optimization/Uncertainty/Multi_Lambda/Multi_Lambda_Optimal_Conditions.csv"

# =============================================================================
# LOAD
# =============================================================================
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_bootstrap():
    preds = np.load(BOOTSTRAP_NPY)
    df    = pd.read_csv(BOOTSTRAP_CSV)
    return preds, df

@st.cache_data
def load_multi_lambda():
    if os.path.exists(MULTI_LAMBDA_CSV):
        return pd.read_csv(MULTI_LAMBDA_CSV)
    return None

model         = load_model()
bootstrap_preds, boot_df = load_bootstrap()
multi_lambda_df = load_multi_lambda()

FACTOR_COLS = [c for c in boot_df.columns
               if c not in ["Predicted_LDOPA","Mean_Predicted_LDOPA",
                             "Std_Predicted_LDOPA","CI_05","CI_95","Robust_Score"]]

# Pre-compute standard strategies at λ=1
opt_df_base = boot_df.copy()
opt_df_base["Robust_Score"] = (opt_df_base["Mean_Predicted_LDOPA"]
                                - 1.0 * opt_df_base["Std_Predicted_LDOPA"])
best_mean   = opt_df_base.loc[opt_df_base["Mean_Predicted_LDOPA"].idxmax()]
best_robust = opt_df_base.loc[opt_df_base["Robust_Score"].idxmax()]
best_safe   = opt_df_base.loc[opt_df_base["Std_Predicted_LDOPA"].idxmin()]

# =============================================================================
# HEADER
# =============================================================================
st.title("🔬 L-DOPA Process Optimization & Decision Support System")
st.caption("ICAR-Indian Institute of Agricultural Biotechnology, Ranchi | "
           "CatBoost + Bootstrap Uncertainty | Multi-λ Robust Optimization")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Total Conditions",  f"{len(boot_df):,}")
col_m2.metric("Bootstrap Samples", f"{bootstrap_preds.shape[0]:,}")
col_m3.metric("Max Mean Yield",    f"{boot_df['Mean_Predicted_LDOPA'].max():,.0f} ng/g")
col_m4.metric("Min Uncertainty",   f"{boot_df['Std_Predicted_LDOPA'].min():.0f} ng/g")

st.divider()

tabs = st.tabs(["🔍 Prediction",
                "📊 Optimization & Multi-λ",
                "📈 Trade-off & Surfaces",
                "🧪 Validation Experiments",
                "📤 Export"])

# =============================================================================
# TAB 1 — PREDICTION
# =============================================================================
with tabs[0]:
    st.header("Predict L-DOPA under New Experimental Conditions")
    st.markdown("Enter a new combination of conditions to get the model prediction with bootstrap uncertainty.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        concentration = st.selectbox("Solvent / Concentration",
                                     sorted(boot_df["Concentration"].unique()))
    with col2:
        sl_ratio = st.selectbox("S/L Ratio",
                                sorted(boot_df["S/L ratio"].unique()))
    with col3:
        pretreat = st.selectbox("Pre-treatment",
                                sorted(boot_df["Pre-treatments"].unique()))
    with col4:
        time = st.selectbox("Extraction Time",
                            sorted(boot_df["Time"].unique()))

    if st.button("▶ Predict L-DOPA", type="primary"):
        X_new = pd.DataFrame([{"Concentration": concentration,
                                "S/L ratio": sl_ratio,
                                "Pre-treatments": pretreat,
                                "Time": time}])
        mean_pred = model.predict(X_new)[0]

        mask = ((boot_df["Concentration"] == concentration) &
                (boot_df["S/L ratio"] == sl_ratio) &
                (boot_df["Pre-treatments"] == pretreat) &
                (boot_df["Time"] == time))
        idx       = boot_df[mask].index[0]
        boot_vals = bootstrap_preds[:, idx]
        std_pred  = boot_vals.std()
        ci05, ci95 = np.percentile(boot_vals, [5, 95])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Point Prediction",  f"{mean_pred:,.1f} ng/g")
        m2.metric("Bootstrap Mean",    f"{boot_vals.mean():,.1f} ng/g")
        m3.metric("Uncertainty (σ)",   f"{std_pred:,.1f} ng/g")
        m4.metric("90% CI",            f"[{ci05:,.0f} – {ci95:,.0f}]")

        fig = go.Figure()
        fig.add_histogram(x=boot_vals, nbinsx=40, name="Bootstrap distribution",
                          marker_color="steelblue", opacity=0.75)
        fig.add_vline(x=boot_vals.mean(), line_color="red",  line_width=2,
                      annotation_text=f"Mean={boot_vals.mean():,.0f}")
        fig.add_vline(x=ci05, line_dash="dash", line_color="gray",
                      annotation_text=f"CI5%={ci05:,.0f}")
        fig.add_vline(x=ci95, line_dash="dash", line_color="gray",
                      annotation_text=f"CI95%={ci95:,.0f}")
        fig.update_layout(title="Bootstrap Prediction Uncertainty Distribution",
                          xaxis_title="Predicted L-DOPA (ng/g DW)",
                          yaxis_title="Frequency", height=400)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2 — OPTIMIZATION & MULTI-LAMBDA
# =============================================================================
with tabs[1]:
    st.header("Optimal Conditions — Multi-λ Robust Analysis")

    st.markdown(r"""
    The **Robust Score** trades expected yield against prediction uncertainty:
    $$\text{Robust Score}(x;\lambda) = \mu(x) - \lambda \cdot \sigma(x)$$
    - **λ = 0** → pure yield maximisation (ignore uncertainty)  
    - **λ = 1** → balanced (standard)  
    - **λ ≥ 1.5** → high risk-aversion (prioritise reproducibility)
    """)

    st.subheader("Select λ Values to Compare")
    col_l, col_r = st.columns([3, 1])
    with col_l:
        PRESET = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        selected_lambdas = st.multiselect(
            "Choose λ values:", options=PRESET,
            default=[0.0, 0.5, 1.0, 1.5, 2.0])
    with col_r:
        custom_lam = st.number_input("Custom λ:", min_value=0.0, max_value=5.0,
                                      value=1.0, step=0.05)
        if st.button("➕ Add"):
            if custom_lam not in selected_lambdas:
                selected_lambdas = sorted(set(selected_lambdas) | {round(custom_lam, 4)})

    if not selected_lambdas:
        st.warning("Select at least one λ.")
        st.stop()

    # Compute best condition per lambda from live data
    rows = []
    for lam in sorted(selected_lambdas):
        tmp = boot_df.copy()
        tmp["RS"] = tmp["Mean_Predicted_LDOPA"] - lam * tmp["Std_Predicted_LDOPA"]
        best = tmp.loc[tmp["RS"].idxmax()]
        row = {"λ": lam,
               "Robust Score": round(best["RS"], 1),
               "Mean (ng/g)": round(best["Mean_Predicted_LDOPA"], 1),
               "σ (ng/g)": round(best["Std_Predicted_LDOPA"], 1),
               "CI 5%": round(best["CI_05"], 1),
               "CI 95%": round(best["CI_95"], 1)}
        for c in FACTOR_COLS:
            row[c] = best[c]
        rows.append(row)

    result_df = pd.DataFrame(rows)
    display_cols = ["λ"] + FACTOR_COLS + ["Mean (ng/g)", "σ (ng/g)", "CI 5%", "CI 95%", "Robust Score"]
    st.subheader("Best Condition at Each Selected λ")
    st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)

    # Pre-computed table from basic.py (if available)
    if multi_lambda_df is not None:
        with st.expander("📂 Pre-computed full λ table from basic.py analysis"):
            st.dataframe(multi_lambda_df, use_container_width=True)

    # Figure: Mean & Robust Score vs lambda
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=result_df["λ"], y=result_df["Mean (ng/g)"],
        mode="lines+markers+text", name="Mean L-DOPA",
        line=dict(color="royalblue", width=2),
        text=[f"{v:,.0f}" for v in result_df["Mean (ng/g)"]],
        textposition="top center"))
    fig1.add_trace(go.Scatter(
        x=result_df["λ"], y=result_df["Robust Score"],
        mode="lines+markers+text", name="Robust Score",
        line=dict(color="tomato", width=2, dash="dash"),
        text=[f"{v:,.0f}" for v in result_df["Robust Score"]],
        textposition="bottom center"))
    # ±σ band
    fig1.add_trace(go.Scatter(
        x=result_df["λ"], y=result_df["Mean (ng/g)"]+result_df["σ (ng/g)"],
        fill=None, mode="lines", line=dict(color="lightblue", width=0), showlegend=False))
    fig1.add_trace(go.Scatter(
        x=result_df["λ"], y=result_df["Mean (ng/g)"]-result_df["σ (ng/g)"],
        fill="tonexty", mode="lines", line=dict(color="lightblue", width=0),
        name="±1σ band", fillcolor="rgba(100,149,237,0.12)"))
    fig1.update_layout(
        title="Mean Yield and Robust Score vs Risk-Aversion (λ)",
        xaxis_title="λ", yaxis_title="L-DOPA (ng/g DW)",
        legend=dict(orientation="h", y=-0.25), height=430)
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = go.Figure()
        fig2.add_bar(x=[str(l) for l in result_df["λ"]], y=result_df["σ (ng/g)"],
                     marker_color="mediumseagreen",
                     text=[f"{v:.0f}" for v in result_df["σ (ng/g)"]],
                     textposition="outside")
        fig2.update_layout(title="Uncertainty (σ) of Best Condition at Each λ",
                           xaxis_title="λ", yaxis_title="σ (ng/g DW)", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=result_df["λ"], y=result_df["CI 95%"],
            mode="lines", name="CI 95%", line=dict(color="tomato", dash="dot")))
        fig3.add_trace(go.Scatter(
            x=result_df["λ"], y=result_df["Mean (ng/g)"],
            mode="lines+markers", name="Mean", line=dict(color="royalblue", width=2)))
        fig3.add_trace(go.Scatter(
            x=result_df["λ"], y=result_df["CI 5%"],
            mode="lines", name="CI 5%", line=dict(color="steelblue", dash="dot")))
        fig3.add_traces([go.Scatter(
            x=result_df["λ"], y=result_df["CI 95%"], fill=None,
            mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False),
            go.Scatter(x=result_df["λ"], y=result_df["CI 5%"],
                       fill="tonexty", mode="lines",
                       line=dict(color="rgba(0,0,0,0)"),
                       fillcolor="rgba(100,149,237,0.12)", name="90% CI band")])
        fig3.update_layout(title="90% CI Width vs λ",
                           xaxis_title="λ", yaxis_title="L-DOPA (ng/g DW)", height=380)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Reference Strategies at λ = 1.0")
    st.dataframe(pd.DataFrame([
        {"Strategy": "🎯 Max Yield",      **best_mean[FACTOR_COLS].to_dict(),
         "Mean (ng/g)": round(best_mean["Mean_Predicted_LDOPA"],1),
         "σ (ng/g)": round(best_mean["Std_Predicted_LDOPA"],1)},
        {"Strategy": "⚖ Robust Optimum", **best_robust[FACTOR_COLS].to_dict(),
         "Mean (ng/g)": round(best_robust["Mean_Predicted_LDOPA"],1),
         "σ (ng/g)": round(best_robust["Std_Predicted_LDOPA"],1)},
        {"Strategy": "🛡 Low Risk",       **best_safe[FACTOR_COLS].to_dict(),
         "Mean (ng/g)": round(best_safe["Mean_Predicted_LDOPA"],1),
         "σ (ng/g)": round(best_safe["Std_Predicted_LDOPA"],1)},
    ]), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3 — TRADE-OFF & SURFACES
# =============================================================================
with tabs[2]:
    st.header("Mean–Risk Trade-off & Response Surfaces")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Mean–Risk Pareto")
        fig_p, ax_p = plt.subplots(figsize=(5, 4))
        ax_p.scatter(opt_df_base["Std_Predicted_LDOPA"],
                     opt_df_base["Mean_Predicted_LDOPA"],
                     alpha=0.4, s=15, color="#aec7e8")
        ax_p.scatter(best_mean["Std_Predicted_LDOPA"], best_mean["Mean_Predicted_LDOPA"],
                     c="red", s=120, zorder=5, marker="*", label="Max Yield")
        ax_p.scatter(best_robust["Std_Predicted_LDOPA"], best_robust["Mean_Predicted_LDOPA"],
                     c="green", s=120, zorder=5, marker="D", label="Robust (λ=1)")
        ax_p.scatter(best_safe["Std_Predicted_LDOPA"], best_safe["Mean_Predicted_LDOPA"],
                     c="blue", s=120, zorder=5, marker="s", label="Min Risk")
        ax_p.set_xlabel("Risk σ (ng/g DW)"); ax_p.set_ylabel("Mean L-DOPA (ng/g DW)")
        ax_p.legend(fontsize=8); ax_p.grid(True, alpha=0.3)
        st.pyplot(fig_p)

    with c2:
        st.subheader("3D Response Surface")
        f1 = st.selectbox("X-axis factor", FACTOR_COLS, index=0)
        f2 = st.selectbox("Y-axis factor", FACTOR_COLS, index=1)
        if f1 != f2:
            pivot = opt_df_base.pivot_table(
                index=f1, columns=f2,
                values="Mean_Predicted_LDOPA", aggfunc="mean")
            fig3d = go.Figure(data=[go.Surface(
                z=pivot.values,
                x=list(range(len(pivot.columns))),
                y=list(range(len(pivot.index))),
                colorscale="Viridis")])
            fig3d.update_layout(scene=dict(
                xaxis=dict(title=f2, tickvals=list(range(len(pivot.columns))),
                           ticktext=list(pivot.columns)),
                yaxis=dict(title=f1, tickvals=list(range(len(pivot.index))),
                           ticktext=list(pivot.index)),
                zaxis=dict(title="Mean L-DOPA (ng/g)")), height=500)
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Select two different factors.")

    st.subheader("Uncertainty Heatmap (σ)")
    f_h1 = st.selectbox("Row factor", FACTOR_COLS, index=0, key="h1")
    f_h2 = st.selectbox("Column factor", FACTOR_COLS, index=1, key="h2")
    if f_h1 != f_h2:
        pivot_std = opt_df_base.pivot_table(
            index=f_h1, columns=f_h2,
            values="Std_Predicted_LDOPA", aggfunc="mean")
        fig_hm, ax_hm = plt.subplots(figsize=(9, 4))
        sns.heatmap(pivot_std, ax=ax_hm, cmap="magma", annot=True, fmt=".0f",
                    cbar_kws={"label": "Prediction σ (ng/g)"})
        ax_hm.set_title(f"Prediction Uncertainty: {f_h1} × {f_h2}")
        st.pyplot(fig_hm)

# =============================================================================
# TAB 4 — VALIDATION
# =============================================================================
with tabs[3]:
    st.header("Suggested Validation Experiments")
    st.markdown("""
    Conditions with high **exploration score** (σ / μ) are most informative for 
    further experimental validation — high uncertainty means the model is least certain 
    and new data would most reduce it.
    """)

    opt_df_val = opt_df_base.copy()
    opt_df_val["Exploration_Score"] = (opt_df_val["Std_Predicted_LDOPA"] /
                                        opt_df_val["Mean_Predicted_LDOPA"])
    top_explore = (opt_df_val.sort_values("Exploration_Score", ascending=False)
                   .head(5)[FACTOR_COLS + ["Mean_Predicted_LDOPA",
                                           "Std_Predicted_LDOPA", "Exploration_Score"]])
    top_explore.columns = FACTOR_COLS + ["Mean (ng/g)", "σ (ng/g)", "Exploration Score"]
    st.subheader("Top-5 High-Uncertainty (Explore) Conditions")
    st.dataframe(top_explore.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Top robust conditions
    top_robust_val = (opt_df_val.sort_values("Robust_Score", ascending=False)
                      .head(10)[FACTOR_COLS + ["Mean_Predicted_LDOPA",
                                               "Std_Predicted_LDOPA", "CI_05", "CI_95"]])
    top_robust_val.columns = FACTOR_COLS + ["Mean (ng/g)", "σ (ng/g)", "CI 5%", "CI 95%"]
    st.subheader("Top-10 Robust Conditions (λ=1.0) for Validation")
    st.dataframe(top_robust_val.reset_index(drop=True), use_container_width=True, hide_index=True)

    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(
        x=list(range(10)),
        y=top_robust_val["Mean (ng/g)"].values,
        mode="markers", marker=dict(size=10, color="royalblue"),
        name="Mean", error_y=dict(
            type="data", symmetric=False,
            array=top_robust_val["CI 95%"].values - top_robust_val["Mean (ng/g)"].values,
            arrayminus=top_robust_val["Mean (ng/g)"].values - top_robust_val["CI 5%"].values,
            color="lightblue", thickness=2)))
    fig_ci.update_layout(
        title="Top-10 Robust Conditions: Mean ± 90% Bootstrap CI",
        xaxis=dict(title="Robust Rank", tickvals=list(range(10)),
                   ticktext=[f"#{i+1}" for i in range(10)]),
        yaxis_title="L-DOPA (ng/g DW)", height=400)
    st.plotly_chart(fig_ci, use_container_width=True)

# =============================================================================
# TAB 5 — EXPORT
# =============================================================================
with tabs[4]:
    st.header("Export Decision Report")

    export_lambdas = sorted(selected_lambdas) if 'selected_lambdas' in dir() else [0.0,0.5,1.0,1.5,2.0]
    export_rows = []
    for lam in export_lambdas:
        tmp = boot_df.copy()
        tmp["RS"] = tmp["Mean_Predicted_LDOPA"] - lam * tmp["Std_Predicted_LDOPA"]
        best = tmp.loc[tmp["RS"].idxmax()]
        export_rows.append({
            "Lambda": lam,
            **{c: best[c] for c in FACTOR_COLS},
            "Mean_LDOPA": round(best["Mean_Predicted_LDOPA"], 2),
            "Sigma": round(best["Std_Predicted_LDOPA"], 2),
            "CI_05": round(best["CI_05"], 2),
            "CI_95": round(best["CI_95"], 2),
            "Robust_Score": round(best["RS"], 2)
        })

    def generate_pdf(rows):
        buf = BytesIO()
        doc = SimpleDocTemplate(buf)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("L-DOPA Optimization Report — Multi-λ Decision Summary", styles["Title"]),
            Spacer(1, 12),
            Paragraph("Robust Score = Mean Predicted L-DOPA − λ × σ", styles["Normal"]),
            Spacer(1, 12)
        ]
        # Table
        table_data = [["λ","Concentration","S/L ratio","Pre-treatment","Time",
                        "Mean (ng/g)","σ (ng/g)","CI 5%","CI 95%","Score"]]
        for r in rows:
            table_data.append([
                str(r["Lambda"]), r.get("Concentration",""), r.get("S/L ratio",""),
                r.get("Pre-treatments",""), r.get("Time",""),
                str(r["Mean_LDOPA"]), str(r["Sigma"]),
                str(r["CI_05"]), str(r["CI_95"]), str(r["Robust_Score"])
            ])
        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), rl_colors.HexColor("#2E4057")),
            ("TEXTCOLOR",  (0,0), (-1,0), rl_colors.white),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor("#F5F7FA")]),
            ("GRID",       (0,0), (-1,-1), 0.3, rl_colors.grey),
        ]))
        story.append(t)
        doc.build(story)
        buf.seek(0)
        return buf

    c_pdf, c_csv = st.columns(2)
    with c_pdf:
        if st.button("📄 Generate PDF Report", type="primary"):
            pdf = generate_pdf(export_rows)
            st.download_button("⬇ Download PDF", pdf,
                               "LDOPA_MultiLambda_Report.pdf", "application/pdf")
    with c_csv:
        csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode()
        st.download_button("📊 Download CSV", csv_bytes,
                           "LDOPA_MultiLambda_Summary.csv", "text/csv")

    # Full bootstrap predictions
    st.subheader("Full Bootstrap Predictions Table")
    st.dataframe(boot_df.sort_values("Mean_Predicted_LDOPA", ascending=False),
                 use_container_width=True)
    full_csv = boot_df.to_csv(index=False).encode()
    st.download_button("⬇ Download Full Bootstrap CSV", full_csv,
                       "Bootstrap_All_Conditions.csv", "text/csv")
