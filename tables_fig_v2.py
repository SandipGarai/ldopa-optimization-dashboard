# %%
# =============================================================================
# tables_fig.py — Manuscript figures (Fig 1–10) and tables (Table 1–5)
# =============================================================================
# Run from the same working directory as basic.py after it has completed.
#
# Required files:
#   data final.csv
#   Models/Trained/linear_regression.pkl
#   Models/Trained/random_forest.pkl
#   Models/Trained/catboost_model.cbm
#   Results/ML_Model_Selection/Final_Model/CatBoost_Final_Optuna.cbm
#   Results/Optimization/Uncertainty/Bootstrap_Predictions.csv
#   Results/Optimization/Uncertainty/Robust_Ranking.csv
#   Results/Optimization/LDOPA_Optimization_Full_Ranking.csv
#
# Extra dependency: pip install Pillow
# =============================================================================

from matplotlib.lines import Line2D
import matplotlib as mpl
import os
import io
import warnings
import itertools
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
from PIL import Image as PILImage

# Suppress all third-party warnings cleanly
warnings.filterwarnings("ignore")

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("manuscript_figures", exist_ok=True)
os.makedirs("manuscript_tables",  exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.5)

# Global style: bold axis labels and tick labels, slightly larger
mpl.rcParams.update({
    "axes.labelweight": "bold",
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "legend.fontsize":  10,
    "font.weight":      "normal",   # body text stays normal weight
})

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data final.csv")
X = df.drop(columns=["LDOPA"])
y = df["LDOPA"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Models ────────────────────────────────────────────────────────────────────
lr_pipeline = joblib.load("Models/Trained/linear_regression.pkl")
rf_pipeline = joblib.load("Models/Trained/random_forest.pkl")

cb_baseline = CatBoostRegressor()
cb_baseline.load_model("Models/Trained/catboost_model.cbm")

cb_optuna = CatBoostRegressor()
cb_optuna.load_model(
    "Results/ML_Model_Selection/Final_Model/CatBoost_Final_Optuna.cbm"
)

# Set handle_unknown='ignore' on saved sklearn OneHotEncoders so that any
# category present in X_test but unseen during training is silently zeroed
# instead of raising a ValueError or UserWarning.
for pl in [lr_pipeline, rf_pipeline]:
    for _, step in pl.steps:
        if hasattr(step, "transformers_"):
            for _, sub, _ in step.transformers_:
                if hasattr(sub, "handle_unknown"):
                    sub.handle_unknown = "ignore"

y_pred_lr = lr_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_cb = cb_baseline.predict(X_test)
y_pred_cbo = cb_optuna.predict(X_test)

# ── Optimization results ──────────────────────────────────────────────────────
opt_df = pd.read_csv(
    "Results/Optimization/Uncertainty/Bootstrap_Predictions.csv")
robust_ranking = pd.read_csv(
    "Results/Optimization/Uncertainty/Robust_Ranking.csv")
full_ranking = pd.read_csv(
    "Results/Optimization/LDOPA_Optimization_Full_Ranking.csv")
factors_opt = [c for c in full_ranking.columns if c != "Predicted_LDOPA"]

LAMBDA = 1.0
if "Robust_Score" not in opt_df.columns:
    opt_df["Robust_Score"] = (
        opt_df["Mean_Predicted_LDOPA"] - LAMBDA * opt_df["Std_Predicted_LDOPA"]
    )

# ── SHAP ──────────────────────────────────────────────────────────────────────
explainer_cb = shap.TreeExplainer(cb_optuna)
shap_values_cb = explainer_cb.shap_values(X)

# Ordinal-encoded X for beeswarm colour gradient.
# shap.summary_plot with string/object dtype produces grey dots because it
# cannot build a numeric colour scale from categorical strings.
# OrdinalEncoder maps each level to an integer rank so blue=low, red=high.
X_numeric = OrdinalEncoder().fit_transform(X)


# =============================================================================
# TABLES
# =============================================================================

# Table 1 — Descriptive statistics by factor level
rows = []
for col in ["Concentration", "S/L ratio", "Pre-treatments", "Time"]:
    for level, grp in df.groupby(col)["LDOPA"]:
        rows.append({
            "Factor": col,
            "Level":  level,
            "n":      len(grp),
            "Mean (microg/g DW)": round(grp.mean(), 0),
            "SD":  round(grp.std(),  0),
            "Min": round(grp.min(),  0),
            "Max": round(grp.max(),  0),
        })
pd.DataFrame(rows).to_csv(
    "manuscript_tables/Table1_Descriptive.csv", index=False)
print("Table 1 saved.")

# Table 2 — Kruskal-Wallis results + epsilon-squared effect sizes
df_stat = df.copy()
for col in df_stat.columns[:-1]:
    df_stat[col] = df_stat[col].astype("category")
kw = []
for col in df_stat.columns[:-1]:
    groups = [
        df_stat[df_stat[col] == lv]["LDOPA"].values
        for lv in df_stat[col].cat.categories
    ]
    H, p = stats.kruskal(*groups)
    k = len(groups)
    N = sum(len(g) for g in groups)
    eps2 = round((H - k + 1) / (N - k), 4)   # Tomczak & Tomczak 2014
    effect = "large" if eps2 >= 0.14 else "medium" if eps2 >= 0.06 else "small"
    kw.append({
        "Factor": col,
        "H-statistic": round(H, 3),
        "p-value": f"{p:.2e}",
        "Significance": ("***" if p < 0.001 else
                         "**" if p < 0.01 else
                         "*" if p < 0.05 else "ns"),
        "Epsilon-squared (eps2)": eps2,
        "Effect size": effect,
    })
pd.DataFrame(kw).to_csv(
    "manuscript_tables/Table2_KruskalWallis.csv", index=False)
print("Table 2 saved.")

# Table 3 — ML model performance


def mrow(name, yt, yp):
    return {
        "Model": name,
        "RMSE (microg/g DW)": round(np.sqrt(mean_squared_error(yt, yp))),
        "MAE (microg/g DW)":  round(mean_absolute_error(yt, yp)),
        "R2": round(r2_score(yt, yp), 3),
    }


pd.DataFrame([
    mrow("Linear Regression",       y_test, y_pred_lr),
    mrow("Random Forest",           y_test, y_pred_rf),
    mrow("CatBoost (Baseline)",     y_test, y_pred_cb),
    mrow("CatBoost (Optuna-tuned)", y_test, y_pred_cbo),
]).to_csv("manuscript_tables/Table3_ModelPerformance.csv", index=False)
print("Table 3 saved.")

# Table 4 — SHAP feature importance
simp = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Abs_SHAP (microg/g DW)": np.abs(shap_values_cb).mean(axis=0).round(0),
}).sort_values("Mean_Abs_SHAP (microg/g DW)", ascending=False).reset_index(drop=True)
simp["Rank"] = range(1, len(simp) + 1)
simp.to_csv("manuscript_tables/Table4_SHAP_importance.csv", index=False)
print("Table 4 saved.")

# Table 5 — Top 10 optimal conditions
top10t = (opt_df
          .sort_values("Mean_Predicted_LDOPA", ascending=False)
          .head(10)
          .reset_index(drop=True))
top10t.index += 1
top10t.to_csv("manuscript_tables/Table5_Top10_Conditions.csv")
print("Table 5 saved.")

# Supplementary Table S1 — Model hyperparameter summary
# (Reviewer 1 comment 4-6; Reviewer 2 minor comment)
# Cite in manuscript: "Supplementary Table S1 summarises hyperparameters
#  for all four models; see also Section 3.6."
os.makedirs("manuscript_tables", exist_ok=True)
pd.DataFrame([
    {
        "Model": "Linear Regression",
        "Encoding": "OneHotEncoder (drop='first')",
        "Key hyperparameters": "OLS; no tuning",
        "n_features": "18 (OHE binary)",
        "Random seed": "42 (split only)",
        "Notes": "Analytical solution; baseline comparator",
    },
    {
        "Model": "Random Forest",
        "Encoding": "OneHotEncoder (drop='first')",
        "Key hyperparameters": "n_estimators=300",
        "n_features": "18 (OHE binary)",
        "Random seed": "42 (split + RF init)",
        "Notes": "Default sklearn RF; no hyperparameter tuning",
    },
    {
        "Model": "CatBoost Baseline",
        "Encoding": "Native categorical (ordered target statistics)",
        "Key hyperparameters": "depth=6, learning_rate=0.05, iterations=1000",
        "n_features": "4 (raw categorical)",
        "Random seed": "42 (split + CatBoost)",
        "Notes": "DEPLOYED model for SHAP and bootstrap optimisation",
    },
    {
        "Model": "CatBoost Optuna-tuned",
        "Encoding": "Native categorical (ordered target statistics)",
        "Key hyperparameters": "depth=9, learning_rate=0.0294, iterations=623, l2_leaf_reg=3",
        "n_features": "4 (raw categorical)",
        "Random seed": "42 (split + CatBoost)",
        "Notes": "40 Optuna trials; TPE sampler; 5-fold CV; trained on full dataset (n=792)",
    },
]).to_csv("manuscript_tables/TableS1_Hyperparameters.csv", index=False)
print("Supplementary Table S1 (hyperparameters) saved.")


# =============================================================================
# FIGURES
# =============================================================================

# ---------------------------------------------------------------------------
# Fig 1 — Dataset overview
# Caption: (A) Distribution of observed L-DOPA (µg/g DW) with KDE.
#          (B) L-DOPA by extraction medium (Concentration).
#          (C) L-DOPA by S/L ratio. (D) L-DOPA by time × pre-treatment.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df["LDOPA"], kde=True, color="teal", edgecolor="black",
             ax=axes[0, 0])
axes[0, 0].set_title(
    "(A) Distribution of L-DOPA Concentration", fontweight="bold")
axes[0, 0].set_xlabel("L-DOPA (\u00b5g/g DW)", fontsize=12)
axes[0, 0].set_ylabel("Frequency", fontsize=12)

sns.boxplot(data=df, x="Concentration", y="LDOPA",
            hue="Concentration", palette="Set2", legend=False,
            ax=axes[0, 1])
axes[0, 1].set_xticks(axes[0, 1].get_xticks())
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha="right",
                           fontsize=10)
axes[0, 1].set_title("(B) L-DOPA by Extraction Medium", fontweight="bold")
axes[0, 1].set_xlabel("Extraction Medium (Concentration)", fontsize=12)
axes[0, 1].set_ylabel("L-DOPA (\u00b5g/g DW)", fontsize=12)

sns.boxplot(data=df, x="S/L ratio", y="LDOPA",
            hue="S/L ratio", palette="Set2", legend=False,
            ax=axes[1, 0])
axes[1, 0].set_title("(C) L-DOPA by S/L Ratio", fontweight="bold")
axes[1, 0].set_xlabel("S/L Ratio", fontsize=12)
axes[1, 0].set_ylabel("L-DOPA (\u00b5g/g DW)", fontsize=12)

sns.boxplot(data=df, x="Time", y="LDOPA",
            hue="Pre-treatments", palette="Set1",
            ax=axes[1, 1])
axes[1, 1].set_title(
    "(D) L-DOPA by Time \u00d7 Pre-treatment", fontweight="bold")
axes[1, 1].set_xlabel("Extraction Time", fontsize=12)
axes[1, 1].set_ylabel("L-DOPA (\u00b5g/g DW)", fontsize=12)

plt.tight_layout()
plt.savefig("manuscript_figures/Fig1_data_overview.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 1 saved.")

# ---------------------------------------------------------------------------
# Fig 2 — Main effect plots
# Caption: Main effect of each extraction factor on predicted L-DOPA
#          (µg/g DW). Boxes show median and interquartile range across all
#          predicted values from the full factorial space (n = 264 conditions).
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.flatten()
for i, factor in enumerate(["Concentration", "S/L ratio", "Pre-treatments", "Time"]):
    sns.boxplot(data=full_ranking, x=factor, y="Predicted_LDOPA",
                hue=factor, palette="Set2", legend=False, ax=axes[i])
    axes[i].set_xticks(axes[i].get_xticks())
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right",
                            fontsize=10)
    axes[i].set_xlabel(factor, fontsize=12)
    axes[i].set_ylabel("Predicted L-DOPA (\u00b5g/g DW)", fontsize=12)
    axes[i].set_title(
        f"({chr(65+i)}) Main Effect of {factor}", fontweight="bold")
plt.tight_layout()
plt.savefig("manuscript_figures/Fig2_main_effects.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 2 saved.")

# ---------------------------------------------------------------------------
# Fig 3 — Observed interaction heatmaps (most informative pairs, main figure)
# Caption: Pairwise interaction heatmaps showing observed mean L-DOPA
#          (µg/g DW) for the three most informative factor combinations.
#          Full set of all 6 pairwise interactions: Fig. S1 (supplementary).
# Reviewer note: dense 6-panel figure split — top 3 pairs (involving the two
# most significant factors per KW: Concentration and S/L ratio) in main ms;
# remaining 3 pairs in supplementary Fig. S1.
# ---------------------------------------------------------------------------
# Top 3 pairs (most relevant — involve Concentration and/or S/L ratio)
main_pairs = [("Concentration", "S/L ratio"),
              ("Concentration", "Pre-treatments"),
              ("S/L ratio",     "Pre-treatments")]
# Remaining 3 for supplementary
supp_pairs = [("Concentration", "Time"),
              ("S/L ratio",     "Time"),
              ("Pre-treatments", "Time")]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (f1, f2) in zip(axes, main_pairs):
    pivot = df.groupby([f1, f2])["LDOPA"].mean().unstack()
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".0f",
                annot_kws={"size": 9},
                cbar_kws={
                    "label": "Mean L-DOPA (\u00b5g/g DW)", "shrink": 0.8},
                ax=ax)
    ax.set_title(f"{f1} \u00d7 {f2}", fontweight="bold", fontsize=12)
    ax.set_ylabel(f1, fontsize=11)
    ax.set_xlabel(f2, fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig("manuscript_figures/Fig3_response_surface_observed_main.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 3 (main, 3 pairs) saved.")

# Supplementary Fig. S1 — remaining 3 pairs
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (f1, f2) in zip(axes, supp_pairs):
    pivot = df.groupby([f1, f2])["LDOPA"].mean().unstack()
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".0f",
                annot_kws={"size": 9},
                cbar_kws={
                    "label": "Mean L-DOPA (\u00b5g/g DW)", "shrink": 0.8},
                ax=ax)
    ax.set_title(f"{f1} \u00d7 {f2}", fontweight="bold", fontsize=12)
    ax.set_ylabel(f1, fontsize=11)
    ax.set_xlabel(f2, fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig("manuscript_figures/FigS1_response_surface_observed_supp.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Supplementary Figure S1 (remaining 3 pairs) saved.")

# ---------------------------------------------------------------------------
# Fig 4 — ML model performance bars
# Caption: Predictive performance on hold-out test set (n = 159):
#          (A) RMSE, (B) MAE (both in µg/g DW), (C) R².
# ---------------------------------------------------------------------------
model_labels = ["Linear\nRegression", "Random\nForest",
                "CatBoost\nBaseline", "CatBoost\nOptuna"]
colors = sns.color_palette("Set2", 4)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, vals, lbl, fmt, unit in zip(
    axes,
    [
        [np.sqrt(mean_squared_error(y_test, p))
         for p in [y_pred_lr, y_pred_rf, y_pred_cb, y_pred_cbo]],
        [mean_absolute_error(y_test, p)
         for p in [y_pred_lr, y_pred_rf, y_pred_cb, y_pred_cbo]],
        [r2_score(y_test, p)
         for p in [y_pred_lr, y_pred_rf, y_pred_cb, y_pred_cbo]],
    ],
    ["(A) RMSE", "(B) MAE", "(C) R\u00b2"],
    ["%.0f", "%.0f", "%.3f"],
    ["\u00b5g/g DW", "\u00b5g/g DW", ""],
):
    bars = ax.bar(model_labels, vals, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt=fmt, fontsize=10, padding=3)
    ax.set_title(lbl, fontweight="bold", fontsize=13)
    ax.set_ylabel(f"{lbl.split(') ')[1]} {unit}".strip(), fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("manuscript_figures/Fig4_model_comparison.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 4 saved.")

# ---------------------------------------------------------------------------
# Fig 5 — Actual vs. predicted (4-panel)
# Caption: Actual vs. predicted L-DOPA. Dashed line = 1:1 perfect prediction.
# ---------------------------------------------------------------------------
lims = [y_test.min() - 1000, y_test.max() + 1000]
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()
for ax, (title, yp) in zip(
    axes,
    {
        "(A) Linear Regression": y_pred_lr,
        "(B) Random Forest":     y_pred_rf,
        "(C) CatBoost Baseline": y_pred_cb,
        "(D) CatBoost Optuna":   y_pred_cbo,
    }.items(),
):
    ax.scatter(y_test, yp, alpha=0.5, s=18)
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual L-DOPA")
    ax.set_ylabel("Predicted L-DOPA")
    ax.set_title(f"{title}  (R\u00b2 = {r2_score(y_test, yp):.3f})")
plt.tight_layout()
plt.savefig("manuscript_figures/Fig5_actual_vs_predicted.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 5 saved.")

# ---------------------------------------------------------------------------
# Fig 6 — SHAP feature importance: bar (6A) + beeswarm (6B) combined
# Caption: (A) SHAP bar plot: mean |SHAP| per feature.
#          (B) SHAP beeswarm: distribution of SHAP values across all
#          observations. Colour = feature value (blue = low, red = high).
# Subtitles are added via matplotlib suptitle BEFORE saving to buffer so they
# render at the correct size in the same font as all other figures.
# ---------------------------------------------------------------------------

# (A) Bar plot — add centered bold subtitle via suptitle
shap.summary_plot(shap_values_cb, X, plot_type="bar", show=False)
plt.gcf().suptitle("(A) SHAP Bar Plot: Mean |SHAP| per Feature",
                   fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
_buf_bar = io.BytesIO()
plt.gcf().savefig(_buf_bar, format="png", dpi=150, bbox_inches="tight")
_buf_bar.seek(0)
_img_bar = PILImage.open(_buf_bar).convert("RGB")
plt.close()

# (B) Beeswarm — X_numeric for colour gradient; centered bold subtitle
shap.summary_plot(
    shap_values_cb,
    X_numeric,
    feature_names=X.columns.tolist(),
    show=False,
)
plt.gcf().suptitle("(B) SHAP Beeswarm: Colour = Feature Level (Blue: Low, Red: High)",
                   fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
_buf_bee = io.BytesIO()
plt.gcf().savefig(_buf_bee, format="png", dpi=150, bbox_inches="tight")
_buf_bee.seek(0)
_img_bee = PILImage.open(_buf_bee).convert("RGB")
plt.close()

# Composite A + B side-by-side, matched height
_h = max(_img_bar.height, _img_bee.height)


def _resize_h(img, th):
    r = th / img.height
    return img.resize((int(img.width * r), th), PILImage.LANCZOS)


_bar_r = _resize_h(_img_bar, _h)
_bee_r = _resize_h(_img_bee, _h)
_combined = PILImage.new(
    "RGB", (_bar_r.width + _bee_r.width, _h), (255, 255, 255))
_combined.paste(_bar_r, (0, 0))
_combined.paste(_bee_r, (_bar_r.width, 0))
_combined.save("manuscript_figures/Fig6_SHAP_combined.png", dpi=(300, 300))
print("Figure 6 saved (A+B combined).")

# ---------------------------------------------------------------------------
# Fig 7 — Predicted response surface heatmaps (main: top 3 informative pairs)
# Caption: Response surface heatmaps (predicted mean L-DOPA, µg/g DW) from
#          CatBoost Baseline model for the three most informative factor
#          combinations. Supplementary Fig. S2: full set of 6 pairs.
# ---------------------------------------------------------------------------
# Top 3 pairs for main figure (consistent with Fig 3 split)
main_pairs_opt = [("Concentration", "S/L ratio"),
                  ("Concentration", "Pre-treatments"),
                  ("S/L ratio",     "Pre-treatments")]
supp_pairs_opt = [("Concentration", "Time"),
                  ("S/L ratio",     "Time"),
                  ("Pre-treatments", "Time")]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (f1, f2) in zip(axes, main_pairs_opt):
    pivot = full_ranking.pivot_table(
        index=f1, columns=f2, values="Predicted_LDOPA", aggfunc="mean"
    )
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".0f",
                annot_kws={"size": 9},
                cbar_kws={
                    "label": "Predicted L-DOPA (\u00b5g/g DW)", "shrink": 0.8},
                ax=ax)
    ax.set_title(f"{f1} \u00d7 {f2}", fontweight="bold", fontsize=12)
    ax.set_ylabel(f1, fontsize=11)
    ax.set_xlabel(f2, fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig("manuscript_figures/Fig7_response_surface_predicted_main.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 7 (main, 3 pairs) saved.")

# Supplementary Fig. S2 — remaining 3 response surface pairs
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (f1, f2) in zip(axes, supp_pairs_opt):
    pivot = full_ranking.pivot_table(
        index=f1, columns=f2, values="Predicted_LDOPA", aggfunc="mean"
    )
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".0f",
                annot_kws={"size": 9},
                cbar_kws={
                    "label": "Predicted L-DOPA (\u00b5g/g DW)", "shrink": 0.8},
                ax=ax)
    ax.set_title(f"{f1} \u00d7 {f2}", fontweight="bold", fontsize=12)
    ax.set_ylabel(f1, fontsize=11)
    ax.set_xlabel(f2, fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig("manuscript_figures/FigS2_response_surface_predicted_supp.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Supplementary Figure S2 saved.")

# ---------------------------------------------------------------------------
# Fig 8 — Top-10 optimal conditions (90% bootstrap CI)
# ---------------------------------------------------------------------------
top10p = (opt_df
          .sort_values("Mean_Predicted_LDOPA", ascending=False)
          .head(10)
          .reset_index(drop=True))

x_pos = np.arange(10)
means = top10p["Mean_Predicted_LDOPA"].values
err_lo = means - top10p["CI_05"].values
err_hi = top10p["CI_95"].values - means
labels = [f"Cond.\n#{i + 1}" for i in range(10)]

# Gradient colour from deep teal (#1B7A6E) for rank 1 → lighter for rank 10
# Gold highlight for the global optimum (rank 1)
cmap_bars = plt.cm.YlGnBu_r(np.linspace(0.25, 0.85, 10))
bar_colors = [matplotlib.colors.to_hex(cmap_bars[i]) for i in range(10)]
bar_colors[0] = "#E8A000"   # gold for global optimum

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.bar(
    x_pos, means,
    color=bar_colors, edgecolor="white", linewidth=0.6,
    yerr=[err_lo, err_hi], capsize=6,
    error_kw={"elinewidth": 1.8, "ecolor": "black", "capthick": 1.8},
)

# NO bar-top numbers — removed to prevent overlap with error bars

# Annotate global optimum with arrow pointing to bar top
ax.annotate(
    "\u2605 Global Optimum\n(Lemon 10%, A1:100\nSonication, 10 min)",
    xy=(0, means[0] + err_hi[0]),
    xytext=(1.5, means[0] + err_hi[0] + 1800),
    fontsize=9.5, fontweight="bold", color="#B87800",
    arrowprops=dict(arrowstyle="->", color="#B87800", lw=1.5),
)

# Legend for error bars
legend_elements = [
    matplotlib.patches.Patch(facecolor="#E8A000", edgecolor="black",
                             label="Global Optimum"),
    matplotlib.patches.Patch(facecolor=bar_colors[1], edgecolor="black",
                             label="Top conditions"),
    Line2D([0], [0], color="black", linewidth=1.8,
           marker="|", markersize=8, label="Error bars = 90% bootstrap CI"),
]
ax.legend(handles=legend_elements, fontsize=10, loc="upper right",
          framealpha=0.9)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel("Predicted L-DOPA (\u00b5g/g DW)")
ax.set_xlabel("Rank (by Bootstrap Mean Predicted L-DOPA)")
ax.set_title(
    "Top 10 Optimal Extraction Conditions (90% Bootstrap Confidence Interval)")
ax.set_ylim(0, max(means + err_hi) * 1.28)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
plt.savefig("manuscript_figures/Fig8_top10_optimal.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 8 saved.")

# ---------------------------------------------------------------------------
# Fig 9 — Mean-uncertainty Pareto + uncertainty distribution
# ---------------------------------------------------------------------------
idx_max = opt_df["Mean_Predicted_LDOPA"].idxmax()
idx_robust = opt_df["Robust_Score"].idxmax()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (A) Pareto scatter — colour by Robust Score
sc = axes[0].scatter(
    opt_df["Std_Predicted_LDOPA"],
    opt_df["Mean_Predicted_LDOPA"],
    c=opt_df["Robust_Score"], cmap="plasma",
    alpha=0.75, s=45, edgecolors="none",
)
cbar = plt.colorbar(sc, ax=axes[0])
cbar.set_label("Robust Score (\u00b5\u2212\u03bb\u03c3)", fontweight="bold")

axes[0].scatter(
    opt_df.loc[idx_max, "Std_Predicted_LDOPA"],
    opt_df.loc[idx_max, "Mean_Predicted_LDOPA"],
    c="red", s=180, marker="*", zorder=6, label="Max Yield",
    edgecolors="darkred", linewidths=0.5,
)
axes[0].scatter(
    opt_df.loc[idx_robust, "Std_Predicted_LDOPA"],
    opt_df.loc[idx_robust, "Mean_Predicted_LDOPA"],
    c="limegreen", s=130, marker="D", zorder=6, label="Robust Optimum",
    edgecolors="darkgreen", linewidths=0.5,
)

# Max Yield annotation — point downward-right from top
axes[0].annotate(
    "Max Yield\n(Lemon 10%, A1:100\nSonication, 10 min)",
    xy=(opt_df.loc[idx_max, "Std_Predicted_LDOPA"],
        opt_df.loc[idx_max, "Mean_Predicted_LDOPA"]),
    xytext=(40, -60), textcoords="offset points",
    fontsize=8.5, color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkred",
              alpha=0.85, lw=0.8),
)
# Robust Optimum annotation — point downward/below the marker to avoid title
axes[0].annotate(
    "Robust Optimum\n(Lemon 10%, A1:100\nSonication, 20 min)",
    xy=(opt_df.loc[idx_robust, "Std_Predicted_LDOPA"],
        opt_df.loc[idx_robust, "Mean_Predicted_LDOPA"]),
    xytext=(40, -100), textcoords="offset points",
    fontsize=8.5, color="darkgreen",
    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkgreen",
              alpha=0.85, lw=0.8),
)
axes[0].set_xlabel("Prediction Uncertainty (\u03c3, \u00b5g/g DW)")
axes[0].set_ylabel("Mean Predicted L-DOPA (\u00b5g/g DW)")
axes[0].set_title("(A) Mean\u2013Uncertainty Trade-off")
axes[0].legend(fontsize=9, loc="lower right")

# (B) Uncertainty distribution with median line
med_sig = opt_df["Std_Predicted_LDOPA"].median()
axes[1].hist(opt_df["Std_Predicted_LDOPA"], bins=30,
             color="#4C9B82", edgecolor="black", alpha=0.85)
axes[1].axvline(med_sig, color="crimson", linestyle="--", linewidth=1.8,
                label=f"Median \u03c3 = {med_sig:.0f} \u00b5g/g DW")
axes[1].set_xlabel("Prediction Std (\u03c3, \u00b5g/g DW)")
axes[1].set_ylabel("Number of Conditions")
axes[1].set_title(
    "(B) Distribution of Prediction Uncertainty\n(model uncertainty only)")
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("manuscript_figures/Fig9_uncertainty.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 9 saved.")

# ---------------------------------------------------------------------------
# Fig 10 — Residual diagnostics (CatBoost Optuna-tuned)
# Caption: (A) Residuals vs. fitted values, (B) residual distribution
#          histogram, (C) normal Q-Q plot.
# ---------------------------------------------------------------------------
res_cbo = y_test.values - y_pred_cbo

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].scatter(y_pred_cbo, res_cbo, alpha=0.55, color="#2E75B6",
                edgecolors="none", s=22)
axes[0].axhline(0, linestyle="--", color="crimson", linewidth=1.5)
axes[0].set_xlabel("Predicted L-DOPA (\u00b5g/g DW)")
axes[0].set_ylabel("Residual (\u00b5g/g DW)")
axes[0].set_title("(A) Residuals vs. Fitted")

axes[1].hist(res_cbo, bins=25, color="#4C9B82", edgecolor="black", alpha=0.85)
axes[1].axvline(0, linestyle="--", color="crimson", linewidth=1.5)
axes[1].set_xlabel("Residual (\u00b5g/g DW)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("(B) Residual Distribution")

stats.probplot(res_cbo, dist="norm", plot=axes[2])
axes[2].get_lines()[0].set(color="#2E75B6", markersize=4, alpha=0.6)
axes[2].get_lines()[1].set(color="crimson", linewidth=1.5)
axes[2].set_title("(C) Normal Q-Q Plot")

plt.tight_layout()
plt.savefig("manuscript_figures/Fig10_residuals.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 10 saved.")

print("\n\u2705 All done.  Figures \u2192 manuscript_figures/  |  Tables \u2192 manuscript_tables/")

# %%
