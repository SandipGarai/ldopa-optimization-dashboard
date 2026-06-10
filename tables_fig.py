# %% tables_fig.py — Manuscript figures (Fig 1–10) and tables (Table 1–5)
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

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.2)

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

# Table 2 — Kruskal-Wallis results
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
    kw.append({
        "Factor": col,
        "H-statistic": round(H, 3),
        "p-value": f"{p:.2e}",
        "Significance": ("***" if p < 0.001 else
                         "**" if p < 0.01 else
                         "*" if p < 0.05 else "ns"),
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


# =============================================================================
# FIGURES
# =============================================================================

# ---------------------------------------------------------------------------
# Fig 1 — Dataset overview
# Caption: (A) Distribution histogram with KDE. (B) Boxplot by extraction
#          medium. (C) Boxplot by S/L ratio. (D) Boxplot by time x pre-treatment.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.histplot(df["LDOPA"], kde=True, color="teal", edgecolor="black",
             ax=axes[0, 0])
axes[0, 0].set_title("(A) Distribution of L-DOPA Concentration")
axes[0, 0].set_xlabel("L-DOPA Concentration")
axes[0, 0].set_ylabel("Frequency")

# FutureWarning fix: assign x variable to hue, set legend=False
sns.boxplot(data=df, x="Concentration", y="LDOPA",
            hue="Concentration", palette="Set2", legend=False,
            ax=axes[0, 1])
# UserWarning fix: set_xticks before set_xticklabels
axes[0, 1].set_xticks(axes[0, 1].get_xticks())
axes[0, 1].set_xticklabels(
    axes[0, 1].get_xticklabels(), rotation=45, ha="right")
axes[0, 1].set_title("(B) L-DOPA by Extraction Medium")
axes[0, 1].set_xlabel("Concentration")
axes[0, 1].set_ylabel("L-DOPA Concentration")

sns.boxplot(data=df, x="S/L ratio", y="LDOPA",
            hue="S/L ratio", palette="Set2", legend=False,
            ax=axes[1, 0])
axes[1, 0].set_title("(C) L-DOPA by S/L Ratio")

# hue="Pre-treatments" is already the grouping variable — no change needed
sns.boxplot(data=df, x="Time", y="LDOPA",
            hue="Pre-treatments", palette="Set1",
            ax=axes[1, 1])
axes[1, 1].set_title("(D) L-DOPA by Time \u00d7 Pre-treatment")

plt.tight_layout()
plt.savefig("manuscript_figures/Fig1_data_overview.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 1 saved.")

# ---------------------------------------------------------------------------
# Fig 2 — Main effect plots
# Caption: Main effect plots for each extraction factor on predicted L-DOPA.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, factor in enumerate(["Concentration", "S/L ratio", "Pre-treatments", "Time"]):
    # FutureWarning fix: hue=factor, legend=False
    sns.boxplot(data=full_ranking, x=factor, y="Predicted_LDOPA",
                hue=factor, palette="Set2", legend=False, ax=axes[i])
    # UserWarning fix: set_xticks first
    axes[i].set_xticks(axes[i].get_xticks())
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")
    axes[i].set_xlabel(factor)
    axes[i].set_ylabel("Predicted L-DOPA Concentration")
    axes[i].set_title(f"Main Effect of {factor} on Predicted L-DOPA")
plt.tight_layout()
plt.savefig("manuscript_figures/Fig2_main_effects.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 2 saved.")

# ---------------------------------------------------------------------------
# Fig 3 — Observed interaction heatmaps
# Caption: Pairwise interaction heatmaps (observed mean L-DOPA, microg/g DW).
# ---------------------------------------------------------------------------
feature_pairs = list(combinations(
    ["Concentration", "S/L ratio", "Pre-treatments", "Time"], 2
))
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for ax, (f1, f2) in zip(axes, feature_pairs):
    pivot = df.groupby([f1, f2])["LDOPA"].mean().unstack()
    sns.heatmap(pivot, cmap="viridis", annot=False, ax=ax)
    ax.set_title(f"Conditional Mean LDOPA: {f1} \u00d7 {f2}")
    ax.set_ylabel(f1)
    ax.set_xlabel(f2)
plt.tight_layout()
plt.savefig("manuscript_figures/Fig3_interaction_heatmaps.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 3 saved.")

# ---------------------------------------------------------------------------
# Fig 4 — ML model performance bars
# Caption: (A) RMSE, (B) MAE, (C) R² on hold-out test set (n = 159).
# ---------------------------------------------------------------------------
model_labels = ["Linear\nRegression", "Random\nForest",
                "CatBoost\nBaseline", "CatBoost\nOptuna"]
colors = sns.color_palette("Set2", 4)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, vals, lbl, fmt in zip(
    axes,
    [
        [np.sqrt(mean_squared_error(y_test, p))
         for p in [y_pred_lr, y_pred_rf, y_pred_cb, y_pred_cbo]],
        [mean_absolute_error(y_test, p)
         for p in [y_pred_lr, y_pred_rf, y_pred_cb, y_pred_cbo]],
        [r2_score(y_test, p)
         for p in [y_pred_lr, y_pred_rf, y_pred_cb, y_pred_cbo]],
    ],
    ["(A) RMSE (microg/g DW)", "(B) MAE (microg/g DW)", "(C) R\u00b2"],
    ["%.0f", "%.0f", "%.3f"],
):
    bars = ax.bar(model_labels, vals, color=colors)
    ax.bar_label(bars, fmt=fmt, fontsize=9)
    ax.set_title(lbl)
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
# Fig 7 — Predicted response surface heatmaps
# Caption: Pairwise response surface heatmaps of predicted mean L-DOPA
#          (microg/g DW) from CatBoost Optuna-tuned model.
# ---------------------------------------------------------------------------
pairwise_factors = list(itertools.combinations(factors_opt, 2))
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for ax, (f1, f2) in zip(axes, pairwise_factors):
    pivot = full_ranking.pivot_table(
        index=f1, columns=f2, values="Predicted_LDOPA", aggfunc="mean"
    )
    sns.heatmap(pivot, cmap="viridis", annot=False,
                cbar_kws={"label": "Predicted L-DOPA"}, ax=ax)
    ax.set_title(f"Response Surface Heatmap: {f1} \u00d7 {f2}")
    ax.set_xlabel(f2)
    ax.set_ylabel(f1)
plt.tight_layout()
plt.savefig("manuscript_figures/Fig7_response_surface_heatmaps.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 7 saved.")

# ---------------------------------------------------------------------------
# Fig 8 — Top-10 optimal conditions with 90% CI (bar chart)
# Caption: Top 10 optimal extraction conditions with 90% bootstrap CI.
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

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(
    x_pos, means,
    color="#5B8DB8", edgecolor="white", linewidth=0.5,
    yerr=[err_lo, err_hi], capsize=5,
    error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
)
for bar, val in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(err_hi) * 0.05,
        f"{int(round(val))}",
        ha="center", va="bottom", fontsize=8, rotation=90, color="black",
    )
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Predicted L-DOPA Concentration (\u00b5g/g DW)", fontsize=11)
ax.set_title(
    "Top 10 Optimal Extraction Conditions (90% Confidence Interval)",
    fontsize=12, fontweight="bold",
)
ax.set_ylim(0, max(means + err_hi) * 1.18)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
plt.savefig("manuscript_figures/Fig8_top10_optimal.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 8 saved.")

# ---------------------------------------------------------------------------
# Fig 9 — Mean-uncertainty Pareto + uncertainty distribution
# Caption: (A) Mean-uncertainty trade-off (red = max yield, green = robust
#          optimum). (B) Distribution of prediction uncertainty (sigma).
# ---------------------------------------------------------------------------
idx_max = opt_df["Mean_Predicted_LDOPA"].idxmax()
idx_robust = opt_df["Robust_Score"].idxmax()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].scatter(opt_df["Std_Predicted_LDOPA"],
                opt_df["Mean_Predicted_LDOPA"], alpha=0.6)
axes[0].scatter(
    opt_df.loc[idx_max,    "Std_Predicted_LDOPA"],
    opt_df.loc[idx_max,    "Mean_Predicted_LDOPA"],
    c="red", label="Max Yield", zorder=5,
)
axes[0].scatter(
    opt_df.loc[idx_robust, "Std_Predicted_LDOPA"],
    opt_df.loc[idx_robust, "Mean_Predicted_LDOPA"],
    c="green", label="Robust Optimum", zorder=5,
)
axes[0].set_xlabel("Prediction Std (Risk)")
axes[0].set_ylabel("Prediction Mean")
axes[0].set_title("Mean\u2013Risk Trade-off (Pareto View)")
axes[0].legend()

axes[1].hist(opt_df["Std_Predicted_LDOPA"], bins=30, edgecolor="black")
axes[1].set_xlabel("Prediction Std (\u03c3)")
axes[1].set_ylabel("Number of Conditions")
axes[1].set_title("Distribution of Prediction Uncertainty")

plt.tight_layout()
plt.savefig("manuscript_figures/Fig9_uncertainty.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 9 saved.")

# ---------------------------------------------------------------------------
# Fig 10 — Residual diagnostics (CatBoost Optuna-tuned)
# Caption: (A) Residuals vs. fitted, (B) residual histogram, (C) Q-Q plot.
# ---------------------------------------------------------------------------
res_cbo = y_test.values - y_pred_cbo

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].scatter(y_pred_cbo, res_cbo, alpha=0.6)
axes[0].axhline(0, linestyle="--")
axes[0].set_xlabel("Predicted LDOPA")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residual Plot \u2013 CatBoost Optuna")

axes[1].hist(res_cbo, bins=25, edgecolor="black")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")

stats.probplot(res_cbo, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot \u2013 CatBoost Optuna")

plt.tight_layout()
plt.savefig("manuscript_figures/Fig10_residuals.png", dpi=300,
            bbox_inches="tight")
plt.close()
print("Figure 10 saved.")

print("\n\u2705 All done.  Figures \u2192 manuscript_figures/  |  Tables \u2192 manuscript_tables/")

# %%
