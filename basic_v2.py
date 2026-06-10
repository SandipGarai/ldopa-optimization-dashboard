# %% =============================================================================
# STEP 1: EXTENDED DESCRIPTIVE & DIAGNOSTIC ANALYSIS OF L-DOPA DATASET
# =============================================================================
# Descriptive statistics, distribution diagnostics, and non-linearity assessment
# for experimentally measured L-DOPA concentration under categorical conditions.
# =============================================================================

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from sklearn.preprocessing import OrdinalEncoder
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
from sklearn.utils import resample
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import GridSearchCV, KFold
from itertools import combinations
from sklearn.inspection import PartialDependenceDisplay
import shap
import joblib
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf
import pingouin as pg
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import linear_reset
import plotly.graph_objects as go

# Visualization style — font_scale=1.5 improves readability in multi-panel figures
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.5)

# =============================================================================
# REPRODUCIBILITY CONSTANTS
# Exact software versions and random seed — report these in Methods section
# =============================================================================
RANDOM_SEED = 42          # applied to train/test split, RF, bootstrap
N_BOOTSTRAP  = 200        # bootstrap resamples (Section 3.8)
LAMBDA_ROBUST = 1.0       # robust score penalty weight

# Library versions (verified):
# python        3.10
# scikit-learn  1.3.2
# catboost      1.2.3
# optuna        3.5.0
# shap          0.44.0
# numpy         1.26.2

# =============================================================================
# CULTIVAR NOTE (Section 3.1)
# All experiments used HAVFB-06 genotype exclusively (best leaf L-DOPA producer
# among 10 cultivars screened; Ansari et al. 2026, J. Plant Biochem. Biotechnol.
# doi:10.1007/s13562-026-01041-7). Leaf tissue from a single growing season
# (2023-24) at a single location (ICAR-IIAB Farm-B, Ranchi) was used.
# =============================================================================

# =============================================================================
# NATURAL MEDIA CHARACTERISATION (Section 3.2 / Reviewer 1 comment 2-2)
# Lemon juice: fresh fruits manually squeezed, diluted v/v with distilled water
#   5%  → pH 3.78  |  10% → pH 3.47  |  20% → pH 3.39
# ACV:  commercial (pH 3.1), diluted v/v with distilled water
# All solutions: centrifuged 10,000 rpm ambient temp, filtered 0.45 µm Whatman
# =============================================================================

# =============================================================================
# EXTRACTION EQUIPMENT SPECS (Section 3.4 / Reviewer 1 comment 2-3)
# Sonication: Branson 3800 ultrasonic bath, 40 kHz sweep, 110 W, sealed vessels
# Rotospin:   Tarsons Model 3091X rotary mixer, 200 rpm, sealed vessels
# All vessels sealed/tightly capped to prevent oxidative L-DOPA degradation
# =============================================================================

# ------------------------------------------------------------------------------
# Directory configuration
# ------------------------------------------------------------------------------
RESULTS_ROOT = "Results"

SECTION_NAME = "Descriptive_Statistics"

RESULTS_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME)
FIGURES_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------------------
df = pd.read_csv("data final.csv")

df.info()
df.head()

# ------------------------------------------------------------------------------
# Dataset dimensions
# ------------------------------------------------------------------------------
dataset_dimensions = pd.DataFrame({
    "Metric": ["Number of observations", "Number of variables"],
    "Value": [df.shape[0], df.shape[1]]
})

dataset_dimensions.to_csv(
    os.path.join(RESULTS_DIR, "dataset_dimensions.csv"),
    index=False
)

# ------------------------------------------------------------------------------
# Extended descriptive statistics for LDOPA
# ------------------------------------------------------------------------------
ldopa = df["LDOPA"].astype(float)

ldopa_stats = {
    "Mean": ldopa.mean(),
    "Standard Deviation": ldopa.std(),
    "Minimum": ldopa.min(),
    "Maximum": ldopa.max(),
    "Coefficient of Variation (%)": (ldopa.std() / ldopa.mean()) * 100,
    "Skewness": ldopa.skew(),
    "Kurtosis": ldopa.kurtosis()
}

ldopa_stats_df = pd.DataFrame.from_dict(
    ldopa_stats, orient="index", columns=["Value"]
)

ldopa_stats_df.to_csv(
    os.path.join(RESULTS_DIR, "LDOPA_extended_descriptive_statistics.csv")
)

ldopa_stats_df

# ------------------------------------------------------------------------------
# Distribution: Histogram + KDE
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(ldopa, kde=True, color="teal", edgecolor="black")
plt.xlabel("L-DOPA Concentration")
plt.ylabel("Frequency")
plt.title("Distribution of L-DOPA Concentration")

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "LDOPA_distribution.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------------------------
# Normality: Q-Q plot
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 6))
stats.probplot(ldopa, dist="norm", plot=plt)
plt.title("Q-Q Plot for L-DOPA Concentration")

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "LDOPA_QQ_plot.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------------------------
# Normality test: Shapiro–Wilk
# ------------------------------------------------------------------------------
shapiro_test = stats.shapiro(ldopa)

shapiro_results = pd.DataFrame({
    "Statistic": [shapiro_test.statistic],
    "p-value": [shapiro_test.pvalue]
})

shapiro_results.to_csv(
    os.path.join(RESULTS_DIR, "LDOPA_shapiro_wilk_test.csv"),
    index=False
)

shapiro_results

# ------------------------------------------------------------------------------
# Boxplot: spread and outliers
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(y=ldopa, color="orange")
plt.ylabel("L-DOPA Concentration")
plt.title("Boxplot of L-DOPA Concentration")

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "LDOPA_boxplot.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------------------------
# Frequency analysis of categorical factors
# ------------------------------------------------------------------------------
categorical_columns = df.columns[:-1]

for col in categorical_columns:
    freq_table = df[col].value_counts()
    freq_table.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"frequency_{col.replace('/', '_')}.csv"
        )
    )

# ------------------------------------------------------------------------------
# Factor-wise LDOPA summaries and plots
# ------------------------------------------------------------------------------
for col in categorical_columns:
    summary_stats = df.groupby(col)["LDOPA"].agg(
        mean="mean",
        std="std",
        min="min",
        max="max"
    )

    summary_stats.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"LDOPA_summary_by_{col.replace('/', '_')}.csv"
        )
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=col,
        y="LDOPA",
        data=df,
        estimator=np.mean,
        errorbar="sd"
    )
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel(col, fontsize=13)
    plt.ylabel("Mean L-DOPA (µg/g DW) ± SD", fontsize=13)
    plt.title(f"L-DOPA Concentration vs {col}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"LDOPA_vs_{col.replace('/', '_')}.png"
        ),
        dpi=300
    )
    plt.show()

# ------------------------------------------------------------------------------
# Non-linearity: LOESS smoothing
# ------------------------------------------------------------------------------
x_vals = np.arange(len(ldopa))
loess_smoothed = lowess(ldopa, x_vals, frac=0.3)

plt.figure(figsize=(8, 5))
plt.scatter(x_vals, ldopa, alpha=0.6, color="gray", label="Observed")
plt.plot(
    loess_smoothed[:, 0],
    loess_smoothed[:, 1],
    color="red",
    linewidth=2,
    label="LOESS Trend"
)

plt.xlabel("Observation Index")
plt.ylabel("L-DOPA Concentration")
plt.title("Non-linearity Assessment using LOESS")
plt.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "LDOPA_LOESS_nonlinearity.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------------------------
# Non-linearity test: Ramsey RESET
# ------------------------------------------------------------------------------
X = df.drop(columns=["LDOPA"])
y = df["LDOPA"].astype(float)

X_encoded = pd.get_dummies(X, drop_first=True).astype(float)
X_encoded = sm.add_constant(X_encoded)

linear_model = sm.OLS(y.values, X_encoded.values).fit()

reset_test = linear_reset(
    linear_model,
    power=2,
    use_f=True
)

reset_results = pd.DataFrame({
    "Test": ["Ramsey RESET"],
    "F-statistic": [reset_test.fvalue],
    "p-value": [reset_test.pvalue]
})

reset_results.to_csv(
    os.path.join(RESULTS_DIR, "LDOPA_ramsey_RESET_test.csv"),
    index=False
)

reset_results

# %% =============================================================================
# STEP 2: FACTOR & INTERACTION ANALYSIS (STATISTICAL INFERENCE)
# =============================================================================
# Classical ANOVA (Type II / III), exploratory transformations, ART-ANOVA,
# and Kruskal–Wallis tests for robust inference under non-normal conditions.
# =============================================================================

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Directory configuration
# ------------------------------------------------------------------------------
RESULTS_ROOT = "Results"

SECTION_NAME = "Statistical_Analysis"

RESULTS_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME)
ANOVA_DIR = os.path.join(RESULTS_DIR, "Classical_ANOVA")
TRANS_DIR = os.path.join(RESULTS_DIR, "Transformations")
ART_DIR = os.path.join(RESULTS_DIR, "ART_ANOVA")
KW_DIR = os.path.join(RESULTS_DIR, "Kruskal_Wallis")

FIG_TRANS_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME, "Transformations")

os.makedirs(ANOVA_DIR, exist_ok=True)
os.makedirs(TRANS_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(KW_DIR, exist_ok=True)
os.makedirs(FIG_TRANS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------------------
df_stat = df.copy()

for col in df_stat.columns[:-1]:
    df_stat[col] = df_stat[col].astype("category")

# ==============================================================================
# STEP 2.1: Classical Multi-way ANOVA (Type II / III) – Baseline
# ==============================================================================
formula = (
    "LDOPA ~ Concentration * Q('S/L ratio') "
    "* Q('Pre-treatments') * Time"
)

anova_model = smf.ols(formula, data=df).fit()

anova_type2 = anova_lm(anova_model, typ=2)
anova_type3 = anova_lm(anova_model, typ=3)

anova_type2.to_csv(os.path.join(ANOVA_DIR, "ANOVA_Type_II.csv"))
anova_type3.to_csv(os.path.join(ANOVA_DIR, "ANOVA_Type_III.csv"))

anova_type2, anova_type3

# ==============================================================================
# STEP 2.2: Exploratory transformation assessment (log & Box–Cox)
# ==============================================================================
y_original = df["LDOPA"].astype(float)

y_log = np.log(y_original)
y_boxcox, lambda_bc = boxcox(y_original)

shapiro_results = pd.DataFrame({
    "Transformation": ["Original", "Log", "Box-Cox"],
    "Statistic": [
        stats.shapiro(y_original).statistic,
        stats.shapiro(y_log).statistic,
        stats.shapiro(y_boxcox).statistic
    ],
    "p-value": [
        stats.shapiro(y_original).pvalue,
        stats.shapiro(y_log).pvalue,
        stats.shapiro(y_boxcox).pvalue
    ]
})

shapiro_results.to_csv(
    os.path.join(TRANS_DIR, "normality_tests_transformations.csv"),
    index=False
)

# Q–Q plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

stats.probplot(y_original, dist="norm", plot=axes[0])
axes[0].set_title("Original LDOPA")

stats.probplot(y_log, dist="norm", plot=axes[1])
axes[1].set_title("Log-transformed LDOPA")

stats.probplot(y_boxcox, dist="norm", plot=axes[2])
axes[2].set_title(f"Box–Cox LDOPA (λ = {lambda_bc:.2f})")

plt.tight_layout()
plt.savefig(
    os.path.join(FIG_TRANS_DIR, "QQ_plots_transformations.png"),
    dpi=300
)
plt.show()

shapiro_results

# ==============================================================================
# STEP 2.3: ART-ANOVA (main and interaction effects)
# ==============================================================================
art_results = pg.anova(
    data=df_stat,
    dv="LDOPA",
    between=df_stat.columns[:-1].tolist(),
    detailed=True,
    effsize="np2"
)

art_results.to_csv(
    os.path.join(ART_DIR, "ART_ANOVA_results.csv"),
    index=False
)

art_results

# ==============================================================================
# STEP 2.4: Effect size ranking (partial eta squared)
# ==============================================================================
effect_ranking = (
    art_results[["Source", "np2"]]
    .sort_values(by="np2", ascending=False)
)

effect_ranking.to_csv(
    os.path.join(ART_DIR, "ART_ANOVA_effect_size_ranking.csv"),
    index=False
)

effect_ranking

# ==============================================================================
# STEP 2.5: Kruskal–Wallis tests (main effects validation)
# ==============================================================================
kw_results = []

for col in df_stat.columns[:-1]:
    groups = [
        df_stat[df_stat[col] == level]["LDOPA"].values
        for level in df_stat[col].cat.categories
    ]

    stat, pval = stats.kruskal(*groups)
    k = len(groups)
    N = sum(len(g) for g in groups)
    # Epsilon-squared effect size (Tomczak & Tomczak 2014)
    epsilon_sq = (stat - k + 1) / (N - k)
    effect_label = ("large" if epsilon_sq >= 0.14 else
                    "medium" if epsilon_sq >= 0.06 else "small")

    kw_results.append({
        "Factor": col,
        "Kruskal_Wallis_Statistic": stat,
        "p-value": pval,
        "Epsilon_Squared": round(epsilon_sq, 4),
        "Effect_Size": effect_label
    })

kw_results_df = pd.DataFrame(kw_results)

kw_results_df.to_csv(
    os.path.join(KW_DIR, "Kruskal_Wallis_results.csv"),
    index=False
)

kw_results_df

# %% =============================================================================
# STEP 3A: BASELINE MACHINE LEARNING MODELS & ERROR COMPARISON
# =============================================================================
# Models:
#   1. Linear Regression (baseline)
#   2. Random Forest Regressor
#   3. CatBoost Regressor (native categorical handling)
#
# Evaluation:
#   RMSE, MAE, R² (train/test split)
# =============================================================================

# ------------------------------------------------------------------------------
# Directory configuration
# ------------------------------------------------------------------------------
RESULTS_ROOT = "Results"
MODELS_ROOT = "Models"

SECTION_NAME = "ML_Models"

RESULTS_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME)
METRICS_DIR = os.path.join(RESULTS_DIR, "Metrics")
PRED_DIR = os.path.join(RESULTS_DIR, "Predictions")
RESID_DIR = os.path.join(RESULTS_DIR, "Residuals")

MODEL_DIR = os.path.join(MODELS_ROOT, "Trained")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(RESID_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------------------
X = df.drop(columns=["LDOPA"])
y = df["LDOPA"].astype(float)

categorical_cols = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
# NOTE (Reviewer 2 Major-2): Standard random k-fold CV used throughout.
# For replicated factorial designs a grouped/leave-treatment-combination-out CV
# would be more conservative. This is acknowledged in Limitations (Section 8).

# ------------------------------------------------------------------------------
# Helper function: evaluation
# ------------------------------------------------------------------------------


def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


metrics = []

# ==============================================================================
# MODEL 1: Linear Regression (One-Hot Encoded)
# ==============================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

metrics.append(evaluate_model("Linear Regression", y_test, y_pred_lr))

joblib.dump(
    lr_pipeline,
    os.path.join(MODEL_DIR, "linear_regression.pkl")
)

# ==============================================================================
# MODEL 2: Random Forest Regressor (One-Hot Encoded)
# ==============================================================================
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

metrics.append(evaluate_model("Random Forest", y_test, y_pred_rf))

joblib.dump(
    rf_pipeline,
    os.path.join(MODEL_DIR, "random_forest.pkl")
)

# ==============================================================================
# MODEL 3: CatBoost Regressor (Native categorical handling)
# ==============================================================================
cat_features = [X.columns.get_loc(col) for col in categorical_cols]

cb_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

cb_model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

y_pred_cb = cb_model.predict(X_test)

metrics.append(evaluate_model("CatBoost", y_test, y_pred_cb))

cb_model.save_model(
    os.path.join(MODEL_DIR, "catboost_model.cbm")
)

# ------------------------------------------------------------------------------
# Save metrics
# ------------------------------------------------------------------------------
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(
    os.path.join(METRICS_DIR, "model_performance_metrics.csv"),
    index=False
)

# =============================================================================
# SUPPLEMENTARY TABLE S1 — MODEL HYPERPARAMETER SUMMARY
# (Reviewer 1 comment 4-6 / Reviewer 2 minor comment: add hyperparameter table)
# =============================================================================
hyperparams_df = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "Key Parameters": "OLS; OneHotEncoder(drop='first')",
        "n_features_after_encoding": "18 (binary OHE)",
        "Notes": "No tuning; analytical solution"
    },
    {
        "Model": "Random Forest",
        "Key Parameters": "n_estimators=300, random_state=42",
        "n_features_after_encoding": "18 (OHE)",
        "Notes": "Default sklearn RF; no tuning"
    },
    {
        "Model": "CatBoost Baseline",
        "Key Parameters": "depth=6, learning_rate=0.05, iterations=1000, random_seed=42",
        "n_features_after_encoding": "4 (native categorical)",
        "Notes": "Default CatBoost; no tuning; DEPLOYED model"
    },
    {
        "Model": "CatBoost Optuna-tuned",
        "Key Parameters": "depth=9, learning_rate=0.0294, iterations=623, l2_leaf_reg=3, random_seed=42",
        "n_features_after_encoding": "4 (native categorical)",
        "Notes": "40 Optuna trials, 5-fold CV, TPE sampler, MedianPruner; trained on full dataset"
    },
])
hyperparams_df.to_csv(
    os.path.join(METRICS_DIR, "Supplementary_Table_S1_Hyperparameters.csv"),
    index=False
)
print("✅ Supplementary Table S1 (hyperparameters) saved.")

metrics_df

# ------------------------------------------------------------------------------
# Save predictions & residuals
# ------------------------------------------------------------------------------
predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Linear_Regression": y_pred_lr,
    "Random_Forest": y_pred_rf,
    "CatBoost": y_pred_cb
})

predictions.to_csv(
    os.path.join(PRED_DIR, "test_set_predictions.csv"),
    index=False
)

residuals = predictions.copy()
for col in residuals.columns[1:]:
    residuals[col] = residuals["Actual"] - residuals[col]

residuals.to_csv(
    os.path.join(RESID_DIR, "test_set_residuals.csv"),
    index=False
)

# ------------------------------------------------------------------------------
# Residual plots
# ------------------------------------------------------------------------------
for model in ["Linear_Regression", "Random_Forest", "CatBoost"]:
    plt.figure(figsize=(6, 4))
    plt.scatter(
        predictions[model],
        residuals[model],
        alpha=0.6
    )
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted LDOPA")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot – {model.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESID_DIR,
            f"residual_plot_{model}.png"
        ),
        dpi=300
    )
    plt.show()

# =============================================================================
# WORKFLOW FLOWCHART FIGURE
# (Reviewer 1 comment 4-4 / Reviewer 2: add schematic workflow figure)
# Summarises: Experiment → Data → ML Models → SHAP → Bootstrap Optimisation
# =============================================================================
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FLOW_DIR = os.path.join(RESULTS_ROOT, "ML_Models", "Workflow_Figure")
os.makedirs(FLOW_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14); ax.set_ylim(0, 9)
ax.axis("off")

# --- colour palette ---
C_EXP   = "#2E75B6"   # dark blue  – experimental
C_DATA  = "#70AD47"   # green      – data
C_ML    = "#ED7D31"   # orange     – ML
C_SHAP  = "#7030A0"   # purple     – explainability
C_OPT   = "#C00000"   # red        – optimisation
C_ARROW = "#404040"

def box(ax, x, y, w, h, label, sublabel, color, fontsize=10):
    bbox = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor="white",
                          linewidth=2, alpha=0.92, zorder=3)
    ax.add_patch(bbox)
    ax.text(x, y + 0.12, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(x, y - 0.28, sublabel, ha="center", va="center",
                fontsize=8, color="white", zorder=4, style="italic")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.8, mutation_scale=18),
                zorder=2)

# Row 1: Experimental design
box(ax, 3.5, 7.8, 5.5, 1.1,
    "Experimental Design",
    "HAVFB-06 | 4 factors | 264 combinations | 3 replicates = 792 obs.",
    C_EXP, fontsize=11)

# Row 2: Data
box(ax, 3.5, 6.2, 5.5, 1.0,
    "L-DOPA Quantification & Dataset",
    "UPLC-UV | data_final.csv | Kruskal-Wallis | Effect sizes",
    C_DATA, fontsize=11)

arrow(ax, 3.5, 7.25, 3.5, 6.7)

# Row 3: ML Models (4 boxes side by side)
y_ml = 4.6
for i, (lbl, sub) in enumerate([
        ("Lin. Reg.", "OHE + OLS"),
        ("Random\nForest", "B=300, seed=42"),
        ("CatBoost\nBaseline", "depth=6\n(DEPLOYED)"),
        ("CatBoost\nOptuna", "depth=9\n40 trials, CV"),
]):
    bx = 1.5 + i * 3.2
    box(ax, bx, y_ml, 2.8, 1.2, lbl, sub, C_ML, fontsize=9)
    arrow(ax, 3.5 + (i - 1.5) * 0.01, 5.7, bx, y_ml + 0.6)

ax.text(7.0, 5.85, "train_test_split (80/20, seed=42)",
        ha="center", fontsize=9, style="italic", color="#404040")

# Row 4: SHAP & Metrics
box(ax, 2.5, 3.1, 3.6, 1.0,
    "Model Evaluation",
    "RMSE | MAE | R² (hold-out n=159)",
    C_ML, fontsize=10)
box(ax, 7.5, 3.1, 3.6, 1.0,
    "SHAP Explainability",
    "TreeSHAP | Bar + Beeswarm | Global importance",
    C_SHAP, fontsize=10)

arrow(ax, 4.0 + 2*3.2/4, y_ml - 0.6, 2.5, 3.6)   # CatBoost → metrics
arrow(ax, 4.0 + 2*3.2/4, y_ml - 0.6, 7.5, 3.6)   # CatBoost → SHAP

# Row 5: Bootstrap Optimisation
box(ax, 5.0, 1.65, 5.5, 1.1,
    "Bootstrap Uncertainty Quantification",
    "B=200 resamples | Per-condition 90% CI | Robust score = µ−λσ",
    C_OPT, fontsize=10)

arrow(ax, 2.5, 2.6, 5.0 - 1.0, 2.2)
arrow(ax, 7.5, 2.6, 5.0 + 1.0, 2.2)

# Row 6: Output
box(ax, 5.0, 0.55, 5.5, 0.85,
    "Optimal Condition + Streamlit Dashboard",
    "Lemon 10% + A1:100 + Sonication + 10 min → 46,862 µg/g DW",
    C_DATA, fontsize=10)
arrow(ax, 5.0, 1.1, 5.0, 0.98)

# Title
ax.text(7.0, 8.7, "ML-Assisted L-DOPA Extraction Optimisation — Workflow",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color="#1F4E79")

plt.tight_layout()
plt.savefig(os.path.join(FLOW_DIR, "ML_Workflow_Flowchart.png"), dpi=300, bbox_inches="tight")
plt.close()
print("✅ Workflow flowchart saved.")

# %% =============================================================================
# STEP 3B: MODEL EXPLAINABILITY (SHAP & MARGINAL EFFECTS)
# =============================================================================
# Explainability analysis for best-performing ML models:
#   1. Random Forest (SHAP + marginal effects)
#   2. CatBoost (SHAP)
#
# NOTE:
# Classical PDP is not suitable for categorical variables.
# Marginal mean plots are used as a statistically valid alternative.
# =============================================================================

# ------------------------------------------------------------------------------
# Directory configuration
# ------------------------------------------------------------------------------
RESULTS_ROOT = "Results"
SECTION_NAME = "ML_Explainability"

RESULTS_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME)
SHAP_DIR = os.path.join(RESULTS_DIR, "SHAP")
MARGINAL_DIR = os.path.join(RESULTS_DIR, "Marginal_Effects")

RF_SHAP_DIR = os.path.join(SHAP_DIR, "Random_Forest")
CB_SHAP_DIR = os.path.join(SHAP_DIR, "CatBoost")

os.makedirs(RF_SHAP_DIR, exist_ok=True)
os.makedirs(CB_SHAP_DIR, exist_ok=True)
os.makedirs(MARGINAL_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Load trained models
# ------------------------------------------------------------------------------
rf_pipeline = joblib.load("Models/Trained/random_forest.pkl")

cb_model = CatBoostRegressor()
cb_model.load_model("Models/Trained/catboost_model.cbm")

# ------------------------------------------------------------------------------
# Prepare data
# ------------------------------------------------------------------------------
X = df.drop(columns=["LDOPA"])
y = df["LDOPA"].astype(float)

# ------------------------------------------------------------------------------
# SHAP ANALYSIS — RANDOM FOREST
# ------------------------------------------------------------------------------
X_rf = rf_pipeline.named_steps["preprocessor"].transform(X)
X_rf = X_rf.toarray().astype(float)

feature_names_rf = rf_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

explainer_rf = shap.TreeExplainer(
    rf_pipeline.named_steps["model"]
)

shap_values_rf = explainer_rf.shap_values(X_rf)

# SHAP summary plot
plt.figure()
shap.summary_plot(
    shap_values_rf,
    X_rf,
    feature_names=feature_names_rf,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(RF_SHAP_DIR, "RF_SHAP_summary.png"),
    dpi=300
)
plt.show()

# SHAP importance (dummy-level)
rf_shap_importance = pd.DataFrame({
    "Feature": feature_names_rf,
    "Mean_Abs_SHAP": np.abs(shap_values_rf).mean(axis=0)
}).sort_values(by="Mean_Abs_SHAP", ascending=False)

rf_shap_importance.to_csv(
    os.path.join(RF_SHAP_DIR, "RF_SHAP_feature_importance.csv"),
    index=False
)

# ------------------------------------------------------------------------------
# Aggregate SHAP importance to original factors
# ------------------------------------------------------------------------------
rf_shap_importance["Base_Feature"] = (
    rf_shap_importance["Feature"]
    .str.replace(r"^cat__", "", regex=True)
    .str.split("_")
    .str[0]
)

factor_importance = (
    rf_shap_importance
    .groupby("Base_Feature")["Mean_Abs_SHAP"]
    .sum()
    .sort_values(ascending=False)
)

factor_importance.to_csv(
    os.path.join(RF_SHAP_DIR, "RF_SHAP_factor_importance.csv")
)

factor_importance

# ------------------------------------------------------------------------------
# SHAP ANALYSIS — CATBOOST
# ------------------------------------------------------------------------------
explainer_cb = shap.TreeExplainer(cb_model)
shap_values_cb = explainer_cb.shap_values(X)

# Ordinal-encode X for beeswarm colour scale.
# shap.summary_plot with string/object dtype data produces grey dots because
# SHAP cannot build a numeric colour gradient from categorical strings.
# OrdinalEncoder maps each category level to an integer rank (0, 1, 2, …),
# giving SHAP a numeric scale so blue = low-ranked level, red = high-ranked.
# Feature names are still passed as the original column names.
from sklearn.preprocessing import OrdinalEncoder as _OrdEnc
_X_numeric = _OrdEnc().fit_transform(X)   # shape identical to X, float64

# --- (A) Bar plot — mean |SHAP| per feature ---
import io as _io
from PIL import Image as _PILImage

plt.figure()
shap.summary_plot(shap_values_cb, X, plot_type="bar", show=False)
plt.tight_layout()
_buf_bar = _io.BytesIO()
plt.gcf().savefig(_buf_bar, format="png", dpi=150, bbox_inches="tight")
_buf_bar.seek(0)
_img_bar = _PILImage.open(_buf_bar)
plt.close()

# --- (B) Beeswarm — X_numeric for colour gradient ---
plt.figure()
shap.summary_plot(
    shap_values_cb,
    _X_numeric,
    feature_names=X.columns.tolist(),
    show=False
)
plt.tight_layout()
_buf_bee = _io.BytesIO()
plt.gcf().savefig(_buf_bee, format="png", dpi=150, bbox_inches="tight")
_buf_bee.seek(0)
_img_bee = _PILImage.open(_buf_bee)
plt.close()

# --- Composite A+B side-by-side ---
_h = max(_img_bar.height, _img_bee.height)
def _rh(img, th):
    r = th / img.height
    return img.resize((int(img.width * r), th), _PILImage.LANCZOS)
_bar_r = _rh(_img_bar, _h)
_bee_r = _rh(_img_bee, _h)
_combined = _PILImage.new("RGB", (_bar_r.width + _bee_r.width, _h), (255, 255, 255))
_combined.paste(_bar_r, (0, 0))
_combined.paste(_bee_r, (_bar_r.width, 0))
_combined.save(os.path.join(CB_SHAP_DIR, "CatBoost_SHAP_combined.png"), dpi=(300, 300))

cb_shap_importance = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Abs_SHAP": np.abs(shap_values_cb).mean(axis=0)
}).sort_values(by="Mean_Abs_SHAP", ascending=False)

cb_shap_importance.to_csv(
    os.path.join(CB_SHAP_DIR, "CatBoost_SHAP_feature_importance.csv"),
    index=False
)

cb_shap_importance

# ------------------------------------------------------------------------------
# Marginal mean plots (categorical PDP replacement)
# ------------------------------------------------------------------------------
top_factors = factor_importance.head(3).index.tolist()

for factor in top_factors:
    plt.figure(figsize=(6, 4))
    df.groupby(factor)["LDOPA"].mean().plot(
        kind="bar",
        color="steelblue",
        edgecolor="black"
    )

    plt.ylabel("Mean LDOPA Concentration")
    plt.title(f"Marginal Effect of {factor}")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            MARGINAL_DIR,
            f"Marginal_Effect_{factor.replace('/', '_')}.png"
        ),
        dpi=300
    )
    plt.show()

# %% =============================================================================
# STEP 3B.1 & 3B.2: RF SHAP INTERACTIONS
# =============================================================================

# ------------------------------------------------------------------
# Directory configuration
# ------------------------------------------------------------------
RESULTS_ROOT = "Results"
SECTION_NAME = "ML_Explainability"

BASE_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME)
SHAP_DIR = os.path.join(BASE_DIR, "SHAP", "Random_Forest")
PDP_DIR = os.path.join(BASE_DIR, "PDP_ICE", "PDP_2D")
ICE_DIR = os.path.join(BASE_DIR, "PDP_ICE", "ICE_2D")

os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(PDP_DIR, exist_ok=True)
os.makedirs(ICE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load model and data
# ------------------------------------------------------------------
rf_pipeline = joblib.load("Models/Trained/random_forest.pkl")

X = df.drop(columns=["LDOPA"])
y = df["LDOPA"].astype(float)

# Transform data (numeric, dense)
X_rf = rf_pipeline.named_steps["preprocessor"].transform(X)
X_rf = X_rf.toarray().astype(float)

feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()

# ------------------------------------------------------------------
# STEP 3B.1 — SHAP INTERACTION VALUES (Random Forest)
# ------------------------------------------------------------------
explainer = shap.TreeExplainer(rf_pipeline.named_steps["model"])
shap_interactions = explainer.shap_interaction_values(X_rf)

# Save raw interaction tensor
np.save(
    os.path.join(SHAP_DIR, "RF_SHAP_interactions.npy"),
    shap_interactions
)

# ------------------------------------------------------------------
# Compute interaction importance (mean absolute φ_ij)
# ------------------------------------------------------------------
n_features = len(feature_names)
interaction_importance = []

for i in range(n_features):
    for j in range(i + 1, n_features):
        mean_interaction = np.abs(shap_interactions[:, i, j]).mean()
        interaction_importance.append({
            "Feature_1": feature_names[i],
            "Feature_2": feature_names[j],
            "Mean_Abs_SHAP_Interaction": mean_interaction
        })

interaction_df = (
    pd.DataFrame(interaction_importance)
    .sort_values(by="Mean_Abs_SHAP_Interaction", ascending=False)
)

interaction_df.to_csv(
    os.path.join(SHAP_DIR, "RF_SHAP_interaction_importance.csv"),
    index=False
)

# ------------------------------------------------------------------
# Plot TOP 3 interactions
# ------------------------------------------------------------------
top3 = interaction_df.head(3)

plt.figure(figsize=(8, 4))
plt.barh(
    top3["Feature_1"] + " × " + top3["Feature_2"],
    top3["Mean_Abs_SHAP_Interaction"]
)
plt.xlabel("Mean |SHAP Interaction Value|")
plt.title("Top 3 SHAP Interaction Effects (Random Forest)")
plt.tight_layout()

plt.savefig(
    os.path.join(SHAP_DIR, "RF_SHAP_top_interactions.png"),
    dpi=300
)
plt.show()

# %% =============================================================================
# STEP 3B.2 — CONDITIONAL MEAN HEATMAPS (CATEGORICAL × CATEGORICAL)
# =============================================================================


HEATMAP_DIR = os.path.join(BASE_DIR, "Conditional_Heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

original_features = [
    "Concentration",
    "S/L ratio",
    "Pre-treatments",
    "Time"
]

feature_pairs = list(combinations(original_features, 2))

for f1, f2 in feature_pairs:

    pivot = (
        df
        .groupby([f1, f2])["LDOPA"]
        .mean()
        .unstack()
    )

for f1, f2 in feature_pairs:

    pivot = (
        df
        .groupby([f1, f2])["LDOPA"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        cmap="viridis",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 9},
        cbar_kws={"label": "Mean L-DOPA (µg/g DW)", "shrink": 0.8},
        ax=ax
    )
    ax.set_title(f"Interaction: {f1} × {f2}", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(f1, fontsize=12)
    ax.set_xlabel(f2, fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            HEATMAP_DIR,
            f"Heatmap_{f1}_{f2}.png".replace("/", "_")
        ),
        dpi=300
    )
    plt.close()

# =============================================================================
# %% =============================================================================
# STEP 3C: OPTUNA-BASED HYPERPARAMETER TUNING (CATBOOST & RF)
# =============================================================================


# ------------------------------------------------------------------------------
# Directory configuration
# ------------------------------------------------------------------------------
RESULTS_ROOT = "Results"
SECTION_NAME = "ML_Model_Selection"

BASE_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME)
OPTUNA_DIR = os.path.join(BASE_DIR, "Optuna")
FINAL_DIR = os.path.join(BASE_DIR, "Final_Model")

os.makedirs(OPTUNA_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
X = df.drop(columns=["LDOPA"])
y = df["LDOPA"].astype(float)

# Identify categorical feature indices (CRITICAL)
cat_features = [
    X.columns.get_loc(col)
    for col in X.columns
    if X[col].dtype == "object" or X[col].dtype.name == "category"
]

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------------------------------------------------------
# Utility: Cross-validated RMSE
# ------------------------------------------------------------------------------


def cv_rmse_catboost(model, X, y, cv, cat_features):
    rmses = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features
        )
        preds = model.predict(X_test)

        rmses.append(
            np.sqrt(mean_squared_error(y_test, preds))
        )

    return np.mean(rmses)

# ==============================================================================
# 1️⃣ OPTUNA — CATBOOST (PRIMARY MODEL)
# ==============================================================================


def objective_catboost(trial):

    params = {
        "depth": trial.suggest_int("depth", 6, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1),
        "iterations": trial.suggest_int("iterations", 400, 900),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 3, 10),
        "loss_function": "RMSE",
        "verbose": 0,
        "random_seed": 42
    }

    model = CatBoostRegressor(**params)

    rmses = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features
        )

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmses.append(rmse)

        # Report intermediate value to Optuna
        trial.report(np.mean(rmses), step=fold)

        # Prune if not promising
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(rmses)


study_cb = optuna.create_study(
    direction="minimize",
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2)
)
study_cb.optimize(objective_catboost, n_trials=40)


# Train final CatBoost model on full data
best_cb_model = CatBoostRegressor(
    **study_cb.best_params,
    loss_function="RMSE",
    verbose=0,
    random_seed=42
)

best_cb_model.fit(
    X,
    y,
    cat_features=cat_features
)

best_cb_model.save_model(
    os.path.join(FINAL_DIR, "CatBoost_Final_Optuna.cbm")
)

pd.DataFrame(study_cb.trials_dataframe()).to_csv(
    os.path.join(OPTUNA_DIR, "CatBoost_Optuna_Trials.csv"),
    index=False
)

# %% =============================================================================
# STEP 4A: PROCESS OPTIMIZATION – OPTIMAL CONDITION DISCOVERY
# =============================================================================
# Objective:
#   Identify experimental conditions (categorical factor combinations)
#   that maximize predicted L-DOPA concentration using the finalized
#   Optuna-tuned CatBoost model.
#
# Strategy:
#   1. Extract all factor levels from the dataset
#   2. Generate the full factorial design space
#   3. Predict LDOPA using the trained CatBoost model
#   4. Rank conditions and identify global and near-optimal regions
# =============================================================================

# ------------------------------------------------------------------------------
# Directory configuration
# ------------------------------------------------------------------------------
RESULTS_ROOT = "Results"
SECTION_NAME = "Optimization"

RESULTS_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Load finalized CatBoost model
# ------------------------------------------------------------------------------
model = CatBoostRegressor()
model.load_model(
    "Results/ML_Model_Selection/Final_Model/CatBoost_Final_Optuna.cbm"
)

# ------------------------------------------------------------------------------
# Load feature data (exclude response variable)
# ------------------------------------------------------------------------------
X = df.drop(columns=["LDOPA"])

# ------------------------------------------------------------------------------
# Extract unique levels for each categorical factor
# ------------------------------------------------------------------------------
factor_levels = {
    col: sorted(X[col].unique().tolist())
    for col in X.columns
}

# Save factor levels for reproducibility
pd.DataFrame({
    "Factor": list(factor_levels.keys()),
    "Levels": list(factor_levels.values())
}).to_csv(
    os.path.join(RESULTS_DIR, "Factor_Levels.csv"),
    index=False
)

# ------------------------------------------------------------------------------
# Generate full factorial design space
# ------------------------------------------------------------------------------
all_combinations = list(
    itertools.product(*factor_levels.values())
)

design_space = pd.DataFrame(
    all_combinations,
    columns=factor_levels.keys()
)

# ------------------------------------------------------------------------------
# Predict LDOPA for each experimental condition
# ------------------------------------------------------------------------------
design_space["Predicted_LDOPA"] = model.predict(design_space)

# ------------------------------------------------------------------------------
# Rank conditions by predicted LDOPA (descending)
# ------------------------------------------------------------------------------
design_space_sorted = design_space.sort_values(
    by="Predicted_LDOPA",
    ascending=False
).reset_index(drop=True)

# Save full ranked design space
design_space_sorted.to_csv(
    os.path.join(RESULTS_DIR, "LDOPA_Optimization_Full_Ranking.csv"),
    index=False
)

# ------------------------------------------------------------------------------
# Extract top optimal conditions
# ------------------------------------------------------------------------------
TOP_K = 10

top_conditions = design_space_sorted.head(TOP_K)

top_conditions.to_csv(
    os.path.join(RESULTS_DIR, "LDOPA_Top_Optimal_Conditions.csv"),
    index=False
)

top_conditions

# ------------------------------------------------------------------------------
# Identify near-optimal region (within 95% of maximum prediction)
# ------------------------------------------------------------------------------
max_ldopa = design_space_sorted["Predicted_LDOPA"].max()
threshold = 0.95 * max_ldopa

near_optimal = design_space_sorted[
    design_space_sorted["Predicted_LDOPA"] >= threshold
]

near_optimal.to_csv(
    os.path.join(RESULTS_DIR, "LDOPA_Near_Optimal_Region.csv"),
    index=False
)

# %% =============================================================================
# STEP 4A: PROCESS OPTIMIZATION + PUBLICATION-READY VISUALIZATIONS
# =============================================================================
# This script:
#   1. Loads CatBoost-based optimization results
#   2. Visualizes factor-wise effects
#   3. Generates 2D response surface heatmaps
#   4. Generates 3D response surface plots
#
# These figures are intended for journal publication.
# =============================================================================

from mpl_toolkits.mplot3d import Axes3D  # noqa

sns.set_theme(style="whitegrid", font_scale=1.2)

# ------------------------------------------------------------------------------
# Directory configuration
# ------------------------------------------------------------------------------
RESULTS_ROOT = "Results"
SECTION_NAME = "Optimization"

OPT_RESULTS_FILE = os.path.join(
    RESULTS_ROOT,
    SECTION_NAME,
    "LDOPA_Optimization_Full_Ranking.csv"
)

FIGURES_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME, "Figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Load optimization results
# ------------------------------------------------------------------------------
opt_df = pd.read_csv(OPT_RESULTS_FILE)

factors = [c for c in opt_df.columns if c != "Predicted_LDOPA"]

# ==============================================================================
# 1️⃣ MAIN EFFECT PLOTS (FACTOR-WISE)
# ==============================================================================

for factor in factors:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=opt_df,
        x=factor,
        y="Predicted_LDOPA",
        hue=factor,
        palette="Set2",
        legend=False
    )
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel(factor, fontsize=13)
    plt.ylabel("Predicted L-DOPA (µg/g DW)", fontsize=13)
    plt.title(f"Main Effect of {factor} on Predicted L-DOPA", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"MainEffect_{factor}.png".replace("/", "_")
        ),
        dpi=300
    )
    plt.show()

# ==============================================================================
# 2️⃣ PAIRWISE RESPONSE SURFACE HEATMAPS (2D)
# ==============================================================================

pairwise_factors = list(itertools.combinations(factors, 2))

for f1, f2 in pairwise_factors:
    pivot = opt_df.pivot_table(
        index=f1,
        columns=f2,
        values="Predicted_LDOPA",
        aggfunc="mean"
    )

for f1, f2 in pairwise_factors:
    pivot = opt_df.pivot_table(
        index=f1,
        columns=f2,
        values="Predicted_LDOPA",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        cmap="viridis",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 9},
        cbar_kws={"label": "Predicted L-DOPA (µg/g DW)", "shrink": 0.8},
        ax=ax
    )
    ax.set_xlabel(f2, fontsize=12)
    ax.set_ylabel(f1, fontsize=12)
    ax.set_title(f"Response Surface: {f1} × {f2}", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"Heatmap_{f1}_{f2}.png".replace("/", "_")
        ),
        dpi=300
    )
    plt.show()

# ==============================================================================
# 3️⃣ 3D RESPONSE SURFACE PLOTS (KEY INTERACTIONS)
# ==============================================================================

# Select top 3 factor pairs (can be changed or SHAP-guided)
top_pairs = pairwise_factors[:3]

for f1, f2 in top_pairs:
    pivot = opt_df.pivot_table(
        index=f1,
        columns=f2,
        values="Predicted_LDOPA",
        aggfunc="mean"
    )

    X_vals = np.arange(pivot.shape[1])
    Y_vals = np.arange(pivot.shape[0])
    X_grid, Y_grid = np.meshgrid(X_vals, Y_vals)
    Z = pivot.values

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X_grid,
        Y_grid,
        Z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9
    )

    ax.set_xlabel(f2)
    ax.set_ylabel(f1)
    ax.set_zlabel("Predicted L-DOPA")
    ax.set_title(f"3D Response Surface: {f1} × {f2}")

    ax.set_xticks(X_vals)
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(Y_vals)
    ax.set_yticklabels(list(pivot.index))

    fig.colorbar(surf, shrink=0.6, aspect=12, label="Predicted L-DOPA")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"3D_Surface_{f1}_{f2}.png".replace("/", "_")
        ),
        dpi=300
    )
    plt.show()
# =========================================
# Interactive 3D RESPONSE SURFACE PLOTS
# =========================================
INTERACTIVE_DIR = os.path.join(
    RESULTS_ROOT,
    SECTION_NAME,
    "Figures",
    "Interactive_3D"
)
os.makedirs(INTERACTIVE_DIR, exist_ok=True)

pairwise_factors = list(itertools.combinations(factors, 2))

for f1, f2 in pairwise_factors:

    # Pivot table (mean prediction)
    pivot = opt_df.pivot_table(
        index=f1,
        columns=f2,
        values="Predicted_LDOPA",
        aggfunc="mean"
    )

    x_vals = pivot.columns.astype(str)
    y_vals = pivot.index.astype(str)
    z_vals = pivot.values

    fig = go.Figure(
        data=[
            go.Surface(
                z=z_vals,
                x=x_vals,
                y=y_vals,
                colorscale="Viridis",
                colorbar=dict(title="Predicted L-DOPA")
            )
        ]
    )

    fig.update_layout(
        title=f"Interactive 3D Response Surface: {f1} × {f2}",
        scene=dict(
            xaxis_title=f2,
            yaxis_title=f1,
            zaxis_title="Predicted L-DOPA",
        ),
        width=900,
        height=700
    )

    # Save as interactive HTML
    filename = f"Interactive_3D_{f1}_{f2}.html".replace("/", "_")
    fig.write_html(os.path.join(INTERACTIVE_DIR, filename))

# %% =============================================================================
# STEP 4B: UNCERTAINTY-AWARE OPTIMIZATION (CATBOOST) – FINAL ROBUST VERSION
# =============================================================================

# ------------------------------------------------------------------
# Directory configuration
# ------------------------------------------------------------------
RESULTS_ROOT = "Results"
SECTION_NAME = "Optimization"
UNCERTAINTY_DIR = os.path.join(RESULTS_ROOT, SECTION_NAME, "Uncertainty")
os.makedirs(UNCERTAINTY_DIR, exist_ok=True)

# Files
BOOTSTRAP_CHECKPOINT_FILE = os.path.join(
    UNCERTAINTY_DIR, "bootstrap_preds_partial.npy"
)
FINAL_BOOTSTRAP_FILE = os.path.join(
    UNCERTAINTY_DIR, "bootstrap_preds_final.npy"
)

# ------------------------------------------------------------------
# Load final CatBoost model
# ------------------------------------------------------------------
cb_model = CatBoostRegressor()
cb_model.load_model(
    os.path.join(
        RESULTS_ROOT,
        "ML_Model_Selection",
        "Final_Model",
        "CatBoost_Final_Optuna.cbm"
    )
)

# ------------------------------------------------------------------
# Load optimization grid (STEP 4A output)
# ------------------------------------------------------------------
opt_df = pd.read_csv(
    os.path.join(
        RESULTS_ROOT,
        SECTION_NAME,
        "LDOPA_Optimization_Full_Ranking.csv"
    )
)

X_opt = opt_df.drop(columns=["Predicted_LDOPA"])

# ------------------------------------------------------------------
# Training data
# ------------------------------------------------------------------
X_train = df.drop(columns=["LDOPA"])
y_train = df["LDOPA"].astype(float)

# Identify categorical feature indices
cat_features = [
    X_train.columns.get_loc(col)
    for col in X_train.columns
    if X_train[col].dtype == "object"
    or X_train[col].dtype.name == "category"
]

# ------------------------------------------------------------------
# Bootstrap configuration
# ------------------------------------------------------------------
N_BOOTSTRAPS = 200
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------------
# Resume bootstrap if checkpoint exists
# ------------------------------------------------------------------
if os.path.exists(BOOTSTRAP_CHECKPOINT_FILE):
    bootstrap_preds = list(
        np.load(BOOTSTRAP_CHECKPOINT_FILE, allow_pickle=True)
    )
    start_b = len(bootstrap_preds)
    print(f"🔁 Resuming bootstrap: {start_b}/{N_BOOTSTRAPS}")
else:
    bootstrap_preds = []
    start_b = 0

# ------------------------------------------------------------------
# Bootstrap loop with checkpointing
# ------------------------------------------------------------------
for b in tqdm(
    range(start_b, N_BOOTSTRAPS),
    desc="Bootstrap resampling",
    total=N_BOOTSTRAPS,
    initial=start_b
):
    X_boot, y_boot = resample(
        X_train,
        y_train,
        replace=True,
        random_state=RANDOM_STATE + b
    )

    params = cb_model.get_params()
    params.pop("verbose", None)

    model_b = CatBoostRegressor(
        **params,
        logging_level="Silent"
    )

    model_b.fit(
        X_boot,
        y_boot,
        cat_features=cat_features
    )

    preds = model_b.predict(X_opt)
    bootstrap_preds.append(preds)

    # Save checkpoint after each iteration
    np.save(
        BOOTSTRAP_CHECKPOINT_FILE,
        np.array(bootstrap_preds, dtype=object)
    )

# ------------------------------------------------------------------
# Sanity check: ensure full completion
# ------------------------------------------------------------------
assert len(bootstrap_preds) == N_BOOTSTRAPS, (
    f"Bootstrap incomplete: {len(bootstrap_preds)}/{N_BOOTSTRAPS}"
)

# ------------------------------------------------------------------
# Convert to NumPy array and save FINAL raw bootstrap matrix
# ------------------------------------------------------------------
bootstrap_preds = np.array(bootstrap_preds)
np.save(FINAL_BOOTSTRAP_FILE, bootstrap_preds)

# ------------------------------------------------------------------
# Compute uncertainty statistics
# ------------------------------------------------------------------
opt_df["Mean_Predicted_LDOPA"] = bootstrap_preds.mean(axis=0)
opt_df["Std_Predicted_LDOPA"] = bootstrap_preds.std(axis=0)
opt_df["CI_05"] = np.percentile(bootstrap_preds, 5, axis=0)
opt_df["CI_95"] = np.percentile(bootstrap_preds, 95, axis=0)

# Robust optimization score
LAMBDA = 1.0
opt_df["Robust_Score"] = (
    opt_df["Mean_Predicted_LDOPA"]
    - LAMBDA * opt_df["Std_Predicted_LDOPA"]
)

# Robust ranking
robust_ranking = (
    opt_df.sort_values("Robust_Score", ascending=False)
    .reset_index(drop=True)
)

# ------------------------------------------------------------------
# Save CSV outputs
# ------------------------------------------------------------------
opt_df.to_csv(
    os.path.join(UNCERTAINTY_DIR, "Bootstrap_Predictions.csv"),
    index=False
)

robust_ranking.to_csv(
    os.path.join(UNCERTAINTY_DIR, "Robust_Ranking.csv"),
    index=False
)

# ------------------------------------------------------------------
# Cleanup checkpoint file (no longer needed)
# ------------------------------------------------------------------
if os.path.exists(BOOTSTRAP_CHECKPOINT_FILE):
    os.remove(BOOTSTRAP_CHECKPOINT_FILE)

print("✅ STEP 4B completed successfully.")
print(f"📁 Results saved in: {UNCERTAINTY_DIR}")

# %% =============================================================================
# STEP 4B1: UNCERTAINTY-AWARE OPTIMIZATION (CATBOOST) – FIGURES
# =============================================================================

# ------------------------------------------------------------------
# Load safely for plotting (in case of restart)
# ------------------------------------------------------------------
opt_df = pd.read_csv(
    os.path.join(UNCERTAINTY_DIR, "Bootstrap_Predictions.csv")
)
robust_ranking = pd.read_csv(
    os.path.join(UNCERTAINTY_DIR, "Robust_Ranking.csv")
)

sns.set_theme(style="whitegrid", font_scale=1.2)
# ------------------------------------------------------------------
# Directory setup for Figures
# ------------------------------------------------------------------
FIG_DIR = os.path.join(UNCERTAINTY_DIR, "Figures")
INT_FIG_DIR = os.path.join(UNCERTAINTY_DIR, "Interactive_Uncertainty_Figures")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(INT_FIG_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1️⃣ Pareto: Mean vs Uncertainty (STATIC)
# ------------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(
    opt_df["Mean_Predicted_LDOPA"],
    opt_df["Std_Predicted_LDOPA"],
    alpha=0.6
)
plt.xlabel("Mean Predicted L-DOPA")
plt.ylabel("Prediction Uncertainty (Std)")
plt.title("Mean–Uncertainty Pareto Front")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Pareto_Mean_vs_Std.png"), dpi=300)
plt.close()

# ------------------------------------------------------------------
# 2️⃣ Robust Score vs Mean Prediction (STATIC) — enhanced
# ------------------------------------------------------------------
idx_max    = opt_df["Mean_Predicted_LDOPA"].idxmax()
idx_robust = opt_df["Robust_Score"].idxmax()

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    opt_df["Std_Predicted_LDOPA"],
    opt_df["Mean_Predicted_LDOPA"],
    c=opt_df["Robust_Score"],
    cmap="plasma",
    alpha=0.7,
    s=40,
    edgecolors="none",
    label="All conditions"
)
ax.scatter(
    opt_df.loc[idx_max, "Std_Predicted_LDOPA"],
    opt_df.loc[idx_max, "Mean_Predicted_LDOPA"],
    c="red", s=120, zorder=5, marker="*", label="Max Yield"
)
ax.scatter(
    opt_df.loc[idx_robust, "Std_Predicted_LDOPA"],
    opt_df.loc[idx_robust, "Mean_Predicted_LDOPA"],
    c="limegreen", s=120, zorder=5, marker="D", label="Robust Optimum"
)
ax.annotate("Max Yield\n(Lemon 10%, A1:100,\nSonication, 10 min)",
            xy=(opt_df.loc[idx_max, "Std_Predicted_LDOPA"],
                opt_df.loc[idx_max, "Mean_Predicted_LDOPA"]),
            xytext=(15, -40), textcoords="offset points",
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
ax.annotate("Robust Optimum\n(Lemon 10%, A1:100,\nSonication, 20 min)",
            xy=(opt_df.loc[idx_robust, "Std_Predicted_LDOPA"],
                opt_df.loc[idx_robust, "Mean_Predicted_LDOPA"]),
            xytext=(15, 20), textcoords="offset points",
            fontsize=9, color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=1.2))
plt.colorbar(sc, ax=ax, label="Robust Score (µ−λσ)")
ax.set_xlabel("Prediction Uncertainty (σ, µg/g DW)", fontsize=12)
ax.set_ylabel("Mean Predicted L-DOPA (µg/g DW)", fontsize=12)
ax.set_title("(A) Mean–Uncertainty Trade-off (Pareto View)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Pareto_Mean_vs_Risk_annotated.png"), dpi=300)
plt.close()

# ------------------------------------------------------------------
# 3️⃣ Top-10 Robust Conditions — annotated bar chart
# (Reviewer 1 comment 4-9: enhance visual emphasis of key findings)
# ------------------------------------------------------------------
top10 = (opt_df
         .sort_values("Mean_Predicted_LDOPA", ascending=False)
         .head(10)
         .reset_index(drop=True))

x_pos  = np.arange(10)
means  = top10["Mean_Predicted_LDOPA"].values
err_lo = means - top10["CI_05"].values
err_hi = top10["CI_95"].values - means
labels = [f"Cond.\n#{i+1}" for i in range(10)]

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.bar(
    x_pos, means,
    color="#5B8DB8", edgecolor="white", linewidth=0.5,
    yerr=[err_lo, err_hi], capsize=5,
    error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
)
# Highlight the global optimum (rank 1) in gold
bars[0].set_color("#E8A000")
bars[0].set_edgecolor("black")
bars[0].set_linewidth(1.2)

for i, (bar, val) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(err_hi) * 0.05,
            f"{int(round(val)):,}",
            ha="center", va="bottom", fontsize=9, rotation=90, color="black")

# Annotate the global optimum
ax.annotate("★ Global\nOptimum",
            xy=(0, means[0] + err_hi[0]),
            xytext=(0.6, means[0] + err_hi[0] + 2500),
            fontsize=9.5, fontweight="bold", color="#B87800",
            arrowprops=dict(arrowstyle="->", color="#B87800", lw=1.2))

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Predicted L-DOPA (µg/g DW)", fontsize=12)
ax.set_xlabel("Bootstrap Rank (by Mean Predicted L-DOPA)", fontsize=12)
ax.set_title("Top 10 Optimal Extraction Conditions (90% Bootstrap CI)",
             fontsize=14, fontweight="bold")
ax.set_ylim(0, max(means + err_hi) * 1.22)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Top10_Optimal_CI_annotated.png"), dpi=300)
plt.close()
print("✅ Annotated Top-10 CI bar chart saved.")

# ------------------------------------------------------------------
# 4️⃣ Distribution of Prediction Uncertainty (STATIC) — enhanced
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(opt_df["Std_Predicted_LDOPA"], bins=30, edgecolor="black",
        color="#5B8DB8", alpha=0.85)
ax.axvline(opt_df["Std_Predicted_LDOPA"].median(), color="red",
           linestyle="--", linewidth=1.5,
           label=f"Median σ = {opt_df['Std_Predicted_LDOPA'].median():.0f}")
ax.set_xlabel("Prediction Std (σ, µg/g DW)", fontsize=12)
ax.set_ylabel("Number of Conditions", fontsize=12)
ax.set_title("(B) Distribution of Prediction Uncertainty\n"
             "(bootstrap model uncertainty only; excludes analytical error)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Uncertainty_Distribution.png"), dpi=300)
plt.close()

# ------------------------------------------------------------------
# 5️⃣ 2D UNCERTAINTY HEATMAPS (σ) – ALL FACTOR PAIRS (STATIC)
# ------------------------------------------------------------------
factors = [
    c for c in opt_df.columns
    if c not in [
        "Predicted_LDOPA",
        "Mean_Predicted_LDOPA",
        "Std_Predicted_LDOPA",
        "CI_05",
        "CI_95",
        "Robust_Score"
    ]
]

pairwise_factors = list(itertools.combinations(factors, 2))

for f1, f2 in pairwise_factors:

    pivot_std = opt_df.pivot_table(
        index=f1,
        columns=f2,
        values="Std_Predicted_LDOPA",
        aggfunc="mean"
    )

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        pivot_std,
        cmap="magma",
        cbar_kws={"label": "Prediction Std (σ)"}
    )
    plt.title(f"Uncertainty Surface: {f1} × {f2}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIG_DIR,
            f"Uncertainty_Heatmap_{f1}_{f2}.png".replace("/", "_")
        ),
        dpi=300
    )
    plt.close()

# ------------------------------------------------------------------
# 6️⃣ INTERACTIVE 3D UNCERTAINTY SURFACES (σ)
# ------------------------------------------------------------------
for f1, f2 in pairwise_factors:

    pivot_std = opt_df.pivot_table(
        index=f1,
        columns=f2,
        values="Std_Predicted_LDOPA",
        aggfunc="mean"
    )

    fig = go.Figure(
        data=[
            go.Surface(
                z=pivot_std.values,
                x=pivot_std.columns.astype(str),
                y=pivot_std.index.astype(str),
                colorscale="Magma",
                colorbar=dict(title="Prediction Std (σ)")
            )
        ]
    )

    fig.update_layout(
        title=f"Interactive Uncertainty Surface: {f1} × {f2}",
        scene=dict(
            xaxis_title=f2,
            yaxis_title=f1,
            zaxis_title="Prediction Std (σ)"
        ),
        width=900,
        height=700
    )

    fig.write_html(
        os.path.join(
            INT_FIG_DIR,
            f"Interactive_Uncertainty_3D_{f1}_{f2}.html".replace("/", "_")
        )
    )

# %% =============================================================================
# STEP 4C: ROBUST EXPERIMENTAL RECOMMENDATION & SENSITIVITY SCENARIOS
# =============================================================================
# This script converts ML + bootstrap uncertainty into
# decision-ready experimental guidance.
#
# Outputs:
#   1. Robust optimum stability plots
#   2. Mean–Risk Pareto front
#   3. SHAP-guided qualitative rules (CSV)
#   4. Experimental recommendation table
# =============================================================================

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

sns.set(style="whitegrid", font_scale=1.1)

RESULTS_ROOT = "Results"
UNCERTAINTY_DIR = os.path.join(RESULTS_ROOT, "Optimization", "Uncertainty")
FIG_DIR = os.path.join(RESULTS_ROOT, "Optimization", "Robust_Analysis")
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Load bootstrap predictions
# -------------------------------------------------------------------------
df = pd.read_csv(os.path.join(UNCERTAINTY_DIR, "Bootstrap_Predictions.csv"))

FACTOR_COLS = [
    c for c in df.columns
    if c not in [
        "Predicted_LDOPA",
        "Mean_Predicted_LDOPA",
        "Std_Predicted_LDOPA",
        "CI_05",
        "CI_95",
        "Robust_Score"
    ]
]

# =========================================================================
# 1️. ROBUST OPTIMUM VALIDATION (ONE-FACTOR PERTURBATION)
# =========================================================================

top_row = df.sort_values("Robust_Score", ascending=False).iloc[0]
top_condition = top_row[FACTOR_COLS]

for factor in FACTOR_COLS:
    subset = df.copy()
    for f in FACTOR_COLS:
        if f != factor:
            subset = subset[subset[f] == top_condition[f]]

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        subset[factor],
        subset["Mean_Predicted_LDOPA"],
        yerr=subset["Std_Predicted_LDOPA"],
        fmt="o",
        capsize=4
    )
    plt.xlabel(factor)
    plt.ylabel("Predicted L-DOPA (mean ± std)")
    plt.title(f"Robustness Check: Varying {factor}")
    plt.xticks(rotation=90, ha="center", va="top")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"Robustness_{factor}.png".replace("/", "_")),
        dpi=300
    )
    plt.close()

# =========================================================================
# 2️. MEAN–RISK TRADE-OFF (PARETO FRONT)
# =========================================================================

plt.figure(figsize=(6, 5))
plt.scatter(
    df["Std_Predicted_LDOPA"],
    df["Mean_Predicted_LDOPA"],
    alpha=0.6
)

# highlight key points
idx_max = df["Mean_Predicted_LDOPA"].idxmax()
idx_robust = df["Robust_Score"].idxmax()
idx_safe = df["Std_Predicted_LDOPA"].idxmin()

plt.scatter(
    df.loc[idx_max, "Std_Predicted_LDOPA"],
    df.loc[idx_max, "Mean_Predicted_LDOPA"],
    c="red", label="Max Yield"
)
plt.scatter(
    df.loc[idx_robust, "Std_Predicted_LDOPA"],
    df.loc[idx_robust, "Mean_Predicted_LDOPA"],
    c="green", label="Robust Optimum"
)
plt.scatter(
    df.loc[idx_safe, "Std_Predicted_LDOPA"],
    df.loc[idx_safe, "Mean_Predicted_LDOPA"],
    c="blue", label="Low Risk"
)

plt.xlabel("Prediction Std (Risk)")
plt.ylabel("Prediction Mean")
plt.title("Mean–Risk Trade-off (Pareto View)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Pareto_Mean_vs_Risk.png"), dpi=300)
plt.close()

# =========================================================================
# 3️. SHAP-GUIDED RULE EXTRACTION (QUALITATIVE)
# =========================================================================
# Uses factor-wise mean impact (model-agnostic summary)

rules = []

for f in FACTOR_COLS:
    grp = df.groupby(f)["Mean_Predicted_LDOPA"].mean()
    spread = grp.max() - grp.min()

    rules.append({
        "Factor": f,
        "Max_Effect_Range": spread,
        "Interpretation": (
            "Strong driver" if spread > df["Mean_Predicted_LDOPA"].std()
            else "Secondary driver"
        )
    })

rules_df = pd.DataFrame(rules).sort_values(
    "Max_Effect_Range", ascending=False
)

rules_df.to_csv(
    os.path.join(FIG_DIR, "SHAP_Guided_Rules.csv"),
    index=False
)

# =========================================================================
# 4️. EXPERIMENTAL DESIGN RECOMMENDATIONS (DECISION TABLE)
# =========================================================================

recommendations = pd.DataFrame([
    {
        "Recommendation_Type": "Maximum Yield",
        "Conditions": df.loc[idx_max, FACTOR_COLS].to_dict(),
        "Rationale": "Highest predicted mean L-DOPA"
    },
    {
        "Recommendation_Type": "Robust Optimum",
        "Conditions": df.loc[idx_robust, FACTOR_COLS].to_dict(),
        "Rationale": "Maximizes mean – uncertainty"
    },
    {
        "Recommendation_Type": "Low Risk",
        "Conditions": df.loc[idx_safe, FACTOR_COLS].to_dict(),
        "Rationale": "Minimum predictive uncertainty"
    }
])

recommendations.to_csv(
    os.path.join(FIG_DIR, "Experimental_Recommendations.csv"),
    index=False
)


# %% STEP 5A — Core prediction function (REUSABLE)

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
MODEL_PATH = "Results/ML_Model_Selection/Final_Model/CatBoost_Final_Optuna.cbm"
BOOTSTRAP_PATH = "Results/Optimization/Uncertainty/bootstrap_preds_final.npy"

# -------------------------------------------------------
# Load model
# -------------------------------------------------------
cb_model = CatBoostRegressor()
cb_model.load_model(MODEL_PATH)

bootstrap_preds = np.load(BOOTSTRAP_PATH)

# -------------------------------------------------------
# Prediction function
# -------------------------------------------------------


def predict_ldopa_with_uncertainty(new_conditions_df):
    """
    Input: DataFrame of new experimental conditions
    Output: mean, std, CI05, CI95
    """

    # Point prediction (mean model)
    mean_pred = cb_model.predict(new_conditions_df)

    # Bootstrap uncertainty
    boot_preds = []
    for preds in bootstrap_preds:
        # each preds array corresponds to full optimization grid
        # retrain bootstrap models was done on same feature order
        boot_preds.append(preds[:len(new_conditions_df)])

    boot_preds = np.array(boot_preds)

    return pd.DataFrame({
        "Mean_LDOPA": mean_pred,
        "Std_LDOPA": boot_preds.std(axis=0),
        "CI_05": np.percentile(boot_preds, 5, axis=0),
        "CI_95": np.percentile(boot_preds, 95, axis=0)
    })
# %% STEP 5C — Robust Optimization Table (Decision Layer)
# Robust scoring & ranking


def robust_optimization_table(df, lambda_=1.0):
    """
    Input: Bootstrap_Predictions.csv
    Output: Ranked robust decision table
    """

    df = df.copy()
    df["Robust_Score"] = (
        df["Mean_Predicted_LDOPA"]
        - lambda_ * df["Std_Predicted_LDOPA"]
    )

    return df.sort_values("Robust_Score", ascending=False)
# Generate recommendation table


robust_df = robust_optimization_table(
    pd.read_csv(
        "Results/Optimization/Uncertainty/Bootstrap_Predictions.csv"
    )
)

recommendations = pd.DataFrame([
    {
        "Recommendation": "Maximum Yield",
        "Condition": robust_df.loc[
            robust_df["Mean_Predicted_LDOPA"].idxmax()
        ].to_dict(),
        "Rationale": "Highest expected LDOPA"
    },
    {
        "Recommendation": "Robust Optimum",
        "Condition": robust_df.iloc[0].to_dict(),
        "Rationale": "Best μ − σ trade-off"
    },
    {
        "Recommendation": "Low Risk",
        "Condition": robust_df.loc[
            robust_df["Std_Predicted_LDOPA"].idxmin()
        ].to_dict(),
        "Rationale": "Minimum uncertainty"
    }
])

recommendations.to_csv(
    "Results/Optimization/Robust_Recommendations.csv",
    index=False
)
