# L-DOPA Process Optimization & Decision Support Dashboard

An interactive, uncertainty-aware decision-support system for **L-DOPA production optimization**, integrating:

- Optuna-tuned **CatBoost regression**
- **Bootstrap-based uncertainty quantification**
- Robust optimization (μ − λσ)
- Experimental recommendation and planning
- Interactive visualization and PDF reporting

---

## Scientific Motivation

Experimental optimization of L-DOPA yield involves complex interactions between:

- Concentration
- Solid-to-liquid (S/L) ratio
- Pre-treatment method
- Processing time

This application translates machine learning predictions into **actionable experimental decisions**, emphasizing:

- **Robustness** over point-optimality
- **Risk-aware trade-offs**
- **Interpretability for chemical insight**

---

## Key Features

### Prediction Under New Conditions

- Input new experimental combinations
- Output:
  - Mean predicted L-DOPA
  - Prediction uncertainty (σ)
  - Confidence interval (CI)

### Optimization Strategies

- **Maximum Yield** (highest μ)
- **Robust Optimum** (μ − λσ)
- **Low-Risk Condition** (minimum σ)

### Trade-off Analysis

- Mean–risk (Pareto) visualization
- Color-coded optimal operating regions

### Experimental Planning

- Suggested next validation experiments
- High-uncertainty regions for model refinement

### Visualization

- 2D and 3D response surfaces
- Interactive Plotly-based exploration

### Reporting

- Downloadable PDF decision report
- Ready for laboratory documentation

---

## Modeling Approach

- **Primary Model**: CatBoost Regressor
- **Hyperparameter Tuning**: Optuna (cross-validated RMSE)
- **Uncertainty Estimation**: Bootstrap ensemble predictions
- **Decision Metric**:  
  [Robust Score = mu - lambda*sigma]

---


