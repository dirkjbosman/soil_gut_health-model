# 🌱 Soil-Plant-Animal-Gut Health Causal Inference Pipeline

A robust, modular causal inference and machine learning pipeline modeling complex multi-trophic pathways along the Soil-Plant-Animal-Gut axis. This repository combines structural causal modeling (DoWhy), interactive DAG visualization (Pyvis), machine learning feature attribution (XGBoost & SHAP), and an automated reporting engine that compiles all findings into a single, unified executive dashboard (analysis/index.html).

---

## 📂 Repository Architecture

```
soil_gut_health-model/
├── analysis/
│   ├── index.html                      # Unified executive dashboard (HTML)
│   ├── interactive_dag.html            # Interactive vis.js causal network graph
│   └── img/
│       └── shap_summary.png            # Multi-feature SHAP impact summary plot
├── data/
│   ├── soil_gut_dataset.csv            # Raw synthetic dataset (1,500 samples)
│   └── processed_soil_gut_dataset.csv  # Scaled and outlier-capped dataset
├── src/
│   ├── seed.py                         # Synthetic data generation engine
│   ├── mung.py                         # Data preprocessing, scaling & outlier clipping
│   ├── model.py                        # Causal DAG modeling & DoWhy ATE estimation
│   ├── refute.py                       # Dual-refutation validation suite (Placebo & Random Common Cause)
│   ├── shap.py                         # XGBoost training & SHAP interpretability engine
│   └── report.py                       # Automated unified HTML dashboard compiler
├── main.py                             # Master orchestration script
├── pyproject.toml                      # Project metadata and strict dependency pinning
└── uv.lock                             # Locked dependency environment
```

---

## ⚙️ Prerequisites & Dependency Management

This project uses `uv` for lightning-fast, reproducible dependency management and virtual environment configuration.

1. Clone & Sync Environment:
   uv sync

2. Core Dependencies:
   - Python == 3.11.*
   - Pandas & NumPy
   - NetworkX & Pyvis (for structural DAG modeling and interactive visualization)
   - DoWhy (for nonparametric causal estimation and refutation testing)
   - XGBoost & SHAP (for machine learning explainability)

---

## 🚀 Execution Instructions

You can execute the pipeline modules individually or run the master orchestrator to automatically generate all datasets, models, plots, and the unified HTML report.

### Option A: Run via Master Orchestrator (Recommended)
uv run main.py

### Option B: Execute Scripts Individually per Step
uv run src/seed.py
uv run src/mung.py
uv run src/model.py
uv run src/refute.py
uv run src/shap.py
uv run src/report.py

---

## 🔬 Pipeline Workflow Breakdown

* Script 1 (seed.py): Generates 1,500 synthetic samples tracking the multi-trophic agricultural ecosystem and exports raw data to data/soil_gut_dataset.csv.
* Script 2 (mung.py): Applies robust outlier capping via interquartile ranges (IQR), missing value handling, and standard scaling.
* Script 3 (model.py): Constructs a native NetworkX directed acyclic graph (DAG), renders an interactive HTML visualization, and computes the Nonparametric Average Treatment Effect (ATE) via DoWhy.
* Script 4 (refute.py): Stress-tests the causal model using Placebo Treatment and Random Common Cause refuters to validate structural robustness.
* Script 5 (shap.py): Trains an optimized XGBoost regressor and computes SHAP feature importance values, saving the summary plot under analysis/img/shap_summary.png.
* Script 6 (report.py): Compiles all causal stats, refutation results, interactive networks, SHAP plots, and domain interpretations into a single, executive-ready dashboard at analysis/index.html.

---

## 📜 License

Distributed under the MIT License. See LICENSE for more information.