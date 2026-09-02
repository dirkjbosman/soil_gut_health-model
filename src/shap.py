import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import os
import matplotlib.pyplot as plt
from src.model import run_causal_model

print("\n--- SCRIPT 5: XGBOOST MODEL & SHAP ANALYSIS ---")
def run_shap_analysis():
    df_processed = pd.read_csv('data/processed_soil_gut_dataset.csv')
    
    # Unpack 3 values as returned by the unmodified model.py
    _, _, causal_estimate = run_causal_model()

    X = df_processed[[
        'soil_health', 'soil_organic_carbon_pct', 'irrigation_water_quality',
        'synthetic_fertilizer_index', 'pesticide_residue_ppb', 'plant_quality', 
        'dietary_quality', 'pasture_quality', 'meat_quality', 'antibiotic_use', 'genetics_score'
    ]]
    y = df_processed['gut_health']

    ml_model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    ml_model.fit(X, y)

    explainer = shap.Explainer(ml_model.predict, X)
    shap_values = explainer(X)

    # Save SHAP plot under analysis/img/
    os.makedirs('analysis/img', exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('analysis/img/shap_summary.png', bbox_inches='tight')
    plt.close()
    print("-> SHAP Summary Plot saved to 'analysis/img/shap_summary.png'.")

    print("\n" + "="*60)
    print("🤖 AUTOMATED SYSTEMS ECOLOGY CAUSAL REPORT")
    print("="*60)
    print(f"1. Causal Impact Analysis:")
    print(f"   - Adjusted Effect Size (ATE): {causal_estimate.value:.4f}")

    shap_vals_arr = shap_values.values if hasattr(shap_values, 'values') else shap_values
    mean_abs_shap = np.abs(shap_vals_arr).mean(axis=0)
    top_idx = np.argmax(mean_abs_shap)
    print(f"\n2. Primary Predictive Feature Driver: '{X.columns[top_idx]}'")
    print(f"\n3. Generated Local Assets:")
    print(f"   - Data Folder: ./data/")
    print(f"   - Analysis Folder: ./analysis/ (index.html & img/)")
    print("="*60)

if __name__ == "__main__":
    run_shap_analysis()