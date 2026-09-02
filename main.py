from src.seed import generate_ultimate_soil_gut_dataset
from src.mung import process_dataset
from src.model import run_causal_model
from src.refute import run_refutations
from src.shap import run_shap_analysis
from src.report import generate_unified_dashboard

if __name__ == "__main__":
    print("🌱 Starting Full Soil-Gut Causal Pipeline Execution...")
    
    # Step 1: Generate Raw Data & Export CSV
    generate_ultimate_soil_gut_dataset()
    
    # Step 2: Clean, Cap Outliers, and Scale Features
    process_dataset()
    
    # Step 3: Build Causal DAG & Run DoWhy Estimation
    run_causal_model()
    
    # Step 4: Execute Statistical Refutation Tests
    run_refutations()
    
    # Step 5: Train XGBoost, Compute SHAP, and Save Plot under analysis/img/
    run_shap_analysis()
    
    # Step 6: Compile Everything into Single Unified Dashboard (analysis/index.html)
    generate_unified_dashboard()
    
    print("\n✨ Pipeline execution complete! Open './analysis/index.html' to view your unified dashboard.")