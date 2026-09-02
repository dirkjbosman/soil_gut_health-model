import numpy as np
import pandas as pd
import os

print("\n--- SCRIPT 1: GENERATING DATASET & CSV EXPORT ---")
def generate_ultimate_soil_gut_dataset(n_samples=1500, seed=42):
    np.random.seed(seed)
    
    antibiotic_use = np.random.binomial(1, 0.2, n_samples)
    genetics_score = np.random.normal(0, 1, n_samples)
    soil_health = np.clip(np.random.normal(60, 20, n_samples), 5, 100)
    
    soil_organic_carbon_pct = np.clip(0.08 * soil_health + np.random.normal(0, 0.5, n_samples), 0.5, 10.0)
    irrigation_water_quality = np.clip(0.7 * soil_health + np.random.normal(0, 10, n_samples), 10, 100)
    
    synthetic_fertilizer_index = np.clip(110 - soil_health + np.random.normal(0, 12, n_samples), 5, 100)
    pesticide_residue_ppb = np.clip(
        (100 - soil_health) * 4.2 + np.random.exponential(scale=20, size=n_samples), 
        5, 450
    )
    
    plant_quality = np.where(
        soil_health < 40,
        (2.5 * soil_organic_carbon_pct) + (0.2 * irrigation_water_quality) - (0.04 * pesticide_residue_ppb**0.7) + np.random.normal(0, 3, n_samples),
        (5.0 * soil_organic_carbon_pct) + (0.3 * irrigation_water_quality) - (0.05 * pesticide_residue_ppb**0.7) + np.random.normal(0, 4, n_samples)
    )
    plant_quality = np.clip(plant_quality, 0, 100)
    dietary_quality = np.clip(0.65 * plant_quality + np.random.normal(0, 8, n_samples), 0, 100)
    
    pasture_quality = np.clip((6.0 * soil_organic_carbon_pct) + (0.25 * irrigation_water_quality) - (0.05 * synthetic_fertilizer_index) + np.random.normal(0, 4, n_samples), 0, 100)
    meat_quality = np.clip(0.75 * pasture_quality - (0.03 * pesticide_residue_ppb**0.6) + np.random.normal(0, 5, n_samples), 0, 100)
    
    gut_health = np.clip(
        0.2 * dietary_quality + 
        0.2 * meat_quality +
        0.2 * (soil_organic_carbon_pct * 10) - 
        0.04 * pesticide_residue_ppb - 
        18 * antibiotic_use + 
        4 * genetics_score + 
        np.random.normal(0, 4, n_samples), 
        0, 100
    )
    
    df = pd.DataFrame({
        'soil_health': soil_health,
        'soil_organic_carbon_pct': soil_organic_carbon_pct,
        'irrigation_water_quality': irrigation_water_quality,
        'synthetic_fertilizer_index': synthetic_fertilizer_index,
        'pesticide_residue_ppb': pesticide_residue_ppb,
        'plant_quality': plant_quality,
        'dietary_quality': dietary_quality,
        'pasture_quality': pasture_quality,
        'meat_quality': meat_quality,
        'antibiotic_use': antibiotic_use,
        'genetics_score': genetics_score,
        'gut_health': gut_health
    })
    
    os.makedirs('data', exist_ok=True)
    df_cleaned = df.dropna()
    df_cleaned.to_csv('data/soil_gut_dataset.csv', index=False)
    print(f"-> Generated {len(df_cleaned)} samples and saved to 'data/soil_gut_dataset.csv'.")
    return df_cleaned

if __name__ == "__main__":
    generate_ultimate_soil_gut_dataset()