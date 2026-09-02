import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

print("\n--- SCRIPT 2: DATA MUNGING, OUTLIER CAPPING & SCALING ---")
def process_dataset():
    df_soil_gut = pd.read_csv('data/soil_gut_dataset.csv')
    continuous_features = [
        'soil_health', 'soil_organic_carbon_pct', 'irrigation_water_quality',
        'synthetic_fertilizer_index', 'pesticide_residue_ppb', 'plant_quality', 
        'dietary_quality', 'pasture_quality', 'meat_quality', 'genetics_score', 'gut_health'
    ]
    df_clean = df_soil_gut.copy()

    for col in continuous_features:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean[col] = df_clean[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
    input_features = [f for f in continuous_features if f != 'gut_health']
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_clean[input_features])

    df_scaled = pd.DataFrame(scaled_array, columns=input_features)
    df_processed = pd.concat([
        df_scaled, 
        df_clean[['antibiotic_use']].reset_index(drop=True),
        df_clean[['gut_health']].reset_index(drop=True)
    ], axis=1)
    
    os.makedirs('data', exist_ok=True)
    df_processed.to_csv('data/processed_soil_gut_dataset.csv', index=False)
    print("-> Outlier treatment and scaling applied successfully. Saved to 'data/processed_soil_gut_dataset.csv'.")
    return df_processed

if __name__ == "__main__":
    process_dataset()