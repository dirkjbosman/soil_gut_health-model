import networkx as nx

# --- ROBUST NETWORKX 3.x PATCH FOR DOWHY ---
import sys
from types import ModuleType

if not hasattr(nx, 'algorithms') or not hasattr(nx.algorithms, 'd_separated'):
    if hasattr(nx, 'd_sep') and hasattr(nx.d_sep, 'd_separated'):
        nx.algorithms.d_separated = nx.d_sep.d_separated
    elif hasattr(nx, 'd_separated'):
        nx.algorithms.d_separated = nx.d_separated
# -------------------------------------------

import pandas as pd
import os
from dowhy import CausalModel
from pyvis.network import Network

print("\n--- SCRIPT 3: INTERACTIVE DRAGGABLE DAG & CAUSAL MODELING ---")
def run_causal_model():
    df_processed = pd.read_csv('data/processed_soil_gut_dataset.csv')
    
    os.makedirs('analysis', exist_ok=True)
    net = Network(height='700px', width='100%', notebook=False, directed=True)
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150, spring_strength=0.05, damping=0.9)

    edges = [
        ('soil_health', 'soil_organic_carbon_pct'),
        ('soil_health', 'irrigation_water_quality'),
        ('soil_health', 'synthetic_fertilizer_index'),
        ('soil_health', 'pesticide_residue_ppb'),
        ('soil_organic_carbon_pct', 'plant_quality'),
        ('soil_organic_carbon_pct', 'pasture_quality'),
        ('irrigation_water_quality', 'plant_quality'),
        ('synthetic_fertilizer_index', 'plant_quality'),
        ('pesticide_residue_ppb', 'plant_quality'),
        ('pesticide_residue_ppb', 'meat_quality'),
        ('pesticide_residue_ppb', 'gut_health'),
        ('soil_health', 'plant_quality'), 
        ('plant_quality', 'dietary_quality'),
        ('soil_health', 'pasture_quality'), 
        ('pasture_quality', 'meat_quality'),
        ('meat_quality', 'gut_health'), 
        ('dietary_quality', 'gut_health'),
        ('soil_health', 'gut_health'), 
        ('antibiotic_use', 'gut_health'),
        ('genetics_score', 'gut_health')
    ]

    for source, target in edges:
        net.add_node(source, label=source, color='#85E3D8', size=25)
        net.add_node(target, label=target, color='#FFB7B2', size=25)
        net.add_edge(source, target)

    net.save_graph('analysis/interactive_dag.html')
    print("-> Interactive network saved to 'analysis/interactive_dag.html'.")

    dag = nx.DiGraph(edges)

    model = CausalModel(
        data=df_processed,
        treatment='soil_health',
        outcome='gut_health',
        common_causes=['antibiotic_use', 'genetics_score'],
        graph=dag
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    causal_estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True
    )

    print("\n--- Causal Estimation Results ---")
    print(causal_estimate)
    return model, identified_estimand, causal_estimate

if __name__ == "__main__":
    run_causal_model()