import networkx as nx
# --- CORRECTED NETWORKX 3.x PATCH ---
if hasattr(nx, 'd_separated'):
    nx.algorithms.d_separated = nx.d_separated
elif hasattr(nx, 'd_separation'):
    nx.algorithms.d_separated = nx.d_separation
# ------------------------------------

from src.model import run_causal_model

print("\n--- SCRIPT 4: DUAL-REFUTATION VALIDATION SUITE ---")
def run_refutations():
    model, identified_estimand, causal_estimate = run_causal_model()

    refutation_placebo = model.refute_estimate(
        identified_estimand, causal_estimate, method_name="placebo_treatment_refuter", placebo_type="permute"
    )
    print("\n[Test 1: Placebo Treatment Refuter]")
    print(refutation_placebo)

    refutation_common_cause = model.refute_estimate(
        identified_estimand, causal_estimate, method_name="random_common_cause"
    )
    print("\n[Test 2: Random Common Cause Refuter]")
    print(refutation_common_cause)

if __name__ == "__main__":
    run_refutations()