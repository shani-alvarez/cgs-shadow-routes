# run.py
import pandas as pd
from model.model import ShadowRoutesModel

# Minimal dummy inputs so it runs
munis = pd.DataFrame(
    {
        "muni_id": [1, 2, 3, 4],
        "state": ["GRO", "GRO", "MIC", "MEX"],
        "name": ["A", "B", "C", "D"],
        "pop": [10000, 12000, 15000, 20000],
        "poverty": [0.5, 0.4, 0.6, 0.3],
        "mining_idx": [0.7, 0.2, 0.1, 0.0],
        "prot_idx": [0.3, 0.5, 0.2, 0.7],
        "collusion_idx": [0.6, 0.3, 0.5, 0.2],
    }
)
edges = pd.DataFrame(
    {
        "src": [1, 2, 2, 3],
        "dst": [2, 3, 4, 4],
        "road_cost": [1, 1, 1, 1],
    }
)

# Optional: crime panel (monthly)
crime_panel = None
# Example schema if you add it:
# crime_panel = pd.DataFrame({
#   "t":[0,0,0,0],
#   "muni_id":[1,2,3,4],
#   "extortion":[0.4,0.2,0.1,0.05],
#   "kidnapping":[0.1,0.05,0.0,0.0],
#   "drugs":[0.2,0.3,0.1,0.05],
# })

model = ShadowRoutesModel(
    munis_df=munis,
    edges_df=edges,
    crime_panel=crime_panel,
    start_t=0,
    end_t=24,  # 2 simulated years for a quick test
    n_ocg=5,
    civ_per_muni=10,
    mining_weight=1.0,
    v_escalation=0.4,
    enable_sink=True,
    seed=7,
)

model.run()
results = model.datacollector.get_model_vars_dataframe()
print(results.head())
