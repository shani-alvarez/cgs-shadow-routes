# run.py

import pandas as pd
from model import Model

# -----------------------------------------------------
# 1. Load preprocessed datasets
# -----------------------------------------------------

df_munis = pd.read_csv("data/processed/df_munis.csv", dtype={"muni_id": str})
edges_df = pd.read_csv("data/processed/edges.csv", dtype={"source": str, "target": str})
crime_panel = pd.read_csv("data/processed/crime_panel.csv", dtype={"muni_id": str})

# Ensure index consistency
crime_panel.set_index("muni_id", inplace=True)

# Figure out T range
t_min = crime_panel["t"].min()
t_max = crime_panel["t"].max()

print(f"Loaded crime panel with t in [{t_min}, {t_max}]")

# -----------------------------------------------------
# 2. Instantiate model
# -----------------------------------------------------

model = Model(
    munis_df=df_munis,
    edges_df=edges_df,
    crime_panel=crime_panel,
    start_t=t_min,
    end_t=t_max,
    n_ocg=40,  # ~ tuning
    civ_per_muni=10,  # ~ tuning
    mining_weight=1.0,
    v_escalation=0.4,
    enable_sink=True,
    seed=42,
)

# -----------------------------------------------------
# 3. Run simulation
# -----------------------------------------------------

model.run()

# -----------------------------------------------------
# 4. Extract outputs
# -----------------------------------------------------

df_out = model.datacollector.get_model_vars_dataframe()
df_out.to_csv("outputs/abm_model_timeseries.csv", index=False)

print("Simulation finished!")
print(df_out.head())
