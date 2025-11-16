# model/env.py
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_env_graph(munis_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """
    Creates the static set up of the environment.
    munis_df: columns ['muni_id','state','name','pop','poverty','mining_idx',
                      'prot_idx','collusion_idx']
    edges_df: columns ['src','dst','road_cost']
    """
    G = nx.Graph()
    for _, r in munis_df.iterrows():
        G.add_node(
            int(r["muni_id"]),
            state=r["state"],
            name=r["name"],
            pop=int(r["pop"]),
            poverty=float(r.get("poverty", 0.0)),
            mining_idx=float(r.get("mining_idx", 0.0)),
            prot_idx=float(r.get("prot_idx", 0.0)),  # enforcement/protection
            collusion_idx=float(r.get("collusion_idx", 0.0)),
            # dynamic attributes updated each step:
            violence=0.0,
            extortion=0.0,
            kidnapping=0.0,
            drugs=0.0,
        )
    for _, e in edges_df.iterrows():
        G.add_edge(int(e["src"]), int(e["dst"]), road_cost=float(e.get("road_cost", 1.0)))
    return G


def update_node_exogenous(G, t: int, crime_panel: pd.DataFrame):
    """
    Apply monthly exogenous updates to each node from the crime panel (SESNSP).
    crime_panel columns: ['t','muni_id','extortion','kidnapping','homicide','drug']
    """
    if crime_panel is None:
        return
    frame = crime_panel[crime_panel["t"] == t]
    for _, r in frame.iterrows():
        n = int(r["muni_id"])
        if n in G:
            G.nodes[n]["extortion"] = float(r.get("extortion", 0.0))
            G.nodes[n]["kidnapping"] = float(r.get("kidnapping", 0.0))
            G.nodes[n]["drugs"] = float(r.get("drugs", 0.0))
            G.nodes[n]["homicide"] = float(r.get("homicide", 0.0))

            G.nodes[n]["violence"] = (
                1.0 * G.nodes[n]["homicide"]
                + 0.8 * G.nodes[n]["kidnapping"]
                + 0.7 * G.nodes[n]["extortion"]
                + 0.3 * G.nodes[n]["drugs"]
            )
