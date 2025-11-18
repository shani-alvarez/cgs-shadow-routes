# model/env.py
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_env_graph(munis_df: pd.DataFrame, edges_df: None) -> nx.Graph:
    """
    Build the environment graph from the municipal dataframe.

    Parameters
    ----------
    munis_df : pd.DataFrame
        One row per municipality, with columns:
        - node_id
        - muni_id
        - state_id
        - municipality
        - state
        - pop_total
        - prot_idx          # already rescaled to [0, 1] upstream
        - mining_idx
        - collusion_idx
    edges_df : pd.DataFrame or None
        DataFrame with columns 'source', 'target' for adjacency.
        If None, graph starts with nodes only (no edges).

    Returns
    -------
    G : networkx.Graph
        Graph where each node is a municipality with static attributes.
    """
    G = nx.Graph()

    valid_nodes = set()
    for _, row in munis_df.iterrows():
        node = row["node_id"]
        valid_nodes.add(node)

        G.add_node(
            node,
            muni_id=row["muni_id"],
            state_id=row["state_id"],
            state=row["state"],
            municipality=row["municipality"],
            pop_total=row["pop_total"],
            prot_idx=row["prot_idx"],
            mining_idx=row["mining_idx"],
            collusion_idx=row["collusion_idx"],
            # Dynamic attributes, filled each tick from crime_panel
            extortion_rate=0.0,
            homicide_rate=0.0,
            drug_dealing_rate=0.0,
            kidnapping_rate=0.0,
            human_trafficking_rate=0.0,
            violence=0.0,  # normalized composite violence index [0, 1]
        )

    if edges_df is not None:
        for _, row in edges_df.iterrows():
            u = row["source"]
            v = row["target"]
            if (u in valid_nodes) and (v in valid_nodes):
                G.add_edge(u, v)

    return G


def update_env_from_crime(G: nx.Graph, crime_panel: pd.DataFrame):
    """
    Update node attributes in the graph using a crime panel.

    Parameters
    ----------
    G : networkx.Graph
        Environment graph with node attributes including muni_id.
    crime_panel : pd.DataFrame
        DataFrame indexed by muni_id with columns:
        - extortion_rate
        - homicide_rate
        - drug_dealing_rate
        - kidnapping_rate
        - human_trafficking_rate
        - violence_idx   # normalized [0, 1], precomputed upstream
    """
    for _, data in G.nodes(data=True):
        muni_id = data.get("muni_id")
        if muni_id is None:
            continue  # orphan node or sink

        if muni_id in crime_panel.index:
            row = crime_panel.loc[muni_id]
            data["extortion_rate"] = float(row["extortion_rate"])
            data["homicide_rate"] = float(row["homicide_rate"])
            data["drug_dealing_rate"] = float(row["drug_dealing_rate"])
            data["kidnapping_rate"] = float(row["kidnapping_rate"])
            data["human_trafficking_rate"] = float(row["human_trafficking_rate"])
            data["violence"] = float(row["violence_idx"])
        else:
            data["extortion_rate"] = 0.0
            data["homicide_rate"] = 0.0
            data["drug_dealing_rate"] = 0.0
            data["kidnapping_rate"] = 0.0
            data["human_trafficking_rate"] = 0.0
            data["violence"] = 0.0


def get_crime_slice(crime_panel: pd.DataFrame, t: int) -> pd.DataFrame:
    """
    Return crime rates for a given time index t, indexed by muni_id.

    Parameters
    ----------
    crime_panel : pd.DataFrame
        Must contain columns:
        - t
        - muni_id
        - extortion_rate
        - homicide_rate
        - drug_dealing_rate
        - kidnapping_rate
        - human_trafficking_rate
        - violence_idx
    t : int
        Time index used in the panel.

    Returns
    -------
    slice_df : pd.DataFrame
        Indexed by muni_id with crime rate columns.
    """
    slice_df = crime_panel.loc[crime_panel["t"] == t].set_index("muni_id")[
        [
            "extortion_rate",
            "homicide_rate",
            "drug_dealing_rate",
            "kidnapping_rate",
            "human_trafficking_rate",
            "violence_idx",
        ]
    ]

    return slice_df
