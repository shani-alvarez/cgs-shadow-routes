# model/model.py
from __future__ import annotations

import random

import networkx as nx
import pandas as pd
from mesa import Model as MesaModel
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from mesa.time import RandomActivation

from .agents import CivilianAgent, OCGAgent
from .env import build_env_graph, get_crime_slice, update_env_from_crime


def update_node_exogenous(G: nx.Graph, t: int, crime_panel: pd.DataFrame | None) -> None:
    """
    Wrapper to update node attributes from the crime panel at time t.
    """
    if crime_panel is None:
        return

    crime_t = get_crime_slice(crime_panel, t)
    update_env_from_crime(G, crime_t)


class Model(MesaModel):
    """
    Core Shadow Routes ABM.
    """

    def __init__(
        self,
        munis_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        crime_panel: pd.DataFrame | None = None,
        start_t: int = 0,
        end_t: int = 120,
        n_ocg: int = 6,
        civ_per_muni: int = 5,
        mining_weight: float = 1.0,
        v_escalation: float = 0.1,
        enable_sink: bool = True,
        seed: int | None = 42,
    ):
        super().__init__(seed=seed)
        self.random.seed(seed)
        random.seed(seed)

        self.G: nx.Graph = build_env_graph(munis_df, edges_df)
        self.grid = NetworkGrid(self.G)

        self.crime_panel = crime_panel
        self.t = start_t
        self.end_t = end_t

        self.mining_weight = mining_weight
        self.v_escalation = v_escalation
        self.enable_sink = enable_sink

        self.scheduler = RandomActivation(self)

        self._ocg_presence: dict = {n: set() for n in self.G.nodes}
        self._flows: list[tuple[int, str, str]] = []

        self._init_ocgs(n_ocg)
        self._init_civilians(civ_per_muni)

        self.datacollector = DataCollector(
            model_reporters={
                "t": lambda m: m.t,
                "violence_mean": lambda m: sum(m.G.nodes[n]["violence"] for n in m.G)
                / m.G.number_of_nodes(),
                "contested_munis": lambda m: sum(1 for n in m.G if len(m._ocg_presence[n]) >= 2),
                "displacement_flows": lambda m: len(m._flows),
            }
        )

        self._flows_history: list[tuple[int, str, str]] = []

    # ---- initialization helpers ----
    def _init_ocgs(self, n_ocg: int) -> None:
        """
        Spawn OCG agents, biased toward high-mining / high-violence municipalities.
        """
        muni_ids = list(self.G.nodes)

        # weights: mining importance + a bit of baseline violence
        weights = [
            self.G.nodes[m].get("mining_idx", 0.0)
            + 0.5 * self.G.nodes[m].get("violence", 0.0)
            + 0.05  # keep non-zero so everyone has a chance
            for m in muni_ids
        ]
        total_w = sum(weights)
        if total_w == 0:
            weights = [1.0 for _ in muni_ids]
            total_w = len(muni_ids)

        def weighted_choice():
            r = self.random.random() * total_w
            acc = 0.0
            for m, w in zip(muni_ids, weights, strict=False):
                acc += w
                if acc >= r:
                    return m
            return muni_ids[-1]

        for i in range(n_ocg):
            m = weighted_choice()
            a = OCGAgent(
                unique_id=f"ocg_{i}",
                model=self,
                muni_id=m,
                violence_propensity=self.random.uniform(0.2, 0.6),
                expansion_tendency=self.random.uniform(0.1, 0.3),
            )
            self.scheduler.add(a)
            self.grid.place_agent(a, m)
            self.tag_ocg_presence(m, a)

    def _init_civilians(self, civ_per_muni: int) -> None:
        """
        Spawn civ_per_muni civilian agents in each municipality.
        """
        uid = 0
        for m in self.G.nodes:
            for _ in range(civ_per_muni):
                a = CivilianAgent(
                    unique_id=f"civ_{uid}",
                    model=self,
                    muni_id=m,
                    mobility=self.random.uniform(0.7, 1.0),
                    risk_threshold=self.random.uniform(0.4, 0.9),
                )
                uid += 1
                self.scheduler.add(a)
                self.grid.place_agent(a, m)

    # ---- utilities for agents ----
    def tag_ocg_presence(self, muni_id, agent: OCGAgent) -> None:
        self._ocg_presence[muni_id].add(agent.unique_id)

    def ocg_presence_count(self, muni_id) -> int:
        return len(self._ocg_presence[muni_id])

    def register_flow(self, origin, dest) -> None:
        record = (self.t, origin, dest)
        self._flows.append(record)  # for per-tick stats
        self._flows_history.append(record)  # for after-run analysis

    # ---- one step ----
    def step(self) -> None:
        self._ocg_presence = {n: set() for n in self.G.nodes}
        self._flows = []

        update_node_exogenous(self.G, self.t, self.crime_panel)
        self.scheduler.step()
        self.datacollector.collect(self)
        self.t += 1

    def run(self) -> None:
        while self.t <= self.end_t:
            self.step()
