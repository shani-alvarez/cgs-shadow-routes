# model/model.py
from __future__ import annotations

import random

import networkx as nx
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from mesa.time import RandomActivation

from .agents import CivilianAgent, OCGAgent
from .env import build_env_graph, update_node_exogenous


class ShadowRoutesModel(Model):
    def __init__(
        self,
        munis_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        crime_panel: pd.DataFrame | None = None,
        start_t: int = 0,
        end_t: int = 120,  # 2015-01..2024-12 = 120 months
        n_ocg: int = 50,
        civ_per_muni: int = 5,
        mining_weight: float = 1.0,
        v_escalation: float = 0.5,
        enable_sink: bool = True,
        seed: int | None = 42,
    ):
        super().__init__(seed=seed)
        self.random.seed(seed)
        random.seed(seed)

        # Environment graph and grid
        self.G: nx.Graph = build_env_graph(munis_df, edges_df)
        self.grid = NetworkGrid(self.G)
        self.crime_panel = crime_panel
        self.t = start_t
        self.end_t = end_t

        # Params
        self.mining_weight = mining_weight
        self.v_escalation = v_escalation
        self.enable_sink = enable_sink

        self.scheduler = RandomActivation(self)

        # Bookkeeping
        self._ocg_presence = {n: set() for n in self.G.nodes}
        self._flows = []  # list of (t, origin, dest)

        # Create agents
        self._init_ocgs(n_ocg)
        self._init_civilians(civ_per_muni)

        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "t": lambda m: m.t,
                "violence_mean": lambda m: sum(m.G.nodes[n]["violence"] for n in m.G)
                / m.G.number_of_nodes(),
                "contested_munis": lambda m: sum(1 for n in m.G if len(m._ocg_presence[n]) >= 2),
                "displacement_flows": lambda m: len(m._flows),  # flows this tick
            }
        )

    # ---- initialization helpers ----
    def _init_ocgs(self, n_ocg: int):
        muni_ids = list(self.G.nodes)
        for i in range(n_ocg):
            m = self.random.choice(muni_ids)
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

    def _init_civilians(self, civ_per_muni: int):
        uid = 0
        for m in self.G.nodes:
            for _ in range(civ_per_muni):
                a = CivilianAgent(
                    unique_id=f"civ_{uid}",
                    model=self,
                    muni_id=m,
                    mobility=self.random.uniform(0.7, 1.3),
                    risk_threshold=self.random.uniform(0.8, 1.6),
                )
                uid += 1
                self.scheduler.add(a)
                self.grid.place_agent(a, m)

    # ---- utilities for agents ----
    def tag_ocg_presence(self, muni_id: int, agent: OCGAgent):
        self._ocg_presence[muni_id].add(agent.unique_id)

    def ocg_presence_count(self, muni_id: int) -> int:
        return len(self._ocg_presence[muni_id])

    def register_flow(self, origin, dest):
        self._flows.append((self.t, origin, dest))

    # ---- one step ----
    def step(self):
        # Reset presence map for this tick
        self._ocg_presence = {n: set() for n in self.G.nodes}
        self._flows = []

        # 1) Exogenous updates (if provided)
        update_node_exogenous(self.G, self.t, self.crime_panel)

        # 2) Run all agents (OCGs and Civilians) in randomized order
        self.scheduler.step()

        # 3) Collect data and advance time
        self.datacollector.collect(self)
        self.t += 1

    def run(self):
        while self.t <= self.end_t:
            self.step()
