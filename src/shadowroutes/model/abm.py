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

    - If crime_panel is None, it does nothing.
    - Otherwise, it:
        1) slices the panel for time t (t is a precomputed month index)
        2) calls update_env_from_crime to push those values into the graph.

    This respects the design you already tested manually in the notebook.
    """
    if crime_panel is None:
        return

    # get_crime_slice should:
    #   - filter crime_panel to t
    #   - set index to muni_id (as string)
    crime_t = get_crime_slice(crime_panel, t)
    update_env_from_crime(G, crime_t)


class Model(MesaModel):
    """
    Core Shadow Routes ABM.

    Agents:
    - OCGAgent: organized criminal groups, competing over extortion/drug/mining payoffs.
    - CivilianAgent: households/individuals deciding whether to stay or move.

    Environment:
    - Network of municipalities (nodes) with:
        pop_total, prot_idx, mining_idx, collusion_idx
      + crime rates injected each tick from crime_panel, if provided.
    """

    def __init__(
        self,
        munis_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        crime_panel: pd.DataFrame | None = None,
        start_t: int = 0,
        end_t: int = 120,  # 2015-01..2024-12 = 120 months if t is precomputed
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

        # Parameters
        self.mining_weight = mining_weight
        self.v_escalation = v_escalation
        self.enable_sink = enable_sink

        # Scheduler
        self.scheduler = RandomActivation(self)

        # Bookkeeping
        #   - which OCGs are present in each muni this tick (for contestation)
        #   - displacement flows (origin, dest, time)
        self._ocg_presence: dict = {n: set() for n in self.G.nodes}
        self._flows: list[tuple[int, str, str]] = []

        # Initialize agents
        self._init_ocgs(n_ocg)
        self._init_civilians(civ_per_muni)

        # DataCollector: coarse proof-of-concept indicators
        self.datacollector = DataCollector(
            model_reporters={
                "t": lambda m: m.t,
                "violence_mean": lambda m: sum(m.G.nodes[n]["violence"] for n in m.G)
                / m.G.number_of_nodes(),
                "contested_munis": lambda m: sum(1 for n in m.G if len(m._ocg_presence[n]) >= 2),
                "displacement_flows": lambda m: len(m._flows),
            }
        )

    # ---- initialization helpers ----
    def _init_ocgs(self, n_ocg: int) -> None:
        """
        Spawn n_ocg OCG agents, randomly placed in municipalities,
        with heterogeneous violence propensity and expansion tendency.
        """
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

    def _init_civilians(self, civ_per_muni: int) -> None:
        """
        Spawn civ_per_muni civilian agents in each municipality.
        Each has heterogeneous mobility and risk threshold.
        """
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
    def tag_ocg_presence(self, muni_id, agent: OCGAgent) -> None:
        """
        Register that a given OCG is present in muni_id this tick.
        Used to compute contestation (number of groups per muni).
        """
        self._ocg_presence[muni_id].add(agent.unique_id)

    def ocg_presence_count(self, muni_id) -> int:
        """
        Number of distinct OCGs present in a municipality in the current tick.
        """
        return len(self._ocg_presence[muni_id])

    def register_flow(self, origin, dest) -> None:
        """
        Store that a civilian displaced from origin to dest at time t.
        """
        self._flows.append((self.t, origin, dest))

    # ---- one step ----
    def step(self) -> None:
        """
        One tick of the simulation:
        1) Reset presence and flows.
        2) Inject exogenous crime rates from crime_panel (if provided).
        3) Step all agents.
        4) Collect model-level indicators and advance time.
        """
        # Reset presence and flows for this tick
        self._ocg_presence = {n: set() for n in self.G.nodes}
        self._flows = []

        # 1) Exogenous updates (crime rates, etc.)
        update_node_exogenous(self.G, self.t, self.crime_panel)

        # 2) Run all agents (OCGs and Civilians) in randomized order
        self.scheduler.step()

        # 3) Collect data and advance time
        self.datacollector.collect(self)
        self.t += 1

    def run(self) -> None:
        """
        Run the model from start_t to end_t (inclusive).
        """
        while self.t <= self.end_t:
            self.step()
