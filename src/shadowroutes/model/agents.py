# model/agents.py
from __future__ import annotations

import random
from collections.abc import Hashable
from typing import Any

from mesa import Agent


class OCGAgent(Agent):
    """
    Organized Crime Group agent.

    - Lives on a municipality node (muni_id).
    - Evaluates local and neighboring municipalities based on:
        * drug_dealing_rate
        * extortion_rate
        * mining_idx
        * prot_idx (state capacity)
        * collusion_idx (state-crime embeddedness)
    - Can escalate violence and expand to neighbors.
    """

    def __init__(
        self,
        unique_id: Hashable,
        model: Any,
        muni_id: Hashable,
        violence_propensity: float = 0.4,
        expansion_tendency: float = 0.2,
    ):
        super().__init__(unique_id, model)
        self.muni_id = muni_id
        self.violence_propensity = violence_propensity
        self.expansion_tendency = expansion_tendency
        self.resources = 1.0
        self.alive = True

    def _local_payoff(self, node_attrs: dict) -> float:
        """Heuristic payoff for a municipality, based on rents and markets."""
        return (
            0.6 * node_attrs.get("drug_dealing_rate", 0.0)
            + 0.5 * node_attrs.get("extortion_rate", 0.0)
            + 0.4 * self.model.mining_weight * node_attrs.get("mining_idx", 0.0)
        )

    def _enforcement_risk(self, node_attrs: dict) -> float:
        """
        Risk from state presence. High protection and low collusion = high risk.
        """
        prot = node_attrs.get("prot_idx", 0.0)
        coll = node_attrs.get("collusion_idx", 0.0)
        return prot * (1.0 - coll)

    def step(self):
        if not self.alive:
            return

        G = self.model.G
        node = G.nodes[self.muni_id]
        self.model.tag_ocg_presence(
            self.muni_id, self
        )  # register presence in current muni for this tick

        # 1) Local payoff & risk
        profit = self._local_payoff(node)
        enforcement = self._enforcement_risk(node)
        rivals = max(self.model.ocg_presence_count(self.muni_id) - 1, 0)
        contest_risk = 0.3 * rivals

        # 2) Violence escalation (nudges normalized violence upward)
        p_escalate = self.violence_propensity + 0.2 * contest_risk - 0.2 * enforcement
        p_escalate = max(min(p_escalate, 1.0), 0.0)

        if random.random() < p_escalate:
            # bump homicide_rate and violence index a bit, capped at 1
            current_hom = node.get("homicide_rate", 0.0)
            node["homicide_rate"] = max(current_hom + self.model.v_escalation, 0.0)
            node["violence"] = min(node.get("violence", 0.0) + 0.05, 1.0)

        # 3) Expansion
        if random.random() < self.expansion_tendency:
            candidates = list(G.neighbors(self.muni_id))
            if candidates:
                scored = []
                for nb in candidates:
                    n = G.nodes[nb]
                    payoff = self._local_payoff(n)
                    risk = self._enforcement_risk(n)
                    scored.append((payoff - risk, nb))

                scored.sort(reverse=True)
                best_score, best_nb = scored[0]
                current_score = profit - enforcement

                if best_score > current_score + 0.1:
                    self.model.tag_ocg_presence(best_nb, self)
                    if random.random() < 0.5:
                        self.muni_id = best_nb


class CivilianAgent(Agent):
    """
    Civilian (household / population block) agent.

    - Lives on a municipality node (muni_id).
    - Reads local crime + governance signals and decides to stay or move.
    - If it moves to SINK, it's treated as displaced out of the modeled region.
    """

    def __init__(
        self,
        unique_id: Hashable,
        model: Any,
        muni_id: Hashable,
        mobility: float = 1.0,
        risk_threshold: float = 1.0,
    ):
        super().__init__(unique_id, model)
        self.muni_id = muni_id
        self.mobility = mobility
        self.risk_threshold = risk_threshold
        self.displaced = False

    def _base_violence(self, node_attrs: dict) -> float:
        """
        Use the normalized violence index in [0, 1] as core signal.
        """
        return node_attrs.get("violence", 0.0)

    def step(self):
        if self.muni_id == "SINK":
            return

        G = self.model.G
        node = G.nodes[self.muni_id]

        base_v = self._base_violence(node)
        rivals = max(self.model.ocg_presence_count(self.muni_id) - 1, 0)
        protection = node.get("prot_idx", 0.0) * (1.0 - node.get("collusion_idx", 0.0))

        # violence [0,1], rivals (0,1,2...), protection [0,1]
        perceived_risk = 2 * base_v + 0.8 * rivals - 0.3 * protection

        if perceived_risk > self.risk_threshold and random.random() < self.mobility:
            dest = self.choose_destination()
            if dest is not None and dest != self.muni_id:
                self.model.register_flow(self.muni_id, dest)
                if dest == "SINK":
                    self.muni_id = "SINK"
                    self.displaced = True
                else:
                    self.muni_id = dest
                    self.displaced = True

    def _node_violence_for_dest(self, node_attrs: dict, muni_id: Hashable) -> float:
        """Violence proxy when scoring destinations."""
        base_v = self._base_violence(node_attrs)
        rivals = max(self.model.ocg_presence_count(muni_id) - 1, 0)
        return base_v + 0.3 * rivals

    def choose_destination(self):
        G = self.model.G
        candidates = list(G.neighbors(self.muni_id))

        if self.model.enable_sink:
            candidates.append("SINK")

        best, best_score = self.muni_id, -1e9

        for c in candidates:
            if c == "SINK":
                score = -0.2
            else:
                n = G.nodes[c]
                violence = self._node_violence_for_dest(n, c)
                capacity = n.get("prot_idx", 0.0)
                road_cost = 1.0  # placeholder

                score = -violence + 0.2 * capacity - 0.05 * road_cost

            if score > best_score:
                best, best_score = c, score

        return best
