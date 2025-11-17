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
    - Can escalate violence (by bumping homicide_rate) and expand to neighbors.
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
        self.violence_propensity = violence_propensity  # baseline willingness to use violence
        self.expansion_tendency = expansion_tendency  # chance to probe neighbors
        self.resources = 1.0  # crude revenue stock
        self.alive = True

    def _local_payoff(self, node_attrs: dict) -> float:
        """
        Heuristic payoff for a municipality, based on rents and markets.
        """
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

        # 1) Assess local payoff and risk
        profit = self._local_payoff(node)
        enforcement = self._enforcement_risk(node)

        rivals = max(self.model.ocg_presence_count(self.muni_id) - 1, 0)
        contest_risk = 0.3 * rivals

        # 2) Decide whether to escalate violence (very simple rule)
        p_escalate = self.violence_propensity + 0.2 * contest_risk - 0.2 * enforcement
        if random.random() < max(min(p_escalate, 1.0), 0.0):
            # We bump homicide_rate as a proxy for lethal violence escalation
            current_hom = node.get("homicide_rate", 0.0)
            node["homicide_rate"] = max(current_hom + self.model.v_escalation, 0.0)

        # 3) Consider expansion to a neighbor
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

                # Compare best neighbor to current node with a small margin
                current_score = profit - enforcement
                if best_score > current_score + 0.1:
                    # Register presence (for contestation / civilian risk)
                    self.model.tag_ocg_presence(best_nb, self)

                    # 50% chance we actually move our "headquarters" to that node
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
        self.mobility = mobility  # ability to move (income, networks, transport)
        self.risk_threshold = risk_threshold  # tolerance to violence
        self.displaced = False

    def _base_violence(self, node_attrs: dict) -> float:
        """
        Aggregate local 'violence' as a function of observed crime rates.

        We weight extortion and kidnapping slightly higher than homicide
        to reflect their strong link to chronic threats and forced displacement.
        """
        h = node_attrs.get("homicide_rate", 0.0)
        e = node_attrs.get("extortion_rate", 0.0)
        k = node_attrs.get("kidnapping_rate", 0.0)

        # Simple, monotone weights; subject to later calibration
        return 0.35 * h + 0.40 * e + 0.25 * k

    def step(self):
        # If already displaced out of region, do nothing
        if self.muni_id == "SINK":
            return

        G = self.model.G
        node = G.nodes[self.muni_id]

        # local violence signal from crime rates (plus any OCG escalation)
        base_v = self._base_violence(node)

        # contestation: more groups -> more risk
        rivals = max(self.model.ocg_presence_count(self.muni_id) - 1, 0)

        # effective state protection: high prot_idx, low collusion
        protection = node.get("prot_idx", 0.0) * (1.0 - node.get("collusion_idx", 0.0))

        perceived_risk = base_v + 0.4 * rivals - 0.3 * protection

        if perceived_risk > self.risk_threshold and random.random() < self.mobility:
            dest = self.choose_destination()
            if dest is not None and dest != self.muni_id:
                # register flow for stats
                self.model.register_flow(self.muni_id, dest)
                if dest == "SINK":
                    # Agent exits the modeled region
                    self.muni_id = "SINK"
                    self.displaced = True
                    return
                else:
                    self.muni_id = dest
                    self.displaced = True

    def _node_violence_for_dest(self, node_attrs: dict, muni_id: Hashable) -> float:
        """
        Violence component used when scoring destinations.
        Uses the same crime-based index plus contestation.
        """
        base_v = self._base_violence(node_attrs)
        rivals = max(self.model.ocg_presence_count(muni_id) - 1, 0)
        return base_v + 0.3 * rivals

    def choose_destination(self):
        G = self.model.G
        candidates = list(G.neighbors(self.muni_id))

        if self.model.enable_sink:
            candidates.append("SINK")  # outside region

        best, best_score = self.muni_id, -1e9

        for c in candidates:
            if c == "SINK":
                # Outside option: constant, relatively safe but costly.
                # You can tune this later.
                score = 0.3
            else:
                n = G.nodes[c]
                violence = self._node_violence_for_dest(n, c)
                capacity = n.get("prot_idx", 0.0)
                # road cost is just a stub; could use edge attribute if you want
                road_cost = 1.0

                # safer + more capacity = more attractive
                score = -violence + 0.2 * capacity - 0.05 * road_cost

            if score > best_score:
                best, best_score = c, score

        return best
