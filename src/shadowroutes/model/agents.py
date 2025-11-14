# model/agents.py
from __future__ import annotations

import random

from mesa import Agent


class OCGAgent(Agent):
    def __init__(
        self,
        unique_id,
        model,
        muni_id: int,
        violence_propensity: float = 0.4,
        expansion_tendency: float = 0.2,
    ):
        super().__init__(unique_id, model)
        self.muni_id = muni_id
        self.violence_propensity = violence_propensity  # baseline willingness to use violence
        self.expansion_tendency = expansion_tendency  # chance to probe neighbors
        self.resources = 1.0  # crude revenue stock
        self.alive = True

    def step(self):
        if not self.alive:
            return
        G = self.model.G
        node = G.nodes[self.muni_id]

        # 1) Assess local payoff and risk
        profit = (
            0.6 * node["drugs"]
            + 0.5 * node["extortion"]
            + 0.4 * self.model.mining_weight * node["mining_idx"]
        )
        enforcement = node["prot_idx"] * (
            1.0 - node["collusion_idx"]
        )  # protection reduces you, collusion increases you
        rivals = self.model.ocg_presence_count(self.muni_id) - 1
        contest_risk = 0.3 * rivals

        # 2) Decide violence level (very simple rule)
        escalate = random.random() < (
            self.violence_propensity + 0.2 * contest_risk - 0.2 * enforcement
        )
        if escalate:
            node["violence"] += self.model.v_escalation

        # 3) Consider expansion to a neighbor
        if random.random() < self.expansion_tendency:
            candidates = list(G.neighbors(self.muni_id))
            if candidates:
                # choose neighbor with best payoff - risk
                scored = []
                for nb in candidates:
                    n = G.nodes[nb]
                    payoff = (
                        0.6 * n["drugs"]
                        + 0.5 * n["extortion"]
                        + 0.4 * self.model.mining_weight * n["mining_idx"]
                    )
                    risk = n["prot_idx"] * (1.0 - n["collusion_idx"])
                    scored.append((payoff - risk, nb))
                scored.sort(reverse=True)
                best_nb = scored[0][1]
                # stochastic move if it beats current by margin
                if scored[0][0] > (profit - enforcement) + 0.1:
                    # "enter" neighbor = spawn light presence
                    self.model.tag_ocg_presence(
                        best_nb, self
                    )  # registers presence for contestation
                    # 50% chance to actually move headquarters
                    if random.random() < 0.5:
                        self.muni_id = best_nb


class CivilianAgent(Agent):
    def __init__(
        self, unique_id, model, muni_id: int, mobility: float = 1.0, risk_threshold: float = 1.0
    ):
        super().__init__(unique_id, model)
        self.muni_id = muni_id
        self.mobility = mobility  # ability to move (income, transport)
        self.risk_threshold = risk_threshold  # tolerance to violence
        self.displaced = False

    def step(self):
        G = self.model.G
        node = G.nodes[self.muni_id]
        # perceived risk: violence amplified by contestation and weak protection
        rivals = self.model.ocg_presence_count(self.muni_id)
        protection = node["prot_idx"] * (1.0 - node["collusion_idx"])
        perceived_risk = node["violence"] + 0.4 * (rivals - 1) - 0.3 * protection

        if perceived_risk > self.risk_threshold:
            # evaluate neighbors + optional sink
            dest = self.choose_destination()
            if dest is not None and dest != self.muni_id:
                self.model.register_flow(self.muni_id, dest)
                self.muni_id = dest
                self.displaced = True

    def choose_destination(self):
        G = self.model.G
        candidates = list(G.neighbors(self.muni_id))
        if self.model.enable_sink:
            candidates += ["SINK"]  # outside region
        # score each candidate
        best, best_score = self.muni_id, -1e9
        for c in candidates:
            if c == "SINK":
                score = 0.3  # constant outside option
            else:
                n = G.nodes[c]
                safety = -n["violence"] - 0.3 * (self.model.ocg_presence_count(c) - 1)
                capacity = n["prot_idx"]
                road = 1.0  # stub; could use edge cost
                score = safety + 0.2 * capacity - 0.05 * road
            if score > best_score:
                best, best_score = c, score
        return best
