"""
social.py — Dynamic Social Network and Interaction system for MARL City Simulator.

Features implemented:
  1. Social Graph Network (edges with trust, friendship)
  2. Memory & Reputation System
  3. Trust-Based Lending
  4. Job Referral Network
  5. Mentorship (Knowledge Transfer)
  6. Social Influence Engine (Peer pressure)
  7. Family System
"""

import random
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from config import (
    SocialConfig, SOCIAL_CONFIG,
    AgentType, AGENT_TYPE_NAMES,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# INTERACTION & GRAPH DATA
# ═══════════════════════════════════════════

@dataclass
class InteractionResult:
    interaction_type: str
    agent_a_id: int
    agent_b_id: int
    description: str
    amount: float = 0.0
    success: bool = True

@dataclass
class EdgeData:
    friendship_strength: float = 0.0
    trust_score: float = 0.5
    interaction_frequency: int = 0
    loan_history: List[Dict[str, Any]] = field(default_factory=list)
    mentorship_history: List[Dict[str, Any]] = field(default_factory=list)
    is_family: bool = False
    family_type: str = ""

class SocialGraph:
    def __init__(self):
        self.edges: Dict[Tuple[int, int], EdgeData] = {}

    def get_edge(self, u: int, v: int) -> EdgeData:
        key = tuple(sorted((u, v)))
        if key not in self.edges:
            self.edges[key] = EdgeData()
        return self.edges[key]

    def has_edge(self, u: int, v: int) -> bool:
        return tuple(sorted((u, v))) in self.edges

    def get_friends(self, u: int, min_friendship=0.3) -> List[int]:
        friends = []
        for (a, b), data in self.edges.items():
            if data.friendship_strength >= min_friendship or data.is_family:
                if a == u: friends.append(b)
                elif b == u: friends.append(a)
        return friends
        
    def serialize_edges(self) -> List[Dict]:
        out = []
        for (a, b), data in self.edges.items():
            if data.friendship_strength > 0.1 or data.is_family or data.loan_history:
                out.append({
                    "source": a,
                    "target": b,
                    "friendship": round(data.friendship_strength, 3),
                    "trust": round(data.trust_score, 3),
                    "is_family": data.is_family,
                    "family_type": data.family_type,
                    "has_loan": len(data.loan_history) > 0
                })
        return out

# ═══════════════════════════════════════════
# SOCIAL SYSTEM
# ═══════════════════════════════════════════

class SocialSystem:
    def __init__(self, config: SocialConfig = SOCIAL_CONFIG):
        self.config = config
        self.graph = SocialGraph()

        # Statistics
        self.total_interactions: int = 0
        self.interactions_this_step: int = 0
        self.total_money_lent: float = 0.0
        self.total_money_repaid: float = 0.0
        self.interaction_counts: Dict[str, int] = {
            "job_referral": 0,
            "borrow_money": 0,
            "repay_loan": 0,
            "mentorship": 0,
            "economic_gossip": 0,
            "family_transfer": 0,
        }

    def initialize_families(self, all_agents: list):
        """Randomly group some agents into families at timestep 0."""
        # Simple heuristic: Group same types into families or random mixes
        unassigned = [a.id for a in all_agents]
        random.shuffle(unassigned)
        
        while len(unassigned) >= 2:
            if random.random() < 0.3: # 30% chance to form a family unit
                size = random.choice([2, 3, 4])
                if len(unassigned) < size:
                    break
                family_members = [unassigned.pop() for _ in range(size)]
                # Link them in the graph
                for i in range(len(family_members)):
                    for j in range(i + 1, len(family_members)):
                        u, v = family_members[i], family_members[j]
                        edge = self.graph.get_edge(u, v)
                        edge.is_family = True
                        edge.family_type = "spouse" if size == 2 else "relative"
                        edge.friendship_strength = 0.8
                        edge.trust_score = 0.9
                        all_agents[u].family_links[v] = edge.family_type
                        all_agents[v].family_links[u] = edge.family_type
            else:
                unassigned.pop()

    def check_interactions(self, all_agents: list, city, timestep: int) -> List[str]:
        events: List[str] = []
        self.interactions_this_step = 0
        interacted_pairs = set()

        # ── Step 1: Proximity Interactions (Creates new bonds, gossip) ──
        for agent in all_agents:
            nearby = city.get_agents_in_radius(agent, self.config.interaction_radius, all_agents)
            for other in nearby:
                pair_key = tuple(sorted((agent.id, other.id)))
                if pair_key in interacted_pairs: continue
                interacted_pairs.add(pair_key)
                
                # Proximity boosts interaction frequency and friendship
                edge = self.graph.get_edge(agent.id, other.id)
                edge.interaction_frequency += 1
                if random.random() < 0.1:
                    edge.friendship_strength = min(1.0, edge.friendship_strength + 0.05)
                
                # Economic Gossip
                if random.random() < self.config.prob_economic_gossip:
                    self._do_gossip(agent, other, edge)
                    events.append(f"[T{timestep}] Gossip between Agent_{agent.id} and Agent_{other.id}")

        # ── Step 2: Global Graph Interactions ──
        # These don't require physical proximity, just a strong graph edge
        for agent in all_agents:
            friends = self.graph.get_friends(agent.id)
            if not friends: continue
            
            # Pick a random friend to potentially interact with
            friend_id = random.choice(friends)
            other = all_agents[friend_id]
            edge = self.graph.get_edge(agent.id, other.id)

            # Mentorship
            res = self._try_mentorship(agent, other, edge, timestep)
            if res: events.append(res.description)

            # Job Referral
            res = self._try_job_referral(agent, other, edge, timestep)
            if res: events.append(res.description)

            # Trust-Based Lending
            res = self._try_trust_lending(agent, other, edge, timestep)
            if res: events.append(res.description)
            
            # Family Transfer
            if edge.is_family:
                res = self._try_family_transfer(agent, other, edge, timestep)
                if res: events.append(res.description)

        # ── Step 3: Influence Engine (Peer Pressure) ──
        self._apply_social_influence(all_agents)

        # ── Step 4: Loan Repayments ──
        repay_events = self._check_repayments(all_agents, timestep)
        events.extend(repay_events)

        # ── Step 5: Decay ──
        for agent in all_agents:
            agent.tick_job_tip()
            
        for key, edge in list(self.graph.edges.items()):
            if not edge.is_family:
                edge.friendship_strength = max(0.0, edge.friendship_strength - 0.001)

        return events

    # ─────────────────────────────────────────
    # INTERACTION IMPLEMENTATIONS
    # ─────────────────────────────────────────

    def _do_gossip(self, a, b, edge):
        self.total_interactions += 1
        self.interactions_this_step += 1
        self.interaction_counts["economic_gossip"] += 1
        a.current_emotion = "social"
        b.current_emotion = "social"

    def _try_mentorship(self, a, b, edge, timestep) -> Optional[InteractionResult]:
        if random.random() > 0.05: return None
        # Agent with much higher skill mentors the other
        if a.skill_level > b.skill_level + 0.3:
            mentor, mentee = a, b
        elif b.skill_level > a.skill_level + 0.3:
            mentor, mentee = b, a
        else:
            return None

        # Transfer skill
        skill_gain = 0.05
        mentee.skill_level = min(1.0, mentee.skill_level + skill_gain)
        
        # Mentor reputation goes up
        mentor.reputation = min(1.0, mentor.reputation + 0.02)
        edge.friendship_strength = min(1.0, edge.friendship_strength + 0.1)
        edge.trust_score = min(1.0, edge.trust_score + 0.1)
        
        # Memory
        mentee.memory.append({"agent_id": mentor.id, "event": "mentorship", "outcome": "positive", "trust_change": 0.1})
        mentor.memory.append({"agent_id": mentee.id, "event": "mentored", "outcome": "positive", "trust_change": 0.1})

        self.interaction_counts["mentorship"] += 1
        return InteractionResult("mentorship", mentor.id, mentee.id, 
                                 f"[T{timestep}] Agent_{mentor.id} mentored Agent_{mentee.id}")

    def _try_job_referral(self, a, b, edge, timestep) -> Optional[InteractionResult]:
        if random.random() > 0.10: return None
        a_emp = getattr(a.type_state, 'job_security', getattr(a.type_state, 'work_found', 0.5))
        b_emp = getattr(b.type_state, 'job_security', getattr(b.type_state, 'work_found', 0.5))

        if a_emp > 0.7 and b_emp < 0.3:
            referrer, receiver = a, b
        elif b_emp > 0.7 and a_emp < 0.3:
            referrer, receiver = b, a
        else:
            return None

        receiver.apply_job_tip(boost=0.3, duration=10)
        edge.friendship_strength = min(1.0, edge.friendship_strength + 0.15)
        edge.trust_score = min(1.0, edge.trust_score + 0.1)
        
        referrer.reputation = min(1.0, referrer.reputation + 0.01)
        
        receiver.memory.append({"agent_id": referrer.id, "event": "job_referral", "outcome": "positive", "trust_change": 0.1})
        
        self.interaction_counts["job_referral"] += 1
        return InteractionResult("job_referral", referrer.id, receiver.id,
                                 f"[T{timestep}] Agent_{referrer.id} referred Agent_{receiver.id} for a job")

    def _try_trust_lending(self, a, b, edge, timestep) -> Optional[InteractionResult]:
        if a.finance.wealth < 2000 and b.finance.wealth > 15000:
            borrower, lender = a, b
        elif b.finance.wealth < 2000 and a.finance.wealth > 15000:
            borrower, lender = b, a
        else:
            return None

        if lender.id in borrower.pending_loans: return None

        # Trust based approval
        # Probability depends on lender's trust in borrower AND borrower's global reputation
        prob = (edge.trust_score * 0.6) + (borrower.reputation * 0.4)
        if random.random() > prob:
            # Rejected!
            return None

        loan_amount = min(lender.finance.wealth * 0.1, 5000.0)
        loan_amount = round(loan_amount, 0)

        lender.finance.wealth -= loan_amount
        borrower.finance.wealth += loan_amount

        borrower.pending_loans[lender.id] = loan_amount
        lender.loans_given[borrower.id] = loan_amount
        
        edge.loan_history.append({"timestep": timestep, "amount": loan_amount, "status": "pending"})

        borrower.memory.append({"agent_id": lender.id, "event": "received_loan", "outcome": "positive", "trust_change": 0.0})
        lender.memory.append({"agent_id": borrower.id, "event": "gave_loan", "outcome": "pending", "trust_change": 0.0})

        self.interaction_counts["borrow_money"] += 1
        self.total_money_lent += loan_amount
        return InteractionResult("borrow_money", borrower.id, lender.id,
                                 f"[T{timestep}] Agent_{lender.id} lent ₹{loan_amount} to Agent_{borrower.id} (Trust: {edge.trust_score:.2f})")

    def _try_family_transfer(self, a, b, edge, timestep) -> Optional[InteractionResult]:
        if random.random() > 0.05: return None
        # Rich family member gives gift to poorer one
        if a.finance.wealth > b.finance.wealth * 3 and a.finance.wealth > 10000:
            giver, receiver = a, b
        elif b.finance.wealth > a.finance.wealth * 3 and b.finance.wealth > 10000:
            giver, receiver = b, a
        else:
            return None

        amount = round(giver.finance.wealth * 0.05, 0)
        giver.finance.wealth -= amount
        receiver.finance.wealth += amount
        
        self.interaction_counts["family_transfer"] += 1
        return InteractionResult("family_transfer", giver.id, receiver.id,
                                 f"[T{timestep}] Agent_{giver.id} gave ₹{amount} family support to Agent_{receiver.id}")

    def _check_repayments(self, all_agents, timestep) -> List[str]:
        events = []
        for agent in all_agents:
            loans_to_repay = []
            for lender_id, amount in agent.pending_loans.items():
                if agent.finance.wealth > amount * 2:
                    loans_to_repay.append((lender_id, amount))

            for lender_id, amount in loans_to_repay:
                lender = all_agents[lender_id]
                agent.finance.wealth -= amount
                lender.finance.wealth += amount
                del agent.pending_loans[lender_id]
                if agent.id in lender.loans_given:
                    del lender.loans_given[agent.id]

                # Boost trust
                edge = self.graph.get_edge(agent.id, lender_id)
                edge.trust_score = min(1.0, edge.trust_score + 0.2)
                agent.reputation = min(1.0, agent.reputation + 0.05)
                
                agent.memory.append({"agent_id": lender_id, "event": "repaid_loan", "outcome": "positive", "trust_change": 0.2})
                lender.memory.append({"agent_id": agent.id, "event": "loan_repaid", "outcome": "positive", "trust_change": 0.2})

                self.interaction_counts["repay_loan"] += 1
                self.total_money_repaid += amount
                events.append(f"[T{timestep}] Agent_{agent.id} repaid ₹{amount} to Agent_{lender_id}")
        return events

    def _apply_social_influence(self, all_agents):
        """Agents adopt spending behaviors of their friends (Herd Behavior)."""
        from config import Action
        for agent in all_agents:
            friends = self.graph.get_friends(agent.id)
            if not friends:
                agent.social_influence_pressure = 0.0
                continue
                
            spending_friends = 0
            for fid in friends:
                if all_agents[fid].current_action == Action.SPEND:
                    spending_friends += 1
                    
            ratio = spending_friends / len(friends)
            agent.social_influence_pressure = ratio * 0.5  # Max 0.5 pressure

    def get_stats(self, all_agents) -> Dict[str, Any]:
        net_summary = self.get_network_summary(all_agents)
        return {
            "total_interactions": self.total_interactions,
            "total_money_lent": round(self.total_money_lent, 0),
            "total_money_repaid": round(self.total_money_repaid, 0),
            "interaction_counts": dict(self.interaction_counts),
            **net_summary
        }

    def get_network_summary(self, all_agents) -> Dict[str, Any]:
        edges = [e for e in self.graph.edges.values() if e.friendship_strength > 0.1 or e.is_family]
        avg_trust = np.mean([e.trust_score for e in edges]) if edges else 0.5
        avg_reputation = np.mean([a.reputation for a in all_agents])
        density = len(edges) / (len(all_agents) * (len(all_agents)-1) / 2) if len(all_agents) > 1 else 0.0
        
        return {
            "avg_trust": round(float(avg_trust), 3),
            "avg_reputation": round(float(avg_reputation), 3),
            "network_density": round(float(density), 4),
            "active_loans": sum(len(a.pending_loans) for a in all_agents),
            "total_edges": len(edges)
        }

    def get_snapshot(self) -> Dict[str, Any]:
        return {"total_interactions": self.total_interactions}

    def restore_snapshot(self, snapshot: Dict[str, Any]):
        self.total_interactions = snapshot.get("total_interactions", 0)

    def reset_step_counter(self):
        self.interactions_this_step = 0
