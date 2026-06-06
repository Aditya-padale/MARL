"""
life_events.py — Stochastic life event system for MARL City Simulator.

Events fire based on configurable per-step probabilities from config.py.
Events are type-specific (crop failure only hits farmers, etc.) and
appear in the city event log with visual indicators.

Event categories:
  - Financial shocks (medical emergency, loan default, job loss)
  - Windfalls (good harvest, business boom, stock windfall, promotion)
  - Support (family remittance, scholarship, pension increase)
  - Career (job found, skill training, startup opportunity)
  - Seasonal (crop failure, good harvest — monsoon-dependent)

All event probabilities live in config.py for easy tuning.
Each event modifies the agent's financial state and/or type-specific
variables, and sets an appropriate emotion icon.

Author: Aditya Padale (B.Tech Final Year Project)
"""

import random
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

from config import (
    AgentType, AGENT_CONFIGS, AGENT_TYPE_NAMES,
    STEPS_PER_MONTH, STEPS_PER_YEAR,
    MONSOON_MONTHS, HARVEST_SEASONS, MONTH_TO_SEASON, Season,
    EMOTION_ICONS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# LIFE EVENT RECORD
# ═══════════════════════════════════════════

@dataclass
class LifeEvent:
    """Record of a single life event that occurred."""
    event_type: str           # Internal event name
    agent_id: int             # Affected agent ID
    agent_type: str           # Agent type name
    description: str          # Human-readable description
    financial_impact: float   # Net ₹ impact (positive = gain)
    emotion: str              # Emotion icon key to display
    timestep: int             # When it occurred
    duration_steps: int = 0   # How long the effect lasts (0 = instant)


# ═══════════════════════════════════════════
# LIFE EVENT SYSTEM
# ═══════════════════════════════════════════

class LifeEventSystem:
    """
    Manages stochastic life events for all 100 agents.

    Each timestep, for each agent, the system rolls dice against
    type-specific event probabilities. When an event fires, it
    modifies the agent's financial state and records the event
    for logging and the Inspector panel.

    Events add realism and create the divergence that makes same-type
    agents behave differently over time — a farmer who experiences
    crop failure twice develops very different PPO behavior from one
    who got two good harvests.
    """

    def __init__(self):
        # Event history for analysis
        self.total_events: int = 0
        self.events_this_step: int = 0
        self.event_counts: Dict[str, int] = {}

        # Active timed effects: agent_id → list of active effects
        self._active_effects: Dict[int, List[Dict[str, Any]]] = {}

    def step(self, all_agents: list, economics, timestep: int) -> List[str]:
        """
        Process life events for all agents this timestep.

        For each agent, checks all type-relevant events against their
        configured probabilities. Multiple events can fire for the
        same agent in one step (rare but possible).

        Args:
            all_agents: List of all 100 Agent objects
            economics:  EconomicsEngine instance (for context)
            timestep:   Current simulation timestep

        Returns:
            List of event log strings for the city event log
        """
        events: List[str] = []
        self.events_this_step = 0

        month = (timestep % STEPS_PER_YEAR) // STEPS_PER_MONTH
        season = MONTH_TO_SEASON.get(month, Season.KHARIF_GROWING)
        is_monsoon = month in MONSOON_MONTHS
        is_harvest = season in HARVEST_SEASONS

        for agent in all_agents:
            # Reset spending influence decay each step
            agent.spending_influence = max(agent.spending_influence - 0.02, 0.0)

            agent_events = self._check_agent_events(
                agent, economics, timestep, month, is_monsoon, is_harvest
            )
            events.extend(agent_events)

        # Process expiring timed effects
        expiry_events = self._process_expiring_effects(all_agents, timestep)
        events.extend(expiry_events)

        return events

    def _check_agent_events(self, agent, economics, timestep: int,
                            month: int, is_monsoon: bool,
                            is_harvest: bool) -> List[str]:
        """
        Check all possible events for a single agent.

        Each event type is checked independently against its probability.
        Type-specific events only fire for the correct agent type.

        Returns:
            List of event description strings
        """
        events = []
        probs = agent.config.life_event_probs

        # ── MEDICAL EMERGENCY — All types (seniors 5x more likely) ──
        med_prob = probs.get("medical_emergency", 0.001)
        if agent.agent_type == AgentType.SENIOR:
            med_prob = probs.get("health_crisis", med_prob)
        if random.random() < med_prob:
            event = self._medical_emergency(agent, timestep)
            if event:
                events.append(event)

        # ── TYPE-SPECIFIC EVENTS ──

        if agent.agent_type == AgentType.FARMER:
            # Crop failure (monsoon months only)
            if is_monsoon and random.random() < probs.get("crop_failure", 0.003):
                event = self._crop_failure(agent, timestep)
                if event:
                    events.append(event)

            # Good harvest (harvest season only)
            if is_harvest and random.random() < probs.get("good_harvest", 0.004):
                event = self._good_harvest(agent, timestep)
                if event:
                    events.append(event)

            # Family remittance
            if random.random() < probs.get("family_remittance", 0.003):
                event = self._family_remittance(agent, timestep)
                if event:
                    events.append(event)

        elif agent.agent_type == AgentType.GIG_WORKER:
            # Work drought
            if random.random() < probs.get("work_drought", 0.003):
                event = self._work_drought(agent, timestep)
                if event:
                    events.append(event)

            # Platform ban
            if random.random() < probs.get("platform_ban", 0.001):
                event = self._platform_ban(agent, timestep)
                if event:
                    events.append(event)

            # Gig boom
            if random.random() < probs.get("gig_boom", 0.002):
                event = self._gig_boom(agent, timestep)
                if event:
                    events.append(event)

        elif agent.agent_type == AgentType.STUDENT:
            # Scholarship
            if random.random() < probs.get("scholarship", 0.002):
                event = self._scholarship(agent, timestep)
                if event:
                    events.append(event)

            # Exam failure
            if random.random() < probs.get("exam_failure", 0.001):
                event = self._exam_failure(agent, timestep)
                if event:
                    events.append(event)

            # Family support cut
            if random.random() < probs.get("family_support_cut", 0.001):
                event = self._family_support_cut(agent, timestep)
                if event:
                    events.append(event)

            # Family remittance
            if random.random() < probs.get("family_remittance", 0.003):
                event = self._family_remittance(agent, timestep)
                if event:
                    events.append(event)

        elif agent.agent_type == AgentType.SMALL_BIZ_OWNER:
            # Business boom
            if random.random() < probs.get("business_boom", 0.002):
                event = self._business_boom(agent, timestep)
                if event:
                    events.append(event)

            # GST audit
            if random.random() < probs.get("gst_audit", 0.001):
                event = self._gst_audit(agent, timestep)
                if event:
                    events.append(event)

            # Shop fire
            if random.random() < probs.get("shop_fire", 0.0005):
                event = self._shop_fire(agent, timestep)
                if event:
                    events.append(event)

        elif agent.agent_type == AgentType.SALARIED_MID:
            # Layoff
            if random.random() < probs.get("layoff", 0.002):
                event = self._layoff(agent, timestep)
                if event:
                    events.append(event)

            # Promotion
            if random.random() < probs.get("promotion", 0.001):
                event = self._promotion(agent, timestep)
                if event:
                    events.append(event)

        elif agent.agent_type == AgentType.SALARIED_HIGH:
            # Stock windfall
            if random.random() < probs.get("stock_windfall", 0.001):
                event = self._stock_windfall(agent, timestep)
                if event:
                    events.append(event)

            # Corporate scandal
            if random.random() < probs.get("corporate_scandal", 0.0005):
                event = self._corporate_scandal(agent, timestep)
                if event:
                    events.append(event)

            # Bonus cut
            if random.random() < probs.get("bonus_cut", 0.001):
                event = self._bonus_cut(agent, timestep)
                if event:
                    events.append(event)

        elif agent.agent_type == AgentType.YOUNG_PROFESSIONAL:
            # Promotion
            if random.random() < probs.get("promotion", 0.001):
                event = self._promotion(agent, timestep)
                if event:
                    events.append(event)

            # Startup opportunity
            if random.random() < probs.get("startup_opportunity", 0.0005):
                event = self._startup_opportunity(agent, timestep)
                if event:
                    events.append(event)

            # Job switch
            if random.random() < probs.get("job_switch", 0.001):
                event = self._job_switch(agent, timestep)
                if event:
                    events.append(event)

        elif agent.agent_type == AgentType.GOVT_EMPLOYEE:
            # DA hike
            if random.random() < probs.get("da_hike", 0.001):
                event = self._da_hike(agent, timestep)
                if event:
                    events.append(event)

            # Transfer
            if random.random() < probs.get("transfer", 0.0005):
                event = self._transfer(agent, timestep)
                if event:
                    events.append(event)

            # Pension reform
            if random.random() < probs.get("pension_reform", 0.0003):
                event = self._pension_reform(agent, timestep)
                if event:
                    events.append(event)

        elif agent.agent_type == AgentType.SENIOR:
            # Pension increase
            if random.random() < probs.get("pension_increase", 0.0005):
                event = self._pension_increase(agent, timestep)
                if event:
                    events.append(event)

            # Family support
            if random.random() < probs.get("family_support", 0.002):
                event = self._family_remittance(agent, timestep)
                if event:
                    events.append(event)

        elif agent.agent_type == AgentType.UNEMPLOYED:
            # Job found
            if random.random() < probs.get("job_found", 0.003):
                event = self._job_found(agent, timestep)
                if event:
                    events.append(event)

            # Welfare cut
            if random.random() < probs.get("welfare_cut", 0.001):
                event = self._welfare_cut(agent, timestep)
                if event:
                    events.append(event)

            # Skill training
            if random.random() < probs.get("skill_training", 0.002):
                event = self._skill_training(agent, timestep)
                if event:
                    events.append(event)

            # Family remittance
            if random.random() < probs.get("family_remittance", 0.003):
                event = self._family_remittance(agent, timestep)
                if event:
                    events.append(event)

        return events

    # ═══════════════════════════════════════════
    # EVENT IMPLEMENTATIONS
    # ═══════════════════════════════════════════

    def _record_event(self, agent, event_type: str, description: str,
                      financial_impact: float, emotion: str,
                      timestep: int, duration: int = 0) -> str:
        """Helper to record event in agent history and update stats."""
        agent.current_emotion = emotion
        agent.life_events.append({
            "timestep": timestep,
            "event": event_type,
            "description": description,
            "financial_impact": financial_impact,
        })

        self.total_events += 1
        self.events_this_step += 1
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

        log_str = (
            f"[T{timestep}] Agent_{agent.id:03d} ({agent.type_name}): "
            f"{description}"
        )
        logger.info(log_str)
        return log_str

    # ── MEDICAL EMERGENCY ──

    def _medical_emergency(self, agent, timestep: int) -> str:
        """One-time expense ₹10,000–₹50,000. All types, seniors 5x more likely."""
        base_cost = random.uniform(10_000, 50_000)
        # Seniors face higher medical costs
        if agent.agent_type == AgentType.SENIOR:
            base_cost *= agent.type_state.health_multiplier

        # Apply healthcare subsidy
        cost = base_cost * (1.0 - agent.finance.healthcare_subsidy)

        agent.finance.wealth -= cost
        if agent.finance.wealth < 0:
            agent.finance.debt += abs(agent.finance.wealth)
            agent.finance.wealth = 0

        desc = f"Medical emergency — expense ₹{cost:,.0f}"
        return self._record_event(agent, "medical_emergency", desc,
                                  -cost, "medical", timestep)

    # ── CROP FAILURE (Farmer) ──

    def _crop_failure(self, agent, timestep: int) -> str:
        """Season income → ₹0, debt += ₹5,000. Farmer only, monsoon months."""
        agent.finance.monthly_income = 0
        debt_increase = random.uniform(3_000, 8_000)
        agent.finance.debt += debt_increase
        agent.type_state.crop_price_idx = max(agent.type_state.crop_price_idx - 0.2, 0.1)

        desc = f"Crop failure — income lost, debt +₹{debt_increase:,.0f}"
        return self._record_event(agent, "crop_failure", desc,
                                  -debt_increase, "sad", timestep)

    # ── GOOD HARVEST (Farmer) ──

    def _good_harvest(self, agent, timestep: int) -> str:
        """Season income × 2.0 this harvest."""
        bonus = agent.finance.base_income * 2.0
        agent.finance.wealth += bonus
        agent.finance.lifetime_earnings += bonus
        agent.type_state.crop_price_idx = min(agent.type_state.crop_price_idx + 0.15, 1.0)

        desc = f"Excellent harvest — bonus income ₹{bonus:,.0f}"
        return self._record_event(agent, "good_harvest", desc,
                                  bonus, "celebration", timestep)

    # ── FAMILY REMITTANCE (Unemployed, Student, Farmer, Senior) ──

    def _family_remittance(self, agent, timestep: int) -> str:
        """One-time ₹3,000–₹8,000 cash transfer from family."""
        amount = random.uniform(3_000, 8_000)
        agent.finance.wealth += amount
        agent.finance.lifetime_earnings += amount

        desc = f"Family remittance received — ₹{amount:,.0f}"
        return self._record_event(agent, "family_remittance", desc,
                                  amount, "remittance", timestep)

    # ── WORK DROUGHT (Gig Worker) ──

    def _work_drought(self, agent, timestep: int) -> str:
        """No work available for 2–4 weeks. Income drops significantly."""
        agent.type_state.work_found = max(agent.type_state.work_found - 0.4, 0.0)
        income_loss = agent.finance.monthly_income * 0.6
        agent.finance.monthly_income *= 0.4

        # Add timed effect to restore work_found
        self._add_timed_effect(agent.id, "work_drought_recovery", timestep,
                               duration=random.randint(14, 28),
                               data={"restore_work_found": 0.3})

        desc = f"Work drought — income dropped by ₹{income_loss:,.0f}"
        return self._record_event(agent, "work_drought", desc,
                                  -income_loss, "sad", timestep)

    # ── PLATFORM BAN (Gig Worker) ──

    def _platform_ban(self, agent, timestep: int) -> str:
        """Temporary ban from gig platform. Severe income hit."""
        agent.type_state.platform_access = 0.0
        agent.finance.monthly_income = 0

        # Restore after 30–60 days
        self._add_timed_effect(agent.id, "platform_restore", timestep,
                               duration=random.randint(30, 60),
                               data={"restore_platform": True})

        desc = "Platform ban — income suspended until reinstated"
        return self._record_event(agent, "platform_ban", desc,
                                  0, "shock", timestep)

    # ── GIG BOOM (Gig Worker) ──

    def _gig_boom(self, agent, timestep: int) -> str:
        """High demand period. Income × 1.5 for 2–4 weeks."""
        agent.type_state.work_found = min(agent.type_state.work_found + 0.3, 1.0)
        bonus = agent.finance.base_income * 0.5
        agent.finance.wealth += bonus

        self._add_timed_effect(agent.id, "gig_boom_end", timestep,
                               duration=random.randint(14, 28),
                               data={"reduce_work_found": 0.2})

        desc = f"Gig boom — high demand, bonus ₹{bonus:,.0f}"
        return self._record_event(agent, "gig_boom", desc,
                                  bonus, "celebration", timestep)

    # ── SCHOLARSHIP (Student) ──

    def _scholarship(self, agent, timestep: int) -> str:
        """Tuition expense = 0 for 12 months. Education progress +0.1."""
        agent.finance.education_subsidy = min(
            agent.finance.education_subsidy + 0.8, 1.0
        )
        agent.type_state.edu_progress = min(
            agent.type_state.edu_progress + 0.1, 1.0
        )

        # Expires after 12 months
        self._add_timed_effect(agent.id, "scholarship_end", timestep,
                               duration=STEPS_PER_YEAR,
                               data={"reduce_edu_subsidy": 0.8})

        desc = "Scholarship awarded — tuition waived for 12 months"
        return self._record_event(agent, "scholarship", desc,
                                  0, "celebration", timestep)

    # ── EXAM FAILURE (Student) ──

    def _exam_failure(self, agent, timestep: int) -> str:
        """Education progress stalls. Family support may decrease."""
        agent.type_state.edu_progress = max(
            agent.type_state.edu_progress - 0.05, 0.0
        )
        agent.type_state.family_support = max(
            agent.type_state.family_support - 0.1, 0.0
        )

        desc = "Exam failure — education progress stalled"
        return self._record_event(agent, "exam_failure", desc,
                                  0, "sad", timestep)

    # ── FAMILY SUPPORT CUT (Student) ──

    def _family_support_cut(self, agent, timestep: int) -> str:
        """Family reduces financial support. Income drops."""
        cut_fraction = random.uniform(0.2, 0.5)
        income_cut = agent.finance.base_income * cut_fraction
        agent.finance.base_income *= (1.0 - cut_fraction)
        agent.type_state.family_support = max(
            agent.type_state.family_support - 0.3, 0.0
        )

        desc = f"Family support cut — income reduced by ₹{income_cut:,.0f}/month"
        return self._record_event(agent, "family_support_cut", desc,
                                  -income_cut, "sad", timestep)

    # ── BUSINESS BOOM (Small Biz Owner) ──

    def _business_boom(self, agent, timestep: int) -> str:
        """Income multiplier × 1.5 for 3 months."""
        agent.finance.income_multiplier *= 1.5
        agent.type_state.biz_health = min(agent.type_state.biz_health + 0.2, 1.0)
        agent.type_state.customers = min(agent.type_state.customers + 0.2, 1.0)

        self._add_timed_effect(agent.id, "business_boom_end", timestep,
                               duration=STEPS_PER_MONTH * 3,
                               data={"restore_income_mult": 1.5})

        desc = "Business boom — revenue ×1.5 for 3 months"
        return self._record_event(agent, "business_boom", desc,
                                  0, "celebration", timestep)

    # ── GST AUDIT (Small Biz Owner) ──

    def _gst_audit(self, agent, timestep: int) -> str:
        """Surprise audit. One-time penalty ₹5,000–₹20,000."""
        penalty = random.uniform(5_000, 20_000)
        agent.finance.wealth -= penalty
        if agent.finance.wealth < 0:
            agent.finance.debt += abs(agent.finance.wealth)
            agent.finance.wealth = 0

        desc = f"GST audit — penalty ₹{penalty:,.0f}"
        return self._record_event(agent, "gst_audit", desc,
                                  -penalty, "shock", timestep)

    # ── SHOP FIRE (Small Biz Owner) ──

    def _shop_fire(self, agent, timestep: int) -> str:
        """Catastrophic loss. Major financial hit."""
        loss = random.uniform(30_000, 80_000)
        agent.finance.wealth -= loss
        if agent.finance.wealth < 0:
            agent.finance.debt += abs(agent.finance.wealth)
            agent.finance.wealth = 0
        agent.type_state.biz_health = max(agent.type_state.biz_health - 0.4, 0.0)
        agent.type_state.customers = max(agent.type_state.customers - 0.3, 0.0)

        desc = f"Shop fire — loss ₹{loss:,.0f}, business severely damaged"
        return self._record_event(agent, "shop_fire", desc,
                                  -loss, "shock", timestep)

    # ── LAYOFF (Salaried Mid, Young Prof) ──

    def _layoff(self, agent, timestep: int) -> str:
        """Income = 0 for 2–6 months. Job security drops."""
        agent.finance.monthly_income = 0
        agent.type_state.job_security = max(agent.type_state.job_security - 0.5, 0.0)

        duration = random.randint(STEPS_PER_MONTH * 2, STEPS_PER_MONTH * 6)
        self._add_timed_effect(agent.id, "layoff_recovery", timestep,
                               duration=duration,
                               data={"restore_income": True,
                                     "restore_job_security": 0.3})

        desc = f"Laid off — income suspended for ~{duration // STEPS_PER_MONTH} months"
        return self._record_event(agent, "layoff", desc,
                                  0, "shock", timestep)

    # ── PROMOTION (Salaried Mid, Young Prof, Govt Employee) ──

    def _promotion(self, agent, timestep: int) -> str:
        """Income multiplier += 0.20 permanently."""
        raise_amount = agent.finance.base_income * 0.20
        agent.finance.base_income *= 1.20
        agent.type_state.promotion_progress = 0.0  # Reset progress
        agent.type_state.job_security = min(agent.type_state.job_security + 0.1, 1.0)

        desc = f"Promoted — income increased by ₹{raise_amount:,.0f}/month permanently"
        return self._record_event(agent, "promotion", desc,
                                  raise_amount, "celebration", timestep)

    # ── STOCK WINDFALL (Salaried High) ──

    def _stock_windfall(self, agent, timestep: int) -> str:
        """Large investment return. One-time gain."""
        gain = random.uniform(50_000, 200_000)
        agent.finance.wealth += gain
        agent.finance.invested_amount += gain * 0.5  # Reinvest half
        agent.finance.lifetime_earnings += gain

        desc = f"Stock windfall — gained ₹{gain:,.0f}"
        return self._record_event(agent, "stock_windfall", desc,
                                  gain, "celebration", timestep)

    # ── CORPORATE SCANDAL (Salaried High) ──

    def _corporate_scandal(self, agent, timestep: int) -> str:
        """Stock/investments lose value. Reputation hit."""
        loss_fraction = random.uniform(0.1, 0.3)
        inv_loss = agent.finance.invested_amount * loss_fraction
        agent.finance.invested_amount -= inv_loss
        agent.finance.wealth -= inv_loss
        agent.type_state.job_security = max(agent.type_state.job_security - 0.2, 0.0)

        desc = f"Corporate scandal — investment loss ₹{inv_loss:,.0f}"
        return self._record_event(agent, "corporate_scandal", desc,
                                  -inv_loss, "shock", timestep)

    # ── BONUS CUT (Salaried High) ──

    def _bonus_cut(self, agent, timestep: int) -> str:
        """Annual bonus reduced or eliminated."""
        lost_bonus = agent.finance.base_income * random.uniform(0.3, 1.0)

        desc = f"Bonus cut — lost potential ₹{lost_bonus:,.0f}"
        return self._record_event(agent, "bonus_cut", desc,
                                  -lost_bonus, "sad", timestep)

    # ── STARTUP OPPORTUNITY (Young Professional) ──

    def _startup_opportunity(self, agent, timestep: int) -> str:
        """Chance to invest in a startup. Risky but potential high reward."""
        investment = min(agent.finance.wealth * 0.15, 30_000)
        if investment < 5_000:
            return None  # Can't afford to invest

        agent.finance.wealth -= investment

        # 40% chance of 3x return, 60% chance of total loss
        if random.random() < 0.4:
            returns = investment * 3
            agent.finance.wealth += returns
            agent.finance.lifetime_earnings += returns - investment
            desc = f"Startup investment paid off — ₹{investment:,.0f} → ₹{returns:,.0f}"
            return self._record_event(agent, "startup_success", desc,
                                      returns - investment, "celebration", timestep)
        else:
            desc = f"Startup investment failed — lost ₹{investment:,.0f}"
            return self._record_event(agent, "startup_failure", desc,
                                      -investment, "sad", timestep)

    # ── JOB SWITCH (Young Professional) ──

    def _job_switch(self, agent, timestep: int) -> str:
        """Switch to new job. Usually comes with 10–30% raise."""
        raise_pct = random.uniform(0.10, 0.30)
        old_income = agent.finance.base_income
        agent.finance.base_income *= (1.0 + raise_pct)
        raise_amount = agent.finance.base_income - old_income
        agent.type_state.promotion_progress = 0.0  # Reset at new company

        desc = f"Job switch — salary increase ₹{raise_amount:,.0f}/month (+{raise_pct:.0%})"
        return self._record_event(agent, "job_switch", desc,
                                  raise_amount, "celebration", timestep)

    # ── DA HIKE (Govt Employee) ──

    def _da_hike(self, agent, timestep: int) -> str:
        """Dearness Allowance increase. Income boost."""
        da_increase = agent.finance.base_income * random.uniform(0.03, 0.08)
        agent.finance.base_income += da_increase

        desc = f"DA hike — income increased by ₹{da_increase:,.0f}/month"
        return self._record_event(agent, "da_hike", desc,
                                  da_increase, "happy", timestep)

    # ── TRANSFER (Govt Employee) ──

    def _transfer(self, agent, timestep: int) -> str:
        """Government transfer. One-time moving expense."""
        expense = random.uniform(5_000, 15_000)
        agent.finance.wealth -= expense
        if agent.finance.wealth < 0:
            agent.finance.debt += abs(agent.finance.wealth)
            agent.finance.wealth = 0

        desc = f"Government transfer — relocation expense ₹{expense:,.0f}"
        return self._record_event(agent, "transfer", desc,
                                  -expense, "sad", timestep)

    # ── PENSION REFORM (Govt Employee) ──

    def _pension_reform(self, agent, timestep: int) -> str:
        """Pension policy change. Could be positive or negative."""
        if random.random() < 0.6:
            # Positive reform — slightly higher future pension
            agent.type_state.promotion_progress += 0.05
            desc = "Pension reform — positive changes for retirement benefits"
            return self._record_event(agent, "pension_reform", desc,
                                      0, "happy", timestep)
        else:
            # Negative reform — reduced benefits
            desc = "Pension reform — reduced retirement benefits announced"
            return self._record_event(agent, "pension_reform_negative", desc,
                                      0, "sad", timestep)

    # ── PENSION INCREASE (Senior) ──

    def _pension_increase(self, agent, timestep: int) -> str:
        """Income += ₹2,000 permanently. Rare event."""
        increase = random.uniform(1_500, 3_000)
        agent.finance.base_income += increase
        agent.type_state.pension_stable = min(agent.type_state.pension_stable + 0.1, 1.0)

        desc = f"Pension increase — income +₹{increase:,.0f}/month permanently"
        return self._record_event(agent, "pension_increase", desc,
                                  increase, "happy", timestep)

    # ── JOB FOUND (Unemployed) ──

    def _job_found(self, agent, timestep: int) -> str:
        """Unemployed agent finds work. Income starts flowing."""
        base_salary = random.uniform(10_000, 25_000)
        agent.finance.base_income = base_salary
        agent.finance.monthly_income = base_salary
        agent.type_state.job_search_score = 0.0  # Reset
        agent.type_state.welfare_active = 0.0     # No longer needs welfare

        desc = f"Job found — starting salary ₹{base_salary:,.0f}/month"
        return self._record_event(agent, "job_found", desc,
                                  base_salary, "celebration", timestep)

    # ── WELFARE CUT (Unemployed) ──

    def _welfare_cut(self, agent, timestep: int) -> str:
        """Welfare benefits reduced or cut entirely."""
        agent.type_state.welfare_active = 0.0
        agent.finance.welfare_payment = 0.0

        desc = "Welfare cut — benefits suspended"
        return self._record_event(agent, "welfare_cut", desc,
                                  0, "shock", timestep)

    # ── SKILL TRAINING (Unemployed) ──

    def _skill_training(self, agent, timestep: int) -> str:
        """Skill training program. Improves job search score."""
        agent.type_state.skills_level = min(agent.type_state.skills_level + 0.15, 1.0)
        agent.type_state.job_search_score = min(
            agent.type_state.job_search_score + 0.1, 1.0
        )

        desc = "Skill training completed — job prospects improved"
        return self._record_event(agent, "skill_training", desc,
                                  0, "happy", timestep)

    # ═══════════════════════════════════════════
    # TIMED EFFECTS
    # ═══════════════════════════════════════════

    def _add_timed_effect(self, agent_id: int, effect_type: str,
                          timestep: int, duration: int,
                          data: Dict[str, Any]):
        """Add a timed effect that will be processed when it expires."""
        if agent_id not in self._active_effects:
            self._active_effects[agent_id] = []

        self._active_effects[agent_id].append({
            "type": effect_type,
            "started_at": timestep,
            "expires_at": timestep + duration,
            "data": data,
        })

    def _process_expiring_effects(self, all_agents: list,
                                  timestep: int) -> List[str]:
        """Process and remove expired timed effects."""
        events = []
        agents_by_id = {a.id: a for a in all_agents}

        for agent_id in list(self._active_effects.keys()):
            effects = self._active_effects[agent_id]
            remaining = []

            for effect in effects:
                if timestep >= effect["expires_at"]:
                    agent = agents_by_id.get(agent_id)
                    if agent:
                        event = self._resolve_expired_effect(agent, effect, timestep)
                        if event:
                            events.append(event)
                else:
                    remaining.append(effect)

            if remaining:
                self._active_effects[agent_id] = remaining
            else:
                del self._active_effects[agent_id]

        return events

    def _resolve_expired_effect(self, agent, effect: Dict, timestep: int) -> Optional[str]:
        """Resolve a single expired timed effect."""
        data = effect["data"]
        effect_type = effect["type"]

        if effect_type == "work_drought_recovery":
            agent.type_state.work_found = min(
                agent.type_state.work_found + data.get("restore_work_found", 0.3), 1.0
            )
            return f"[T{timestep}] Agent_{agent.id:03d}: Work drought ended — opportunities returning"

        elif effect_type == "platform_restore":
            agent.type_state.platform_access = 1.0
            return f"[T{timestep}] Agent_{agent.id:03d}: Platform access restored"

        elif effect_type == "gig_boom_end":
            agent.type_state.work_found = max(
                agent.type_state.work_found - data.get("reduce_work_found", 0.2), 0.0
            )
            return f"[T{timestep}] Agent_{agent.id:03d}: Gig boom period ended"

        elif effect_type == "scholarship_end":
            agent.finance.education_subsidy = max(
                agent.finance.education_subsidy - data.get("reduce_edu_subsidy", 0.8), 0.0
            )
            return f"[T{timestep}] Agent_{agent.id:03d}: Scholarship period ended"

        elif effect_type == "business_boom_end":
            mult = data.get("restore_income_mult", 1.5)
            agent.finance.income_multiplier /= mult
            agent.type_state.biz_health = max(agent.type_state.biz_health - 0.1, 0.0)
            return f"[T{timestep}] Agent_{agent.id:03d}: Business boom period ended"

        elif effect_type == "layoff_recovery":
            agent.finance.monthly_income = agent.finance.base_income
            agent.type_state.job_security = min(
                agent.type_state.job_security + data.get("restore_job_security", 0.3), 0.8
            )
            return f"[T{timestep}] Agent_{agent.id:03d}: Found new employment after layoff"

        return None

    # ─────────────────────────────────────────
    # STATISTICS & DASHBOARD
    # ─────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics."""
        return {
            "total_events": self.total_events,
            "events_this_step": self.events_this_step,
            "event_counts": dict(self.event_counts),
            "active_timed_effects": sum(
                len(effects) for effects in self._active_effects.values()
            ),
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """Get serializable snapshot for comparison mode."""
        return {
            "total_events": self.total_events,
            "event_counts": dict(self.event_counts),
            "active_effects": {
                str(k): [dict(e) for e in v]
                for k, v in self._active_effects.items()
            },
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]):
        """Restore from snapshot."""
        self.total_events = snapshot["total_events"]
        self.event_counts = dict(snapshot["event_counts"])
        self._active_effects = {
            int(k): [dict(e) for e in v]
            for k, v in snapshot["active_effects"].items()
        }

    def __repr__(self) -> str:
        return (
            f"LifeEventSystem(total={self.total_events}, "
            f"this_step={self.events_this_step}, "
            f"active_effects={sum(len(v) for v in self._active_effects.values())})"
        )
