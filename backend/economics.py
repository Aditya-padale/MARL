"""
economics.py — Global economic simulation layer for MARL City Simulator.

Manages macroeconomic variables that are OBSERVABLE by all agents in their
state vectors. These variables create the economic environment agents operate
in, and policies from policy_engine.py modify them at runtime.

Tracked variables:
  - Inflation rate (multiplies expenses)
  - Base interest rate (grows debt, earns on savings)
  - Housing cost index
  - Employment availability
  - Market return rate (investment returns)
  - Credit availability (loan access)
  - Healthcare cost index
  - Education cost index

Monthly update logic:
  - All debt balances × (1 + monthly_interest_rate)
  - All expenses × (1 + inflation_rate)
  - Investment returns = invested_amount × market_return_rate
  - Savings interest = savings × (base_interest_rate × 0.4 / 12)
  - Loan default check: if debt > 3x monthly_income for 3 months

Author: Aditya Padale (B.Tech Final Year Project)
"""

import random
import math
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from config import (
    EconomicsConfig, ECONOMICS_CONFIG,
    AgentType, AGENT_TYPE_NAMES,
    STEPS_PER_MONTH, STEPS_PER_YEAR,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# ECONOMIC HISTORY ENTRY
# ═══════════════════════════════════════════

@dataclass
class EconomicSnapshot:
    """A timestamped snapshot of all economic variables for trend tracking."""
    timestep: int
    inflation_rate: float
    base_interest_rate: float
    housing_cost_index: float
    employment_availability: float
    market_return_rate: float
    credit_availability: float
    healthcare_cost_index: float
    education_cost_index: float
    total_wealth: float = 0.0
    total_debt: float = 0.0
    avg_income: float = 0.0


# ═══════════════════════════════════════════
# ECONOMICS ENGINE
# ═══════════════════════════════════════════

class EconomicsEngine:
    """
    Manages the global economic environment for the simulation.

    All 100 agents observe these variables in their state vectors.
    Policies modify these variables to create different economic conditions.
    The engine also handles monthly financial processing for all agents:
    interest accrual, investment returns, inflation adjustments, and
    loan default detection.

    This is the bridge between policy_engine.py (which sets parameters)
    and agent.py (which reads them for state vectors and financial updates).
    """

    def __init__(self, config: EconomicsConfig = ECONOMICS_CONFIG):
        """
        Initialize with default economic parameters from config.

        Args:
            config: EconomicsConfig dataclass with default values
        """
        # ── Core Economic Variables ──
        # These are the knobs that policies turn

        self.inflation_rate: float = config.inflation_rate
        self.base_interest_rate: float = config.base_interest_rate
        self.housing_cost_index: float = config.housing_cost_index
        self.employment_availability: float = config.employment_availability
        self.market_return_rate: float = config.market_return_rate
        self.credit_availability: float = config.credit_availability
        self.healthcare_cost_index: float = config.healthcare_cost_index
        self.education_cost_index: float = config.education_cost_index

        # ── Derived / Internal ──
        self.savings_rate_fraction: float = config.savings_rate_fraction
        self.loan_default_months: int = config.loan_default_months
        self.loan_default_multiplier: float = config.loan_default_multiplier

        # ── Natural Economic Drift ──
        # Small random fluctuations to make the economy feel alive
        # even without policy interventions
        self._drift_enabled: bool = True
        self._drift_magnitude: float = 0.002  # ±0.2% per month

        # ── History ──
        self.history: List[EconomicSnapshot] = []
        self.max_history: int = 500  # Keep last 500 snapshots

        # ── Policy Overrides ──
        # Active global effects from policies, with expiry tracking
        self._active_effects: Dict[str, Dict[str, Any]] = {}

        # ── Defaults (for reset / comparison) ──
        self._defaults = {
            "inflation_rate": config.inflation_rate,
            "base_interest_rate": config.base_interest_rate,
            "housing_cost_index": config.housing_cost_index,
            "employment_availability": config.employment_availability,
            "market_return_rate": config.market_return_rate,
            "credit_availability": config.credit_availability,
            "healthcare_cost_index": config.healthcare_cost_index,
            "education_cost_index": config.education_cost_index,
        }

        logger.info("EconomicsEngine initialized with defaults: %s", self._defaults)

    # ─────────────────────────────────────────
    # MONTHLY UPDATE
    # ─────────────────────────────────────────

    def monthly_update(self, agents: list, timestep: int) -> List[str]:
        """
        Process monthly economic updates for all agents and the economy.

        Called every STEPS_PER_MONTH (30) timesteps. This is where the
        economic rubber meets the road:

        1. Natural economic drift (small random fluctuations)
        2. Apply inflation to all agent expenses
        3. Accrue interest on all debts
        4. Pay interest on all savings
        5. Process investment returns
        6. Check for loan defaults
        7. Process each agent's monthly income
        8. Update employment availability based on economic conditions
        9. Record economic snapshot for trend tracking

        Args:
            agents:   List of all Agent objects
            timestep: Current simulation timestep

        Returns:
            List of event log strings for notable economic events
        """
        events: List[str] = []

        # ── Step 1: Natural economic drift ──
        if self._drift_enabled:
            self._apply_natural_drift()

        # ── Step 2: Clamp all variables to sane ranges ──
        self._clamp_variables()

        # ── Step 3: Process each agent's monthly finances ──
        economics_state = self.get_state_dict()
        total_wealth = 0.0
        total_debt = 0.0
        total_income = 0.0
        defaults_this_month = 0

        for agent in agents:
            # 3a. Process monthly income (type-specific logic in agent.py)
            agent.process_monthly_income(economics_state, timestep)

            # 3b. Apply interest to debt
            if agent.finance.debt > 0:
                monthly_rate = self.base_interest_rate
                # Agent-specific debt rate override (from policy)
                agent_rate = agent.finance.debt_interest_rate / 12.0
                effective_rate = max(monthly_rate, agent_rate)
                interest = agent.finance.debt * effective_rate
                agent.finance.debt += interest

            # 3c. Apply interest to savings
            if agent.finance.savings > 0:
                savings_rate = (self.base_interest_rate *
                                self.savings_rate_fraction / 12.0)
                # Agent-specific savings rate override
                agent_savings_rate = agent.finance.savings_interest_rate / 12.0
                effective_savings_rate = max(savings_rate, agent_savings_rate)
                interest_earned = agent.finance.savings * effective_savings_rate
                agent.finance.savings += interest_earned
                agent.finance.wealth += interest_earned

            # 3d. Process investment returns
            if agent.finance.invested_amount > 0:
                # Market returns with some randomness
                base_return = self.market_return_rate
                agent_return = agent.finance.investment_return_rate
                effective_return = (base_return + agent_return) / 2.0
                # Add market noise (±30% of expected return)
                noise = random.uniform(-0.3, 0.3)
                actual_return = effective_return * (1.0 + noise)
                returns = agent.finance.invested_amount * actual_return
                agent.finance.invested_amount += returns
                agent.finance.wealth += returns

                # Small chance of market crash affecting investments
                if random.random() < 0.005:  # 0.5% chance per month
                    crash_loss = agent.finance.invested_amount * random.uniform(0.05, 0.15)
                    agent.finance.invested_amount -= crash_loss
                    agent.finance.wealth -= crash_loss
                    events.append(
                        f"[T{timestep}] Agent_{agent.id:03d} ({agent.type_name}): "
                        f"Market downturn — lost ₹{crash_loss:,.0f} on investments"
                    )

            # 3e. Apply inflation to expenses
            inflation_factor = 1.0 + self.inflation_rate
            agent.finance.monthly_expenses = (
                sum(agent.config.expenses.values()) *
                inflation_factor *
                agent.finance.expense_multiplier
            )

            # Apply housing cost index
            housing_keys = ["rent", "home_emi", "luxury_housing", "housing"]
            for key in housing_keys:
                if key in agent.config.expenses:
                    base = agent.config.expenses[key]
                    adjusted = base * self.housing_cost_index * inflation_factor
                    # This is already captured in monthly_expenses calculation
                    # but we track for housing_cost_norm in state vector

            # Apply healthcare cost index
            healthcare_keys = ["healthcare", "medicine"]
            healthcare_cost = sum(
                agent.config.expenses.get(k, 0.0) for k in healthcare_keys
            )
            healthcare_cost *= self.healthcare_cost_index
            healthcare_cost *= (1.0 - agent.finance.healthcare_subsidy)

            # Apply education cost index
            if "tuition" in agent.config.expenses:
                edu_cost = agent.config.expenses["tuition"]
                edu_cost *= self.education_cost_index
                edu_cost *= (1.0 - agent.finance.education_subsidy)

            # 3f. Loan default check
            if agent.finance.debt > 0 and agent.finance.monthly_income > 0:
                debt_ratio = agent.finance.debt / agent.finance.monthly_income
                if debt_ratio > self.loan_default_multiplier:
                    agent.finance.months_high_debt += 1
                    if agent.finance.months_high_debt >= self.loan_default_months:
                        # LOAN DEFAULT triggered
                        penalty = agent.finance.debt * 0.20  # 20% penalty
                        agent.finance.debt += penalty

                        # Reduce credit access
                        if hasattr(agent.type_state, 'credit_access'):
                            agent.type_state.credit_access = max(
                                agent.type_state.credit_access - 0.2, 0.0
                            )

                        defaults_this_month += 1
                        agent.finance.months_high_debt = 0  # Reset counter

                        event_str = (
                            f"[T{timestep}] Agent_{agent.id:03d} ({agent.type_name}): "
                            f"LOAN DEFAULT — debt increased by ₹{penalty:,.0f} "
                            f"(total debt: ₹{agent.finance.debt:,.0f})"
                        )
                        events.append(event_str)

                        # Record in agent's life events
                        agent.life_events.append({
                            "timestep": timestep,
                            "event": "loan_default",
                            "description": event_str,
                        })

                        logger.info(event_str)
                else:
                    agent.finance.months_high_debt = max(
                        agent.finance.months_high_debt - 1, 0
                    )

            # 3g. Ensure wealth stays consistent with savings
            agent.finance.wealth = max(agent.finance.wealth, 0.0)
            agent.finance.savings = min(agent.finance.savings, agent.finance.wealth)
            agent.finance.debt = max(agent.finance.debt, 0.0)

            # Accumulate totals for snapshot
            total_wealth += agent.finance.wealth
            total_debt += agent.finance.debt
            total_income += agent.finance.monthly_income

        # ── Step 4: Update employment availability ──
        # Employment responds to overall economic conditions
        old_emp = self.employment_availability
        self._update_employment(agents, timestep)

        if abs(self.employment_availability - old_emp) > 0.05:
            direction = "improved" if self.employment_availability > old_emp else "declined"
            events.append(
                f"[T{timestep}] ECONOMY: Employment availability {direction} "
                f"to {self.employment_availability:.1%}"
            )

        # ── Step 5: Record snapshot ──
        avg_income = total_income / max(len(agents), 1)
        snapshot = EconomicSnapshot(
            timestep=timestep,
            inflation_rate=self.inflation_rate,
            base_interest_rate=self.base_interest_rate,
            housing_cost_index=self.housing_cost_index,
            employment_availability=self.employment_availability,
            market_return_rate=self.market_return_rate,
            credit_availability=self.credit_availability,
            healthcare_cost_index=self.healthcare_cost_index,
            education_cost_index=self.education_cost_index,
            total_wealth=total_wealth,
            total_debt=total_debt,
            avg_income=avg_income,
        )
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Log summary
        if defaults_this_month > 0:
            events.append(
                f"[T{timestep}] ECONOMY: {defaults_this_month} loan default(s) this month"
            )

        logger.info(
            "Monthly update T=%d: inflation=%.2f%%, interest=%.2f%%, "
            "employment=%.1f%%, total_wealth=₹%.0f, total_debt=₹%.0f",
            timestep, self.inflation_rate * 100, self.base_interest_rate * 100,
            self.employment_availability * 100, total_wealth, total_debt,
        )

        return events

    # ─────────────────────────────────────────
    # NATURAL ECONOMIC DRIFT
    # ─────────────────────────────────────────

    def _apply_natural_drift(self):
        """
        Apply small random fluctuations to economic variables.

        This makes the economy feel alive even without policy interventions.
        Drift is mean-reverting toward default values to prevent runaway
        inflation/deflation.
        """
        mag = self._drift_magnitude

        # Inflation: tends to slowly increase (realistic for developing economy)
        drift = random.uniform(-mag, mag * 1.2)
        # Mean-revert toward default
        revert = (self._defaults["inflation_rate"] - self.inflation_rate) * 0.05
        self.inflation_rate += drift + revert

        # Interest rate: follows inflation loosely
        drift = random.uniform(-mag * 0.5, mag * 0.5)
        revert = (self._defaults["base_interest_rate"] - self.base_interest_rate) * 0.05
        self.base_interest_rate += drift + revert

        # Housing: slow upward trend (urbanization pressure)
        drift = random.uniform(-mag * 0.3, mag * 0.5)
        revert = (self._defaults["housing_cost_index"] - self.housing_cost_index) * 0.03
        self.housing_cost_index += drift + revert

        # Market returns: volatile
        drift = random.uniform(-mag * 2, mag * 2)
        revert = (self._defaults["market_return_rate"] - self.market_return_rate) * 0.1
        self.market_return_rate += drift + revert

        # Healthcare: slow upward trend
        drift = random.uniform(-mag * 0.2, mag * 0.4)
        revert = (self._defaults["healthcare_cost_index"] - self.healthcare_cost_index) * 0.03
        self.healthcare_cost_index += drift + revert

        # Education: relatively stable
        drift = random.uniform(-mag * 0.1, mag * 0.2)
        revert = (self._defaults["education_cost_index"] - self.education_cost_index) * 0.05
        self.education_cost_index += drift + revert

    def _clamp_variables(self):
        """Clamp all economic variables to reasonable ranges."""
        self.inflation_rate = max(-0.02, min(self.inflation_rate, 0.15))
        self.base_interest_rate = max(0.001, min(self.base_interest_rate, 0.03))
        self.housing_cost_index = max(0.3, min(self.housing_cost_index, 3.0))
        self.employment_availability = max(0.1, min(self.employment_availability, 1.0))
        self.market_return_rate = max(-0.10, min(self.market_return_rate, 0.20))
        self.credit_availability = max(0.1, min(self.credit_availability, 1.0))
        self.healthcare_cost_index = max(0.3, min(self.healthcare_cost_index, 3.0))
        self.education_cost_index = max(0.3, min(self.education_cost_index, 3.0))

    # ─────────────────────────────────────────
    # EMPLOYMENT DYNAMICS
    # ─────────────────────────────────────────

    def _update_employment(self, agents: list, timestep: int):
        """
        Update employment availability based on economic conditions.

        Employment is affected by:
          - Inflation (high inflation → less hiring)
          - Credit availability (more credit → more business → more jobs)
          - Overall wealth trend (economic growth → more jobs)
          - Seasonal effects (some months have more hiring)
        """
        month = (timestep % STEPS_PER_YEAR) // STEPS_PER_MONTH

        # Base: mean-revert toward default
        revert = (self._defaults["employment_availability"] -
                  self.employment_availability) * 0.03

        # Inflation pressure: high inflation → fewer jobs
        inflation_effect = -max(self.inflation_rate - 0.05, 0) * 0.5

        # Credit effect: more credit → more business activity
        credit_effect = (self.credit_availability - 0.5) * 0.02

        # Seasonal hiring (India: hiring peaks in Q1 and Q3)
        seasonal = 0.0
        if month in [0, 1, 2]:      # Jan–Mar: financial year end hiring
            seasonal = 0.01
        elif month in [6, 7, 8]:     # Jul–Sep: festival season, business ramp-up
            seasonal = 0.015
        elif month in [4, 5]:        # May–Jun: typically slower
            seasonal = -0.01

        # Small random component
        noise = random.uniform(-0.005, 0.005)

        self.employment_availability += revert + inflation_effect + credit_effect + seasonal + noise
        self.employment_availability = max(0.1, min(self.employment_availability, 1.0))

    # ─────────────────────────────────────────
    # POLICY EFFECTS
    # ─────────────────────────────────────────

    def apply_global_effect(self, param: str, value: float,
                            policy_name: str = "unknown",
                            duration_steps: int = -1,
                            timestep: int = 0):
        """
        Apply a global economic effect from a parsed policy.

        Called by policy_engine.py when a policy modifies global parameters.

        Args:
            param:          Parameter name (must match a known variable)
            value:          New value or modifier
            policy_name:    Name of the policy (for tracking)
            duration_steps: How long the effect lasts (-1 = permanent)
            timestep:       When the effect was applied
        """
        # Map policy parameter names to internal variables
        param_map = {
            "global_inflation_rate":      "inflation_rate",
            "inflation_rate":             "inflation_rate",
            "market_return_multiplier":   "market_return_rate",
            "market_return_rate":         "market_return_rate",
            "credit_availability":        "credit_availability",
            "employment_availability":    "employment_availability",
            "housing_cost_index":         "housing_cost_index",
            "healthcare_cost_index":      "healthcare_cost_index",
            "education_cost_index":       "education_cost_index",
            "base_interest_rate":         "base_interest_rate",
        }

        internal_param = param_map.get(param, param)

        if hasattr(self, internal_param):
            old_value = getattr(self, internal_param)
            setattr(self, internal_param, value)
            self._clamp_variables()

            # Track the active effect for expiry
            effect_key = f"{policy_name}:{param}"
            self._active_effects[effect_key] = {
                "param": internal_param,
                "old_value": old_value,
                "new_value": value,
                "policy_name": policy_name,
                "applied_at": timestep,
                "duration": duration_steps,
                "expires_at": timestep + duration_steps if duration_steps > 0 else -1,
            }

            logger.info(
                "Global effect applied: %s = %.4f → %.4f (policy: %s, duration: %s)",
                internal_param, old_value, value, policy_name,
                f"{duration_steps} steps" if duration_steps > 0 else "permanent"
            )
        else:
            logger.warning(
                "Unknown global parameter '%s' (mapped: '%s') from policy '%s'",
                param, internal_param, policy_name
            )

    def expire_effects(self, timestep: int) -> List[str]:
        """
        Check and expire time-limited global effects.

        Called each timestep. When an effect expires, the parameter
        reverts to its value before the policy was applied.

        Args:
            timestep: Current simulation timestep

        Returns:
            List of expiry event log strings
        """
        events = []
        expired_keys = []

        for key, effect in self._active_effects.items():
            if effect["expires_at"] > 0 and timestep >= effect["expires_at"]:
                param = effect["param"]
                old_value = effect["old_value"]
                current = getattr(self, param, None)

                if current is not None:
                    setattr(self, param, old_value)
                    self._clamp_variables()

                    event_str = (
                        f"[T{timestep}] POLICY EXPIRED: '{effect['policy_name']}' — "
                        f"{param} reverted from {current:.4f} to {old_value:.4f}"
                    )
                    events.append(event_str)
                    logger.info(event_str)

                expired_keys.append(key)

        for key in expired_keys:
            del self._active_effects[key]

        return events

    def get_active_effects(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently active global policy effects."""
        return dict(self._active_effects)

    # ─────────────────────────────────────────
    # STATE DICT (for agent state vectors)
    # ─────────────────────────────────────────

    def get_state_dict(self) -> Dict[str, float]:
        """
        Get current economic variables as a dict.

        This is what agents read to build their state vectors.
        Every agent sees the same global economic indicators —
        these are the "publicly observable" part of the economy.

        Returns:
            Dict mapping parameter names to float values
        """
        return {
            "inflation_rate": self.inflation_rate,
            "base_interest_rate": self.base_interest_rate,
            "housing_cost_index": self.housing_cost_index,
            "employment_availability": self.employment_availability,
            "market_return_rate": self.market_return_rate,
            "credit_availability": self.credit_availability,
            "healthcare_cost_index": self.healthcare_cost_index,
            "education_cost_index": self.education_cost_index,
        }

    # ─────────────────────────────────────────
    # SNAPSHOT & RESTORE (for comparison mode)
    # ─────────────────────────────────────────

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of economic state for scenario cloning.

        Returns:
            Serializable dict of all economic state
        """
        return {
            "variables": self.get_state_dict(),
            "savings_rate_fraction": self.savings_rate_fraction,
            "loan_default_months": self.loan_default_months,
            "loan_default_multiplier": self.loan_default_multiplier,
            "drift_enabled": self._drift_enabled,
            "active_effects": {
                k: dict(v) for k, v in self._active_effects.items()
            },
            "defaults": dict(self._defaults),
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]):
        """
        Restore economic state from a snapshot.

        Args:
            snapshot: Dict from get_snapshot()
        """
        variables = snapshot["variables"]
        self.inflation_rate = variables["inflation_rate"]
        self.base_interest_rate = variables["base_interest_rate"]
        self.housing_cost_index = variables["housing_cost_index"]
        self.employment_availability = variables["employment_availability"]
        self.market_return_rate = variables["market_return_rate"]
        self.credit_availability = variables["credit_availability"]
        self.healthcare_cost_index = variables["healthcare_cost_index"]
        self.education_cost_index = variables["education_cost_index"]

        self.savings_rate_fraction = snapshot["savings_rate_fraction"]
        self.loan_default_months = snapshot["loan_default_months"]
        self.loan_default_multiplier = snapshot["loan_default_multiplier"]
        self._drift_enabled = snapshot["drift_enabled"]
        self._active_effects = {
            k: dict(v) for k, v in snapshot["active_effects"].items()
        }
        self._defaults = dict(snapshot["defaults"])

    # ─────────────────────────────────────────
    # ANALYSIS & DASHBOARD DATA
    # ─────────────────────────────────────────

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get economic data formatted for the frontend dashboard.

        Returns trend data for Chart.js visualization.
        """
        if not self.history:
            return {"current": self.get_state_dict(), "trends": {}}

        # Extract trend arrays from history
        trends = {
            "timesteps": [s.timestep for s in self.history[-60:]],
            "inflation": [s.inflation_rate for s in self.history[-60:]],
            "interest": [s.base_interest_rate for s in self.history[-60:]],
            "employment": [s.employment_availability for s in self.history[-60:]],
            "market_return": [s.market_return_rate for s in self.history[-60:]],
            "total_wealth": [s.total_wealth for s in self.history[-60:]],
            "total_debt": [s.total_debt for s in self.history[-60:]],
            "avg_income": [s.avg_income for s in self.history[-60:]],
        }

        return {
            "current": self.get_state_dict(),
            "trends": trends,
            "active_effects_count": len(self._active_effects),
        }

    def get_economic_summary(self) -> str:
        """Generate a human-readable economic summary."""
        return (
            f"Inflation: {self.inflation_rate:.1%} | "
            f"Interest: {self.base_interest_rate * 12:.1%}/yr | "
            f"Employment: {self.employment_availability:.0%} | "
            f"Market Return: {self.market_return_rate:.1%}/mo | "
            f"Credit: {self.credit_availability:.0%} | "
            f"Housing: {self.housing_cost_index:.2f}x | "
            f"Healthcare: {self.healthcare_cost_index:.2f}x | "
            f"Education: {self.education_cost_index:.2f}x"
        )

    def reset(self):
        """Reset all economic variables to defaults."""
        for param, value in self._defaults.items():
            setattr(self, param, value)
        self._active_effects.clear()
        self.history.clear()
        logger.info("EconomicsEngine reset to defaults")

    def __repr__(self) -> str:
        return (
            f"EconomicsEngine(inflation={self.inflation_rate:.2%}, "
            f"interest={self.base_interest_rate:.4f}, "
            f"employment={self.employment_availability:.0%}, "
            f"effects={len(self._active_effects)})"
        )
