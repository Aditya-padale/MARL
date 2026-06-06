"""
agent.py — Individual Agent class for MARL City Simulator.

Each of the 100 agents is a fully independent economic actor with:
  - Its own financial state (wealth, savings, debt, income, expenses)
  - Its own PPO decision-making (via PPOAgent with independent optimizer)
  - Its own position on the city canvas with smooth movement
  - Its own social relationships (proximity-based, top-10 tracked)
  - Its own life event history and policy impact log

NO agent has telepathy. They observe only their own financial state
plus publicly observable economic indicators (inflation, interest rates).

Author: Aditya Padale (B.Tech Final Year Project)
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from config import (
    AgentType, AgentTypeConfig, AGENT_CONFIGS, AGENT_TYPE_NAMES,
    Action, ACTION_NAMES, ACTION_EMOJIS, NUM_ACTIONS,
    WealthTier, WEALTH_TIERS, BANKRUPTCY_THRESHOLD, POVERTY_THRESHOLD,
    SURVIVAL_THRESHOLD,
    EMOTION_ICONS,
    LERP_FACTOR, BANKRUPT_SPEED_MULT,
    Season, MONTH_TO_SEASON, HARVEST_SEASONS, MONSOON_MONTHS,
    NORM_WEALTH_MAX, NORM_INCOME_MAX, NORM_SAVINGS_MAX, NORM_DEBT_MAX,
    NUM_POLICY_SLOTS, SHARED_STATE_DIM,
    REWARD_CONFIG, ACTION_CONFIG,
    STEPS_PER_MONTH, STEPS_PER_YEAR,
    CANVAS_WIDTH, CANVAS_HEIGHT,
)
from ppo import PPOAgent, Actor, Critic


# ═══════════════════════════════════════════
# FINANCIAL STATE
# ═══════════════════════════════════════════

@dataclass
class FinancialState:
    """
    Complete financial snapshot of one agent.
    All values in INR (₹). Updated every simulation step.
    """
    wealth: float = 0.0              # Total liquid assets (cash + savings)
    savings: float = 0.0             # Long-term savings (subset of wealth)
    debt: float = 0.0                # Outstanding loans / obligations
    monthly_income: float = 0.0      # Current month's income
    base_income: float = 0.0         # Base income before modifiers
    monthly_expenses: float = 0.0    # Current month's total expenses

    # Modifiers applied by policies or life events
    income_multiplier: float = 1.0
    expense_multiplier: float = 1.0
    tax_rate: float = 0.0
    investment_return_rate: float = 0.05
    savings_interest_rate: float = 0.02
    debt_interest_rate: float = 0.08
    welfare_payment: float = 0.0
    healthcare_subsidy: float = 0.0
    education_subsidy: float = 0.0
    cash_freeze_fraction: float = 0.0

    # Investment tracking
    invested_amount: float = 0.0     # Currently invested capital

    # Action permission flags (policies can restrict these)
    trade_allowed: bool = True
    invest_allowed: bool = True

    # Tracking
    lifetime_earnings: float = 0.0
    lifetime_spending: float = 0.0
    bankruptcies: int = 0
    is_bankrupt: bool = False
    months_high_debt: int = 0        # Consecutive months with debt > 3x income
    missed_obligations: int = 0      # Count of unpaid bills this step

    # Last step deltas (for reward computation)
    wealth_delta: float = 0.0
    consumption_this_step: float = 0.0


# ═══════════════════════════════════════════
# TYPE-SPECIFIC STATE
# ═══════════════════════════════════════════

@dataclass
class TypeSpecificState:
    """
    Type-specific state variables appended to the shared state vector.
    Only the relevant fields are used based on agent type.
    """
    # Farmer
    season: float = 0.0               # Current season (0–5 normalized)
    crop_price_idx: float = 0.5       # Crop price index (0–1)
    loan_due: float = 0.0             # 1.0 if loan payment due this month
    harvest_soon: float = 0.0         # 1.0 if harvest within 2 months

    # Gig Worker
    work_found: float = 0.5           # 1.0 if currently has work
    income_variance_3mo: float = 0.3  # Rolling 3-month income variance
    platform_access: float = 1.0      # 1.0 if platform access active

    # Student
    edu_progress: float = 0.0         # Education progress (0–1)
    family_support: float = 0.5       # Family support level (0–1)
    tuition_due: float = 0.0          # 1.0 if tuition payment due

    # Small Business Owner
    biz_health: float = 0.5           # Business health score (0–1)
    gst_rate: float = 0.18            # Current GST rate
    credit_access: float = 0.7        # Credit access score (0–1)
    customers: float = 0.5            # Customer flow (0–1)

    # Salaried / Young Professional / Govt Employee
    emi_burden: float = 0.3           # EMI as fraction of income
    job_security: float = 0.7         # Job security score (0–1)
    promotion_progress: float = 0.0   # Progress toward promotion (0–1)

    # Senior
    health_multiplier: float = 1.0    # Healthcare cost multiplier (increases)
    pension_stable: float = 0.8       # Pension stability score (0–1)
    medical_due: float = 0.0          # 1.0 if medical appointment due

    # Unemployed
    job_search_score: float = 0.3     # Job search effectiveness (0–1)
    welfare_active: float = 0.0       # 1.0 if receiving welfare
    skills_level: float = 0.3         # Skill level (0–1)


# ═══════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════

class Agent:
    """
    A single autonomous agent in the MARL city simulation.

    Each agent independently:
      1. Observes its own financial state + global economic indicators
      2. Builds a state vector for its PPO network
      3. Selects an action (Save/Spend/Invest/Trade)
      4. Executes the action, affecting its financial state
      5. Receives a reward based on financial outcomes
      6. Moves toward the appropriate city zone for its action
      7. May interact with nearby agents (social system)
      8. May experience stochastic life events

    There are 100 agents total, each fully independent.
    """

    def __init__(self, agent_id: int, agent_type: AgentType,
                 config: AgentTypeConfig, ppo_agent: PPOAgent,
                 initial_position: Tuple[float, float]):
        """
        Args:
            agent_id:          Unique ID (0–99)
            agent_type:        AgentType enum
            config:            AgentTypeConfig from config.py
            ppo_agent:         PPOAgent instance (owns buffer + optimizer)
            initial_position:  Starting (x, y) on city canvas
        """
        # ── Identity ──
        self.id: int = agent_id
        self.agent_type: AgentType = agent_type
        self.type_name: str = AGENT_TYPE_NAMES[agent_type]
        self.config: AgentTypeConfig = config

        # ── PPO Decision-Making ──
        self.ppo: PPOAgent = ppo_agent

        # ── Position & Movement ──
        self.x: float = initial_position[0]
        self.y: float = initial_position[1]
        self.target_x: float = self.x
        self.target_y: float = self.y
        self.base_speed: float = config.move_speed
        self.current_speed: float = config.move_speed
        self.is_moving: bool = False
        self.walk_frame: int = 0       # Animation frame (0 or 1)
        self.walk_timer: float = 0.0   # Milliseconds since last frame toggle

        # ── Financial State ──
        self.finance = FinancialState()
        self._initialize_finances()

        # ── Type-Specific State ──
        self.type_state = TypeSpecificState()
        self._initialize_type_state()

        # ── Social & Network Attributes ──
        # relationships will be managed partially by the SocialGraph globally, 
        # but we keep local copies of trust and relationships for quick access
        self.relationships: Dict[int, float] = {}
        self.trust_scores: Dict[int, float] = {}     # Maps agent_id -> trust score (-1.0 to 1.0)
        self.memory: List[Dict[str, Any]] = []       # List of past interactions
        self.reputation: float = 0.5                 # Global reputation (0.0 to 1.0)
        self.skill_level: float = random.uniform(0.1, 0.9)  # Agent's expertise level
        self.family_links: Dict[int, str] = {}       # Maps agent_id -> relationship type (e.g. 'parent', 'spouse')

        # Social interaction cooldowns and pending loans
        self.pending_loans: Dict[int, float] = {}    # lender_id → amount owed
        self.loans_given: Dict[int, float] = {}      # borrower_id → amount lent
        self.job_tip_boost: float = 0.0              # Temporary employment boost
        self.job_tip_timer: int = 0                  # Steps remaining for boost
        self.spending_influence: float = 0.0         # Temp spend prob boost
        self.social_influence_pressure: float = 0.0  # Pressure from friends' actions

        # ── Current State ──
        self.current_action: Optional[int] = None
        self.current_emotion: str = "happy"
        self.last_reward: float = 0.0
        self.current_wealth_tier: int = 1            # 0=poor, 1=mid, 2=rich

        # ── History (for Inspector Panel) ──
        self.life_events: List[Dict[str, Any]] = []
        self.policy_impacts: List[Dict[str, Any]] = []
        self.reward_history: List[float] = []        # Last 100 rewards
        self.wealth_history: List[float] = []        # Last 100 wealth values
        self.income_history: List[float] = []        # Last 12 monthly incomes

        # ── Active Policy Effects ──
        self.active_policies: Dict[str, Dict] = {}   # policy_name → effect_dict
        self.policy_slots: List[float] = [0.0] * NUM_POLICY_SLOTS

        # ── Timing ──
        self.steps_alive: int = 0

    # ─────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────

    def _initialize_finances(self):
        """Set starting financial state from config ranges with randomization."""
        cfg = self.config

        # Randomize starting wealth within configured range
        self.finance.wealth = random.uniform(*cfg.wealth_range)
        self.finance.savings = self.finance.wealth * random.uniform(0.1, 0.4)

        # Base monthly income (randomized within range)
        self.finance.base_income = random.uniform(*cfg.income_range)
        self.finance.monthly_income = self.finance.base_income

        # Starting debt (type-dependent)
        if self.agent_type == AgentType.FARMER:
            self.finance.debt = random.uniform(5_000, 20_000)
        elif self.agent_type == AgentType.YOUNG_PROFESSIONAL:
            self.finance.debt = random.uniform(50_000, 200_000)  # Education loan
        elif self.agent_type == AgentType.SALARIED_MID:
            self.finance.debt = random.uniform(100_000, 500_000)  # Home loan
        elif self.agent_type == AgentType.SMALL_BIZ_OWNER:
            self.finance.debt = random.uniform(20_000, 100_000)  # Business loan
        else:
            self.finance.debt = random.uniform(0, 10_000)

        # Compute total monthly expenses from config
        self.finance.monthly_expenses = sum(cfg.expenses.values())

        # Tax rate from config
        self.finance.tax_rate = cfg.base_tax_rate

    def _initialize_type_state(self):
        """Set type-specific state variables with randomized starting values."""
        ts = self.type_state

        if self.agent_type == AgentType.FARMER:
            ts.season = 0.0
            ts.crop_price_idx = random.uniform(0.3, 0.7)
            ts.loan_due = 1.0 if self.finance.debt > 0 else 0.0
            ts.harvest_soon = 0.0

        elif self.agent_type == AgentType.GIG_WORKER:
            ts.work_found = random.uniform(0.3, 0.8)
            ts.income_variance_3mo = random.uniform(0.2, 0.5)
            ts.platform_access = 1.0

        elif self.agent_type == AgentType.STUDENT:
            ts.edu_progress = random.uniform(0.0, 0.3)
            ts.family_support = random.uniform(0.3, 0.8)
            ts.tuition_due = 0.0

        elif self.agent_type == AgentType.SMALL_BIZ_OWNER:
            ts.biz_health = random.uniform(0.4, 0.7)
            ts.gst_rate = 0.18
            ts.credit_access = random.uniform(0.5, 0.9)
            ts.customers = random.uniform(0.3, 0.7)

        elif self.agent_type in (AgentType.SALARIED_MID, AgentType.SALARIED_HIGH,
                                  AgentType.YOUNG_PROFESSIONAL, AgentType.GOVT_EMPLOYEE):
            ts.emi_burden = (sum(v for k, v in self.config.expenses.items()
                                 if 'emi' in k.lower() or 'loan' in k.lower())
                             / max(self.finance.monthly_income, 1.0))
            ts.job_security = random.uniform(0.5, 0.9)
            ts.promotion_progress = random.uniform(0.0, 0.2)

        elif self.agent_type == AgentType.SENIOR:
            ts.health_multiplier = random.uniform(1.0, 1.5)
            ts.pension_stable = random.uniform(0.6, 0.9)
            ts.medical_due = 0.0

        elif self.agent_type == AgentType.UNEMPLOYED:
            ts.job_search_score = random.uniform(0.1, 0.4)
            ts.welfare_active = 0.0
            ts.skills_level = random.uniform(0.1, 0.4)

    # ─────────────────────────────────────────
    # STATE VECTOR CONSTRUCTION
    # ─────────────────────────────────────────

    def build_state_vector(self, economics_state: Dict[str, float],
                           timestep: int) -> np.ndarray:
        """
        Build the full state vector for PPO input.

        Combines:
          - 18 shared features (own finances + global economic indicators)
          - 3–4 type-specific features

        All values normalized to roughly [0, 1] range for neural network input.
        Agents observe ONLY their own state — no telepathy.

        Args:
            economics_state: Dict of global economic variables from EconomicsEngine
            timestep:        Current simulation timestep

        Returns:
            numpy array of shape (state_dim,)
        """
        f = self.finance
        month_in_year = (timestep % STEPS_PER_YEAR) // STEPS_PER_MONTH

        # ── Shared features (21 values) ──
        shared = [
            # Own financial state (normalized)
            min(f.wealth / NORM_WEALTH_MAX, 1.0),                          # 0
            min(f.monthly_income / NORM_INCOME_MAX, 1.0),                  # 1
            min(f.savings / NORM_SAVINGS_MAX, 1.0),                        # 2
            min(f.debt / NORM_DEBT_MAX, 1.0),                              # 3
            min(f.monthly_expenses / max(f.monthly_income, 1.0), 2.0),     # 4: expense_ratio
            # Global economic indicators (publicly observable)
            economics_state.get("inflation_rate", 0.03),                   # 5
            economics_state.get("base_interest_rate", 0.0067),             # 6
            min(economics_state.get("housing_cost_index", 1.0) *
                self._get_housing_expense() /
                max(f.monthly_income, 1.0), 2.0),                          # 7: housing_cost_norm
            economics_state.get("employment_availability", 0.75),          # 8
            # Wealth tier as float (0=bankrupt/poor, 0.5=mid, 1.0=rich)
            self.current_wealth_tier / 4.0,                                # 9
            # Time features
            month_in_year / 11.0,                                          # 10
            # Day in month (for finer temporal resolution)
            (timestep % STEPS_PER_MONTH) / float(STEPS_PER_MONTH),         # 11
            # Social inputs
            self.reputation,                                               # 12
            self.skill_level,                                              # 13
            self.social_influence_pressure,                                # 14
        ]

        # Active policy effect slots (one-hot, 6 slots)
        shared.extend(self.policy_slots)                                    # 15–20

        # ── Type-specific features ──
        type_features = self._get_type_specific_features(month_in_year)

        # Concatenate
        state = np.array(shared + type_features, dtype=np.float32)

        # Safety: clamp any NaN or Inf values
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)

        return state

    def _get_housing_expense(self) -> float:
        """Extract housing-related expense from the config."""
        housing_keys = ["rent", "home_emi", "luxury_housing", "housing"]
        return sum(self.config.expenses.get(k, 0.0) for k in housing_keys)

    def _get_type_specific_features(self, month_in_year: int) -> List[float]:
        """Build the type-specific portion of the state vector."""
        ts = self.type_state

        if self.agent_type == AgentType.FARMER:
            season_val = MONTH_TO_SEASON.get(month_in_year, Season.KHARIF_GROWING)
            ts.season = season_val / 5.0
            ts.harvest_soon = 1.0 if season_val in HARVEST_SEASONS else 0.0
            ts.loan_due = 1.0 if self.finance.debt > 5_000 else 0.0
            return [ts.season, ts.crop_price_idx, ts.loan_due, ts.harvest_soon]

        elif self.agent_type == AgentType.GIG_WORKER:
            return [ts.work_found, ts.income_variance_3mo, ts.platform_access]

        elif self.agent_type == AgentType.STUDENT:
            return [ts.edu_progress, ts.family_support, ts.tuition_due]

        elif self.agent_type == AgentType.SMALL_BIZ_OWNER:
            return [ts.biz_health, ts.gst_rate, ts.credit_access, ts.customers]

        elif self.agent_type in (AgentType.SALARIED_MID, AgentType.SALARIED_HIGH,
                                  AgentType.YOUNG_PROFESSIONAL, AgentType.GOVT_EMPLOYEE):
            return [ts.emi_burden, ts.job_security, ts.promotion_progress]

        elif self.agent_type == AgentType.SENIOR:
            return [ts.health_multiplier / 3.0, ts.pension_stable, ts.medical_due]

        elif self.agent_type == AgentType.UNEMPLOYED:
            return [ts.job_search_score, ts.welfare_active, ts.skills_level]

        # Fallback — should never reach here
        return [0.0, 0.0, 0.0]

    # ─────────────────────────────────────────
    # REWARD COMPUTATION
    # ─────────────────────────────────────────

    def calculate_reward(self) -> float:
        """
        Compute the scalar reward for the current timestep.

        Reward function (from spec):
            reward = (
                w1 * log(1 + max(wealth_delta, 0))         # wealth gain
              - w2 * log(1 + max(-wealth_delta, 0))        # wealth loss
              + w3 * log(1 + consumption_spend)             # consumption utility
              + w4 * savings_security_bonus                 # savings buffer
              - w5 * debt_stress_score                      # debt burden
              - w6 * housing_stress                         # housing cost pressure
              - w7 * health_stress                          # healthcare burden
              - 10.0 * float(is_bankrupt)                   # bankruptcy penalty
              - 2.0  * float(missed_obligation)             # unpaid bill penalty
              + 1.0  * float(social_interaction_gain)       # social benefit
            )

        Returns:
            float — scalar reward for this timestep
        """
        f = self.finance
        w = self.config.reward_weights

        # ── Component 1: Wealth gain (asymmetric — gains reward less than losses hurt) ──
        wealth_gain = w["w1_wealth_gain"] * math.log1p(max(f.wealth_delta, 0.0))
        wealth_loss = w["w2_wealth_loss"] * math.log1p(max(-f.wealth_delta, 0.0))

        # ── Component 2: Consumption utility (diminishing returns via log) ──
        consumption = w["w3_consumption"] * math.log1p(f.consumption_this_step)

        # ── Component 3: Savings security (having a buffer is comforting) ──
        # Normalized: savings as fraction of 3 months' expenses
        three_month_expenses = f.monthly_expenses * 3.0
        savings_ratio = min(f.savings / max(three_month_expenses, 1.0), 1.0)
        savings_security = w["w4_savings_security"] * savings_ratio

        # ── Component 4: Debt stress (debt as fraction of annual income) ──
        annual_income = f.monthly_income * 12.0
        debt_ratio = min(f.debt / max(annual_income, 1.0), 1.0)
        debt_stress = w["w5_debt_stress"] * debt_ratio

        # ── Component 5: Housing stress (housing cost as fraction of income) ──
        housing_expense = self._get_housing_expense()
        housing_ratio = min(housing_expense / max(f.monthly_income, 1.0), 1.0)
        housing_stress = w["w6_housing_stress"] * housing_ratio

        # ── Component 6: Health stress (healthcare as fraction of income) ──
        health_expense = self.config.expenses.get("healthcare", 0.0) + \
                         self.config.expenses.get("medicine", 0.0)
        health_ratio = min(health_expense / max(f.monthly_income, 1.0), 1.0)
        health_stress = w["w7_health_stress"] * health_ratio

        # ── Fixed penalties / bonuses ──
        bankruptcy_penalty = REWARD_CONFIG.bankruptcy_penalty * float(f.is_bankrupt)
        obligation_penalty = REWARD_CONFIG.missed_obligation_penalty * float(f.missed_obligations > 0)

        # Social interaction bonus (set by social.py when interaction occurs)
        social_bonus = 0.0  # Updated externally via self.social_reward_this_step

        # ── Total reward ──
        reward = (
            wealth_gain
            - wealth_loss
            + consumption
            + savings_security
            - debt_stress
            - housing_stress
            - health_stress
            + bankruptcy_penalty
            + obligation_penalty
            + social_bonus
        )

        return reward

    # ─────────────────────────────────────────
    # ACTION EXECUTION
    # ─────────────────────────────────────────

    def decide_and_act(self, economics_state: Dict[str, float],
                       timestep: int) -> Tuple[int, float]:
        """
        Full decision cycle: observe → decide → act → reward.

        1. Build state vector from current financial state
        2. Feed state to PPO Actor → sample action
        3. Execute the chosen action (modify finances)
        4. Compute reward
        5. Store transition in PPO buffer
        6. Return action and reward for logging

        Args:
            economics_state: Global economic variables dict
            timestep:        Current simulation timestep

        Returns:
            (action, reward) tuple
        """
        # Step 1: Build state
        state = self.build_state_vector(economics_state, timestep)

        # Step 2: PPO selects action
        action, log_prob, value = self.ppo.select_action(state)

        # Step 3: Execute action
        prev_wealth = self.finance.wealth
        self._execute_action(action, economics_state)

        # Update wealth delta for reward computation
        self.finance.wealth_delta = self.finance.wealth - prev_wealth

        # Step 4: Compute reward
        reward = self.calculate_reward()

        # Step 5: Store transition
        done = self.finance.is_bankrupt
        self.ppo.store_transition(state, action, reward, value, log_prob, done)

        # Step 6: Update agent state
        self.current_action = action
        self.last_reward = reward
        self.steps_alive += 1

        # Update emotion based on reward
        self._update_emotion(action, reward)

        # Update wealth tier
        self._update_wealth_tier()

        # Track history (keep last 100 entries)
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        self.wealth_history.append(self.finance.wealth)
        if len(self.wealth_history) > 100:
            self.wealth_history.pop(0)

        return action, reward

    def _execute_action(self, action: int, economics_state: Dict[str, float]):
        """
        Execute the chosen action, modifying financial state.

        Actions:
            SAVE:   Move a fraction of available cash into savings
            SPEND:  Consume goods (lose money, gain utility)
            INVEST: Put money into market (delayed returns, risk)
            TRADE:  Active income generation (boost this step's income)
        """
        f = self.finance
        available_cash = max(f.wealth - f.savings, 0.0)

        # Apply cash freeze if policy active
        if f.cash_freeze_fraction > 0:
            available_cash *= (1.0 - f.cash_freeze_fraction)

        f.consumption_this_step = 0.0
        f.missed_obligations = 0

        if action == Action.SAVE:
            # ── SAVE: Transfer cash to savings ──
            save_amount = available_cash * ACTION_CONFIG.save_fraction
            f.savings += save_amount
            # No wealth change — just reclassification

        elif action == Action.SPEND:
            # ── SPEND: Consume goods/services ──
            spend_amount = min(
                f.wealth * ACTION_CONFIG.spend_fraction,
                available_cash
            )
            f.wealth -= spend_amount
            f.consumption_this_step = spend_amount
            f.lifetime_spending += spend_amount

        elif action == Action.INVEST:
            # ── INVEST: Put money into market ──
            if f.invest_allowed and available_cash > 0:
                invest_amount = min(
                    available_cash * ACTION_CONFIG.invest_fraction,
                    available_cash
                )
                f.wealth -= invest_amount
                f.invested_amount += invest_amount
            # Returns are processed in monthly_update via economics.py

        elif action == Action.TRADE:
            # ── TRADE: Active income generation ──
            if f.trade_allowed:
                # Boost income for this step
                trade_income = f.monthly_income * ACTION_CONFIG.trade_income_boost
                f.wealth += trade_income
                f.lifetime_earnings += trade_income

        # ── Process fixed monthly obligations (prorated per step) ──
        daily_expenses = (f.monthly_expenses * f.expense_multiplier) / STEPS_PER_MONTH

        # Apply subsidies
        if f.healthcare_subsidy > 0:
            healthcare_exp = (self.config.expenses.get("healthcare", 0.0) +
                              self.config.expenses.get("medicine", 0.0))
            daily_expenses -= (healthcare_exp * f.healthcare_subsidy) / STEPS_PER_MONTH

        if f.education_subsidy > 0:
            edu_exp = self.config.expenses.get("tuition", 0.0)
            daily_expenses -= (edu_exp * f.education_subsidy) / STEPS_PER_MONTH

        daily_expenses = max(daily_expenses, 0.0)

        # Pay expenses
        if f.wealth >= daily_expenses:
            f.wealth -= daily_expenses
        else:
            # Can't afford expenses — missed obligation
            f.missed_obligations += 1
            f.debt += daily_expenses - f.wealth
            f.wealth = 0.0

        # ── Apply tax (prorated daily) ──
        daily_tax = (f.monthly_income * f.tax_rate) / STEPS_PER_MONTH
        if f.wealth >= daily_tax:
            f.wealth -= daily_tax
        else:
            f.debt += daily_tax - f.wealth
            f.wealth = 0.0

        # ── Add welfare payment (prorated daily) ──
        if f.welfare_payment > 0:
            f.wealth += f.welfare_payment / STEPS_PER_MONTH

        # ── Bankruptcy check ──
        if f.wealth < BANKRUPTCY_THRESHOLD and not f.is_bankrupt:
            f.is_bankrupt = True
            f.bankruptcies += 1

        # ── Bankruptcy recovery (if wealth rises above threshold) ──
        if f.is_bankrupt and f.wealth > POVERTY_THRESHOLD:
            f.is_bankrupt = False

    # ─────────────────────────────────────────
    # MONTHLY INCOME PROCESSING
    # ─────────────────────────────────────────

    def process_monthly_income(self, economics_state: Dict[str, float],
                               timestep: int):
        """
        Process monthly income arrival. Called every STEPS_PER_MONTH steps.

        Income varies by type:
          - Farmers: seasonal (high at harvest, low off-season)
          - Gig workers: ±40% random variance
          - Salaried: stable with small variance
          - Unemployed: 0 unless welfare active
          - Young professionals: grows 0.5% per month

        Args:
            economics_state: Global economic variables
            timestep:        Current timestep
        """
        f = self.finance
        month = (timestep % STEPS_PER_YEAR) // STEPS_PER_MONTH

        # ── Base income calculation ──
        if self.agent_type == AgentType.FARMER:
            season = MONTH_TO_SEASON.get(month, Season.KHARIF_GROWING)
            if season in HARVEST_SEASONS:
                # Harvest months: high income
                f.monthly_income = f.base_income * 4.0
            else:
                # Off-season: low income
                f.monthly_income = f.base_income * 0.2

        elif self.agent_type == AgentType.GIG_WORKER:
            # Highly variable income
            emp_avail = economics_state.get("employment_availability", 0.75)
            variance = self.config.income_variance * (1.0 + self.spending_influence)
            f.monthly_income = f.base_income * random.uniform(1.0 - variance,
                                                               1.0 + variance)
            f.monthly_income *= emp_avail  # Scaled by job market
            self.type_state.work_found = min(emp_avail + self.job_tip_boost, 1.0)

        elif self.agent_type == AgentType.UNEMPLOYED:
            # Zero income unless welfare is active
            if self.type_state.welfare_active > 0.5:
                f.monthly_income = f.welfare_payment
            else:
                f.monthly_income = 0.0

        elif self.agent_type == AgentType.YOUNG_PROFESSIONAL:
            # Income grows 0.5% per month (career progression)
            f.base_income *= 1.005
            variance = self.config.income_variance
            f.monthly_income = f.base_income * random.uniform(1.0 - variance,
                                                               1.0 + variance)

        elif self.agent_type == AgentType.SALARIED_HIGH:
            # Annual bonus in month 3 (March — Indian financial year end)
            variance = self.config.income_variance
            f.monthly_income = f.base_income * random.uniform(1.0 - variance,
                                                               1.0 + variance)
            if month == 3:  # March — bonus month
                f.monthly_income += f.base_income * random.uniform(0.5, 2.0)

        else:
            # Standard income with configured variance
            variance = self.config.income_variance
            f.monthly_income = f.base_income * random.uniform(1.0 - variance,
                                                               1.0 + variance)

        # ── Apply income multiplier (from policies) ──
        f.monthly_income *= f.income_multiplier

        # ── Apply inflation to expenses ──
        inflation = economics_state.get("inflation_rate", 0.03)
        f.monthly_expenses = sum(self.config.expenses.values()) * (1.0 + inflation)
        f.monthly_expenses *= f.expense_multiplier

        # ── Add monthly income to wealth ──
        net_income = f.monthly_income * (1.0 - f.tax_rate)
        f.wealth += net_income
        f.lifetime_earnings += net_income

        # ── Process investment returns ──
        if f.invested_amount > 0:
            returns = f.invested_amount * f.investment_return_rate
            f.invested_amount += returns
            f.wealth += returns

        # ── Process savings interest ──
        if f.savings > 0:
            interest = f.savings * (f.savings_interest_rate / 12.0)
            f.savings += interest
            f.wealth += interest

        # ── Process debt interest ──
        if f.debt > 0:
            debt_interest = f.debt * (f.debt_interest_rate / 12.0)
            f.debt += debt_interest

        # ── Track income history ──
        self.income_history.append(f.monthly_income)
        if len(self.income_history) > 12:
            self.income_history.pop(0)

        # ── Update type-specific state after income ──
        self._update_type_state_monthly(economics_state, month)

    def _update_type_state_monthly(self, economics_state: Dict[str, float],
                                    month: int):
        """Update type-specific state variables at month boundary."""
        ts = self.type_state
        f = self.finance

        if self.agent_type == AgentType.STUDENT:
            ts.edu_progress = min(ts.edu_progress + 1.0 / 36.0, 1.0)  # 3-year degree
            ts.tuition_due = 1.0 if month in [0, 6] else 0.0  # Semester fees

        elif self.agent_type == AgentType.GIG_WORKER:
            # Update 3-month income variance
            if len(self.income_history) >= 3:
                recent = self.income_history[-3:]
                mean_inc = sum(recent) / 3.0
                var = sum((x - mean_inc) ** 2 for x in recent) / 3.0
                ts.income_variance_3mo = min(math.sqrt(var) / max(mean_inc, 1.0), 1.0)

        elif self.agent_type == AgentType.SMALL_BIZ_OWNER:
            # Business health tracks recent income trend
            if len(self.income_history) >= 2:
                trend = self.income_history[-1] / max(self.income_history[-2], 1.0)
                ts.biz_health = min(max(ts.biz_health * 0.8 + trend * 0.2, 0.0), 1.0)
            ts.credit_access = min(
                economics_state.get("credit_availability", 0.7) *
                (1.0 - min(f.debt / max(f.monthly_income * 12, 1.0), 1.0)),
                1.0
            )

        elif self.agent_type == AgentType.SENIOR:
            # Health costs slowly increase over time
            ts.health_multiplier = min(ts.health_multiplier + 0.005, 3.0)

        elif self.agent_type == AgentType.UNEMPLOYED:
            # Job search score improves slowly with skill training
            ts.job_search_score = min(ts.job_search_score + 0.01, 1.0)

        # Update EMI burden for salaried types
        if self.agent_type in (AgentType.SALARIED_MID, AgentType.SALARIED_HIGH,
                                AgentType.YOUNG_PROFESSIONAL, AgentType.GOVT_EMPLOYEE):
            emi_expenses = sum(v for k, v in self.config.expenses.items()
                               if 'emi' in k.lower() or 'loan' in k.lower())
            ts.emi_burden = min(emi_expenses / max(f.monthly_income, 1.0), 1.0)

        # Debt high-debt tracking (for loan default trigger)
        if f.debt > f.monthly_income * 3.0 and f.monthly_income > 0:
            f.months_high_debt += 1
        else:
            f.months_high_debt = 0

    # ─────────────────────────────────────────
    # MOVEMENT
    # ─────────────────────────────────────────

    def set_target(self, target_x: float, target_y: float):
        """
        Set a new movement target. Agent will lerp toward this position.

        Args:
            target_x: Target x coordinate on canvas
            target_y: Target y coordinate on canvas
        """
        self.target_x = max(0, min(target_x, CANVAS_WIDTH))
        self.target_y = max(0, min(target_y, CANVAS_HEIGHT))
        self.is_moving = True

    def move_step(self, dt: float = 1.0):
        """
        Update position using linear interpolation toward target.

        Smooth movement with configurable speed:
          - Bankrupt agents move at 30% speed (visibly struggling)
          - Seniors move at 40% speed (slow)
          - Rich agents at 130% speed (fast-paced)

        Args:
            dt: Delta time multiplier (usually 1.0)
        """
        # Compute effective speed
        speed = self.base_speed
        if self.finance.is_bankrupt:
            speed *= BANKRUPT_SPEED_MULT
        self.current_speed = speed

        # Lerp toward target
        lerp = LERP_FACTOR * speed * dt
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > 2.0:  # Only move if not already at target
            self.x += dx * lerp
            self.y += dy * lerp
            self.is_moving = True

            # Walking animation timer
            self.walk_timer += dt * 16.67  # Approximate ms per frame at 60fps
            if self.walk_timer > 400:  # Toggle every 400ms
                self.walk_frame = 1 - self.walk_frame
                self.walk_timer = 0.0
        else:
            self.x = self.target_x
            self.y = self.target_y
            self.is_moving = False
            self.walk_frame = 0

        # Clamp to canvas bounds
        self.x = max(0, min(self.x, CANVAS_WIDTH))
        self.y = max(0, min(self.y, CANVAS_HEIGHT))

    # ─────────────────────────────────────────
    # EMOTION & WEALTH TIER
    # ─────────────────────────────────────────

    def _update_emotion(self, action: int, reward: float):
        """Set current emotion icon based on action and reward."""
        if self.finance.is_bankrupt or self.finance.wealth < SURVIVAL_THRESHOLD:
            self.current_emotion = "danger"
        elif action == Action.SPEND:
            self.current_emotion = "spending"
        elif action == Action.INVEST:
            self.current_emotion = "investing"
        elif reward > 0.1:
            self.current_emotion = "happy"
        elif reward < -0.5:
            self.current_emotion = "sad"
        else:
            self.current_emotion = "happy"

    def _update_wealth_tier(self):
        """Update wealth tier based on current wealth."""
        wealth = self.finance.wealth
        if wealth < BANKRUPTCY_THRESHOLD:
            self.current_wealth_tier = 0      # Bankrupt
        elif wealth < POVERTY_THRESHOLD:
            self.current_wealth_tier = 1      # Poor
        elif wealth < 20_000:
            self.current_wealth_tier = 2      # Low-middle
        elif wealth < 100_000:
            self.current_wealth_tier = 3      # Middle
        else:
            self.current_wealth_tier = 4      # Rich

    def get_wealth_dot_color(self) -> str:
        """Get the hex color for the floating wealth dot above the agent."""
        for tier in reversed(WEALTH_TIERS):
            if self.finance.wealth >= tier.min_wealth:
                return tier.color
        return WEALTH_TIERS[0].color

    def get_emotion_icon(self) -> str:
        """Get the current emotion emoji."""
        return EMOTION_ICONS.get(self.current_emotion, "😊")

    # ─────────────────────────────────────────
    # SOCIAL SYSTEM HOOKS
    # ─────────────────────────────────────────

    def update_relationship(self, other_id: int, delta: float):
        """
        Update relationship score with another agent.

        Only top 10 relationships are kept to save memory.

        Args:
            other_id: The other agent's ID
            delta:    Score change (+ve for bonding, -ve for decay)
        """
        current = self.relationships.get(other_id, 0.0)
        new_score = max(0.0, min(1.0, current + delta))

        if new_score > 0.001:
            self.relationships[other_id] = new_score

            # Prune to top 10
            if len(self.relationships) > 10:
                sorted_rels = sorted(self.relationships.items(),
                                      key=lambda x: x[1], reverse=True)
                self.relationships = dict(sorted_rels[:10])
        elif other_id in self.relationships:
            del self.relationships[other_id]

    def decay_relationships(self, decay_rate: float = 0.001):
        """Apply natural decay to all relationship scores."""
        to_remove = []
        for other_id in self.relationships:
            self.relationships[other_id] -= decay_rate
            if self.relationships[other_id] <= 0:
                to_remove.append(other_id)
        for rid in to_remove:
            del self.relationships[rid]

    # ─────────────────────────────────────────
    # JOB TIP BOOST
    # ─────────────────────────────────────────

    def apply_job_tip(self, boost: float, duration: int):
        """Apply a temporary employment availability boost from social tip."""
        self.job_tip_boost = boost
        self.job_tip_timer = duration

    def tick_job_tip(self):
        """Decrement job tip timer; clear boost when expired."""
        if self.job_tip_timer > 0:
            self.job_tip_timer -= 1
            if self.job_tip_timer <= 0:
                self.job_tip_boost = 0.0

    # ─────────────────────────────────────────
    # SERIALIZATION — WebSocket Diffs
    # ─────────────────────────────────────────

    def to_render_dict(self) -> Dict[str, Any]:
        """
        Compact dict for WebSocket diff updates.

        Only essential rendering data — position, visual state,
        core financial indicators. Sent at 10fps.

        Returns:
            Dict with keys: id, x, y, wealth_tier, emotion, action,
            wealth, type, body_color, walk_frame
        """
        return {
            "id": self.id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "wealth_tier": self.current_wealth_tier,
            "wealth_dot_color": self.get_wealth_dot_color(),
            "emotion": self.get_emotion_icon(),
            "action": self.current_action,
            "action_name": ACTION_NAMES.get(self.current_action, "Idle"),
            "wealth": round(self.finance.wealth, 0),
            "type": self.agent_type.value,
            "type_name": self.type_name,
            "body_color": self.config.body_color,
            "walk_frame": self.walk_frame,
            "is_moving": self.is_moving,
            "is_bankrupt": self.finance.is_bankrupt,
            "speed": round(self.current_speed, 2),
        }

    def to_inspect_dict(self) -> Dict[str, Any]:
        """
        Full dict for the Agent Inspector panel.

        Contains everything a researcher needs to understand this agent:
        identity, financials, PPO explainability, history, social state.

        Returns:
            Comprehensive dict for inspector popup UI
        """
        f = self.finance

        # Get current action probabilities from PPO
        dummy_econ = {
            "inflation_rate": 0.03,
            "base_interest_rate": 0.0067,
            "housing_cost_index": 1.0,
            "employment_availability": 0.75,
        }
        state = self.build_state_vector(dummy_econ, self.steps_alive)
        action_probs = self.ppo.get_action_probs(state)
        value_estimate = self.ppo.get_value_estimate(state)

        # Find top 3 most influential state variables
        top_features = self._get_top_influential_features(state)

        # Economic status summary
        status_summary = self._generate_status_summary()

        return {
            # ── Identity ──
            "id": self.id,
            "type": self.agent_type.value,
            "type_name": self.type_name,
            "body_color": self.config.body_color,
            "status": "Bankrupt" if f.is_bankrupt else "Surviving",
            "steps_alive": self.steps_alive,

            # ── Financials ──
            "wealth": round(f.wealth, 2),
            "monthly_income": round(f.monthly_income, 2),
            "savings": round(f.savings, 2),
            "debt": round(f.debt, 2),
            "monthly_expenses": round(f.monthly_expenses, 2),
            "invested_amount": round(f.invested_amount, 2),
            "lifetime_earnings": round(f.lifetime_earnings, 2),
            "lifetime_spending": round(f.lifetime_spending, 2),
            "wealth_tier": self.current_wealth_tier,
            "wealth_dot_color": self.get_wealth_dot_color(),

            # ── PPO Explainability ──
            "action_probabilities": {
                ACTION_NAMES[i]: round(float(action_probs[i]), 4)
                for i in range(NUM_ACTIONS)
            },
            "action_probs_raw": [round(float(p), 4) for p in action_probs],
            "last_reward": round(self.last_reward, 4),
            "value_estimate": round(value_estimate, 4),
            "top_influential_features": top_features,
            "status_summary": status_summary,

            # ── Training Stats ──
            "ppo_stats": self.ppo.get_training_stats(),

            # ── History ──
            "bankruptcies": f.bankruptcies,
            "life_events": self.life_events[-10:],     # Last 10 events
            "policy_impacts": self.policy_impacts[-5:], # Last 5 policies
            "reward_history": self.reward_history[-50:],
            "wealth_history": self.wealth_history[-50:],

            # ── Social ──
            "relationships": dict(sorted(
                self.relationships.items(),
                key=lambda x: x[1], reverse=True
            )[:5]),  # Top 5 relationships
            "trust_scores": dict(sorted(
                self.trust_scores.items(),
                key=lambda item: item[1],
                reverse=True
            )[:5]),
            "reputation": round(self.reputation, 3),
            "skill_level": round(self.skill_level, 3),
            "family_links": self.family_links,
            "memory": self.memory[-10:], # Last 10 memories
            "pending_loans": self.pending_loans.copy(),
            "loans_given": self.loans_given.copy(),

            # ── Position ──
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "target_x": round(self.target_x, 1),
            "target_y": round(self.target_y, 1),
        }

    def _get_top_influential_features(self, state: np.ndarray) -> List[Dict]:
        """
        Identify the top 3 state features most likely driving the agent's
        current behavior (highest absolute normalized values).

        This is a simple heuristic: features with extreme values (near 0 or 1)
        are likely most influential in the network's decision.
        """
        feature_names = [
            "wealth_norm", "income_norm", "savings_norm", "debt_norm",
            "expense_ratio", "inflation_rate", "interest_rate",
            "housing_cost_norm", "employment_avail", "wealth_tier",
            "month", "day_in_month", "reputation", "skill_level", "social_pressure"
        ]
        # Add policy slots
        for i in range(NUM_POLICY_SLOTS):
            feature_names.append(f"policy_slot_{i}")

        # Add type-specific features
        feature_names.extend(self.config.type_specific_features)

        # Score each feature by deviation from 0.5 (neutral)
        scored = []
        for i, name in enumerate(feature_names):
            if i < len(state):
                deviation = abs(state[i] - 0.5)
                scored.append({
                    "name": name,
                    "value": round(float(state[i]), 3),
                    "influence": round(float(deviation), 3),
                })

        # Sort by influence and return top 3
        scored.sort(key=lambda x: x["influence"], reverse=True)
        return scored[:3]

    def _generate_status_summary(self) -> str:
        """Generate a human-readable economic status summary for the inspector."""
        f = self.finance
        parts = []

        # Debt assessment
        if f.debt > f.monthly_income * 6:
            parts.append("Critical debt burden")
        elif f.debt > f.monthly_income * 3:
            parts.append("High debt burden")
        elif f.debt > 0:
            parts.append("Manageable debt")

        # Savings assessment
        if f.savings > f.monthly_expenses * 6:
            parts.append("strong savings buffer")
        elif f.savings > f.monthly_expenses * 3:
            parts.append("adequate savings")
        elif f.savings > 0:
            parts.append("thin savings")
        else:
            parts.append("no savings")

        # Income vs expenses
        if f.monthly_income > f.monthly_expenses * 1.5:
            parts.append("comfortable income margin")
        elif f.monthly_income > f.monthly_expenses:
            parts.append("tight income margin")
        else:
            parts.append("expenses exceed income")

        # Wealth trajectory
        if len(self.wealth_history) >= 5:
            recent_avg = sum(self.wealth_history[-5:]) / 5
            older_avg = sum(self.wealth_history[:5]) / max(len(self.wealth_history[:5]), 1)
            if recent_avg > older_avg * 1.1:
                parts.append("wealth trending upward")
            elif recent_avg < older_avg * 0.9:
                parts.append("wealth declining")
            else:
                parts.append("wealth stable")

        # Bankruptcy
        if f.is_bankrupt:
            parts.append("BANKRUPT — needs immediate support")

        # Type-specific notes
        if self.agent_type == AgentType.FARMER:
            season = MONTH_TO_SEASON.get(
                (self.steps_alive % STEPS_PER_YEAR) // STEPS_PER_MONTH,
                Season.KHARIF_GROWING
            )
            if season in HARVEST_SEASONS:
                parts.append("currently in harvest season (peak income)")
            else:
                parts.append("off-season (low income)")

        elif self.agent_type == AgentType.UNEMPLOYED:
            if self.type_state.welfare_active > 0.5:
                parts.append("receiving welfare support")
            else:
                parts.append("no welfare — surviving on reserves")

        return ". ".join(parts) + "."

    # ─────────────────────────────────────────
    # DEEP COPY (for scenario comparison)
    # ─────────────────────────────────────────

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get a serializable snapshot of all agent state for scenario cloning.

        Returns a dict that can be used to restore agent state via restore_snapshot().
        PPO network weights are NOT included (they're shared and handled separately).
        """
        return {
            "id": self.id,
            "agent_type": self.agent_type.value,
            "x": self.x,
            "y": self.y,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "finance": {
                "wealth": self.finance.wealth,
                "savings": self.finance.savings,
                "debt": self.finance.debt,
                "monthly_income": self.finance.monthly_income,
                "base_income": self.finance.base_income,
                "monthly_expenses": self.finance.monthly_expenses,
                "income_multiplier": self.finance.income_multiplier,
                "expense_multiplier": self.finance.expense_multiplier,
                "tax_rate": self.finance.tax_rate,
                "investment_return_rate": self.finance.investment_return_rate,
                "invested_amount": self.finance.invested_amount,
                "welfare_payment": self.finance.welfare_payment,
                "healthcare_subsidy": self.finance.healthcare_subsidy,
                "education_subsidy": self.finance.education_subsidy,
                "cash_freeze_fraction": self.finance.cash_freeze_fraction,
                "trade_allowed": self.finance.trade_allowed,
                "invest_allowed": self.finance.invest_allowed,
                "lifetime_earnings": self.finance.lifetime_earnings,
                "lifetime_spending": self.finance.lifetime_spending,
                "bankruptcies": self.finance.bankruptcies,
                "is_bankrupt": self.finance.is_bankrupt,
                "months_high_debt": self.finance.months_high_debt,
            },
            "relationships": dict(self.relationships),
            "pending_loans": dict(self.pending_loans),
            "loans_given": dict(self.loans_given),
            "current_wealth_tier": self.current_wealth_tier,
            "steps_alive": self.steps_alive,
            "life_events": list(self.life_events),
            "policy_impacts": list(self.policy_impacts),
            "policy_slots": list(self.policy_slots),
            "active_policies": dict(self.active_policies),
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]):
        """
        Restore agent state from a snapshot dict (for scenario comparison).

        Args:
            snapshot: Dict from get_snapshot()
        """
        self.x = snapshot["x"]
        self.y = snapshot["y"]
        self.target_x = snapshot["target_x"]
        self.target_y = snapshot["target_y"]

        fs = snapshot["finance"]
        self.finance.wealth = fs["wealth"]
        self.finance.savings = fs["savings"]
        self.finance.debt = fs["debt"]
        self.finance.monthly_income = fs["monthly_income"]
        self.finance.base_income = fs["base_income"]
        self.finance.monthly_expenses = fs["monthly_expenses"]
        self.finance.income_multiplier = fs["income_multiplier"]
        self.finance.expense_multiplier = fs["expense_multiplier"]
        self.finance.tax_rate = fs["tax_rate"]
        self.finance.investment_return_rate = fs["investment_return_rate"]
        self.finance.invested_amount = fs["invested_amount"]
        self.finance.welfare_payment = fs["welfare_payment"]
        self.finance.healthcare_subsidy = fs["healthcare_subsidy"]
        self.finance.education_subsidy = fs["education_subsidy"]
        self.finance.cash_freeze_fraction = fs["cash_freeze_fraction"]
        self.finance.trade_allowed = fs["trade_allowed"]
        self.finance.invest_allowed = fs["invest_allowed"]
        self.finance.lifetime_earnings = fs["lifetime_earnings"]
        self.finance.lifetime_spending = fs["lifetime_spending"]
        self.finance.bankruptcies = fs["bankruptcies"]
        self.finance.is_bankrupt = fs["is_bankrupt"]
        self.finance.months_high_debt = fs["months_high_debt"]

        self.relationships = dict(snapshot["relationships"])
        self.pending_loans = dict(snapshot["pending_loans"])
        self.loans_given = dict(snapshot["loans_given"])
        self.current_wealth_tier = snapshot["current_wealth_tier"]
        self.steps_alive = snapshot["steps_alive"]
        self.life_events = list(snapshot["life_events"])
        self.policy_impacts = list(snapshot["policy_impacts"])
        self.policy_slots = list(snapshot["policy_slots"])
        self.active_policies = dict(snapshot["active_policies"])

    def __repr__(self) -> str:
        return (f"Agent(id={self.id}, type={self.type_name}, "
                f"wealth=₹{self.finance.wealth:,.0f}, "
                f"pos=({self.x:.0f},{self.y:.0f}))")
