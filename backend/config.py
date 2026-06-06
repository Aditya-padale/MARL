"""
config.py — Single source of truth for the MARL City Simulator.

Every tunable parameter lives here. Researchers modify ONLY this file
to adjust simulation behavior. All values are India-realistic (INR).

Author: Aditya Padale (B.Tech Final Year Project)
Target: IEEE publication on multi-agent RL for policy analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import IntEnum


# ═══════════════════════════════════════════
# CANVAS & RENDERING
# ═══════════════════════════════════════════

CANVAS_WIDTH: int = 1200
CANVAS_HEIGHT: int = 900
TARGET_FPS: int = 60          # Frontend rendering framerate
BACKEND_TICK_RATE: int = 10   # Backend simulation updates per second
AGENT_SIZE: int = 16          # Agent sprite size in pixels
AGENT_DOT_RADIUS: int = 4    # Wealth indicator dot radius


# ═══════════════════════════════════════════
# TIME SYSTEM
# ═══════════════════════════════════════════

STEPS_PER_DAY: int = 1
STEPS_PER_WEEK: int = 7
STEPS_PER_MONTH: int = 30
STEPS_PER_YEAR: int = 365

# Time acceleration multipliers (steps to advance per tick)
TIME_SPEEDS: Dict[str, int] = {
    "pause":    0,
    "play":     1,       # Real-time: 1 day per tick
    "week":     7,       # Fast: 1 week per tick
    "month":    30,      # Faster: 1 month per tick
    "year":     365,     # Fastest: 1 year per tick
}

# Day/night cycle colors (gradient transitions)
SKY_COLORS: Dict[str, str] = {
    "dawn":   "#f4a261",   # 06:00 – warm orange sunrise
    "day":    "#87ceeb",   # 10:00 – clear sky blue
    "dusk":   "#e76f51",   # 18:00 – sunset red-orange
    "night":  "#1a1a2e",   # 22:00 – deep dark blue
}

# Night speed multiplier (simulation slows at night)
NIGHT_SPEED_MULTIPLIER: float = 0.3


# ═══════════════════════════════════════════
# AGENT TYPE DEFINITIONS
# ═══════════════════════════════════════════

class AgentType(IntEnum):
    """
    10 distinct agent types representing Indian society cross-section.
    IntEnum for array indexing into PPO network pools.
    """
    FARMER            = 0
    GIG_WORKER        = 1
    STUDENT           = 2
    SMALL_BIZ_OWNER   = 3
    SALARIED_MID      = 4
    SALARIED_HIGH     = 5
    YOUNG_PROFESSIONAL = 6
    GOVT_EMPLOYEE     = 7
    SENIOR            = 8
    UNEMPLOYED        = 9


# String labels for display and serialization
AGENT_TYPE_NAMES: Dict[int, str] = {
    AgentType.FARMER:             "Farmer",
    AgentType.GIG_WORKER:         "Gig Worker",
    AgentType.STUDENT:            "Student",
    AgentType.SMALL_BIZ_OWNER:    "Small Business Owner",
    AgentType.SALARIED_MID:       "Salaried Mid",
    AgentType.SALARIED_HIGH:      "Salaried High",
    AgentType.YOUNG_PROFESSIONAL: "Young Professional",
    AgentType.GOVT_EMPLOYEE:      "Govt Employee",
    AgentType.SENIOR:             "Senior / Retired",
    AgentType.UNEMPLOYED:         "Unemployed",
}


@dataclass
class AgentTypeConfig:
    """
    Complete configuration for one agent type.
    All monetary values in INR (₹).
    """
    # --- Identity ---
    agent_type: AgentType
    count: int                              # How many of this type to spawn
    body_color: str                         # Hex color for canvas rendering

    # --- Financial Starting Conditions ---
    wealth_range: Tuple[float, float]       # (min, max) starting wealth in ₹
    income_range: Tuple[float, float]       # (min, max) monthly income in ₹
    income_variance: float                  # ±variance as fraction (0.0–1.0)
    is_seasonal_income: bool                # True for farmers (harvest cycles)

    # --- Monthly Fixed Expenses (₹) ---
    expenses: Dict[str, float]             # Named expense categories

    # --- Movement ---
    home_zone: str                          # Zone name for nighttime return
    work_zone: str                          # Zone name for daytime activity
    move_speed: float                       # Pixels per frame (base speed)

    # --- State Vector ---
    state_dim: int                          # Total state vector dimension
    type_specific_features: List[str]       # Names of type-specific features

    # --- PPO Reward Weights ---
    reward_weights: Dict[str, float]        # w1–w7 for reward computation

    # --- Life Events (per-step probabilities) ---
    life_event_probs: Dict[str, float]      # Event name → probability per step

    # --- Tax ---
    base_tax_rate: float                    # Default income tax rate


# ═══════════════════════════════════════════
# AGENT TYPE CONFIGURATIONS (all 10 types)
# ═══════════════════════════════════════════

AGENT_CONFIGS: Dict[int, AgentTypeConfig] = {

    # ──────────────────────────────────────
    # 1. FARMER — 15 agents
    # ──────────────────────────────────────
    AgentType.FARMER: AgentTypeConfig(
        agent_type=AgentType.FARMER,
        count=15,
        body_color="#8B4513",               # Brown — earthy farmer

        wealth_range=(3_000.0, 15_000.0),
        income_range=(1_000.0, 20_000.0),   # ₹1K off-season, ₹20K post-harvest
        income_variance=0.40,
        is_seasonal_income=True,

        expenses={
            "seed":        2_000.0,
            "fertilizer":  1_500.0,
            "loan_emi":    3_000.0,
            "food":        2_000.0,
        },

        home_zone="residential_south",
        work_zone="outskirts",
        move_speed=1.0,

        state_dim=25,                       # 21 shared + 4 type-specific
        type_specific_features=["season", "crop_price_idx", "loan_due", "harvest_soon"],

        reward_weights={
            "w1_wealth_gain":      0.40,
            "w2_wealth_loss":      0.50,
            "w3_consumption":      0.25,
            "w4_savings_security": 0.30,
            "w5_debt_stress":      0.50,
            "w6_housing_stress":   0.20,
            "w7_health_stress":    0.15,
        },

        life_event_probs={
            "crop_failure":        0.003,    # Monsoon months only
            "good_harvest":        0.004,    # Harvest season only
            "medical_emergency":   0.001,
            "loan_default":        0.002,
            "family_remittance":   0.003,
        },

        base_tax_rate=0.0,                  # Small farmers typically exempt
    ),

    # ──────────────────────────────────────
    # 2. GIG WORKER — 12 agents
    # ──────────────────────────────────────
    AgentType.GIG_WORKER: AgentTypeConfig(
        agent_type=AgentType.GIG_WORKER,
        count=12,
        body_color="#FF8C00",               # Orange — hustler energy

        wealth_range=(1_000.0, 8_000.0),
        income_range=(8_000.0, 25_000.0),
        income_variance=0.40,               # Highly volatile income
        is_seasonal_income=False,

        expenses={
            "rent":   4_000.0,
            "food":   3_000.0,
            "fuel":   1_500.0,
        },

        home_zone="residential_cheap",
        work_zone="commercial",             # Wanders commercial + business
        move_speed=1.2,

        state_dim=24,                       # 21 shared + 3 type-specific
        type_specific_features=["work_found", "income_variance_3mo", "platform_access"],

        reward_weights={
            "w1_wealth_gain":      0.30,
            "w2_wealth_loss":      0.45,
            "w3_consumption":      0.35,
            "w4_savings_security": 0.30,
            "w5_debt_stress":      0.40,
            "w6_housing_stress":   0.25,
            "w7_health_stress":    0.10,
        },

        life_event_probs={
            "work_drought":        0.003,
            "platform_ban":        0.001,
            "gig_boom":            0.002,
            "medical_emergency":   0.001,
            "loan_default":        0.002,
        },

        base_tax_rate=0.05,
    ),

    # ──────────────────────────────────────
    # 3. STUDENT — 10 agents
    # ──────────────────────────────────────
    AgentType.STUDENT: AgentTypeConfig(
        agent_type=AgentType.STUDENT,
        count=10,
        body_color="#4169E1",               # Royal blue — academic

        wealth_range=(500.0, 3_000.0),
        income_range=(3_000.0, 5_000.0),    # Stipend / family transfer
        income_variance=0.15,
        is_seasonal_income=False,

        expenses={
            "tuition":  5_000.0,
            "rent":     3_000.0,
            "food":     2_000.0,
            "books":      500.0,
        },

        home_zone="residential_shared",
        work_zone="education",
        move_speed=1.1,

        state_dim=24,                       # 21 shared + 3 type-specific
        type_specific_features=["edu_progress", "family_support", "tuition_due"],

        reward_weights={
            "w1_wealth_gain":      0.20,
            "w2_wealth_loss":      0.35,
            "w3_consumption":      0.45,
            "w4_savings_security": 0.30,
            "w5_debt_stress":      0.30,
            "w6_housing_stress":   0.30,
            "w7_health_stress":    0.05,
        },

        life_event_probs={
            "scholarship":          0.002,
            "exam_failure":         0.001,
            "family_support_cut":   0.001,
            "medical_emergency":    0.0005,
            "family_remittance":    0.003,
        },

        base_tax_rate=0.0,                  # Students don't pay income tax
    ),

    # ──────────────────────────────────────
    # 4. SMALL BUSINESS OWNER — 10 agents
    # ──────────────────────────────────────
    AgentType.SMALL_BIZ_OWNER: AgentTypeConfig(
        agent_type=AgentType.SMALL_BIZ_OWNER,
        count=10,
        body_color="#8B008B",               # Purple — entrepreneurial

        wealth_range=(30_000.0, 150_000.0),
        income_range=(30_000.0, 80_000.0),
        income_variance=0.30,
        is_seasonal_income=False,

        expenses={
            "shop_rent":    10_000.0,
            "gst":           5_000.0,
            "staff_salary": 15_000.0,
            "inventory":     5_000.0,
            "food":          4_000.0,
        },

        home_zone="residential_mid",
        work_zone="commercial",
        move_speed=1.0,

        state_dim=25,                       # 21 shared + 4 type-specific
        type_specific_features=["biz_health", "gst_rate", "credit_access", "customers"],

        reward_weights={
            "w1_wealth_gain":      0.50,
            "w2_wealth_loss":      0.55,
            "w3_consumption":      0.20,
            "w4_savings_security": 0.25,
            "w5_debt_stress":      0.50,
            "w6_housing_stress":   0.20,
            "w7_health_stress":    0.10,
        },

        life_event_probs={
            "business_boom":       0.002,
            "gst_audit":           0.001,
            "shop_fire":           0.0005,
            "loan_default":        0.002,
            "medical_emergency":   0.001,
        },

        base_tax_rate=0.18,                 # GST bracket
    ),

    # ──────────────────────────────────────
    # 5. SALARIED MID — 15 agents
    # ──────────────────────────────────────
    AgentType.SALARIED_MID: AgentTypeConfig(
        agent_type=AgentType.SALARIED_MID,
        count=15,
        body_color="#008080",               # Teal — corporate stability

        wealth_range=(20_000.0, 80_000.0),
        income_range=(35_000.0, 60_000.0),
        income_variance=0.05,               # Very stable salary
        is_seasonal_income=False,

        expenses={
            "home_emi":     12_000.0,
            "insurance":     3_000.0,
            "food":          8_000.0,
            "transport":     3_000.0,
            "utilities":     2_000.0,
        },

        home_zone="residential_mid",
        work_zone="business_district",
        move_speed=1.0,

        state_dim=24,                       # 21 shared + 3 type-specific
        type_specific_features=["emi_burden", "job_security", "promotion_progress"],

        reward_weights={
            "w1_wealth_gain":      0.40,
            "w2_wealth_loss":      0.45,
            "w3_consumption":      0.30,
            "w4_savings_security": 0.30,
            "w5_debt_stress":      0.40,
            "w6_housing_stress":   0.30,
            "w7_health_stress":    0.15,
        },

        life_event_probs={
            "layoff":              0.002,
            "promotion":           0.001,
            "medical_emergency":   0.001,
        },

        base_tax_rate=0.20,                 # 20% tax bracket
    ),

    # ──────────────────────────────────────
    # 6. SALARIED HIGH — 8 agents
    # ──────────────────────────────────────
    AgentType.SALARIED_HIGH: AgentTypeConfig(
        agent_type=AgentType.SALARIED_HIGH,
        count=8,
        body_color="#FFD700",               # Gold — high earner

        wealth_range=(200_000.0, 1_000_000.0),
        income_range=(150_000.0, 300_000.0),
        income_variance=0.10,
        is_seasonal_income=False,

        expenses={
            "luxury_housing":  40_000.0,
            "lifestyle":       30_000.0,
            "insurance":        8_000.0,
            "food":            10_000.0,
            "investments":     20_000.0,      # Regular SIP/mutual fund
        },

        home_zone="residential_premium",
        work_zone="business_district",
        move_speed=1.3,                     # Rich agents move faster

        state_dim=24,
        type_specific_features=["emi_burden", "job_security", "promotion_progress"],

        reward_weights={
            "w1_wealth_gain":      0.50,
            "w2_wealth_loss":      0.40,
            "w3_consumption":      0.25,
            "w4_savings_security": 0.20,
            "w5_debt_stress":      0.30,
            "w6_housing_stress":   0.15,
            "w7_health_stress":    0.10,
        },

        life_event_probs={
            "stock_windfall":      0.001,
            "corporate_scandal":   0.0005,
            "bonus_cut":           0.001,
            "medical_emergency":   0.001,
        },

        base_tax_rate=0.30,                 # 30% highest bracket
    ),

    # ──────────────────────────────────────
    # 7. YOUNG PROFESSIONAL — 8 agents
    # ──────────────────────────────────────
    AgentType.YOUNG_PROFESSIONAL: AgentTypeConfig(
        agent_type=AgentType.YOUNG_PROFESSIONAL,
        count=8,
        body_color="#00CED1",               # Cyan — fresh energy

        wealth_range=(10_000.0, 50_000.0),
        income_range=(40_000.0, 80_000.0),
        income_variance=0.10,
        is_seasonal_income=False,

        expenses={
            "rent":       12_000.0,
            "loan_emi":    8_000.0,          # Education loan repayment
            "lifestyle":  10_000.0,
            "food":        5_000.0,
            "transport":   3_000.0,
        },

        home_zone="residential_mid_high",
        work_zone="business_district",
        move_speed=1.1,

        state_dim=24,
        type_specific_features=["emi_burden", "job_security", "promotion_progress"],

        reward_weights={
            "w1_wealth_gain":      0.40,
            "w2_wealth_loss":      0.45,
            "w3_consumption":      0.35,
            "w4_savings_security": 0.20,
            "w5_debt_stress":      0.40,
            "w6_housing_stress":   0.35,
            "w7_health_stress":    0.10,
        },

        life_event_probs={
            "promotion":           0.001,
            "startup_opportunity": 0.0005,
            "job_switch":          0.001,
            "medical_emergency":   0.0005,
        },

        base_tax_rate=0.20,
    ),

    # ──────────────────────────────────────
    # 8. GOVT EMPLOYEE — 8 agents
    # ──────────────────────────────────────
    AgentType.GOVT_EMPLOYEE: AgentTypeConfig(
        agent_type=AgentType.GOVT_EMPLOYEE,
        count=8,
        body_color="#00008B",               # Dark blue — government authority

        wealth_range=(40_000.0, 120_000.0),
        income_range=(45_000.0, 70_000.0),
        income_variance=0.03,               # Very stable — government salary
        is_seasonal_income=False,

        expenses={
            "housing":      8_000.0,         # Often subsidized quarters
            "family":      10_000.0,
            "food":         6_000.0,
            "insurance":    2_000.0,
            "transport":    2_000.0,
        },

        home_zone="residential_stable_mid",
        work_zone="government",
        move_speed=0.9,

        state_dim=24,
        type_specific_features=["emi_burden", "job_security", "promotion_progress"],

        reward_weights={
            "w1_wealth_gain":      0.30,
            "w2_wealth_loss":      0.35,
            "w3_consumption":      0.30,
            "w4_savings_security": 0.40,
            "w5_debt_stress":      0.30,
            "w6_housing_stress":   0.25,
            "w7_health_stress":    0.15,
        },

        life_event_probs={
            "da_hike":             0.001,
            "transfer":            0.0005,
            "pension_reform":      0.0003,
            "medical_emergency":   0.001,
        },

        base_tax_rate=0.20,
    ),

    # ──────────────────────────────────────
    # 9. SENIOR / RETIRED — 8 agents
    # ──────────────────────────────────────
    AgentType.SENIOR: AgentTypeConfig(
        agent_type=AgentType.SENIOR,
        count=8,
        body_color="#808080",               # Gray — wisdom and age

        wealth_range=(50_000.0, 300_000.0), # Lifetime savings
        income_range=(8_000.0, 20_000.0),   # Pension / interest only
        income_variance=0.05,
        is_seasonal_income=False,

        expenses={
            "healthcare":   8_000.0,
            "food":         5_000.0,
            "medicine":     3_000.0,
            "utilities":    2_000.0,
        },

        home_zone="residential_quiet",
        work_zone="none",                   # No work — visits bank/hospital/market
        move_speed=0.4,                     # Seniors move slowly

        state_dim=24,                       # 21 shared + 3 type-specific
        type_specific_features=["health_multiplier", "pension_stable", "medical_due"],

        reward_weights={
            "w1_wealth_gain":      0.20,
            "w2_wealth_loss":      0.30,
            "w3_consumption":      0.40,
            "w4_savings_security": 0.35,
            "w5_debt_stress":      0.20,
            "w6_housing_stress":   0.20,
            "w7_health_stress":    0.50,     # Healthcare is #1 concern
        },

        life_event_probs={
            "health_crisis":       0.005,    # 5x higher than others
            "pension_increase":    0.0005,
            "family_support":      0.002,
            "medical_emergency":   0.005,
        },

        base_tax_rate=0.05,                 # Seniors get tax benefits
    ),

    # ──────────────────────────────────────
    # 10. UNEMPLOYED — 6 agents
    # ──────────────────────────────────────
    AgentType.UNEMPLOYED: AgentTypeConfig(
        agent_type=AgentType.UNEMPLOYED,
        count=6,
        body_color="#DC143C",               # Crimson — urgency and struggle

        wealth_range=(500.0, 2_000.0),
        income_range=(0.0, 0.0),            # No income unless welfare active
        income_variance=0.0,
        is_seasonal_income=False,

        expenses={
            "survival":     3_000.0,         # Bare minimum: food + shelter
        },

        home_zone="residential_cheapest",
        work_zone="government",             # Wanders near govt building for welfare
        move_speed=0.8,

        state_dim=24,                       # 21 shared + 3 type-specific
        type_specific_features=["job_search_score", "welfare_active", "skills_level"],

        reward_weights={
            "w1_wealth_gain":      0.20,
            "w2_wealth_loss":      0.40,
            "w3_consumption":      0.55,     # Consumption is primary need
            "w4_savings_security": 0.25,
            "w5_debt_stress":      0.30,
            "w6_housing_stress":   0.20,
            "w7_health_stress":    0.15,
        },

        life_event_probs={
            "job_found":           0.003,
            "welfare_cut":         0.001,
            "skill_training":      0.002,
            "medical_emergency":   0.001,
            "family_remittance":   0.003,
        },

        base_tax_rate=0.0,
    ),
}

# Verify total agent count sums to 100
TOTAL_AGENTS: int = sum(cfg.count for cfg in AGENT_CONFIGS.values())
assert TOTAL_AGENTS == 100, f"Expected 100 agents, got {TOTAL_AGENTS}"


# ═══════════════════════════════════════════
# ACTIONS
# ═══════════════════════════════════════════

class Action(IntEnum):
    """
    Discrete action space for all agents.
    Each step, an agent picks exactly one action.
    """
    SAVE    = 0   # Save a fraction of income (reduce spending, grow savings)
    SPEND   = 1   # Consume goods/services (utility reward but wealth drain)
    INVEST  = 2   # Put money into market (risk/reward, delayed returns)
    TRADE   = 3   # Engage in trade/work activity (income generation)

NUM_ACTIONS: int = 4

# Action labels for UI display
ACTION_NAMES: Dict[int, str] = {
    Action.SAVE:   "Save",
    Action.SPEND:  "Spend",
    Action.INVEST: "Invest",
    Action.TRADE:  "Trade",
}

ACTION_EMOJIS: Dict[int, str] = {
    Action.SAVE:   "💾",
    Action.SPEND:  "💸",
    Action.INVEST: "📈",
    Action.TRADE:  "🤝",
}


# ═══════════════════════════════════════════
# WEALTH TIERS
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class WealthTier:
    """Immutable wealth tier definition with color and threshold."""
    name: str
    min_wealth: float
    color: str

WEALTH_TIERS: List[WealthTier] = [
    WealthTier(name="bankrupt",  min_wealth=float("-inf"), color="#212121"),  # Black
    WealthTier(name="poor",      min_wealth=0.0,           color="#D50000"),  # Red
    WealthTier(name="low",       min_wealth=5_000.0,       color="#FF6D00"),  # Orange
    WealthTier(name="middle",    min_wealth=20_000.0,      color="#FFD600"),  # Yellow
    WealthTier(name="rich",      min_wealth=100_000.0,     color="#00C853"),  # Green
]

# Thresholds for quick lookup
BANKRUPTCY_THRESHOLD: float = 500.0      # Wealth below this = bankrupt
POVERTY_THRESHOLD: float = 5_000.0       # Below this = "poor"
SURVIVAL_THRESHOLD: float = 500.0        # Below this = can't meet basic needs


# ═══════════════════════════════════════════
# EMOTION ICONS
# ═══════════════════════════════════════════

EMOTION_ICONS: Dict[str, str] = {
    "happy":               "😊",    # Positive reward this step
    "sad":                 "😟",    # Negative reward this step
    "spending":            "💸",    # Currently spending
    "investing":           "📈",    # Currently investing
    "income":              "💰",    # Just received income
    "danger":              "🚨",    # Wealth below survival threshold
    "social":              "🤝",    # In social interaction
    "life_event":          "⚡",    # Life event just occurred
    "medical":             "🏥",    # Medical emergency
    "shock":               "😱",    # Job loss
    "celebration":         "🎉",    # Promotion / good event
    "remittance":          "💌",    # Family remittance received
}


# ═══════════════════════════════════════════
# PPO HYPERPARAMETERS
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class PPOConfig:
    """
    Proximal Policy Optimization hyperparameters.
    These values are standard for discrete-action PPO on small networks.
    """
    gamma: float           = 0.99    # Discount factor for future rewards
    gae_lambda: float      = 0.95    # GAE lambda for advantage estimation
    clip_epsilon: float    = 0.2     # PPO clipping range
    learning_rate: float   = 3e-4    # Adam optimizer learning rate
    update_epochs: int     = 4       # Gradient epochs per PPO update
    minibatch_size: int    = 32      # Minibatch size for PPO update
    rollout_length: int    = 128     # Steps collected before each update
    entropy_coef: float    = 0.01    # Entropy bonus coefficient (exploration)
    value_loss_coef: float = 0.5     # Value function loss weight
    max_grad_norm: float   = 0.5     # Gradient clipping norm

    # Network architecture
    hidden_size: int       = 64      # Hidden layer size for Actor & Critic
    num_hidden_layers: int = 2       # Number of hidden layers

PPO_CONFIG = PPOConfig()


# ═══════════════════════════════════════════
# CITY ZONE DEFINITIONS
# ═══════════════════════════════════════════

@dataclass
class ZoneConfig:
    """
    Defines a rectangular zone on the city canvas.
    Coordinates are (x1, y1, x2, y2) pixel boundaries.
    """
    name: str
    label: str                           # Display name
    bounds: Tuple[int, int, int, int]    # (x1, y1, x2, y2) — top-left to bottom-right
    color: str                           # Zone background color (subtle tint)
    building_count: int                  # Number of buildings to generate
    waypoints: List[Tuple[int, int]]     # Navigation waypoints within zone


# City zones — 1200x900 canvas
# Layout:
#   ┌──────────────┬──────────────┐
#   │ RESIDENTIAL  │  COMMERCIAL  │
#   │  (top-left)  │ (top-right)  │
#   ├──────┬───────┼──────────────┤
#   │ EDUC │       │              │
#   │      │INVEST │   BUSINESS   │
#   │ HOSP │(center)│  DISTRICT   │
#   ├──────┴───────┼──────────────┤
#   │  GOVERNMENT  │   BUSINESS   │
#   │ (bottom-left)│(bottom-right)│
#   └──────────────┴──────────────┘

CITY_ZONES: Dict[str, ZoneConfig] = {
    # --- RESIDENTIAL ZONE (top-left quadrant) ---
    "residential": ZoneConfig(
        name="residential",
        label="Residential Zone",
        bounds=(0, 0, 600, 450),
        color="#1a2a1a",                   # Dark green tint — housing area
        building_count=25,
        waypoints=[
            # Poor housing (bottom of zone)
            (80, 380), (180, 400), (280, 390), (400, 410),
            # Mid housing (center)
            (100, 250), (200, 260), (320, 240), (480, 260),
            # Rich housing (top)
            (120, 80), (250, 100), (400, 90), (520, 100),
            # Connectors to roads
            (580, 200), (580, 350), (300, 440),
        ],
    ),

    # Sub-zones within residential for agent home assignments
    "residential_south": ZoneConfig(
        name="residential_south",
        label="Residential South (Farmers)",
        bounds=(20, 340, 580, 440),
        color="#1a2a1a",
        building_count=0,                  # Parent zone handles buildings
        waypoints=[(80, 380), (180, 400), (280, 390), (400, 410)],
    ),
    "residential_cheap": ZoneConfig(
        name="residential_cheap",
        label="Residential Cheap",
        bounds=(20, 300, 300, 440),
        color="#1a2a1a",
        building_count=0,
        waypoints=[(80, 340), (150, 360), (250, 350)],
    ),
    "residential_cheapest": ZoneConfig(
        name="residential_cheapest",
        label="Residential Cheapest",
        bounds=(20, 380, 200, 440),
        color="#1a2a1a",
        building_count=0,
        waypoints=[(60, 410), (120, 420), (180, 400)],
    ),
    "residential_shared": ZoneConfig(
        name="residential_shared",
        label="Residential Shared (Students)",
        bounds=(20, 280, 250, 340),
        color="#1a2a1a",
        building_count=0,
        waypoints=[(80, 300), (160, 310), (220, 300)],
    ),
    "residential_mid": ZoneConfig(
        name="residential_mid",
        label="Residential Mid",
        bounds=(100, 180, 500, 280),
        color="#1a2a1a",
        building_count=0,
        waypoints=[(150, 220), (280, 240), (420, 230)],
    ),
    "residential_mid_high": ZoneConfig(
        name="residential_mid_high",
        label="Residential Mid-High",
        bounds=(250, 100, 500, 200),
        color="#1a2a1a",
        building_count=0,
        waypoints=[(300, 150), (400, 140), (470, 160)],
    ),
    "residential_premium": ZoneConfig(
        name="residential_premium",
        label="Residential Premium",
        bounds=(80, 30, 350, 120),
        color="#1a2a1a",
        building_count=0,
        waypoints=[(120, 70), (220, 80), (300, 60)],
    ),
    "residential_stable_mid": ZoneConfig(
        name="residential_stable_mid",
        label="Residential Stable Mid (Govt)",
        bounds=(350, 130, 580, 230),
        color="#1a2a1a",
        building_count=0,
        waypoints=[(400, 170), (500, 180), (550, 160)],
    ),
    "residential_quiet": ZoneConfig(
        name="residential_quiet",
        label="Residential Quiet (Seniors)",
        bounds=(400, 30, 580, 130),
        color="#1a2a1a",
        building_count=0,
        waypoints=[(440, 70), (520, 80), (560, 60)],
    ),

    # --- COMMERCIAL ZONE (top-right quadrant) ---
    "commercial": ZoneConfig(
        name="commercial",
        label="Commercial Zone",
        bounds=(620, 0, 1200, 350),
        color="#2a1a1a",                   # Dark red tint — market energy
        building_count=20,
        waypoints=[
            (680, 60), (800, 80), (950, 70), (1100, 90),
            (700, 180), (850, 200), (1000, 190), (1130, 180),
            (720, 300), (880, 280), (1050, 310), (1150, 290),
            # Connectors
            (630, 170), (630, 300), (900, 340),
        ],
    ),

    # --- BUSINESS DISTRICT (bottom-right quadrant) ---
    "business_district": ZoneConfig(
        name="business_district",
        label="Business District",
        bounds=(620, 370, 1200, 900),
        color="#1a1a2a",                   # Dark blue tint — corporate
        building_count=18,
        waypoints=[
            (680, 420), (840, 400), (1000, 430), (1130, 410),
            (700, 550), (860, 530), (1020, 560), (1140, 540),
            (720, 700), (880, 680), (1040, 720), (1150, 690),
            (700, 830), (900, 850), (1100, 840),
            # Connectors
            (630, 500), (630, 700), (900, 380),
        ],
    ),

    # --- INVESTMENT ZONE (center) ---
    "investment": ZoneConfig(
        name="investment",
        label="Investment Zone",
        bounds=(420, 370, 600, 550),
        color="#1a1a20",                   # Dark — financial
        building_count=5,
        waypoints=[
            (470, 410), (540, 430),
            (480, 480), (550, 500),
            (510, 540),
        ],
    ),

    # --- GOVERNMENT BUILDING (bottom-left) ---
    "government": ZoneConfig(
        name="government",
        label="Government Building",
        bounds=(0, 560, 400, 900),
        color="#1a201a",                   # Dark greenish — institutional
        building_count=8,
        waypoints=[
            (60, 620), (200, 600), (340, 630),
            (80, 730), (220, 710), (350, 740),
            (100, 840), (250, 860), (370, 830),
            # Connectors
            (200, 570), (390, 700),
        ],
    ),

    # --- HOSPITAL (top-center) ---
    "hospital": ZoneConfig(
        name="hospital",
        label="Hospital",
        bounds=(420, 0, 600, 130),
        color="#1a2020",                   # Dark teal — medical
        building_count=3,
        waypoints=[
            (470, 40), (540, 60), (560, 110),
            (450, 120),
        ],
    ),

    # --- EDUCATION ZONE (left-center) ---
    "education": ZoneConfig(
        name="education",
        label="Education Zone",
        bounds=(0, 460, 400, 550),
        color="#1a1a28",                   # Dark blue — academic
        building_count=5,
        waypoints=[
            (60, 480), (180, 490), (300, 480), (380, 500),
            (200, 540),
        ],
    ),

    # --- OUTSKIRTS (for farmers — seasonal work area) ---
    "outskirts": ZoneConfig(
        name="outskirts",
        label="Outskirts / Farmland",
        bounds=(0, 0, 80, 450),
        color="#0d1a0d",                   # Very dark green — fields
        building_count=0,
        waypoints=[
            (30, 60), (40, 180), (30, 300), (50, 420),
        ],
    ),
}


# ═══════════════════════════════════════════
# ROAD NETWORK (connecting zones)
# ═══════════════════════════════════════════

# Roads are defined as (start_waypoint, end_waypoint) line segments.
# Agents pathfind along these waypoints between zones.
# This is a simple waypoint graph, not full A*.

ROAD_WAYPOINTS: List[Tuple[int, int]] = [
    # Main horizontal roads
    (0, 450), (200, 450), (400, 450), (600, 450), (800, 450), (1000, 450), (1200, 450),
    (0, 350), (200, 350), (400, 350), (600, 350),
    (0, 560), (200, 560), (400, 560), (600, 560), (800, 560), (1000, 560), (1200, 560),

    # Main vertical roads
    (600, 0), (600, 200), (600, 350), (600, 450), (600, 560), (600, 700), (600, 900),
    (400, 0), (400, 200), (400, 350), (400, 450), (400, 560), (400, 700), (400, 900),

    # Cross connectors
    (200, 0), (200, 200), (200, 350), (200, 450), (200, 560), (200, 700), (200, 900),
    (800, 0), (800, 200), (800, 350), (800, 560), (800, 700), (800, 900),
    (1000, 0), (1000, 200), (1000, 350), (1000, 560), (1000, 700), (1000, 900),
]

# Road adjacency — each waypoint connects to nearby waypoints
# Built dynamically in city.py based on proximity threshold
ROAD_PROXIMITY_THRESHOLD: int = 220  # Max pixels between connected waypoints

# Road visual properties
ROAD_WIDTH: int = 6
ROAD_COLOR: str = "#2a2a2a"
ROAD_LINE_COLOR: str = "#3a3a3a"     # Dashed center line


# ═══════════════════════════════════════════
# BUILDING DEFINITIONS
# ═══════════════════════════════════════════

@dataclass
class BuildingConfig:
    """Template for procedurally generated buildings within zones."""
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    color: str
    label: Optional[str] = None

# Building templates per zone type
BUILDING_TEMPLATES: Dict[str, BuildingConfig] = {
    "residential_poor":    BuildingConfig(15, 25, 15, 20, "#3d2b1f", "Home"),
    "residential_mid":     BuildingConfig(25, 40, 20, 35, "#4a3728", "Home"),
    "residential_rich":    BuildingConfig(40, 60, 35, 50, "#5c4033", "Villa"),
    "commercial_shop":     BuildingConfig(20, 35, 20, 30, "#4a1a2a", "Shop"),
    "commercial_market":   BuildingConfig(40, 60, 30, 45, "#5a2a3a", "Market"),
    "business_office":     BuildingConfig(35, 55, 30, 50, "#1a1a4a", "Office"),
    "business_factory":    BuildingConfig(50, 70, 35, 55, "#2a2a5a", "Factory"),
    "investment_bank":     BuildingConfig(45, 60, 35, 50, "#1a1a30", "Bank"),
    "government_building": BuildingConfig(50, 70, 40, 60, "#1a3a1a", "Govt"),
    "hospital_building":   BuildingConfig(50, 65, 40, 55, "#1a3a3a", "Hospital"),
    "education_building":  BuildingConfig(40, 55, 30, 45, "#1a1a3a", "School"),
}

# Seed for reproducible city generation
CITY_GENERATION_SEED: int = 42


# ═══════════════════════════════════════════
# ECONOMIC DEFAULTS
# ═══════════════════════════════════════════

@dataclass
class EconomicsConfig:
    """
    Global economic variables. All values are monthly rates
    unless otherwise specified. Policies modify these at runtime.
    """
    inflation_rate: float           = 0.03     # 3% per month
    base_interest_rate: float       = 0.0067   # 8% per year → 0.67% per month
    housing_cost_index: float       = 1.0      # Multiplier on housing expenses
    employment_availability: float  = 0.75     # 0.0–1.0 job market score
    market_return_rate: float       = 0.05     # 5% monthly return on investments
    credit_availability: float     = 0.70     # 0.0–1.0 loan access score
    healthcare_cost_index: float   = 1.0      # Multiplier on healthcare expenses
    education_cost_index: float    = 1.0      # Multiplier on education expenses

    # Savings interest = base_interest_rate * savings_rate_fraction
    savings_rate_fraction: float   = 0.4      # Savings earn 40% of base rate

    # Loan default trigger
    loan_default_months: int       = 3        # Consecutive months debt > 3x income
    loan_default_multiplier: float = 3.0      # Debt-to-income threshold

ECONOMICS_CONFIG = EconomicsConfig()


# ═══════════════════════════════════════════
# SOCIAL INTERACTION PARAMETERS
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class SocialConfig:
    """Parameters for the proximity-based social interaction system."""
    interaction_radius: float       = 40.0     # Pixels — must be within this to interact
    max_relationships: int          = 10       # Top N relationships stored per agent

    # Interaction probabilities (per step, per qualifying pair)
    prob_job_tip: float             = 0.15
    prob_borrow_money: float        = 0.10
    prob_spending_influence: float  = 0.05
    prob_economic_gossip: float     = 0.20

    # Relationship dynamics
    relationship_gain: float        = 0.05     # Score increase per interaction
    relationship_decay: float       = 0.001    # Score decay per step (natural fade)
    high_relationship_threshold: float = 0.7   # Threshold for bonus effects

    # Borrow money parameters (₹)
    borrow_min: float               = 2_000.0
    borrow_max: float               = 5_000.0
    lender_min_wealth: float        = 20_000.0  # Minimum wealth to be a lender
    borrower_max_wealth: float      = 2_000.0   # Maximum wealth to request loan
    repay_wealth_multiplier: float  = 2.0       # Repay when wealth > 2x loan

    # Job tip effect
    job_tip_boost: float            = 0.2      # Employment availability boost
    job_tip_duration: int           = 5        # Steps the boost lasts

    # Spending influence effect
    spending_influence_boost: float = 0.1      # Spend probability increase

    # Economic gossip (inflation signal blending)
    gossip_blend_factor: float      = 0.1      # How much to blend sender's observed rate

SOCIAL_CONFIG = SocialConfig()


# ═══════════════════════════════════════════
# REWARD FUNCTION PARAMETERS
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class RewardConfig:
    """Fixed reward penalties/bonuses not varying by agent type."""
    bankruptcy_penalty: float            = -10.0
    missed_obligation_penalty: float     = -2.0
    social_interaction_bonus: float      = 1.0
    survival_threshold: float            = 500.0  # Below this → danger state

REWARD_CONFIG = RewardConfig()


# ═══════════════════════════════════════════
# ACTION PARAMETERS
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class ActionConfig:
    """Parameters controlling action effects."""
    save_fraction: float    = 0.30    # Fraction of available cash saved
    spend_fraction: float   = 0.20    # Fraction of wealth spent on consumption
    invest_fraction: float  = 0.15    # Fraction of wealth invested
    trade_income_boost: float = 0.10  # Extra income fraction from active trading

ACTION_CONFIG = ActionConfig()


# ═══════════════════════════════════════════
# MOVEMENT PARAMETERS
# ═══════════════════════════════════════════

LERP_FACTOR: float = 0.08            # Position interpolation speed per frame
WALKING_ANIM_INTERVAL: int = 400     # Milliseconds between walk animation frames
BANKRUPT_SPEED_MULT: float = 0.3     # Bankrupt agents move at 30% speed


# ═══════════════════════════════════════════
# SEASONAL CALENDAR (for Farmers)
# ═══════════════════════════════════════════

class Season(IntEnum):
    """Indian agricultural seasons mapped to simulation months."""
    KHARIF_SOWING   = 0   # June–July      (monsoon sowing)
    KHARIF_GROWING  = 1   # Aug–Oct        (growing season)
    KHARIF_HARVEST  = 2   # Nov–Dec        (harvest — high income)
    RABI_SOWING     = 3   # Dec–Jan        (winter sowing)
    RABI_GROWING    = 4   # Feb–Mar        (growing)
    RABI_HARVEST    = 5   # Apr–May        (harvest — high income)

# Map month-of-year (0–11) to season
MONTH_TO_SEASON: Dict[int, Season] = {
    0: Season.RABI_SOWING,     # January
    1: Season.RABI_GROWING,    # February
    2: Season.RABI_GROWING,    # March
    3: Season.RABI_HARVEST,    # April
    4: Season.RABI_HARVEST,    # May
    5: Season.KHARIF_SOWING,   # June
    6: Season.KHARIF_SOWING,   # July
    7: Season.KHARIF_GROWING,  # August
    8: Season.KHARIF_GROWING,  # September
    9: Season.KHARIF_GROWING,  # October
    10: Season.KHARIF_HARVEST, # November
    11: Season.KHARIF_HARVEST, # December
}

HARVEST_SEASONS: List[Season] = [Season.KHARIF_HARVEST, Season.RABI_HARVEST]
MONSOON_MONTHS: List[int] = [5, 6, 7, 8]  # June–September (crop failure risk)


# ═══════════════════════════════════════════
# POLICY ENGINE PARAMETERS
# ═══════════════════════════════════════════

@dataclass
class PolicyConfig:
    """Configuration for the Gemini-powered policy engine."""
    gemini_model: str = "gemini-3.5-flash"
    max_retries: int = 3
    timeout_seconds: int = 30

    # Cache settings
    cache_file: str = "data/policy_cache.json"
    fallback_file: str = "data/fallback_policies.json"

    # Modifiable agent parameters (whitelist for safety)
    modifiable_agent_params: List[str] = field(default_factory=lambda: [
        "income_multiplier",
        "expense_multiplier",
        "tax_rate",
        "investment_return_rate",
        "savings_interest_rate",
        "debt_interest_rate",
        "welfare_payment",
        "trade_allowed",
        "invest_allowed",
        "cash_freeze_fraction",
        "healthcare_subsidy",
        "education_subsidy",
    ])

    # Modifiable global parameters (whitelist for safety)
    modifiable_global_params: List[str] = field(default_factory=lambda: [
        "global_inflation_rate",
        "market_return_multiplier",
        "credit_availability",
        "employment_availability",
        "housing_cost_index",
        "healthcare_cost_index",
        "education_cost_index",
    ])

POLICY_CONFIG = PolicyConfig()

# System prompt for Gemini policy parsing (verbatim from spec)
GEMINI_SYSTEM_PROMPT: str = """
You are an economic policy parameter interpreter for an
agent-based simulation of Indian society with 10 agent types:
farmer, gig_worker, student, small_biz_owner, salaried_mid,
salaried_high, young_professional, govt_employee, senior, unemployed

Modifiable agent parameters:
  income_multiplier       (float, default 1.0)
  expense_multiplier      (float, default 1.0)
  tax_rate                (float, 0.0–0.9)
  investment_return_rate  (float, default 0.05)
  savings_interest_rate   (float, default 0.02)
  debt_interest_rate      (float, default 0.08)
  welfare_payment         (float, INR per month, default 0)
  trade_allowed           (bool)
  invest_allowed          (bool)
  cash_freeze_fraction    (float, 0.0–1.0)
  healthcare_subsidy      (float, 0.0–1.0)
  education_subsidy       (float, 0.0–1.0)

Modifiable global parameters:
  global_inflation_rate
  market_return_multiplier
  credit_availability
  employment_availability
  housing_cost_index
  healthcare_cost_index
  education_cost_index

Given a government policy in plain English, output ONLY valid JSON:
{
  "policy_name": "...",
  "policy_description": "...",
  "duration_steps": <int, -1 = permanent>,
  "affected_agents": { "<type>": { "<param>": <value> } },
  "global_effects": { "<param>": <value> },
  "reasoning": "..."
}

Rules:
- Only include actually affected agent types and parameters
- Be economically realistic for India (INR values, Indian context)
- Be honest about which agents are hurt vs helped
- Output ONLY the JSON. No markdown. No explanation. No preamble.
""".strip()


# ═══════════════════════════════════════════
# COMPARISON MODE PARAMETERS
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for scenario comparison mode."""
    default_comparison_steps: int = 90    # Default: 3 months comparison
    available_durations: List[int] = field(default_factory=lambda: [30, 90, 365])

    # Policy effectiveness score weights (must sum to 1.0)
    weight_gini: float         = 0.30
    weight_poverty: float      = 0.25
    weight_mobility: float     = 0.20
    weight_bankruptcy: float   = 0.15
    weight_median_wealth: float = 0.10

COMPARISON_CONFIG = ComparisonConfig()


# ═══════════════════════════════════════════
# METRICS THRESHOLDS
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for the 7 research metrics."""
    # Social mobility lookback window (in timesteps)
    mobility_window: int = 30

    # Wealth tier boundaries for mobility tracking
    tier_boundaries: List[float] = field(default_factory=lambda: [
        5_000.0,     # Below = poor
        20_000.0,    # Below = low-middle
        100_000.0,   # Below = middle
        # Above = rich
    ])

    # Top N agents for concentration metric
    top_n_concentration: int = 10

    # CSV export default path
    export_path: str = "data/metrics_export.csv"

METRICS_CONFIG = MetricsConfig()


# ═══════════════════════════════════════════
# WEBSOCKET PROTOCOL
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class WebSocketConfig:
    """WebSocket communication parameters."""
    host: str = "0.0.0.0"
    port: int = 8000
    diff_send_rate: int = 10               # Diffs per second
    full_state_on_connect: bool = True     # Send full state on new connection
    compress_diffs: bool = True            # Only send changed agent data

WEBSOCKET_CONFIG = WebSocketConfig()


# ═══════════════════════════════════════════
# STATE VECTOR NORMALIZATION CONSTANTS
# ═══════════════════════════════════════════

# Used to normalize raw values into [0, 1] range for neural network input
NORM_WEALTH_MAX: float = 1_000_000.0       # ₹10 lakh max for normalization
NORM_INCOME_MAX: float = 50_000.0          # ₹50K monthly income max
NORM_SAVINGS_MAX: float = 1_000_000.0      # ₹10 lakh savings max
NORM_DEBT_MAX: float = 500_000.0           # ₹5 lakh debt max

# Number of active policy slots in state vector
NUM_POLICY_SLOTS: int = 6

# Shared state features count (before type-specific features)
# wealth, income, savings, debt, expense_ratio, inflation, interest,
# housing_cost, employment_avail, wealth_tier, month, day, reputation, skill, influence, policy_slots[6]
SHARED_STATE_DIM: int = 15 + NUM_POLICY_SLOTS  # 21 features


# ═══════════════════════════════════════════
# FRONTEND AESTHETIC CONSTANTS
# ═══════════════════════════════════════════
# (Passed to frontend via full_state message)

UI_THEME = {
    "background": "#0d0d0d",
    "accent_primary": "#00ff88",
    "accent_secondary": "#00bfff",
    "panel_bg": "#1a1a2e",
    "panel_border": "#16213e",
    "text_primary": "#e0e0e0",
    "text_secondary": "#888888",
    "text_accent": "#00ff88",
    "danger": "#ff4444",
    "warning": "#ffaa00",
    "success": "#00ff88",
    "chart_colors": [
        "#00ff88", "#00bfff", "#ff6b6b", "#ffd93d",
        "#6c5ce7", "#fd79a8", "#00cec9", "#e17055",
    ],
}


# ═══════════════════════════════════════════
# LOGGING & DEBUG
# ═══════════════════════════════════════════

LOG_LEVEL: str = "INFO"
LOG_EVENTS: bool = True                    # Log life events to console
LOG_POLICIES: bool = True                  # Log policy applications
LOG_PPO_UPDATES: bool = False              # Log PPO loss values (verbose)
LOG_SOCIAL: bool = False                   # Log social interactions (verbose)

# Event log max display length (frontend)
EVENT_LOG_MAX_ENTRIES: int = 100

FASTAPI_HOST: str = "127.0.0.1"
FASTAPI_PORT: int = 8000
ALLOWED_ORIGINS: List[str] = ["*"]
