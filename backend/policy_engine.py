"""
policy_engine.py — Gemini-powered policy parsing engine for MARL City Simulator.

Gemini 2.5 Flash is used ONLY for natural language policy parsing.
Gemini NEVER touches agent decisions or RL in any way.
Gemini's sole job: convert plain English policy text into structured JSON.

Pipeline:
  1. Researcher types policy in plain English
  2. Hash the text → check cache → return if hit
  3. If cache miss, call Gemini API with system prompt
  4. Parse JSON response → PolicyEffect dataclass
  5. Cache the result (disk-persisted)
  6. Apply effects to agents and economics engine

Fallback: If Gemini API is unavailable (no key, timeout, error),
match against fallback_policies.json by keyword similarity.

Author: Aditya Padale (B.Tech Final Year Project)
"""

import os
import json
import hashlib
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from config import (
    PolicyConfig, POLICY_CONFIG,
    GEMINI_SYSTEM_PROMPT,
    AgentType, AGENT_TYPE_NAMES, AGENT_CONFIGS,
    STEPS_PER_MONTH,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# POLICY EFFECT
# ═══════════════════════════════════════════

@dataclass
class PolicyEffect:
    """
    Structured representation of a parsed government policy.
    This is the output of Gemini parsing or fallback matching.
    """
    policy_name: str                          # Short policy title
    policy_description: str                   # Human-readable description
    duration_steps: int                       # -1 = permanent, else N timesteps
    affected_agents: Dict[str, Dict[str, Any]]  # type_name → {param: value}
    global_effects: Dict[str, float]          # global_param → value
    reasoning: str                            # Why this policy has these effects
    applied_at: int = 0                       # Timestep when applied
    expires_at: int = -1                      # -1 = never, else timestep
    source: str = "gemini"                    # "gemini", "cache", or "fallback"
    raw_text: str = ""                        # Original policy text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage and WebSocket transmission."""
        return {
            "policy_name": self.policy_name,
            "policy_description": self.policy_description,
            "duration_steps": self.duration_steps,
            "affected_agents": self.affected_agents,
            "global_effects": self.global_effects,
            "reasoning": self.reasoning,
            "applied_at": self.applied_at,
            "expires_at": self.expires_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_text: str = "") -> "PolicyEffect":
        """Construct from a parsed JSON dict (from Gemini or cache)."""
        return cls(
            policy_name=data.get("policy_name", "Unknown Policy"),
            policy_description=data.get("policy_description", ""),
            duration_steps=data.get("duration_steps", -1),
            affected_agents=data.get("affected_agents", {}),
            global_effects=data.get("global_effects", {}),
            reasoning=data.get("reasoning", ""),
            source=data.get("source", "unknown"),
            raw_text=raw_text,
        )


# ═══════════════════════════════════════════
# POLICY ENGINE
# ═══════════════════════════════════════════

class PolicyEngine:
    """
    Converts natural language policy descriptions into structured
    simulation parameters using Gemini API, with caching and fallback.

    Architecture role:
      - ONLY component that touches an LLM
      - Output is a deterministic PolicyEffect struct
      - RL loop never depends on LLM calls
      - All parsed policies are cached by text hash
      - Offline fallback ensures demo works without API key

    Usage:
        engine = PolicyEngine()
        effect = await engine.interpret_policy("Give UBI of ₹5000 to all poor")
        engine.apply_policy(effect, agents, economics, timestep=100)
    """

    def __init__(self, config: PolicyConfig = POLICY_CONFIG,
                 data_dir: str = "data"):
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # ── Cache ──
        self.cache: Dict[str, Dict] = {}   # hash → parsed policy dict
        self._cache_path = self.data_dir / "policy_cache.json"
        self._load_cache()

        # ── Fallback Policies ──
        self.fallback_policies: List[Dict] = []
        self._fallback_path = self.data_dir / "fallback_policies.json"
        self._load_fallback_policies()

        # ── Active Policies ──
        self.active_policies: Dict[str, PolicyEffect] = {}  # name → effect
        self.policy_history: List[PolicyEffect] = []

        # ── Visual Events (for frontend announcements) ──
        self._pending_events: List[str] = []

        # ── Gemini Client ──
        self._gemini_available = False
        self._gemini_model = None
        self._init_gemini()

    # ─────────────────────────────────────────
    # GEMINI INITIALIZATION
    # ─────────────────────────────────────────

    def _init_gemini(self):
        """Initialize the Gemini API client if API key is available."""
        api_key = os.environ.get("GEMINI_API_KEY", "")

        if not api_key:
            logger.warning(
                "GEMINI_API_KEY not set. Policy engine will use fallback matching only. "
                "Set the environment variable to enable AI-powered policy parsing."
            )
            self._gemini_available = False
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(
                model_name=self.config.gemini_model,
                system_instruction=GEMINI_SYSTEM_PROMPT,
            )
            self._gemini_available = True
            logger.info("Gemini API initialized successfully (model: %s)",
                        self.config.gemini_model)
        except ImportError:
            logger.warning(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
            self._gemini_available = False
        except Exception as e:
            logger.error("Failed to initialize Gemini API: %s", e)
            self._gemini_available = False

    # ─────────────────────────────────────────
    # POLICY INTERPRETATION (MAIN ENTRY POINT)
    # ─────────────────────────────────────────

    async def interpret_policy(self, policy_text: str,
                               sim_state: Dict[str, Any] = None) -> PolicyEffect:
        """
        Convert plain English policy text into a structured PolicyEffect.

        Pipeline:
          1. Normalize and hash the policy text
          2. Check cache → return if hit
          3. Call Gemini API → parse JSON response
          4. If Gemini fails, try fallback keyword matching
          5. Cache the result
          6. Return PolicyEffect

        Args:
            policy_text: Natural language policy description
            sim_state:   Optional current simulation state for context

        Returns:
            PolicyEffect ready to be applied to the simulation
        """
        # Normalize text
        normalized = policy_text.strip().lower()
        policy_hash = self._hash_policy(normalized)

        # ── Step 1: Check cache ──
        if policy_hash in self.cache:
            logger.info("Policy cache hit: '%s'", policy_text[:50])
            cached = self.cache[policy_hash]
            effect = PolicyEffect.from_dict(cached, raw_text=policy_text)
            effect.source = "cache"
            return effect

        # ── Step 2: Try Gemini API ──
        if self._gemini_available:
            try:
                effect = await self._call_gemini(policy_text, sim_state)
                if effect:
                    # Cache the result
                    self.cache[policy_hash] = effect.to_dict()
                    self._save_cache()
                    effect.source = "gemini"
                    logger.info("Policy parsed by Gemini: '%s' → %s",
                                policy_text[:50], effect.policy_name)
                    return effect
            except Exception as e:
                logger.error("Gemini API error: %s. Falling back to keyword matching.", e)

        # ── Step 3: Fallback to keyword matching ──
        effect = self._fallback_match(policy_text)
        if effect:
            # Cache the fallback result too
            self.cache[policy_hash] = effect.to_dict()
            self._save_cache()
            effect.source = "fallback"
            logger.info("Policy matched by fallback: '%s' → %s",
                        policy_text[:50], effect.policy_name)
            return effect

        # ── Step 4: No match at all — return generic placeholder ──
        logger.warning("Could not interpret policy: '%s'", policy_text)
        return PolicyEffect(
            policy_name="Unrecognized Policy",
            policy_description=policy_text,
            duration_steps=-1,
            affected_agents={},
            global_effects={},
            reasoning="Could not parse this policy. No matching fallback found.",
            source="none",
            raw_text=policy_text,
        )

    def interpret_policy_sync(self, policy_text: str,
                               sim_state: Dict[str, Any] = None) -> PolicyEffect:
        """
        Synchronous version of interpret_policy for non-async contexts.
        Uses fallback matching only (no API call).
        """
        normalized = policy_text.strip().lower()
        policy_hash = self._hash_policy(normalized)

        # Check cache
        if policy_hash in self.cache:
            cached = self.cache[policy_hash]
            effect = PolicyEffect.from_dict(cached, raw_text=policy_text)
            effect.source = "cache"
            return effect

        # Fallback only
        effect = self._fallback_match(policy_text)
        if effect:
            self.cache[policy_hash] = effect.to_dict()
            self._save_cache()
            return effect

        return PolicyEffect(
            policy_name="Unrecognized Policy",
            policy_description=policy_text,
            duration_steps=-1,
            affected_agents={},
            global_effects={},
            reasoning="Synchronous mode — fallback matching failed.",
            source="none",
            raw_text=policy_text,
        )

    # ─────────────────────────────────────────
    # GEMINI API CALL
    # ─────────────────────────────────────────

    async def _call_gemini(self, policy_text: str,
                           sim_state: Dict[str, Any] = None) -> Optional[PolicyEffect]:
        """
        Call Gemini API to parse policy text into JSON.

        The system prompt constrains Gemini to output ONLY valid JSON
        with the exact schema defined in config.py.

        Args:
            policy_text: Raw policy text from researcher
            sim_state:   Optional simulation context

        Returns:
            PolicyEffect if successful, None if parsing failed
        """
        import asyncio

        # Build the prompt
        prompt = f"Policy: {policy_text}"

        if sim_state:
            # Add context about current simulation state
            context = (
                f"\nCurrent state context: "
                f"inflation={sim_state.get('inflation_rate', 0.03):.1%}, "
                f"employment={sim_state.get('employment_availability', 0.75):.0%}, "
                f"avg poverty_rate={sim_state.get('poverty_rate', 0.2):.0%}"
            )
            prompt += context

        # Call Gemini (run sync call in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        for attempt in range(self.config.max_retries):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._gemini_model.generate_content(prompt)
                )

                if response and response.text:
                    # Parse JSON from response
                    json_str = self._extract_json(response.text)
                    if json_str:
                        data = json.loads(json_str)
                        return PolicyEffect.from_dict(data, raw_text=policy_text)

                logger.warning("Gemini returned empty/invalid response (attempt %d)",
                               attempt + 1)

            except json.JSONDecodeError as e:
                logger.warning("Failed to parse Gemini JSON (attempt %d): %s",
                               attempt + 1, e)
            except Exception as e:
                logger.warning("Gemini API call failed (attempt %d): %s",
                               attempt + 1, e)

        return None

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """
        Extract JSON from Gemini response text.

        Handles cases where Gemini wraps JSON in markdown code blocks
        or adds extra text despite instructions.
        """
        # Try to find JSON in code blocks
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to find raw JSON (starts with {)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()

        return None

    # ─────────────────────────────────────────
    # FALLBACK MATCHING
    # ─────────────────────────────────────────

    def _fallback_match(self, policy_text: str) -> Optional[PolicyEffect]:
        """
        Match policy text against pre-built fallback policies by keyword similarity.

        Uses simple keyword overlap scoring — not ML, just string matching.
        This guarantees the system works fully offline for demos and vivas.

        Args:
            policy_text: Normalized policy text

        Returns:
            Best matching PolicyEffect, or None if no match above threshold
        """
        if not self.fallback_policies:
            return None

        normalized = policy_text.lower()
        words = set(re.findall(r'\w+', normalized))

        best_score = 0.0
        best_match = None

        for fallback in self.fallback_policies:
            # Score by keyword overlap
            keywords = set(fallback.get("keywords", []))
            if not keywords:
                continue

            # Count matching keywords
            overlap = len(words & keywords)
            # Weighted: exact name match is worth extra
            name_words = set(re.findall(r'\w+', fallback.get("policy_name", "").lower()))
            name_overlap = len(words & name_words)

            score = overlap + name_overlap * 2.0

            # Boost if key policy terms match
            key_terms = fallback.get("key_terms", [])
            for term in key_terms:
                if term.lower() in normalized:
                    score += 3.0

            if score > best_score:
                best_score = score
                best_match = fallback

        # Minimum threshold to avoid random matches
        if best_score >= 2.0 and best_match:
            effect = PolicyEffect.from_dict(best_match, raw_text=policy_text)
            effect.source = "fallback"
            return effect

        return None

    # ─────────────────────────────────────────
    # POLICY APPLICATION
    # ─────────────────────────────────────────

    def apply_policy(self, effect: PolicyEffect, agents: list,
                     economics, timestep: int) -> List[str]:
        """
        Apply a parsed policy to the simulation.

        Modifies agent parameters and global economic variables
        according to the PolicyEffect specification.

        Args:
            effect:    Parsed PolicyEffect to apply
            agents:    List of all Agent objects
            economics: EconomicsEngine instance
            timestep:  Current simulation timestep

        Returns:
            List of visual event strings for the city log
        """
        events = []
        effect.applied_at = timestep

        if effect.duration_steps > 0:
            effect.expires_at = timestep + effect.duration_steps
        else:
            effect.expires_at = -1

        # ── Apply agent-specific effects ──
        type_name_to_enum = {
            "farmer": AgentType.FARMER,
            "gig_worker": AgentType.GIG_WORKER,
            "student": AgentType.STUDENT,
            "small_biz_owner": AgentType.SMALL_BIZ_OWNER,
            "salaried_mid": AgentType.SALARIED_MID,
            "salaried_high": AgentType.SALARIED_HIGH,
            "young_professional": AgentType.YOUNG_PROFESSIONAL,
            "govt_employee": AgentType.GOVT_EMPLOYEE,
            "senior": AgentType.SENIOR,
            "unemployed": AgentType.UNEMPLOYED,
        }

        for type_name, params in effect.affected_agents.items():
            agent_type = type_name_to_enum.get(type_name.lower())
            if agent_type is None:
                logger.warning("Unknown agent type in policy: '%s'", type_name)
                continue

            affected_count = 0
            for agent in agents:
                if agent.agent_type == agent_type:
                    self._apply_agent_params(agent, params)
                    affected_count += 1

                    # Record policy impact in agent history
                    agent.policy_impacts.append({
                        "timestep": timestep,
                        "policy_name": effect.policy_name,
                        "params": params,
                    })

            if affected_count > 0:
                events.append(
                    f"[T{timestep}] POLICY '{effect.policy_name}': "
                    f"Applied to {affected_count} {AGENT_TYPE_NAMES.get(agent_type, type_name)} agents"
                )

        # ── Apply global effects ──
        for param, value in effect.global_effects.items():
            economics.apply_global_effect(
                param=param,
                value=value,
                policy_name=effect.policy_name,
                duration_steps=effect.duration_steps,
                timestep=timestep,
            )
            events.append(
                f"[T{timestep}] POLICY '{effect.policy_name}': "
                f"Global {param} → {value}"
            )

        # ── Update policy slots in agent state vectors ──
        slot_index = len(self.active_policies) % 6  # Circular buffer of 6 slots
        for agent in agents:
            agent.policy_slots[slot_index] = 1.0

        # ── Track active policy ──
        self.active_policies[effect.policy_name] = effect
        self.policy_history.append(effect)

        # ── Visual event ──
        announcement = (
            f"📜 POLICY ANNOUNCED: {effect.policy_name}\n"
            f"   {effect.policy_description}\n"
            f"   Duration: {'Permanent' if effect.duration_steps < 0 else f'{effect.duration_steps} days'}\n"
            f"   Source: {effect.source}"
        )
        self._pending_events.append(announcement)
        events.insert(0, announcement)

        logger.info("Policy applied: %s (source: %s, affects %d type(s), %d global params)",
                     effect.policy_name, effect.source,
                     len(effect.affected_agents), len(effect.global_effects))

        return events

    def _apply_agent_params(self, agent, params: Dict[str, Any]):
        """Apply parameter modifications to a single agent."""
        f = agent.finance

        for param, value in params.items():
            if param == "income_multiplier":
                f.income_multiplier = float(value)
            elif param == "expense_multiplier":
                f.expense_multiplier = float(value)
            elif param == "tax_rate":
                f.tax_rate = float(value)
            elif param == "investment_return_rate":
                f.investment_return_rate = float(value)
            elif param == "savings_interest_rate":
                f.savings_interest_rate = float(value)
            elif param == "debt_interest_rate":
                f.debt_interest_rate = float(value)
            elif param == "welfare_payment":
                f.welfare_payment = float(value)
                # Also activate welfare status for unemployed
                if agent.agent_type == AgentType.UNEMPLOYED and float(value) > 0:
                    agent.type_state.welfare_active = 1.0
            elif param == "trade_allowed":
                f.trade_allowed = bool(value)
            elif param == "invest_allowed":
                f.invest_allowed = bool(value)
            elif param == "cash_freeze_fraction":
                f.cash_freeze_fraction = float(value)
            elif param == "healthcare_subsidy":
                f.healthcare_subsidy = float(value)
            elif param == "education_subsidy":
                f.education_subsidy = float(value)
            else:
                logger.warning("Unknown agent parameter: '%s' = %s", param, value)

    # ─────────────────────────────────────────
    # POLICY EXPIRY
    # ─────────────────────────────────────────

    def expire_policies(self, timestep: int, agents: list,
                        economics) -> List[str]:
        """
        Check and expire time-limited policies.

        When a policy expires:
          - Agent parameters revert to defaults
          - Global effects are expired via economics engine
          - Policy slots in state vectors are cleared

        Args:
            timestep:  Current simulation timestep
            agents:    All agents
            economics: EconomicsEngine

        Returns:
            List of expiry event strings
        """
        events = []
        expired_names = []

        for name, effect in self.active_policies.items():
            if effect.expires_at > 0 and timestep >= effect.expires_at:
                # Revert agent-specific effects
                type_name_to_enum = {
                    "farmer": AgentType.FARMER,
                    "gig_worker": AgentType.GIG_WORKER,
                    "student": AgentType.STUDENT,
                    "small_biz_owner": AgentType.SMALL_BIZ_OWNER,
                    "salaried_mid": AgentType.SALARIED_MID,
                    "salaried_high": AgentType.SALARIED_HIGH,
                    "young_professional": AgentType.YOUNG_PROFESSIONAL,
                    "govt_employee": AgentType.GOVT_EMPLOYEE,
                    "senior": AgentType.SENIOR,
                    "unemployed": AgentType.UNEMPLOYED,
                }

                for type_name, params in effect.affected_agents.items():
                    agent_type = type_name_to_enum.get(type_name.lower())
                    if agent_type is None:
                        continue

                    for agent in agents:
                        if agent.agent_type == agent_type:
                            self._revert_agent_params(agent, params)

                expired_names.append(name)
                event_str = (
                    f"[T{timestep}] POLICY EXPIRED: '{name}' — "
                    f"all effects reverted after {effect.duration_steps} steps"
                )
                events.append(event_str)
                self._pending_events.append(f"⏰ {event_str}")
                logger.info(event_str)

        for name in expired_names:
            del self.active_policies[name]

        # Expire global effects via economics engine
        econ_events = economics.expire_effects(timestep)
        events.extend(econ_events)

        return events

    def _revert_agent_params(self, agent, params: Dict[str, Any]):
        """Revert agent parameters to defaults when policy expires."""
        f = agent.finance
        base_config = AGENT_CONFIGS.get(agent.agent_type)

        for param in params:
            if param == "income_multiplier":
                f.income_multiplier = 1.0
            elif param == "expense_multiplier":
                f.expense_multiplier = 1.0
            elif param == "tax_rate":
                f.tax_rate = base_config.base_tax_rate if base_config else 0.0
            elif param == "investment_return_rate":
                f.investment_return_rate = 0.05
            elif param == "savings_interest_rate":
                f.savings_interest_rate = 0.02
            elif param == "debt_interest_rate":
                f.debt_interest_rate = 0.08
            elif param == "welfare_payment":
                f.welfare_payment = 0.0
                if agent.agent_type == AgentType.UNEMPLOYED:
                    agent.type_state.welfare_active = 0.0
            elif param == "trade_allowed":
                f.trade_allowed = True
            elif param == "invest_allowed":
                f.invest_allowed = True
            elif param == "cash_freeze_fraction":
                f.cash_freeze_fraction = 0.0
            elif param == "healthcare_subsidy":
                f.healthcare_subsidy = 0.0
            elif param == "education_subsidy":
                f.education_subsidy = 0.0

    # ─────────────────────────────────────────
    # VISUAL EVENTS
    # ─────────────────────────────────────────

    def get_visual_events(self) -> List[str]:
        """Get and clear pending visual events for frontend display."""
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def get_active_policies_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all active policies for the dashboard."""
        return [
            {
                "name": name,
                "description": effect.policy_description,
                "source": effect.source,
                "applied_at": effect.applied_at,
                "expires_at": effect.expires_at,
                "duration": effect.duration_steps,
                "affected_types": list(effect.affected_agents.keys()),
                "global_params": list(effect.global_effects.keys()),
            }
            for name, effect in self.active_policies.items()
        ]

    # ─────────────────────────────────────────
    # CACHE MANAGEMENT
    # ─────────────────────────────────────────

    @staticmethod
    def _hash_policy(text: str) -> str:
        """Create a deterministic hash of normalized policy text."""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _load_cache(self):
        """Load policy cache from disk."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r") as f:
                    self.cache = json.load(f)
                logger.info("Loaded policy cache: %d entries", len(self.cache))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load policy cache: %s", e)
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self):
        """Persist policy cache to disk."""
        try:
            with open(self._cache_path, "w") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error("Failed to save policy cache: %s", e)

    def _load_fallback_policies(self):
        """Load pre-built fallback policies from JSON."""
        if self._fallback_path.exists():
            try:
                with open(self._fallback_path, "r") as f:
                    self.fallback_policies = json.load(f)
                logger.info("Loaded %d fallback policies", len(self.fallback_policies))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load fallback policies: %s", e)
                self.fallback_policies = []
        else:
            logger.info("No fallback_policies.json found at %s", self._fallback_path)
            self.fallback_policies = []

    # ─────────────────────────────────────────
    # SNAPSHOT & RESTORE
    # ─────────────────────────────────────────

    def get_snapshot(self) -> Dict[str, Any]:
        """Get serializable snapshot for comparison mode."""
        return {
            "active_policies": {
                name: effect.to_dict()
                for name, effect in self.active_policies.items()
            },
            "policy_history_count": len(self.policy_history),
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]):
        """Restore from snapshot."""
        self.active_policies = {
            name: PolicyEffect.from_dict(data)
            for name, data in snapshot.get("active_policies", {}).items()
        }

    def __repr__(self) -> str:
        return (
            f"PolicyEngine(gemini={'✓' if self._gemini_available else '✗'}, "
            f"cache={len(self.cache)}, fallbacks={len(self.fallback_policies)}, "
            f"active={len(self.active_policies)})"
        )
