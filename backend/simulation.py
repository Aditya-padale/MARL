"""
simulation.py — Master simulation engine for MARL City Simulator.

Orchestrates all subsystems:
  - 100 independent agents with PPO decision-making
  - Procedurally generated city with zone-based navigation
  - Global economics engine with monthly updates
  - Proximity-based social interaction system
  - Stochastic life event system
  - 7 research metrics tracked every timestep
  - Policy engine with Gemini API integration
  - Scenario comparison mode

The simulation runs as an async loop at configurable speed
(default 10 steps/second), with PPO updates every 128 steps
per agent. State diffs are emitted via callback for WebSocket.

Author: Aditya Padale (B.Tech Final Year Project)
"""

import asyncio
import random
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Tuple

from config import (
    AgentType, AGENT_CONFIGS, AGENT_TYPE_NAMES,
    Action, NUM_ACTIONS,
    PPO_CONFIG,
    BACKEND_TICK_RATE, TIME_SPEEDS,
    STEPS_PER_MONTH, STEPS_PER_YEAR,
    CITY_GENERATION_SEED,
)
from ppo import PPONetworkPool, PPOAgent
from agent import Agent
from city import City
from economics import EconomicsEngine
from social import SocialSystem
from life_events import LifeEventSystem
from metrics import MetricsTracker
from policy_engine import PolicyEngine, PolicyEffect
from comparison import ScenarioComparison

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Master simulation engine managing all 100 agents and subsystems.

    Lifecycle:
        1. __init__()       — Create city, agents, subsystems
        2. initialize()     — Set up PPO networks, place agents, send full state
        3. run_loop()       — Async main loop (10 steps/sec by default)
        4. step_sync()      — Single synchronous step (for comparison mode)

    Each simulation step:
        a. Each agent observes state, selects action via PPO, executes action
        b. Agents move toward action-appropriate zones
        c. Social interactions fire for nearby agent pairs
        d. Life events fire stochastically
        e. Monthly processing (every 30 steps): income, interest, inflation
        f. PPO updates (every 128 steps per agent)
        g. Metrics computed and logged
        h. State diff emitted via callback for WebSocket
    """

    def __init__(self, seed: int = CITY_GENERATION_SEED):
        """
        Initialize all simulation subsystems.

        Args:
            seed: Random seed for reproducible simulations
        """
        self.seed = seed
        random.seed(seed)

        # ── Timestep & Speed ──
        self.timestep: int = 0
        self.speed: str = "play"               # Current speed mode
        self.steps_per_tick: int = 1            # Steps to advance per tick
        self.is_paused: bool = False
        self.is_running: bool = False

        # ── City ──
        self.city = City(seed=seed)
        logger.info("City generated: %s", self.city.get_summary())

        # ── PPO Network Pool (10 type-level networks) ──
        self.network_pool = PPONetworkPool()

        # ── Agents (100 total) ──
        self.agents: List[Agent] = []

        # ── Subsystems ──
        self.economics = EconomicsEngine()
        self.social = SocialSystem()
        self.life_events = LifeEventSystem()
        self.metrics = MetricsTracker()
        self.policy_engine = PolicyEngine(data_dir="data")
        self.comparison = ScenarioComparison()

        # ── Event Log ──
        self.event_log: List[str] = []
        self.max_event_log: int = 200

        # ── State Diff Tracking ──
        self._prev_agent_states: Dict[int, Dict] = {}
        self._state_callback: Optional[Callable] = None

        # ── Performance Tracking ──
        self._step_times: List[float] = []

    # ─────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────

    def initialize(self):
        """
        Set up PPO networks and spawn all 100 agents.

        Creates:
          - 10 Actor-Critic network pairs (one per type)
          - 100 Agent objects with independent PPO agents
          - Places agents at their home zone positions
        """
        logger.info("Initializing simulation with %d agent types...",
                     len(AGENT_CONFIGS))

        agent_id = 0

        for agent_type, config in AGENT_CONFIGS.items():
            # Create type-level networks
            actor, critic = self.network_pool.create_networks(
                agent_type=agent_type,
                state_dim=config.state_dim,
            )

            # Spawn agents of this type
            for i in range(config.count):
                # Each agent gets its own PPOAgent (own buffer + optimizer)
                ppo_agent = PPOAgent(
                    actor=actor,
                    critic=critic,
                    state_dim=config.state_dim,
                    agent_id=agent_id,
                    agent_type_name=AGENT_TYPE_NAMES[agent_type],
                )

                # Get starting position in home zone
                home_pos = self.city.get_home_position(agent_type)

                # Create the agent
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    config=config,
                    ppo_agent=ppo_agent,
                    initial_position=home_pos,
                )

                self.agents.append(agent)
                agent_id += 1

        # Network pool summary
        pool_summary = self.network_pool.summary()
        logger.info(
            "Initialized %d agents with %d type networks (%d total params)",
            len(self.agents), pool_summary["type_count"],
            pool_summary["total_params"],
        )

        # Initialize families
        self.social.initialize_families(self.agents)

        # Initial metrics snapshot
        self.metrics.log_timestep(self.agents, 0)

    # ─────────────────────────────────────────
    # MAIN ASYNC LOOP
    # ─────────────────────────────────────────

    async def run_loop(self, callback: Optional[Callable] = None):
        """
        Main async simulation loop.

        Runs at BACKEND_TICK_RATE (default 10 ticks/second).
        Each tick advances the simulation by steps_per_tick steps.
        Emits state diffs via callback after each tick.

        Args:
            callback: Async function called with state diff dict each tick.
                      Used by WebSocket to stream updates to frontend.
        """
        self._state_callback = callback
        self.is_running = True
        tick_interval = 1.0 / BACKEND_TICK_RATE

        logger.info("Simulation loop started (%.0f ticks/sec, speed=%s)",
                     BACKEND_TICK_RATE, self.speed)

        while self.is_running:
            tick_start = time.time()

            if not self.is_paused and self.steps_per_tick > 0:
                # Run N steps per tick (based on speed setting)
                for _ in range(self.steps_per_tick):
                    self._run_single_step()

                # Emit state diff
                if callback:
                    diff = self._build_state_diff()
                    await callback(diff)

            # Maintain tick rate
            elapsed = time.time() - tick_start
            sleep_time = max(0, tick_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def stop(self):
        """Stop the simulation loop."""
        self.is_running = False
        logger.info("Simulation loop stopped at T=%d", self.timestep)

    # ─────────────────────────────────────────
    # SINGLE STEP (core simulation logic)
    # ─────────────────────────────────────────

    def _run_single_step(self):
        """
        Execute one complete simulation step.

        Order of operations:
          1. Get economic state for agent observations
          2. Each agent: observe → decide → act → reward
          3. Move agents toward action-appropriate zones
          4. Social interactions (proximity-based)
          5. Life events (stochastic)
          6. Monthly processing (every 30 steps)
          7. PPO updates (every 128 steps)
          8. Policy expiry checks
          9. Metrics logging
          10. Increment timestep
        """
        step_start = time.time()
        economics_state = self.economics.get_state_dict()

        # ── 1. Agent decision cycle ──
        for agent in self.agents:
            action, reward = agent.decide_and_act(economics_state, self.timestep)

            # Set movement target based on action
            target = self.city.get_target_for_action(action, agent.agent_type)
            agent.set_target(*target)

        # ── 2. Move all agents ──
        is_night = self.city.is_night(self.timestep)
        for agent in self.agents:
            if is_night:
                # At night, agents go home
                home = self.city.get_home_position(agent.agent_type)
                agent.set_target(*home)
            agent.move_step()

        # ── 3. Social interactions ──
        social_events = self.social.check_interactions(
            self.agents, self.city, self.timestep
        )
        self._add_events(social_events)

        # ── 4. Life events ──
        life_events = self.life_events.step(
            self.agents, self.economics, self.timestep
        )
        self._add_events(life_events)

        # ── 5. Monthly processing ──
        if self.timestep > 0 and self.timestep % STEPS_PER_MONTH == 0:
            econ_events = self.economics.monthly_update(
                self.agents, self.timestep
            )
            self._add_events(econ_events)

        # ── 6. PPO updates (every rollout_length steps) ──
        if self.timestep > 0 and self.timestep % PPO_CONFIG.rollout_length == 0:
            self._run_ppo_updates()

        # ── 7. Policy expiry ──
        policy_events = self.policy_engine.expire_policies(
            self.timestep, self.agents, self.economics
        )
        self._add_events(policy_events)

        # ── 8. Metrics ──
        # Log every step for short runs, every 5 for long runs
        if self.timestep % 5 == 0 or self.timestep < 100:
            self.metrics.log_timestep(
                self.agents, self.timestep,
                list(self.policy_engine.active_policies.keys()),
            )

        # ── 9. Advance time ──
        self.timestep += 1

        # Track performance
        step_time = time.time() - step_start
        self._step_times.append(step_time)
        if len(self._step_times) > 100:
            self._step_times.pop(0)

    def step_sync(self):
        """
        Public synchronous step method for comparison mode.
        Same as _run_single_step but callable from outside.
        """
        self._run_single_step()

    # ─────────────────────────────────────────
    # PPO UPDATE CYCLE
    # ─────────────────────────────────────────

    def _run_ppo_updates(self):
        """
        Run PPO updates for all agents.

        Called every PPO_CONFIG.rollout_length steps (default 128).
        Updates are sequential per type group to avoid conflicts
        on shared network weights.

        Each agent:
          1. Gets bootstrap value V(s_T) from Critic
          2. Computes GAE advantages from its own buffer
          3. Runs 4 epochs of clipped PPO loss
          4. Applies gradients through its own Adam optimizer
          5. Clears its buffer
        """
        economics_state = self.economics.get_state_dict()
        total_updates = 0

        # Group agents by type for sequential weight updates
        for agent_type in AgentType:
            type_agents = [a for a in self.agents if a.agent_type == agent_type]

            for agent in type_agents:
                if agent.ppo.buffer.size > 0:
                    # Bootstrap value for GAE
                    state = agent.build_state_vector(economics_state, self.timestep)
                    last_value = agent.ppo.get_value_estimate(state)

                    # Run PPO update
                    loss_info = agent.ppo.update(last_value)
                    total_updates += 1

        if total_updates > 0:
            logger.debug("PPO updates: %d agents updated at T=%d",
                         total_updates, self.timestep)

    # ─────────────────────────────────────────
    # TIME CONTROL
    # ─────────────────────────────────────────

    def set_speed(self, speed: str):
        """
        Set simulation speed.

        Args:
            speed: One of "pause", "play", "week", "month", "year"
        """
        if speed in TIME_SPEEDS:
            self.speed = speed
            self.steps_per_tick = TIME_SPEEDS[speed]
            self.is_paused = (speed == "pause")
            logger.info("Speed set to '%s' (%d steps/tick)", speed, self.steps_per_tick)
        else:
            logger.warning("Unknown speed: '%s'", speed)

    def skip_steps(self, n_steps: int):
        """
        Skip forward by n_steps (for time jump buttons).

        Runs steps synchronously without emitting diffs.

        Args:
            n_steps: Number of steps to skip
        """
        logger.info("Skipping %d steps from T=%d...", n_steps, self.timestep)
        for _ in range(n_steps):
            self._run_single_step()
        logger.info("Skip complete. Now at T=%d", self.timestep)

    # ─────────────────────────────────────────
    # STATE SERIALIZATION
    # ─────────────────────────────────────────

    def get_full_state(self) -> Dict[str, Any]:
        """
        Build the complete simulation state for initial WebSocket sync.

        Sent once when a client connects. Contains everything needed
        to render the city and all agents from scratch.

        Returns:
            Full state dict with city, agents, metrics, events, policies
        """
        return {
            "type": "full_state",
            "timestep": self.timestep,
            "time_label": self.city.get_time_label(self.timestep),
            "time_of_day": self.city.get_time_of_day_label(self.timestep),
            "sky_color": self.city.get_sky_color(self.timestep),
            "speed": self.speed,
            "is_paused": self.is_paused,
            "city": self.city.draw_data(),
            "agents": [agent.to_render_dict() for agent in self.agents],
            "metrics": self.metrics.get_current_metrics(),
            "trends": self.metrics.get_trend_data(last_n=60),
            "events": self.event_log[-20:],
            "active_policies": self.policy_engine.get_active_policies_summary(),
            "economics": self.economics.get_state_dict(),
            "social_stats": self.social.get_stats(self.agents),
            "social_graph": self.social.graph.serialize_edges(),
            "life_event_stats": self.life_events.get_stats(),
            "comparison_status": self.comparison.get_status(),
            "performance": self._get_performance_stats(),
        }

    def _build_state_diff(self) -> Dict[str, Any]:
        """
        Build a compressed state diff for WebSocket streaming.

        Only includes agents that changed position or state since
        the last diff. This reduces bandwidth from ~500KB/s to ~5-20KB/s.

        Returns:
            Diff dict with only changed data
        """
        # Build current agent render states
        agent_updates = []
        for agent in self.agents:
            current = agent.to_render_dict()
            prev = self._prev_agent_states.get(agent.id)

            # Check if anything changed
            if prev is None or self._agent_changed(prev, current):
                agent_updates.append(current)
                self._prev_agent_states[agent.id] = current

        # Get new events since last diff
        policy_events = self.policy_engine.get_visual_events()

        diff = {
            "type": "diff",
            "timestep": self.timestep,
            "time_label": self.city.get_time_label(self.timestep),
            "time_of_day": self.city.get_time_of_day_label(self.timestep),
            "sky_color": self.city.get_sky_color(self.timestep),
            "agent_updates": agent_updates,
            "metrics": self.metrics.get_current_metrics(),
            "events": self.event_log[-5:],     # Last 5 events
            "policy_events": policy_events,
            "speed": self.speed,
            "is_paused": self.is_paused,
            "social_graph": self.social.graph.serialize_edges(), # Also send edges in diff
        }

        return diff

    @staticmethod
    def _agent_changed(prev: Dict, current: Dict) -> bool:
        """Check if an agent's render state changed since last diff."""
        # Position change (threshold: 1 pixel)
        if abs(prev.get("x", 0) - current.get("x", 0)) > 1.0:
            return True
        if abs(prev.get("y", 0) - current.get("y", 0)) > 1.0:
            return True
        # State changes
        if prev.get("wealth_tier") != current.get("wealth_tier"):
            return True
        if prev.get("emotion") != current.get("emotion"):
            return True
        if prev.get("action") != current.get("action"):
            return True
        if prev.get("is_bankrupt") != current.get("is_bankrupt"):
            return True
        if prev.get("walk_frame") != current.get("walk_frame"):
            return True
        return False

    # ─────────────────────────────────────────
    # AGENT INSPECTION
    # ─────────────────────────────────────────

    def get_agent_inspect(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """
        Get full inspection data for a specific agent.

        Args:
            agent_id: Agent ID (0–99)

        Returns:
            Full inspect dict, or None if agent not found
        """
        if 0 <= agent_id < len(self.agents):
            return self.agents[agent_id].to_inspect_dict()
        return None

    # ─────────────────────────────────────────
    # POLICY APPLICATION
    # ─────────────────────────────────────────

    async def apply_policy_text(self, policy_text: str) -> Dict[str, Any]:
        """
        Process a natural language policy from the researcher.

        Pipeline:
          1. Parse via PolicyEngine (cache → Gemini → fallback)
          2. Record metrics baseline for effectiveness scoring
          3. Apply to agents and economics
          4. Return result for frontend display

        Args:
            policy_text: Plain English policy description

        Returns:
            Dict with policy details and application events
        """
        # Parse policy
        sim_state = self.economics.get_state_dict()
        sim_state["poverty_rate"] = self.metrics.get_current_metrics().get("poverty_rate", 0.2)

        effect = await self.policy_engine.interpret_policy(policy_text, sim_state)

        # Record baseline for effectiveness scoring
        self.metrics.record_policy_baseline(effect.policy_name)

        # Apply
        events = self.policy_engine.apply_policy(
            effect, self.agents, self.economics, self.timestep
        )
        self._add_events(events)

        return {
            "success": True,
            "policy": effect.to_dict(),
            "events": events,
        }

    # ─────────────────────────────────────────
    # COMPARISON MODE
    # ─────────────────────────────────────────

    async def run_comparison(self, policy_text: str,
                              n_steps: int = 90) -> Dict[str, Any]:
        """
        Fork simulation and run A/B comparison with a policy.

        Args:
            policy_text: Policy to apply to Scenario B
            n_steps:     Steps to run each scenario forward

        Returns:
            ComparisonResult as dict
        """
        logger.info("Starting comparison: '%s' for %d steps", policy_text[:50], n_steps)

        # Parse the policy
        sim_state = self.economics.get_state_dict()
        effect = await self.policy_engine.interpret_policy(policy_text, sim_state)

        # Save current state to restore later
        original_timestep = self.timestep
        original_agent_snaps = [a.get_snapshot() for a in self.agents]
        original_econ_snap = self.economics.get_snapshot()
        original_social_snap = self.social.get_snapshot()
        original_life_snap = self.life_events.get_snapshot()
        original_policy_snap = self.policy_engine.get_snapshot()

        # Fork
        state_a, state_b = self.comparison.fork(self)

        # Run Scenario A (baseline — no policy)
        state_a = self.comparison.run_scenario_sync(self, state_a, n_steps)

        # Run Scenario B (with policy)
        state_b = self.comparison.run_scenario_sync(self, state_b, n_steps, policy_effect=effect)

        # Compare
        result = self.comparison.compare(state_a, state_b)

        # Restore original state
        self.timestep = original_timestep
        for agent, snap in zip(self.agents, original_agent_snaps):
            agent.restore_snapshot(snap)
        self.economics.restore_snapshot(original_econ_snap)
        self.social.restore_snapshot(original_social_snap)
        self.life_events.restore_snapshot(original_life_snap)
        self.policy_engine.restore_snapshot(original_policy_snap)

        logger.info("Comparison complete. Simulation restored to T=%d", self.timestep)

        # Log the comparison event
        self._add_events([
            f"[T{self.timestep}] COMPARISON: '{effect.policy_name}' — "
            f"Effectiveness: {result.policy_effectiveness:.1f}/100"
        ])

        return result.to_dict()

    # ─────────────────────────────────────────
    # EVENT LOG
    # ─────────────────────────────────────────

    def _add_events(self, events: List[str]):
        """Add events to the log, maintaining max size."""
        for event in events:
            if event:
                self.event_log.append(event)
        # Trim
        if len(self.event_log) > self.max_event_log:
            self.event_log = self.event_log[-self.max_event_log:]

    # ─────────────────────────────────────────
    # PERFORMANCE & METRICS EXPORT
    # ─────────────────────────────────────────

    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get simulation performance statistics."""
        if not self._step_times:
            return {"avg_step_ms": 0, "steps_per_second": 0}

        avg_ms = sum(self._step_times) * 1000 / len(self._step_times)
        sps = 1000 / avg_ms if avg_ms > 0 else 0

        return {
            "avg_step_ms": round(avg_ms, 2),
            "steps_per_second": round(sps, 0),
            "timestep": self.timestep,
            "agents_active": sum(1 for a in self.agents if not a.finance.is_bankrupt),
            "agents_bankrupt": sum(1 for a in self.agents if a.finance.is_bankrupt),
        }

    def export_metrics(self, path: str = "data/metrics_export.csv"):
        """Export all metrics to CSV for the research paper."""
        self.metrics.export_csv(path)

    def get_year_review(self) -> Dict[str, Any]:
        """Generate Year in Review panel data."""
        return self.metrics.generate_year_review(self.agents, self.timestep)

    def __repr__(self) -> str:
        return (
            f"SimulationEngine(T={self.timestep}, agents={len(self.agents)}, "
            f"speed={self.speed}, paused={self.is_paused})"
        )
