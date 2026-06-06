"""
comparison.py — Scenario comparison system for MARL City Simulator.

Research mode for IEEE paper result generation. Enables the researcher to:
  1. Fork the current simulation at any point into Scenario A (baseline) and Scenario B (policy)
  2. Apply a policy to Scenario B only
  3. Run both scenarios forward for N steps (30/90/365)
  4. Compare side-by-side metrics with deltas

IMPLEMENTATION NOTES:
  - Scenarios run SEQUENTIALLY (A completes, then B), NOT parallel
  - This avoids RAM doubling on student hardware
  - Results are stored as metric arrays and compared after
  - Export comparison to CSV for paper data tables

Author: Aditya Padale (B.Tech Final Year Project)
"""

import copy
import csv
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

from config import (
    ComparisonConfig, COMPARISON_CONFIG,
    STEPS_PER_MONTH, STEPS_PER_YEAR,
    AGENT_TYPE_NAMES,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# SCENARIO STATE
# ═══════════════════════════════════════════

@dataclass
class ScenarioState:
    """
    Complete deep-copyable state of one simulation scenario.

    Contains snapshots of all agents, economics, social system,
    life events, metrics, and active policies — everything needed
    to reconstruct and run the simulation forward independently.
    """
    name: str                                    # "Scenario A" or "Scenario B"
    label: str                                   # User-facing label
    agent_snapshots: List[Dict[str, Any]]        # One snapshot per agent
    economics_snapshot: Dict[str, Any]           # EconomicsEngine state
    social_snapshot: Dict[str, Any]              # SocialSystem state
    life_events_snapshot: Dict[str, Any]         # LifeEventSystem state
    policy_snapshot: Dict[str, Any]              # PolicyEngine state
    metrics_at_fork: Dict[str, Any]              # Metrics when forked
    forked_at_timestep: int                      # When the fork occurred
    run_steps: int = 0                           # Steps run after fork
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    final_metrics: Optional[Dict[str, Any]] = None


# ═══════════════════════════════════════════
# COMPARISON RESULT
# ═══════════════════════════════════════════

@dataclass
class ComparisonResult:
    """
    Side-by-side comparison of two scenarios with deltas.

    This is the main output displayed in the Comparison Panel
    and exported as CSV for the research paper.
    """
    scenario_a_label: str
    scenario_b_label: str
    forked_at: int
    steps_compared: int

    # Core metrics comparison
    gini: Dict[str, float] = field(default_factory=dict)
    poverty_rate: Dict[str, float] = field(default_factory=dict)
    bankruptcy_rate: Dict[str, float] = field(default_factory=dict)
    social_mobility: Dict[str, Any] = field(default_factory=dict)
    total_wealth: Dict[str, float] = field(default_factory=dict)
    median_wealth: Dict[str, float] = field(default_factory=dict)
    top_10_concentration: Dict[str, float] = field(default_factory=dict)
    policy_effectiveness: float = 0.0

    # Per-type wealth comparison
    type_wealth_a: Dict[str, float] = field(default_factory=dict)
    type_wealth_b: Dict[str, float] = field(default_factory=dict)

    # Wealth distribution comparison
    distribution_a: Dict[str, int] = field(default_factory=dict)
    distribution_b: Dict[str, int] = field(default_factory=dict)

    # Time series for charts
    metrics_timeseries_a: List[Dict] = field(default_factory=list)
    metrics_timeseries_b: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for WebSocket/REST transmission."""
        return {
            "scenario_a": self.scenario_a_label,
            "scenario_b": self.scenario_b_label,
            "forked_at": self.forked_at,
            "steps_compared": self.steps_compared,
            "metrics": {
                "gini": self.gini,
                "poverty_rate": self.poverty_rate,
                "bankruptcy_rate": self.bankruptcy_rate,
                "social_mobility": self.social_mobility,
                "total_wealth": self.total_wealth,
                "median_wealth": self.median_wealth,
                "top_10_concentration": self.top_10_concentration,
                "policy_effectiveness": self.policy_effectiveness,
            },
            "per_type_wealth": {
                "scenario_a": self.type_wealth_a,
                "scenario_b": self.type_wealth_b,
            },
            "distribution": {
                "scenario_a": self.distribution_a,
                "scenario_b": self.distribution_b,
            },
            "timeseries": {
                "scenario_a": self.metrics_timeseries_a[-30:],
                "scenario_b": self.metrics_timeseries_b[-30:],
            },
        }

    def summary_text(self) -> str:
        """Human-readable comparison summary for logging."""
        lines = [
            f"═══ SCENARIO COMPARISON ═══",
            f"  Fork point:  T={self.forked_at}",
            f"  Steps run:   {self.steps_compared}",
            f"  A: {self.scenario_a_label}",
            f"  B: {self.scenario_b_label}",
            f"",
            f"  {'Metric':<25} {'A':>10} {'B':>10} {'Δ':>10}",
            f"  {'─'*55}",
        ]

        metrics_list = [
            ("Gini Coefficient", self.gini),
            ("Poverty Rate", self.poverty_rate),
            ("Bankruptcy Rate", self.bankruptcy_rate),
            ("Median Wealth (₹)", self.median_wealth),
            ("Total Wealth (₹)", self.total_wealth),
            ("Top 10% Concentration", self.top_10_concentration),
        ]

        for name, data in metrics_list:
            a_val = data.get("a", 0)
            b_val = data.get("b", 0)
            delta = data.get("delta", 0)

            if isinstance(a_val, float) and a_val < 1:
                lines.append(f"  {name:<25} {a_val:>9.1%} {b_val:>9.1%} {delta:>+9.1%}")
            else:
                lines.append(f"  {name:<25} {a_val:>10,.0f} {b_val:>10,.0f} {delta:>+10,.0f}")

        lines.append(f"\n  Policy Effectiveness Score: {self.policy_effectiveness:.1f}/100")
        return "\n".join(lines)


# ═══════════════════════════════════════════
# SCENARIO COMPARISON ENGINE
# ═══════════════════════════════════════════

class ScenarioComparison:
    """
    Manages scenario forking, sequential execution, and comparison.

    Usage:
        comp = ScenarioComparison()

        # Fork current simulation
        state_a, state_b = comp.fork(simulation)

        # Run baseline (A) forward
        metrics_a = await comp.run_scenario(simulation, state_a, n_steps=90)

        # Apply policy to B and run forward
        metrics_b = await comp.run_scenario(simulation, state_b, n_steps=90, policy=effect)

        # Compare
        result = comp.compare(state_a, state_b)
        comp.export_comparison(result, "data/comparison.csv")
    """

    def __init__(self, config: ComparisonConfig = COMPARISON_CONFIG):
        self.config = config
        self._last_result: Optional[ComparisonResult] = None
        self._is_running: bool = False
        self._progress: float = 0.0  # 0.0 – 1.0

    # ─────────────────────────────────────────
    # FORK
    # ─────────────────────────────────────────

    def fork(self, simulation) -> Tuple[ScenarioState, ScenarioState]:
        """
        Deep-copy the current simulation state into two independent scenarios.

        Creates Scenario A (baseline — no changes) and Scenario B (policy target).
        Both start from the exact same state.

        Args:
            simulation: SimulationEngine instance

        Returns:
            (state_A, state_B) — two independent ScenarioState objects
        """
        timestep = simulation.timestep
        logger.info("Forking simulation at T=%d into two scenarios", timestep)

        # Capture current metrics
        current_metrics = simulation.metrics.get_current_metrics()

        # Snapshot all agents
        agent_snapshots = [agent.get_snapshot() for agent in simulation.agents]

        # Snapshot subsystems
        economics_snap = simulation.economics.get_snapshot()
        social_snap = simulation.social.get_snapshot()
        life_events_snap = simulation.life_events.get_snapshot()
        policy_snap = simulation.policy_engine.get_snapshot()

        # Create Scenario A (baseline)
        state_a = ScenarioState(
            name="scenario_a",
            label="Baseline (No Policy)",
            agent_snapshots=copy.deepcopy(agent_snapshots),
            economics_snapshot=copy.deepcopy(economics_snap),
            social_snapshot=copy.deepcopy(social_snap),
            life_events_snapshot=copy.deepcopy(life_events_snap),
            policy_snapshot=copy.deepcopy(policy_snap),
            metrics_at_fork=copy.deepcopy(current_metrics),
            forked_at_timestep=timestep,
        )

        # Create Scenario B (policy target)
        state_b = ScenarioState(
            name="scenario_b",
            label="Policy Scenario",
            agent_snapshots=copy.deepcopy(agent_snapshots),
            economics_snapshot=copy.deepcopy(economics_snap),
            social_snapshot=copy.deepcopy(social_snap),
            life_events_snapshot=copy.deepcopy(life_events_snap),
            policy_snapshot=copy.deepcopy(policy_snap),
            metrics_at_fork=copy.deepcopy(current_metrics),
            forked_at_timestep=timestep,
        )

        logger.info("Fork complete. Both scenarios initialized at T=%d", timestep)
        return state_a, state_b

    # ─────────────────────────────────────────
    # RUN SCENARIO
    # ─────────────────────────────────────────

    async def run_scenario(self, simulation, scenario: ScenarioState,
                           n_steps: int, policy_effect=None) -> ScenarioState:
        """
        Run a scenario forward for n_steps from its forked state.

        IMPORTANT: This restores the simulation to the scenario's state,
        runs it forward, captures metrics at each step, then the caller
        must restore the original state afterward.

        Scenarios run SEQUENTIALLY to avoid RAM doubling.

        Args:
            simulation:    SimulationEngine instance (will be temporarily modified)
            scenario:      ScenarioState to run
            n_steps:       Number of steps to simulate forward
            policy_effect: Optional PolicyEffect to apply to this scenario

        Returns:
            Updated ScenarioState with metrics_history and final_metrics populated
        """
        self._is_running = True
        self._progress = 0.0
        start_time = time.time()

        logger.info("Running %s for %d steps%s",
                     scenario.label, n_steps,
                     f" with policy '{policy_effect.policy_name}'" if policy_effect else "")

        # ── Step 1: Restore simulation to forked state ──
        self._restore_simulation_state(simulation, scenario)

        # ── Step 2: Apply policy if provided ──
        if policy_effect:
            scenario.label = f"Policy: {policy_effect.policy_name}"
            events = simulation.policy_engine.apply_policy(
                policy_effect, simulation.agents,
                simulation.economics, simulation.timestep
            )
            logger.info("Policy applied: %s (%d events)", policy_effect.policy_name, len(events))

        # ── Step 3: Run simulation forward ──
        scenario.metrics_history = []

        for step in range(n_steps):
            # Run one simulation step (synchronous for comparison mode)
            simulation.step_sync()

            # Log metrics periodically (every step for short runs, every 5 for long)
            log_interval = 1 if n_steps <= 90 else 5
            if step % log_interval == 0:
                metrics = simulation.metrics.get_current_metrics()
                metrics["step"] = step
                metrics["timestep"] = simulation.timestep
                scenario.metrics_history.append(metrics)

            # Update progress
            self._progress = (step + 1) / n_steps

        # ── Step 4: Capture final metrics ──
        scenario.final_metrics = simulation.metrics.get_current_metrics()
        scenario.run_steps = n_steps

        elapsed = time.time() - start_time
        logger.info("%s complete: %d steps in %.1fs (%.0f steps/s)",
                     scenario.label, n_steps, elapsed, n_steps / max(elapsed, 0.001))

        self._is_running = False
        self._progress = 1.0
        return scenario

    def run_scenario_sync(self, simulation, scenario: ScenarioState,
                          n_steps: int, policy_effect=None) -> ScenarioState:
        """
        Synchronous version of run_scenario for non-async contexts.
        Same logic but without async/await.
        """
        self._is_running = True
        self._progress = 0.0

        logger.info("Running %s (sync) for %d steps", scenario.label, n_steps)

        # Restore state
        self._restore_simulation_state(simulation, scenario)

        # Apply policy
        if policy_effect:
            scenario.label = f"Policy: {policy_effect.policy_name}"
            simulation.policy_engine.apply_policy(
                policy_effect, simulation.agents,
                simulation.economics, simulation.timestep
            )

        # Run forward
        scenario.metrics_history = []
        log_interval = 1 if n_steps <= 90 else 5

        for step in range(n_steps):
            simulation.step_sync()

            if step % log_interval == 0:
                metrics = simulation.metrics.get_current_metrics()
                metrics["step"] = step
                metrics["timestep"] = simulation.timestep
                scenario.metrics_history.append(metrics)

            self._progress = (step + 1) / n_steps

        scenario.final_metrics = simulation.metrics.get_current_metrics()
        scenario.run_steps = n_steps
        self._is_running = False
        self._progress = 1.0
        return scenario

    # ─────────────────────────────────────────
    # RESTORE SIMULATION STATE
    # ─────────────────────────────────────────

    def _restore_simulation_state(self, simulation, scenario: ScenarioState):
        """
        Restore the simulation engine to a scenario's saved state.

        Args:
            simulation: SimulationEngine to restore
            scenario:   ScenarioState with saved snapshots
        """
        # Restore timestep
        simulation.timestep = scenario.forked_at_timestep

        # Restore agents
        for agent, snap in zip(simulation.agents, scenario.agent_snapshots):
            agent.restore_snapshot(snap)

        # Restore economics
        simulation.economics.restore_snapshot(scenario.economics_snapshot)

        # Restore social system
        simulation.social.restore_snapshot(scenario.social_snapshot)

        # Restore life events
        simulation.life_events.restore_snapshot(scenario.life_events_snapshot)

        # Restore policy engine
        simulation.policy_engine.restore_snapshot(scenario.policy_snapshot)

        logger.info("Simulation state restored to T=%d for %s",
                     scenario.forked_at_timestep, scenario.label)

    # ─────────────────────────────────────────
    # COMPARE
    # ─────────────────────────────────────────

    def compare(self, state_a: ScenarioState,
                state_b: ScenarioState) -> ComparisonResult:
        """
        Compare the final metrics of two completed scenarios.

        Produces the side-by-side comparison shown in the Comparison Panel:
          Gini:       A: 0.42  → B: 0.38  (Δ -0.04)
          Poverty:    A: 22%   → B: 17%   (Δ -5%)
          ...

        Args:
            state_a: Completed Scenario A (baseline)
            state_b: Completed Scenario B (policy)

        Returns:
            ComparisonResult with all deltas computed
        """
        if not state_a.final_metrics or not state_b.final_metrics:
            logger.error("Cannot compare: one or both scenarios have no final metrics")
            return ComparisonResult(
                scenario_a_label=state_a.label,
                scenario_b_label=state_b.label,
                forked_at=state_a.forked_at_timestep,
                steps_compared=0,
            )

        a = state_a.final_metrics
        b = state_b.final_metrics

        def _metric_dict(key: str) -> Dict[str, float]:
            a_val = a.get(key, 0)
            b_val = b.get(key, 0)
            if isinstance(a_val, dict) or isinstance(b_val, dict):
                return {"a": a_val, "b": b_val, "delta": None}
            delta = b_val - a_val
            pct = (delta / a_val * 100) if a_val != 0 else 0
            return {
                "a": round(a_val, 4),
                "b": round(b_val, 4),
                "delta": round(delta, 4),
                "pct_change": round(pct, 2),
            }

        # Compute policy effectiveness score
        effectiveness = self.policy_effectiveness_score(
            state_a.metrics_at_fork, state_b.final_metrics
        )

        result = ComparisonResult(
            scenario_a_label=state_a.label,
            scenario_b_label=state_b.label,
            forked_at=state_a.forked_at_timestep,
            steps_compared=max(state_a.run_steps, state_b.run_steps),
            gini=_metric_dict("gini"),
            poverty_rate=_metric_dict("poverty_rate"),
            bankruptcy_rate=_metric_dict("bankruptcy_rate"),
            social_mobility={
                "a": a.get("social_mobility", {}),
                "b": b.get("social_mobility", {}),
            },
            total_wealth=_metric_dict("total_wealth"),
            median_wealth=_metric_dict("median_wealth"),
            top_10_concentration=_metric_dict("top_10_concentration"),
            policy_effectiveness=effectiveness,
            type_wealth_a=a.get("type_avg_wealth", {}),
            type_wealth_b=b.get("type_avg_wealth", {}),
            distribution_a=a.get("wealth_distribution", {}),
            distribution_b=b.get("wealth_distribution", {}),
            metrics_timeseries_a=state_a.metrics_history,
            metrics_timeseries_b=state_b.metrics_history,
        )

        self._last_result = result
        logger.info("Comparison complete:\n%s", result.summary_text())
        return result

    # ─────────────────────────────────────────
    # POLICY EFFECTIVENESS SCORE
    # ─────────────────────────────────────────

    def policy_effectiveness_score(self, metrics_before: Dict,
                                    metrics_after: Dict) -> float:
        """
        Compute the composite Policy Effectiveness Score (0–100).

        Formula:
            score = 100 × (
                0.30 × (1 - gini_after / gini_before)        +
                0.25 × (1 - poverty_after / poverty_before)   +
                0.20 × upward_mobility_after                  +
                0.15 × (1 - bankruptcy_after / before)        +
                0.10 × (median_wealth_after / before - 1)
            )

        Higher score = policy improved societal outcomes.

        Args:
            metrics_before: Metrics snapshot from before policy (at fork point)
            metrics_after:  Metrics snapshot after policy has run

        Returns:
            Score from 0.0 to 100.0
        """
        cfg = self.config

        # Extract values with safe defaults
        gini_before = max(metrics_before.get("gini", 0.5), 0.001)
        gini_after = metrics_after.get("gini", 0.5)

        poverty_before = max(metrics_before.get("poverty_rate", 0.2), 0.001)
        poverty_after = metrics_after.get("poverty_rate", 0.2)

        bankruptcy_before = max(metrics_before.get("bankruptcy_rate", 0.05), 0.001)
        bankruptcy_after = metrics_after.get("bankruptcy_rate", 0.05)

        median_before = max(metrics_before.get("median_wealth", 10000), 1.0)
        median_after = metrics_after.get("median_wealth", 10000)

        mobility_after = 0.0
        mobility_data = metrics_after.get("social_mobility", {})
        if isinstance(mobility_data, dict):
            mobility_after = mobility_data.get("upward_pct", 0.0)

        # Compute component scores
        gini_score = 1.0 - (gini_after / gini_before)
        poverty_score = 1.0 - (poverty_after / poverty_before)
        bankruptcy_score = 1.0 - (bankruptcy_after / bankruptcy_before)
        wealth_score = (median_after / median_before) - 1.0
        mobility_score = mobility_after

        # Weighted composite
        raw_score = 100.0 * (
            cfg.weight_gini * gini_score +
            cfg.weight_poverty * poverty_score +
            cfg.weight_mobility * mobility_score +
            cfg.weight_bankruptcy * bankruptcy_score +
            cfg.weight_median_wealth * wealth_score
        )

        # Clamp for display
        return max(0.0, min(100.0, raw_score))

    # ─────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────

    def export_comparison(self, result: ComparisonResult, path: str):
        """
        Export comparison results to CSV for the research paper.

        Creates two files:
          1. {path}_summary.csv — One row with all metric deltas
          2. {path}_timeseries.csv — Time series of both scenarios

        Args:
            result: ComparisonResult to export
            path:   Base path (e.g., "data/comparison")
        """
        # ── Summary CSV ──
        summary_path = f"{path}_summary.csv"
        fieldnames = [
            "scenario_a", "scenario_b", "forked_at", "steps_compared",
            "gini_a", "gini_b", "gini_delta",
            "poverty_a", "poverty_b", "poverty_delta",
            "bankruptcy_a", "bankruptcy_b", "bankruptcy_delta",
            "median_wealth_a", "median_wealth_b", "median_wealth_delta",
            "total_wealth_a", "total_wealth_b", "total_wealth_delta",
            "top10_a", "top10_b", "top10_delta",
            "policy_effectiveness",
        ]

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                "scenario_a": result.scenario_a_label,
                "scenario_b": result.scenario_b_label,
                "forked_at": result.forked_at,
                "steps_compared": result.steps_compared,
                "gini_a": result.gini.get("a", 0),
                "gini_b": result.gini.get("b", 0),
                "gini_delta": result.gini.get("delta", 0),
                "poverty_a": result.poverty_rate.get("a", 0),
                "poverty_b": result.poverty_rate.get("b", 0),
                "poverty_delta": result.poverty_rate.get("delta", 0),
                "bankruptcy_a": result.bankruptcy_rate.get("a", 0),
                "bankruptcy_b": result.bankruptcy_rate.get("b", 0),
                "bankruptcy_delta": result.bankruptcy_rate.get("delta", 0),
                "median_wealth_a": result.median_wealth.get("a", 0),
                "median_wealth_b": result.median_wealth.get("b", 0),
                "median_wealth_delta": result.median_wealth.get("delta", 0),
                "total_wealth_a": result.total_wealth.get("a", 0),
                "total_wealth_b": result.total_wealth.get("b", 0),
                "total_wealth_delta": result.total_wealth.get("delta", 0),
                "top10_a": result.top_10_concentration.get("a", 0),
                "top10_b": result.top_10_concentration.get("b", 0),
                "top10_delta": result.top_10_concentration.get("delta", 0),
                "policy_effectiveness": result.policy_effectiveness,
            })

        logger.info("Comparison summary exported to %s", summary_path)

        # ── Time Series CSV ──
        ts_path = f"{path}_timeseries.csv"
        ts_fields = [
            "scenario", "step", "timestep", "gini", "poverty_rate",
            "bankruptcy_rate", "median_wealth", "total_wealth",
        ]

        with open(ts_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ts_fields)
            writer.writeheader()

            for entry in result.metrics_timeseries_a:
                writer.writerow({
                    "scenario": "A",
                    "step": entry.get("step", 0),
                    "timestep": entry.get("timestep", 0),
                    "gini": entry.get("gini", 0),
                    "poverty_rate": entry.get("poverty_rate", 0),
                    "bankruptcy_rate": entry.get("bankruptcy_rate", 0),
                    "median_wealth": entry.get("median_wealth", 0),
                    "total_wealth": entry.get("total_wealth", 0),
                })

            for entry in result.metrics_timeseries_b:
                writer.writerow({
                    "scenario": "B",
                    "step": entry.get("step", 0),
                    "timestep": entry.get("timestep", 0),
                    "gini": entry.get("gini", 0),
                    "poverty_rate": entry.get("poverty_rate", 0),
                    "bankruptcy_rate": entry.get("bankruptcy_rate", 0),
                    "median_wealth": entry.get("median_wealth", 0),
                    "total_wealth": entry.get("total_wealth", 0),
                })

        logger.info("Comparison time series exported to %s", ts_path)

    def export_per_type_comparison(self, result: ComparisonResult, path: str):
        """
        Export per-agent-type wealth comparison to CSV.

        One row per agent type with Scenario A and B average wealth.
        """
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent_type", "avg_wealth_A", "avg_wealth_B", "delta", "pct_change"])

            all_types = set(list(result.type_wealth_a.keys()) +
                            list(result.type_wealth_b.keys()))

            for type_name in sorted(all_types):
                a_val = result.type_wealth_a.get(type_name, 0)
                b_val = result.type_wealth_b.get(type_name, 0)
                delta = b_val - a_val
                pct = (delta / a_val * 100) if a_val != 0 else 0

                writer.writerow([
                    type_name,
                    round(a_val, 2),
                    round(b_val, 2),
                    round(delta, 2),
                    round(pct, 2),
                ])

        logger.info("Per-type comparison exported to %s", path)

    # ─────────────────────────────────────────
    # STATUS & STATE
    # ─────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """Whether a comparison is currently running."""
        return self._is_running

    @property
    def progress(self) -> float:
        """Progress of current comparison run (0.0 – 1.0)."""
        return self._progress

    @property
    def last_result(self) -> Optional[ComparisonResult]:
        """The most recent comparison result."""
        return self._last_result

    def get_status(self) -> Dict[str, Any]:
        """Get current status for the frontend."""
        return {
            "is_running": self._is_running,
            "progress": round(self._progress, 2),
            "has_result": self._last_result is not None,
            "available_durations": self.config.available_durations,
        }

    def __repr__(self) -> str:
        status = "running" if self._is_running else "idle"
        has_result = "with result" if self._last_result else "no result"
        return f"ScenarioComparison({status}, {has_result})"
