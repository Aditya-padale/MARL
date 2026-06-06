"""
metrics.py — Research metrics tracker for MARL City Simulator.

Computes and logs all 7 research metrics every timestep for IEEE publication:

  1. Gini Coefficient         — Wealth inequality (Lorenz curve formula)
  2. Social Mobility Index    — % agents changing wealth tiers in last 30 steps
  3. Poverty Rate             — % agents below ₹5,000 poverty threshold
  4. Median Wealth            — More robust than mean for skewed distributions
  5. Top-10% Wealth Conc.     — Sum of top 10 agents' wealth / total wealth
  6. Bankruptcy Rate           — % agents below ₹500 survival threshold
  7. Policy Effectiveness      — Composite 0–100 score measuring policy impact

All metrics are logged to arrays for time-series analysis and can be
exported to CSV for direct use in research paper data tables.

Author: Aditya Padale (B.Tech Final Year Project)
"""

import csv
import math
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from config import (
    MetricsConfig, METRICS_CONFIG,
    ComparisonConfig, COMPARISON_CONFIG,
    POVERTY_THRESHOLD, BANKRUPTCY_THRESHOLD,
    AgentType, AGENT_TYPE_NAMES,
    STEPS_PER_MONTH,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# METRICS SNAPSHOT
# ═══════════════════════════════════════════

@dataclass
class MetricsSnapshot:
    """
    Complete metrics snapshot at a single timestep.
    Stored in the time-series log for trend analysis.
    """
    timestep: int
    gini_coefficient: float
    social_mobility: Dict[str, float]    # upward%, downward%, stagnant%
    poverty_rate: float
    median_wealth: float
    mean_wealth: float
    top_10_concentration: float
    bankruptcy_rate: float
    bankruptcy_count: int
    total_wealth: float
    total_debt: float
    policy_effectiveness: float           # Latest composite score
    wealth_distribution: Dict[str, int]   # tier_name → count
    type_avg_wealth: Dict[str, float]     # type_name → average wealth


# ═══════════════════════════════════════════
# METRICS TRACKER
# ═══════════════════════════════════════════

class MetricsTracker:
    """
    Tracks all 7 research metrics across the simulation lifetime.

    Usage:
        tracker = MetricsTracker()
        tracker.log_timestep(agents, timestep, active_policies)
        current = tracker.get_current_metrics()
        impact = tracker.get_policy_impact("UBI", before_ts=100, after_ts=200)
        tracker.export_csv("data/metrics_export.csv")
    """

    def __init__(self, config: MetricsConfig = METRICS_CONFIG):
        self.config = config

        # Time-series storage
        self.history: List[MetricsSnapshot] = []
        self.max_history: int = 5000       # Keep up to ~14 simulation years

        # Wealth tier tracking for mobility computation
        # agent_id → list of (timestep, tier) entries
        self._tier_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        # Policy tracking for effectiveness scoring
        self._policy_baselines: Dict[str, MetricsSnapshot] = {}
        self._policy_results: Dict[str, Dict[str, float]] = {}

        # Current snapshot cache
        self._current: Optional[MetricsSnapshot] = None

    # ─────────────────────────────────────────
    # MAIN LOGGING METHOD
    # ─────────────────────────────────────────

    def log_timestep(self, agents: list, timestep: int,
                     active_policies: List[str] = None) -> MetricsSnapshot:
        """
        Compute all 7 metrics and log a snapshot for this timestep.

        Called once per simulation step (or less frequently for performance).

        Args:
            agents:          List of all 100 Agent objects
            timestep:        Current simulation timestep
            active_policies: List of currently active policy names

        Returns:
            MetricsSnapshot with all computed metrics
        """
        if not agents:
            return self._empty_snapshot(timestep)

        # Extract wealth values
        wealths = [a.finance.wealth for a in agents]
        n = len(wealths)

        # ── 1. Gini Coefficient ──
        gini = self._compute_gini(wealths)

        # ── 2. Social Mobility Index ──
        self._update_tier_history(agents, timestep)
        mobility = self._compute_social_mobility(agents, timestep)

        # ── 3. Poverty Rate ──
        poverty_count = sum(1 for w in wealths if w < POVERTY_THRESHOLD)
        poverty_rate = poverty_count / n

        # ── 4. Median Wealth ──
        sorted_wealths = sorted(wealths)
        if n % 2 == 0:
            median_wealth = (sorted_wealths[n // 2 - 1] + sorted_wealths[n // 2]) / 2
        else:
            median_wealth = sorted_wealths[n // 2]

        # Mean wealth
        mean_wealth = sum(wealths) / n

        # ── 5. Top-10% Wealth Concentration ──
        top_n = self.config.top_n_concentration
        top_wealths = sorted(wealths, reverse=True)[:top_n]
        total_wealth = sum(wealths)
        top_10_concentration = (
            sum(top_wealths) / total_wealth if total_wealth > 0 else 0.0
        )

        # ── 6. Bankruptcy Rate ──
        bankruptcy_count = sum(1 for a in agents if a.finance.is_bankrupt)
        bankruptcy_rate = bankruptcy_count / n

        # ── 7. Policy Effectiveness Score ──
        policy_effectiveness = self._compute_latest_policy_effectiveness()

        # ── Wealth Distribution ──
        wealth_dist = self._compute_wealth_distribution(wealths)

        # ── Per-Type Average Wealth ──
        type_avg = self._compute_type_averages(agents)

        # ── Total Debt ──
        total_debt = sum(a.finance.debt for a in agents)

        # Build snapshot
        snapshot = MetricsSnapshot(
            timestep=timestep,
            gini_coefficient=round(gini, 4),
            social_mobility=mobility,
            poverty_rate=round(poverty_rate, 4),
            median_wealth=round(median_wealth, 2),
            mean_wealth=round(mean_wealth, 2),
            top_10_concentration=round(top_10_concentration, 4),
            bankruptcy_rate=round(bankruptcy_rate, 4),
            bankruptcy_count=bankruptcy_count,
            total_wealth=round(total_wealth, 2),
            total_debt=round(total_debt, 2),
            policy_effectiveness=round(policy_effectiveness, 2),
            wealth_distribution=wealth_dist,
            type_avg_wealth=type_avg,
        )

        # Store
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        self._current = snapshot
        return snapshot

    # ─────────────────────────────────────────
    # METRIC 1: GINI COEFFICIENT
    # ─────────────────────────────────────────

    @staticmethod
    def _compute_gini(wealths: List[float]) -> float:
        """
        Compute the Gini coefficient using the standard formula.

        Gini = (2 × Σ(i × y_i)) / (n × Σ(y_i)) - (n + 1) / n

        Where y_i are sorted wealth values.

        Range: 0 (perfect equality) to 1 (perfect inequality)

        This is the standard Lorenz curve-based formula used in
        economics research. For an IEEE paper, this is the most
        widely recognized inequality metric.
        """
        n = len(wealths)
        if n == 0:
            return 0.0

        # Shift to non-negative (Gini requires non-negative values)
        min_w = min(wealths)
        shifted = [w - min_w for w in wealths] if min_w < 0 else list(wealths)

        sorted_w = sorted(shifted)
        total = sum(sorted_w)

        if total == 0:
            return 0.0

        # Standard Gini formula
        cumulative = 0.0
        weighted_sum = 0.0
        for i, w in enumerate(sorted_w):
            cumulative += w
            weighted_sum += (i + 1) * w

        gini = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n
        return max(0.0, min(gini, 1.0))

    # ─────────────────────────────────────────
    # METRIC 2: SOCIAL MOBILITY INDEX
    # ─────────────────────────────────────────

    def _update_tier_history(self, agents: list, timestep: int):
        """Record current wealth tier for each agent."""
        for agent in agents:
            tier = self._get_wealth_tier(agent.finance.wealth)
            history = self._tier_history[agent.id]
            # Only record if tier changed or first entry
            if not history or history[-1][1] != tier:
                history.append((timestep, tier))
            # Prune old entries
            cutoff = timestep - self.config.mobility_window * 2
            self._tier_history[agent.id] = [
                (ts, t) for ts, t in history if ts >= cutoff
            ]

    def _compute_social_mobility(self, agents: list,
                                  timestep: int) -> Dict[str, float]:
        """
        Compute social mobility: % of agents who changed wealth tier
        in the last mobility_window timesteps.

        Returns:
            Dict with keys: upward_pct, downward_pct, stagnant_pct
        """
        window = self.config.mobility_window
        lookback = timestep - window
        n = len(agents)
        if n == 0:
            return {"upward_pct": 0.0, "downward_pct": 0.0, "stagnant_pct": 1.0}

        upward = 0
        downward = 0
        stagnant = 0

        for agent in agents:
            history = self._tier_history.get(agent.id, [])
            if len(history) < 2:
                stagnant += 1
                continue

            # Find tier at lookback point
            old_tier = history[0][1]  # Earliest recorded
            for ts, tier in history:
                if ts <= lookback:
                    old_tier = tier
                else:
                    break

            current_tier = self._get_wealth_tier(agent.finance.wealth)

            if current_tier > old_tier:
                upward += 1
            elif current_tier < old_tier:
                downward += 1
            else:
                stagnant += 1

        return {
            "upward_pct": round(upward / n, 4),
            "downward_pct": round(downward / n, 4),
            "stagnant_pct": round(stagnant / n, 4),
        }

    def _get_wealth_tier(self, wealth: float) -> int:
        """Classify wealth into tier for mobility tracking."""
        boundaries = self.config.tier_boundaries
        for i, boundary in enumerate(boundaries):
            if wealth < boundary:
                return i
        return len(boundaries)  # Above all boundaries = highest tier

    # ─────────────────────────────────────────
    # METRIC 7: POLICY EFFECTIVENESS
    # ─────────────────────────────────────────

    def record_policy_baseline(self, policy_name: str):
        """
        Record current metrics as baseline before a policy is applied.

        Call this BEFORE applying a policy to capture the "before" state.

        Args:
            policy_name: Name of the policy about to be applied
        """
        if self._current:
            self._policy_baselines[policy_name] = self._current
            logger.info("Recorded baseline for policy '%s' at T=%d",
                        policy_name, self._current.timestep)

    def compute_policy_effectiveness(self, policy_name: str) -> float:
        """
        Compute the Policy Effectiveness Score for a specific policy.

        Formula:
            score = 100 × (
                0.30 × (1 - gini_after / gini_before)      +
                0.25 × (1 - poverty_after / poverty_before) +
                0.20 × mobility_after                        +
                0.15 × (1 - bankruptcy_after / before)       +
                0.10 × (median_wealth_after / before - 1)
            )

        Higher score = policy improved societal outcomes.

        Args:
            policy_name: Name of the policy to evaluate

        Returns:
            Score from 0–100 (can exceed 100 for exceptional policies)
        """
        baseline = self._policy_baselines.get(policy_name)
        if not baseline or not self._current:
            return 0.0

        before = baseline
        after = self._current
        cfg = COMPARISON_CONFIG

        # Gini improvement (lower is better)
        gini_score = 1.0 - (after.gini_coefficient /
                            max(before.gini_coefficient, 0.001))

        # Poverty reduction (lower is better)
        poverty_score = 1.0 - (after.poverty_rate /
                               max(before.poverty_rate, 0.001))

        # Social mobility (higher upward is better)
        mobility_score = after.social_mobility.get("upward_pct", 0.0)

        # Bankruptcy reduction (lower is better)
        bankruptcy_score = 1.0 - (after.bankruptcy_rate /
                                  max(before.bankruptcy_rate, 0.001))

        # Median wealth growth
        wealth_score = (after.median_wealth /
                        max(before.median_wealth, 1.0)) - 1.0

        # Composite score
        score = 100.0 * (
            cfg.weight_gini * gini_score +
            cfg.weight_poverty * poverty_score +
            cfg.weight_mobility * mobility_score +
            cfg.weight_bankruptcy * bankruptcy_score +
            cfg.weight_median_wealth * wealth_score
        )

        # Clamp to [0, 100] for display, but store raw for analysis
        display_score = max(0.0, min(100.0, score))

        self._policy_results[policy_name] = {
            "score": round(score, 2),
            "display_score": round(display_score, 2),
            "gini_delta": round(after.gini_coefficient - before.gini_coefficient, 4),
            "poverty_delta": round(after.poverty_rate - before.poverty_rate, 4),
            "mobility_upward": round(after.social_mobility.get("upward_pct", 0.0), 4),
            "bankruptcy_delta": round(after.bankruptcy_rate - before.bankruptcy_rate, 4),
            "median_wealth_delta": round(after.median_wealth - before.median_wealth, 2),
            "baseline_timestep": before.timestep,
            "eval_timestep": after.timestep,
        }

        logger.info("Policy '%s' effectiveness: %.1f/100", policy_name, display_score)
        return display_score

    def _compute_latest_policy_effectiveness(self) -> float:
        """Compute effectiveness for the most recent policy evaluated."""
        if not self._policy_results:
            return 0.0
        latest = list(self._policy_results.values())[-1]
        return latest.get("display_score", 0.0)

    def get_policy_impact(self, policy_name: str,
                          before_ts: int, after_ts: int) -> Dict[str, Any]:
        """
        Get detailed policy impact analysis between two timesteps.

        Useful for generating comparison tables in the research paper.

        Args:
            policy_name: Policy name
            before_ts:   Timestep to use as "before" baseline
            after_ts:    Timestep to use as "after" measurement

        Returns:
            Dict with detailed metric deltas
        """
        before = self._find_snapshot_at(before_ts)
        after = self._find_snapshot_at(after_ts)

        if not before or not after:
            return {"error": "Snapshots not found for specified timesteps"}

        return {
            "policy_name": policy_name,
            "before_timestep": before.timestep,
            "after_timestep": after.timestep,
            "steps_elapsed": after.timestep - before.timestep,
            "gini": {
                "before": before.gini_coefficient,
                "after": after.gini_coefficient,
                "delta": round(after.gini_coefficient - before.gini_coefficient, 4),
            },
            "poverty_rate": {
                "before": before.poverty_rate,
                "after": after.poverty_rate,
                "delta": round(after.poverty_rate - before.poverty_rate, 4),
            },
            "social_mobility": {
                "before": before.social_mobility,
                "after": after.social_mobility,
            },
            "median_wealth": {
                "before": before.median_wealth,
                "after": after.median_wealth,
                "delta": round(after.median_wealth - before.median_wealth, 2),
                "pct_change": round(
                    (after.median_wealth - before.median_wealth) /
                    max(before.median_wealth, 1.0) * 100, 2
                ),
            },
            "top_10_concentration": {
                "before": before.top_10_concentration,
                "after": after.top_10_concentration,
                "delta": round(
                    after.top_10_concentration - before.top_10_concentration, 4
                ),
            },
            "bankruptcy_rate": {
                "before": before.bankruptcy_rate,
                "after": after.bankruptcy_rate,
                "delta": round(after.bankruptcy_rate - before.bankruptcy_rate, 4),
            },
            "total_wealth": {
                "before": before.total_wealth,
                "after": after.total_wealth,
                "delta": round(after.total_wealth - before.total_wealth, 2),
                "pct_change": round(
                    (after.total_wealth - before.total_wealth) /
                    max(before.total_wealth, 1.0) * 100, 2
                ),
            },
            "wealth_distribution": {
                "before": before.wealth_distribution,
                "after": after.wealth_distribution,
            },
            "type_avg_wealth": {
                "before": before.type_avg_wealth,
                "after": after.type_avg_wealth,
            },
        }

    # ─────────────────────────────────────────
    # HELPER COMPUTATIONS
    # ─────────────────────────────────────────

    @staticmethod
    def _compute_wealth_distribution(wealths: List[float]) -> Dict[str, int]:
        """Classify agents into wealth tiers and count."""
        dist = {
            "bankrupt": 0,     # < ₹500
            "poor": 0,         # ₹500 – ₹5,000
            "low_middle": 0,   # ₹5,000 – ₹20,000
            "middle": 0,       # ₹20,000 – ₹1,00,000
            "rich": 0,         # > ₹1,00,000
        }
        for w in wealths:
            if w < 500:
                dist["bankrupt"] += 1
            elif w < 5_000:
                dist["poor"] += 1
            elif w < 20_000:
                dist["low_middle"] += 1
            elif w < 100_000:
                dist["middle"] += 1
            else:
                dist["rich"] += 1
        return dist

    @staticmethod
    def _compute_type_averages(agents: list) -> Dict[str, float]:
        """Compute average wealth per agent type."""
        type_sums: Dict[int, float] = defaultdict(float)
        type_counts: Dict[int, int] = defaultdict(int)

        for agent in agents:
            type_sums[agent.agent_type] += agent.finance.wealth
            type_counts[agent.agent_type] += 1

        return {
            AGENT_TYPE_NAMES.get(t, f"Type_{t}"): round(
                type_sums[t] / max(type_counts[t], 1), 2
            )
            for t in type_sums
        }

    def _find_snapshot_at(self, timestep: int) -> Optional[MetricsSnapshot]:
        """Find the snapshot closest to a given timestep."""
        if not self.history:
            return None

        best = None
        best_diff = float("inf")
        for snap in self.history:
            diff = abs(snap.timestep - timestep)
            if diff < best_diff:
                best_diff = diff
                best = snap
        return best

    def _empty_snapshot(self, timestep: int) -> MetricsSnapshot:
        """Return an empty snapshot for edge cases."""
        return MetricsSnapshot(
            timestep=timestep,
            gini_coefficient=0.0,
            social_mobility={"upward_pct": 0.0, "downward_pct": 0.0, "stagnant_pct": 1.0},
            poverty_rate=0.0,
            median_wealth=0.0,
            mean_wealth=0.0,
            top_10_concentration=0.0,
            bankruptcy_rate=0.0,
            bankruptcy_count=0,
            total_wealth=0.0,
            total_debt=0.0,
            policy_effectiveness=0.0,
            wealth_distribution={"bankrupt": 0, "poor": 0, "low_middle": 0, "middle": 0, "rich": 0},
            type_avg_wealth={},
        )

    # ─────────────────────────────────────────
    # CURRENT METRICS (for WebSocket)
    # ─────────────────────────────────────────

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics dict for WebSocket diff updates.

        This is sent to the frontend every tick for the live dashboard.

        Returns:
            Dict with all 7 metrics plus supplementary data
        """
        if not self._current:
            return {
                "gini": 0.0,
                "poverty_rate": 0.0,
                "bankruptcy_count": 0,
                "bankruptcy_rate": 0.0,
                "median_wealth": 0.0,
                "mean_wealth": 0.0,
                "top_10_concentration": 0.0,
                "social_mobility": {"upward_pct": 0.0, "downward_pct": 0.0, "stagnant_pct": 1.0},
                "total_wealth": 0.0,
                "total_debt": 0.0,
                "policy_effectiveness": 0.0,
                "wealth_distribution": {},
            }

        s = self._current
        return {
            "gini": s.gini_coefficient,
            "poverty_rate": s.poverty_rate,
            "bankruptcy_count": s.bankruptcy_count,
            "bankruptcy_rate": s.bankruptcy_rate,
            "median_wealth": s.median_wealth,
            "mean_wealth": s.mean_wealth,
            "top_10_concentration": s.top_10_concentration,
            "social_mobility": s.social_mobility,
            "total_wealth": s.total_wealth,
            "total_debt": s.total_debt,
            "policy_effectiveness": s.policy_effectiveness,
            "wealth_distribution": s.wealth_distribution,
            "type_avg_wealth": s.type_avg_wealth,
        }

    def get_trend_data(self, last_n: int = 60) -> Dict[str, List]:
        """
        Get time-series trend data for Chart.js dashboard charts.

        Args:
            last_n: Number of recent snapshots to include

        Returns:
            Dict mapping metric name to array of values
        """
        recent = self.history[-last_n:]
        if not recent:
            return {}

        return {
            "timesteps": [s.timestep for s in recent],
            "gini": [s.gini_coefficient for s in recent],
            "poverty_rate": [s.poverty_rate for s in recent],
            "median_wealth": [s.median_wealth for s in recent],
            "mean_wealth": [s.mean_wealth for s in recent],
            "top_10_concentration": [s.top_10_concentration for s in recent],
            "bankruptcy_rate": [s.bankruptcy_rate for s in recent],
            "total_wealth": [s.total_wealth for s in recent],
            "total_debt": [s.total_debt for s in recent],
            "upward_mobility": [s.social_mobility.get("upward_pct", 0.0) for s in recent],
            "downward_mobility": [s.social_mobility.get("downward_pct", 0.0) for s in recent],
        }

    # ─────────────────────────────────────────
    # CSV EXPORT (for IEEE paper data tables)
    # ─────────────────────────────────────────

    def export_csv(self, path: str = None):
        """
        Export all metrics history to CSV for research paper tables.

        Format: one row per timestep with all 7 metrics.

        Args:
            path: Output file path (defaults to config.export_path)
        """
        path = path or self.config.export_path

        if not self.history:
            logger.warning("No metrics history to export")
            return

        fieldnames = [
            "timestep", "gini_coefficient", "poverty_rate", "median_wealth",
            "mean_wealth", "top_10_concentration", "bankruptcy_rate",
            "bankruptcy_count", "total_wealth", "total_debt",
            "upward_mobility_pct", "downward_mobility_pct",
            "policy_effectiveness",
            "dist_bankrupt", "dist_poor", "dist_low_middle",
            "dist_middle", "dist_rich",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for snap in self.history:
                row = {
                    "timestep": snap.timestep,
                    "gini_coefficient": snap.gini_coefficient,
                    "poverty_rate": snap.poverty_rate,
                    "median_wealth": snap.median_wealth,
                    "mean_wealth": snap.mean_wealth,
                    "top_10_concentration": snap.top_10_concentration,
                    "bankruptcy_rate": snap.bankruptcy_rate,
                    "bankruptcy_count": snap.bankruptcy_count,
                    "total_wealth": snap.total_wealth,
                    "total_debt": snap.total_debt,
                    "upward_mobility_pct": snap.social_mobility.get("upward_pct", 0.0),
                    "downward_mobility_pct": snap.social_mobility.get("downward_pct", 0.0),
                    "policy_effectiveness": snap.policy_effectiveness,
                    "dist_bankrupt": snap.wealth_distribution.get("bankrupt", 0),
                    "dist_poor": snap.wealth_distribution.get("poor", 0),
                    "dist_low_middle": snap.wealth_distribution.get("low_middle", 0),
                    "dist_middle": snap.wealth_distribution.get("middle", 0),
                    "dist_rich": snap.wealth_distribution.get("rich", 0),
                }
                writer.writerow(row)

        logger.info("Metrics exported to %s (%d rows)", path, len(self.history))

    def export_policy_results_csv(self, path: str):
        """Export all policy effectiveness results to CSV."""
        if not self._policy_results:
            logger.warning("No policy results to export")
            return

        fieldnames = [
            "policy_name", "score", "gini_delta", "poverty_delta",
            "mobility_upward", "bankruptcy_delta", "median_wealth_delta",
            "baseline_timestep", "eval_timestep",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for name, results in self._policy_results.items():
                row = {"policy_name": name}
                row.update(results)
                writer.writerow(row)

        logger.info("Policy results exported to %s (%d policies)",
                    path, len(self._policy_results))

    # ─────────────────────────────────────────
    # YEAR IN REVIEW
    # ─────────────────────────────────────────

    def generate_year_review(self, agents: list,
                             timestep: int) -> Dict[str, Any]:
        """
        Generate the Year in Review panel data.

        Summarizes the past year of simulation for the researcher.

        Args:
            agents:   All agents
            timestep: Current timestep

        Returns:
            Dict with year review data for the frontend panel
        """
        year_start_ts = max(0, timestep - 365)

        # Find snapshots from year start and now
        start_snap = self._find_snapshot_at(year_start_ts)
        end_snap = self._current

        if not start_snap or not end_snap:
            return {"error": "Insufficient data for year review"}

        # Top 5 gainers and losers
        agents_sorted = sorted(agents, key=lambda a: a.finance.wealth, reverse=True)
        top_gainers = []
        top_losers = []

        for agent in agents:
            if len(agent.wealth_history) >= 2:
                gain = agent.finance.wealth - agent.wealth_history[0]
                entry = {
                    "id": agent.id,
                    "type": agent.type_name,
                    "gain": round(gain, 0),
                    "current_wealth": round(agent.finance.wealth, 0),
                }
                if gain > 0:
                    top_gainers.append(entry)
                else:
                    top_losers.append(entry)

        top_gainers.sort(key=lambda x: x["gain"], reverse=True)
        top_losers.sort(key=lambda x: x["gain"])

        # Survivors count
        survivors = sum(1 for a in agents if not a.finance.is_bankrupt)

        # Best policy
        best_policy = None
        best_score = 0.0
        for name, results in self._policy_results.items():
            score = results.get("display_score", 0.0)
            if score > best_score:
                best_score = score
                best_policy = name

        return {
            "year": timestep // 365,
            "survivors": survivors,
            "total_agents": len(agents),
            "top_5_gainers": top_gainers[:5],
            "top_5_losers": top_losers[:5],
            "gini_trajectory": {
                "start": start_snap.gini_coefficient,
                "end": end_snap.gini_coefficient,
                "improved": end_snap.gini_coefficient < start_snap.gini_coefficient,
            },
            "poverty_trajectory": {
                "start": start_snap.poverty_rate,
                "end": end_snap.poverty_rate,
            },
            "best_policy": {
                "name": best_policy,
                "score": best_score,
            } if best_policy else None,
            "social_mobility_summary": end_snap.social_mobility,
            "wealth_distribution": end_snap.wealth_distribution,
        }

    # ─────────────────────────────────────────
    # SNAPSHOT & RESTORE
    # ─────────────────────────────────────────

    def get_snapshot(self) -> Dict[str, Any]:
        """Get serializable snapshot for comparison mode."""
        return {
            "history_length": len(self.history),
            "policy_baselines": {
                k: v.timestep for k, v in self._policy_baselines.items()
            },
            "policy_results": dict(self._policy_results),
        }

    def __repr__(self) -> str:
        if self._current:
            return (
                f"MetricsTracker(T={self._current.timestep}, "
                f"gini={self._current.gini_coefficient:.3f}, "
                f"poverty={self._current.poverty_rate:.1%}, "
                f"bankruptcies={self._current.bankruptcy_count})"
            )
        return "MetricsTracker(no data)"
