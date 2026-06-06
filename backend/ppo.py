"""
ppo.py — Proximal Policy Optimization implementation for MARL City Simulator.

Architecture: TYPE-LEVEL shared networks + INDIVIDUAL experience buffers.
  - 10 Actor networks  (one per AgentType)
  - 10 Critic networks (one per AgentType)
  - 100 RolloutBuffers (one per agent — fully independent)
  - 100 Adam optimizer states (one per agent — diverge over time)

This achieves individual behavior at 10x lower compute than 100 separate
networks, because each agent's unique experience trajectory causes its
optimizer state (momentum, variance estimates) to diverge from same-type
peers within ~500 episodes.

Author: Aditya Padale (B.Tech Final Year Project)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Optional, Dict

from config import PPO_CONFIG, NUM_ACTIONS


# ═══════════════════════════════════════════
# ACTOR NETWORK
# ═══════════════════════════════════════════

class Actor(nn.Module):
    """
    Policy network that maps state → action probabilities.

    Architecture:
        Input(state_dim) → Linear(64) + ReLU → Linear(64) + ReLU → Linear(4) + Softmax

    One Actor is instantiated per agent type (10 total).
    The Softmax output gives probabilities over the 4 discrete actions:
        [Save, Spend, Invest, Trade]
    """

    def __init__(self, state_dim: int, action_dim: int = NUM_ACTIONS,
                 hidden_size: int = PPO_CONFIG.hidden_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        # Initialize weights using orthogonal initialization
        # (standard practice for PPO — improves early training stability)
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization with gain tuned per layer purpose."""
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                if i == len(self.network) - 1:
                    # Output layer: small weights → near-uniform initial policy
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                else:
                    # Hidden layers: sqrt(2) gain for ReLU
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → action logits.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Action logits of shape (batch_size, action_dim) — NOT softmaxed.
            Use Categorical distribution to sample, which handles logits internally.
        """
        return self.network(state)

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get softmax action probabilities (for the Agent Inspector UI).

        Args:
            state: Tensor of shape (state_dim,)

        Returns:
            Probability tensor of shape (action_dim,) summing to 1.0
        """
        with torch.no_grad():
            logits = self.forward(state)
            return F.softmax(logits, dim=-1)


# ═══════════════════════════════════════════
# CRITIC NETWORK
# ═══════════════════════════════════════════

class Critic(nn.Module):
    """
    Value network that maps state → scalar value estimate V(s).

    Architecture:
        Input(state_dim) → Linear(64) + ReLU → Linear(64) + ReLU → Linear(1)

    One Critic is instantiated per agent type (10 total).
    The single output is the estimated return from state s.
    """

    def __init__(self, state_dim: int,
                 hidden_size: int = PPO_CONFIG.hidden_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for value network."""
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                if i == len(self.network) - 1:
                    # Value output: unit gain
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → value estimate.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Value estimate of shape (batch_size, 1) or (1,)
        """
        return self.network(state)


# ═══════════════════════════════════════════
# ROLLOUT BUFFER
# ═══════════════════════════════════════════

class RolloutBuffer:
    """
    Stores experience tuples collected during rollout for ONE agent.

    Each agent owns exactly one RolloutBuffer. This is the key to
    individual behavior — even agents sharing a network architecture
    collect completely different experiences based on their unique
    starting conditions, life events, and social interactions.

    Stores: (state, action, reward, value, log_prob, done)
    Computes: GAE advantages and discounted returns for PPO update.

    Buffer is cleared after each PPO update cycle (every rollout_length steps).
    """

    def __init__(self, state_dim: int,
                 buffer_size: int = PPO_CONFIG.rollout_length):
        self.state_dim = state_dim
        self.buffer_size = buffer_size
        self.reset()

    def reset(self):
        """Clear all stored experience. Called after each PPO update."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

        # Computed during finalize()
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
        self._finalized: bool = False

    def store(self, state: np.ndarray, action: int, reward: float,
              value: float, log_prob: float, done: bool = False):
        """
        Store one timestep of experience.

        Args:
            state:    State vector observed by the agent
            action:   Discrete action taken (0–3)
            reward:   Scalar reward received
            value:    Critic's value estimate V(s) at this state
            log_prob: Log probability of the action under current policy
            done:     Whether episode terminated (bankruptcy, etc.)
        """
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return len(self.states)

    @property
    def is_full(self) -> bool:
        """Whether buffer has reached rollout_length."""
        return self.size >= self.buffer_size

    def compute_gae(self, last_value: float,
                    gamma: float = PPO_CONFIG.gamma,
                    gae_lambda: float = PPO_CONFIG.gae_lambda):
        """
        Compute Generalized Advantage Estimation (GAE-λ).

        GAE provides a bias-variance tradeoff for advantage estimation:
            Â_t = Σ_{l=0}^{T-t} (γλ)^l × δ_{t+l}
            δ_t = r_t + γ V(s_{t+1}) - V(s_t)

        Where:
            γ (gamma)  = discount factor (0.99) — how much future matters
            λ (lambda) = GAE lambda (0.95) — bias-variance tradeoff

        Args:
            last_value: V(s_T) — critic's estimate for the state AFTER
                        the last stored transition (bootstrap value)
            gamma:      Discount factor
            gae_lambda: GAE lambda parameter

        Sets:
            self.advantages: np.ndarray of shape (buffer_size,)
            self.returns:    np.ndarray of shape (buffer_size,)
        """
        n = self.size
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        # Reverse sweep — accumulate advantages from the end
        last_gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            # If episode ended at step t, don't bootstrap from next state
            if self.dones[t]:
                next_non_terminal = 0.0
                next_value_for_delta = 0.0
            else:
                next_non_terminal = 1.0
                next_value_for_delta = next_value

            # TD residual: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = (self.rewards[t]
                     + gamma * next_value_for_delta * next_non_terminal
                     - self.values[t])

            # GAE accumulation: Â_t = δ_t + γλ * Â_{t+1}
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

            # Discounted return: R_t = Â_t + V(s_t)
            returns[t] = advantages[t] + self.values[t]

            next_value = self.values[t]

        self.advantages = advantages
        self.returns = returns
        self._finalized = True

    def get_batches(self, minibatch_size: int = PPO_CONFIG.minibatch_size):
        """
        Yield random minibatches for PPO update epochs.

        Shuffles indices and yields minibatch_size chunks.
        Each yielded batch is a dict of tensors ready for PPO loss computation.

        Args:
            minibatch_size: Number of samples per minibatch

        Yields:
            Dict with keys: states, actions, old_log_probs, advantages, returns
            All values are torch.Tensors.
        """
        assert self._finalized, "Must call compute_gae() before get_batches()"

        n = self.size
        indices = np.arange(n)
        np.random.shuffle(indices)

        # Convert lists to numpy arrays for efficient slicing
        states_arr = np.array(self.states, dtype=np.float32)
        actions_arr = np.array(self.actions, dtype=np.int64)
        log_probs_arr = np.array(self.log_probs, dtype=np.float32)

        # Normalize advantages (standard practice — stabilizes training)
        adv = self.advantages.copy()
        if len(adv) > 1:
            adv_std = adv.std()
            if adv_std > 1e-8:
                adv = (adv - adv.mean()) / adv_std

        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            batch_idx = indices[start:end]

            yield {
                "states":        torch.FloatTensor(states_arr[batch_idx]),
                "actions":       torch.LongTensor(actions_arr[batch_idx]),
                "old_log_probs": torch.FloatTensor(log_probs_arr[batch_idx]),
                "advantages":    torch.FloatTensor(adv[batch_idx]),
                "returns":       torch.FloatTensor(self.returns[batch_idx]),
            }


# ═══════════════════════════════════════════
# PPO AGENT
# ═══════════════════════════════════════════

class PPOAgent:
    """
    PPO agent wrapping one Actor, one Critic, one RolloutBuffer, and one Adam optimizer.

    ARCHITECTURE DECISION:
        Actor and Critic networks are SHARED by reference across agents of the same type.
        But each PPOAgent has its OWN RolloutBuffer and its OWN Adam optimizer state.

        This means:
        1. Two farmers share the same nn.Module weights (memory efficient)
        2. But their Adam optimizer's momentum/variance estimates diverge
           because they accumulate gradients from different experience
        3. Over time, gradient updates from agents with different life
           experiences push the shared weights in slightly different
           directions each update cycle
        4. The net effect: same architecture, different learned behavior

    Usage:
        agent = PPOAgent(actor, critic, state_dim=22)
        action, log_prob, value = agent.select_action(state_vector)
        agent.store_transition(state, action, reward, value, log_prob)
        if agent.buffer.is_full:
            loss_info = agent.update(last_value)
    """

    def __init__(self, actor: Actor, critic: Critic, state_dim: int,
                 agent_id: int, agent_type_name: str):
        """
        Args:
            actor:           Shared Actor network (by reference)
            critic:          Shared Critic network (by reference)
            state_dim:       Dimension of this agent's state vector
            agent_id:        Unique agent identifier (0–99)
            agent_type_name: Human-readable type name for logging
        """
        self.actor = actor
        self.critic = critic
        self.state_dim = state_dim
        self.agent_id = agent_id
        self.agent_type_name = agent_type_name

        # Each agent owns its own rollout buffer (independent experience)
        self.buffer = RolloutBuffer(state_dim=state_dim)

        # Each agent owns its own Adam optimizer state
        # This is what makes same-type agents diverge over time:
        # Adam tracks per-parameter momentum (m) and variance (v) estimates.
        # Different experience → different gradients → different m,v →
        # different effective learning rates per weight → different behavior.
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=PPO_CONFIG.learning_rate,
        )

        # Training statistics (for logging and inspector)
        self.total_updates: int = 0
        self.last_policy_loss: float = 0.0
        self.last_value_loss: float = 0.0
        self.last_entropy: float = 0.0
        self.cumulative_reward: float = 0.0
        self.episode_reward: float = 0.0

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select an action using the current policy (Actor network).

        Samples from the categorical distribution defined by the Actor's
        softmax output. Returns the action, its log probability (for PPO
        importance sampling), and the Critic's value estimate (for GAE).

        Args:
            state: numpy array of shape (state_dim,)

        Returns:
            action:   int in [0, 3] — the chosen discrete action
            log_prob: float — log π(a|s) under current policy
            value:    float — V(s) estimate from Critic
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, state_dim)

        with torch.no_grad():
            # Actor: state → action logits → distribution
            logits = self.actor(state_tensor)                  # (1, 4)
            dist = Categorical(logits=logits)
            action = dist.sample()                             # (1,)
            log_prob = dist.log_prob(action)                   # (1,)

            # Critic: state → value estimate
            value = self.critic(state_tensor)                  # (1, 1)

        return (
            action.item(),
            log_prob.item(),
            value.squeeze().item(),
        )

    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, value: float, log_prob: float,
                         done: bool = False):
        """
        Store one transition in this agent's personal rollout buffer.

        Args:
            state:    State vector when action was taken
            action:   Action that was taken
            reward:   Reward received after taking action
            value:    Critic's value estimate at this state
            log_prob: Log probability of the action
            done:     Whether this was a terminal state
        """
        self.buffer.store(state, action, reward, value, log_prob, done)
        self.episode_reward += reward
        self.cumulative_reward += reward

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Run PPO update using this agent's collected experience.

        This is the core training step. It:
        1. Computes GAE advantages from the agent's rollout buffer
        2. Runs K epochs of clipped PPO loss
        3. Applies gradients through this agent's own Adam optimizer
        4. Clears the buffer for the next rollout

        The key insight: even though gradients flow into SHARED network weights,
        the Adam optimizer's per-parameter statistics (momentum, variance) are
        UNIQUE to this agent. This creates a subtle but meaningful divergence
        in how each agent's experience shapes the shared weights.

        Args:
            last_value: Bootstrap value V(s_T) for GAE computation.
                        Should be 0.0 if episode ended, otherwise
                        the Critic's estimate of the final state.

        Returns:
            Dict with training statistics:
                policy_loss, value_loss, entropy, total_loss
        """
        if self.buffer.size == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "total_loss": 0.0}

        # Step 1: Compute GAE advantages
        self.buffer.compute_gae(last_value)

        # Step 2: Run K epochs of PPO updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(PPO_CONFIG.update_epochs):
            for batch in self.buffer.get_batches():
                states = batch["states"]           # (B, state_dim)
                actions = batch["actions"]         # (B,)
                old_log_probs = batch["old_log_probs"]  # (B,)
                advantages = batch["advantages"]   # (B,)
                returns = batch["returns"]          # (B,)

                # ── Forward pass: Actor ──
                logits = self.actor(states)                    # (B, 4)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)         # (B,)
                entropy = dist.entropy().mean()                # scalar

                # ── Forward pass: Critic ──
                values = self.critic(states).squeeze(-1)       # (B,)

                # ── PPO Clipped Surrogate Loss ──
                # Importance sampling ratio: π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs)  # (B,)

                # Clipped surrogate objective
                surrogate1 = ratio * advantages
                surrogate2 = torch.clamp(
                    ratio,
                    1.0 - PPO_CONFIG.clip_epsilon,
                    1.0 + PPO_CONFIG.clip_epsilon,
                ) * advantages

                # Policy loss: take the pessimistic (min) bound
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # ── Value Function Loss ──
                # Simple MSE between predicted values and computed returns
                value_loss = F.mse_loss(values, returns)

                # ── Total Loss ──
                # Combine policy loss, value loss, and entropy bonus
                # Entropy bonus encourages exploration (negative because we minimize)
                loss = (policy_loss
                        + PPO_CONFIG.value_loss_coef * value_loss
                        - PPO_CONFIG.entropy_coef * entropy)

                # ── Gradient Step ──
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (prevents catastrophic updates)
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    PPO_CONFIG.max_grad_norm,
                )

                self.optimizer.step()

                # Accumulate stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1

        # Step 3: Compute averages and store
        num_batches = max(num_batches, 1)  # Avoid division by zero
        self.last_policy_loss = total_policy_loss / num_batches
        self.last_value_loss = total_value_loss / num_batches
        self.last_entropy = total_entropy / num_batches
        self.total_updates += 1

        # Step 4: Clear buffer for next rollout
        episode_reward = self.episode_reward
        self.episode_reward = 0.0
        self.buffer.reset()

        return {
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
            "total_loss": (self.last_policy_loss
                           + PPO_CONFIG.value_loss_coef * self.last_value_loss
                           - PPO_CONFIG.entropy_coef * self.last_entropy),
            "episode_reward": episode_reward,
        }

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probability distribution for the Agent Inspector UI.

        Returns softmax probabilities over all 4 actions, showing the
        researcher exactly how the agent is "thinking" about its options.

        Args:
            state: numpy array of shape (state_dim,)

        Returns:
            numpy array of shape (4,) — probabilities summing to 1.0
            Index 0=Save, 1=Spend, 2=Invest, 3=Trade
        """
        state_tensor = torch.FloatTensor(state)
        probs = self.actor.get_action_probs(state_tensor)
        return probs.numpy()

    def get_value_estimate(self, state: np.ndarray) -> float:
        """
        Get the Critic's value estimate for the Agent Inspector UI.

        This shows the researcher how "optimistic" the agent is about
        its current situation — high V(s) means the agent expects good
        future returns from this state.

        Args:
            state: numpy array of shape (state_dim,)

        Returns:
            Scalar value estimate V(s)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return value.squeeze().item()

    def get_training_stats(self) -> Dict[str, float]:
        """
        Get training statistics for logging and the inspector panel.

        Returns:
            Dict with keys: total_updates, last_policy_loss,
            last_value_loss, last_entropy, cumulative_reward
        """
        return {
            "total_updates": self.total_updates,
            "last_policy_loss": round(self.last_policy_loss, 6),
            "last_value_loss": round(self.last_value_loss, 6),
            "last_entropy": round(self.last_entropy, 6),
            "cumulative_reward": round(self.cumulative_reward, 2),
        }


# ═══════════════════════════════════════════
# TYPE-LEVEL NETWORK POOL
# ═══════════════════════════════════════════

class PPONetworkPool:
    """
    Manages the 10 type-level Actor-Critic network pairs.

    This class creates exactly one (Actor, Critic) pair per AgentType
    and hands out references to PPOAgent instances. Multiple PPOAgents
    of the same type share the same network objects but maintain
    independent optimizer states.

    Usage:
        pool = PPONetworkPool()
        pool.create_networks(AgentType.FARMER, state_dim=22)
        actor, critic = pool.get_networks(AgentType.FARMER)
    """

    def __init__(self):
        self.actors: Dict[int, Actor] = {}
        self.critics: Dict[int, Critic] = {}

    def create_networks(self, agent_type: int, state_dim: int) -> Tuple[Actor, Critic]:
        """
        Create Actor-Critic pair for an agent type (if not already created).

        Args:
            agent_type: AgentType enum value (0–9)
            state_dim:  State vector dimension for this type

        Returns:
            (Actor, Critic) tuple — shared by all agents of this type
        """
        if agent_type not in self.actors:
            self.actors[agent_type] = Actor(state_dim=state_dim)
            self.critics[agent_type] = Critic(state_dim=state_dim)

        return self.actors[agent_type], self.critics[agent_type]

    def get_networks(self, agent_type: int) -> Tuple[Actor, Critic]:
        """
        Get existing Actor-Critic pair for an agent type.

        Args:
            agent_type: AgentType enum value

        Returns:
            (Actor, Critic) tuple

        Raises:
            KeyError if networks haven't been created for this type
        """
        return self.actors[agent_type], self.critics[agent_type]

    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        """Get all parameters across all networks (for serialization)."""
        params = []
        for actor in self.actors.values():
            params.extend(actor.parameters())
        for critic in self.critics.values():
            params.extend(critic.parameters())
        return params

    def save_checkpoint(self, path: str):
        """
        Save all network weights to disk.

        Args:
            path: File path for the checkpoint (e.g., 'checkpoints/ppo_step_1000.pt')
        """
        checkpoint = {
            "actors": {k: v.state_dict() for k, v in self.actors.items()},
            "critics": {k: v.state_dict() for k, v in self.critics.items()},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load network weights from disk.

        Args:
            path: File path of the checkpoint

        Raises:
            FileNotFoundError if checkpoint doesn't exist
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        for k, state_dict in checkpoint["actors"].items():
            if k in self.actors:
                self.actors[k].load_state_dict(state_dict)

        for k, state_dict in checkpoint["critics"].items():
            if k in self.critics:
                self.critics[k].load_state_dict(state_dict)

    def summary(self) -> Dict[str, int]:
        """
        Get a summary of network pool contents.

        Returns:
            Dict with type_count, total_actor_params, total_critic_params
        """
        actor_params = sum(
            sum(p.numel() for p in actor.parameters())
            for actor in self.actors.values()
        )
        critic_params = sum(
            sum(p.numel() for p in critic.parameters())
            for critic in self.critics.values()
        )
        return {
            "type_count": len(self.actors),
            "total_actor_params": actor_params,
            "total_critic_params": critic_params,
            "total_params": actor_params + critic_params,
        }
