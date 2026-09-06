"""Tabular Q-learning (model-free, epsilon-greedy) for a GridWorld MDP."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .gridworld import ACTION_NAMES, GridWorld, State

QTable = Dict[State, Dict[str, float]]


def _init_q(env: GridWorld) -> QTable:
    return {
        s: {a: 0.0 for a in ACTION_NAMES}
        for s in env.states
        if not env.is_terminal(s)
    }


def _init_counts(env: GridWorld) -> Dict[State, Dict[str, int]]:
    return {
        s: {a: 0 for a in ACTION_NAMES}
        for s in env.states
        if not env.is_terminal(s)
    }


def _greedy_action(q_row: Dict[str, float]) -> str:
    """Argmax over ACTION_NAMES order, so ties break deterministically."""
    best_a, best_q = None, float("-inf")
    for a in ACTION_NAMES:
        if q_row[a] > best_q:
            best_a, best_q = a, q_row[a]
    return best_a


def q_learning(
    env: GridWorld,
    gamma: Optional[float] = None,
    episodes: int = 3000,
    alpha: float = 1.0,
    epsilon: float = 0.3,
    epsilon_decay: float = 0.999,
    min_epsilon: float = 0.05,
    max_steps: int = 200,
    seed: Optional[int] = None,
    exploring_starts: bool = True,
) -> Tuple[QTable, List[float]]:
    """Run tabular Q-learning.

    The per-(state, action) learning rate decays with visit count as
    ``alpha / (1 + visits)`` -- i.e. it is a running average by default
    (``alpha=1.0``). This satisfies the Robbins-Monro conditions needed
    for tabular Q-learning to actually converge to the true action
    values (a *constant* learning rate never fully settles, which
    matters once a state has a rare but large-magnitude outcome, such as
    a small chance of stepping into a hazard).

    By default (``exploring_starts=True``) each training episode begins
    from a state sampled uniformly at random over *every* non-terminal
    state, not just the grid's own 'S' cell(s). Without this, states far
    from the real start (e.g. corners the greedy policy quickly learns to
    avoid routing through) get visited only a handful of times even
    across tens of thousands of episodes -- epsilon-greedy exploration
    from a single fixed start is not enough to reliably cover a whole
    gridworld's state space, so their Q-values never converge. This is
    the classic "exploring starts" fix from Sutton & Barto; set it to
    False to instead always start from the grid's own start state(s)
    (closer to how an agent would learn while actually deployed).

    Returns (Q, episode_returns) where Q maps state -> {action: value} for
    every non-terminal state, and episode_returns is the (undiscounted)
    total reward collected in each training episode -- useful for checking
    that learning is actually improving over time. When exploring_starts is
    on, returns are noisier episode-to-episode (since each starts from a
    different, randomly-placed state) but still trend upward on average as
    Q improves.
    """
    gamma = env.gamma if gamma is None else gamma
    rng = random.Random(seed)
    Q = _init_q(env)
    counts = _init_counts(env)
    episode_returns: List[float] = []
    training_starts = tuple(Q.keys()) if exploring_starts else env.start_states

    for episode in range(episodes):
        state = rng.choice(training_starts)
        eps = max(min_epsilon, epsilon * (epsilon_decay ** episode))
        total_reward = 0.0

        for _step in range(max_steps):
            if env.is_terminal(state):
                break
            if rng.random() < eps:
                action = rng.choice(ACTION_NAMES)
            else:
                action = _greedy_action(Q[state])

            next_state, reward, done = env.step(state, action, rng)
            total_reward += reward

            counts[state][action] += 1
            step_size = alpha / (1 + counts[state][action])
            best_next = 0.0 if done else max(Q[next_state].values())
            td_target = reward + gamma * best_next
            Q[state][action] += step_size * (td_target - Q[state][action])

            state = next_state
            if done:
                break

        episode_returns.append(total_reward)

    return Q, episode_returns


def policy_from_q(env: GridWorld, Q: QTable) -> Dict[State, str]:
    """Extract the greedy policy pi(s) = argmax_a Q[s][a]."""
    return {
        state: _greedy_action(q_row)
        for state, q_row in Q.items()
        if not env.is_terminal(state)
    }


def value_from_q(Q: QTable) -> Dict[State, float]:
    """Derive V(s) = max_a Q[s][a] for reporting/comparison purposes."""
    return {state: max(q_row.values()) for state, q_row in Q.items()}
