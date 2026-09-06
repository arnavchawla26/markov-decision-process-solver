"""Value iteration and greedy policy extraction for a GridWorld MDP."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .gridworld import ACTION_NAMES, GridWorld, State


def q_value(env: GridWorld, V: Dict[State, float], state: State, action: str, gamma: float) -> float:
    """Compute Q(state, action) from a value function V under env's dynamics."""
    total = 0.0
    for prob, next_state, reward, _done in env.transitions(state, action):
        total += prob * (reward + gamma * V.get(next_state, 0.0))
    return total


def best_action(
    env: GridWorld, V: Dict[State, float], state: State, gamma: float
) -> Tuple[str, float]:
    """Return the (action, q_value) maximizing Q(state, ·) under V.

    Ties are broken by the fixed action order in ACTION_NAMES, so results
    are deterministic.
    """
    best_a, best_q = None, float("-inf")
    for action in env.actions(state) or ACTION_NAMES:
        q = q_value(env, V, state, action, gamma)
        if q > best_q:
            best_a, best_q = action, q
    return best_a, best_q


def value_iteration(
    env: GridWorld,
    gamma: Optional[float] = None,
    theta: float = 1e-6,
    max_iterations: int = 10000,
) -> Tuple[Dict[State, float], int]:
    """Run value iteration to convergence.

    Returns (V, iterations_used). Terminal states always stay at V=0: the
    reward for *entering* a terminal is already paid out on the transition
    that leads into it (see GridWorld.transitions), so pinning V(terminal)
    to that same reward would double-count it in every predecessor's
    Bellman backup.
    """
    gamma = env.gamma if gamma is None else gamma
    V: Dict[State, float] = {s: 0.0 for s in env.states}

    iterations = 0
    for iterations in range(1, max_iterations + 1):
        delta = 0.0
        new_V = dict(V)
        for state in env.states:
            if env.is_terminal(state):
                continue
            _a, q = best_action(env, V, state, gamma)
            new_V[state] = q
            delta = max(delta, abs(new_V[state] - V[state]))
        V = new_V
        if delta < theta:
            break
    return V, iterations


def extract_policy(
    env: GridWorld, V: Dict[State, float], gamma: Optional[float] = None
) -> Dict[State, str]:
    """Derive the greedy policy pi(s) = argmax_a Q(s, a) from a value function."""
    gamma = env.gamma if gamma is None else gamma
    policy: Dict[State, str] = {}
    for state in env.states:
        if env.is_terminal(state):
            continue
        a, _q = best_action(env, V, state, gamma)
        policy[state] = a
    return policy


def solve(
    env: GridWorld,
    gamma: Optional[float] = None,
    theta: float = 1e-6,
    max_iterations: int = 10000,
) -> Tuple[Dict[State, str], Dict[State, float], int]:
    """Convenience wrapper: value-iterate then extract the greedy policy."""
    gamma = env.gamma if gamma is None else gamma
    V, iterations = value_iteration(env, gamma=gamma, theta=theta, max_iterations=max_iterations)
    policy = extract_policy(env, V, gamma=gamma)
    return policy, V, iterations
