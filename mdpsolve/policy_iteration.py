"""Policy iteration (iterative policy evaluation + greedy improvement)."""
from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

from .gridworld import ACTION_NAMES, GridWorld, State
from .value_iteration import best_action, q_value


def policy_evaluation(
    env: GridWorld,
    policy: Dict[State, str],
    gamma: Optional[float] = None,
    theta: float = 1e-6,
    max_iterations: int = 10000,
) -> Tuple[Dict[State, float], int]:
    """Evaluate a fixed policy to convergence, returning (V, iterations).

    Terminal states always stay at V=0 (see value_iteration.value_iteration
    for why they must not be pinned to their reward).
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
            action = policy[state]
            new_V[state] = q_value(env, V, state, action, gamma)
            delta = max(delta, abs(new_V[state] - V[state]))
        V = new_V
        if delta < theta:
            break
    return V, iterations


def policy_improvement(
    env: GridWorld, V: Dict[State, float], gamma: Optional[float] = None
) -> Tuple[Dict[State, str], bool]:
    """Greedily improve a policy w.r.t. V. Returns (new_policy, is_stable)."""
    gamma = env.gamma if gamma is None else gamma
    new_policy: Dict[State, str] = {}
    for state in env.states:
        if env.is_terminal(state):
            continue
        a, _q = best_action(env, V, state, gamma)
        new_policy[state] = a
    return new_policy


def _random_policy(env: GridWorld, seed: Optional[int] = None) -> Dict[State, str]:
    rng = random.Random(seed)
    return {
        s: rng.choice(ACTION_NAMES) for s in env.states if not env.is_terminal(s)
    }


def policy_iteration(
    env: GridWorld,
    gamma: Optional[float] = None,
    theta: float = 1e-6,
    max_eval_iterations: int = 10000,
    max_policy_iterations: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[Dict[State, str], Dict[State, float], int]:
    """Run policy iteration to a stable (optimal) policy.

    Returns (policy, V, policy_iterations_used).
    """
    gamma = env.gamma if gamma is None else gamma
    policy = _random_policy(env, seed=seed)
    V: Dict[State, float] = {}

    for iteration in range(1, max_policy_iterations + 1):
        V, _eval_iters = policy_evaluation(
            env, policy, gamma=gamma, theta=theta, max_iterations=max_eval_iterations
        )
        new_policy = policy_improvement(env, V, gamma=gamma)
        if new_policy == policy:
            return policy, V, iteration
        policy = new_policy
    return policy, V, max_policy_iterations
