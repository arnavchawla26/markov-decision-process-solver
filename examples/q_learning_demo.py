#!/usr/bin/env python3
"""Train tabular Q-learning on the classic gridworld and check how closely
its learned (model-free) policy matches the value-iteration (model-based,
ground-truth) optimal policy.

Run from the repo root:
    python examples/q_learning_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdpsolve.gridworld import GridWorld
from mdpsolve.q_learning import policy_from_q, q_learning
from mdpsolve.render import render_policy
from mdpsolve.value_iteration import extract_policy, value_iteration


def main() -> None:
    grid_path = os.path.join(os.path.dirname(__file__), "..", "grids", "classic.txt")
    env = GridWorld.from_file(grid_path, gamma=0.99, stochastic=False)

    vi_V, _iters = value_iteration(env, theta=1e-8)
    vi_policy = extract_policy(env, vi_V)

    Q, returns = q_learning(
        env, episodes=4000, alpha=0.4, epsilon=0.4, epsilon_decay=0.999, seed=1
    )
    ql_policy = policy_from_q(env, Q)

    agree = sum(1 for s, a in vi_policy.items() if ql_policy.get(s) == a)
    first_100_avg = sum(returns[:100]) / 100
    last_100_avg = sum(returns[-100:]) / 100

    print(f"Q-learning trained for {len(returns)} episodes.")
    print(f"Mean episode return -- first 100: {first_100_avg:.3f}, last 100: {last_100_avg:.3f}")
    print(f"Q-learning policy agrees with value-iteration's optimal policy on "
          f"{agree}/{len(vi_policy)} states.\n")
    print("Q-learning's greedy policy:")
    print(render_policy(env, ql_policy))


if __name__ == "__main__":
    main()
