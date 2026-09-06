#!/usr/bin/env python3
"""Solve the classic Russell & Norvig 3x4 gridworld with value iteration
and print the resulting values and greedy policy.

Run from the repo root:
    python examples/value_iteration_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdpsolve.gridworld import GridWorld
from mdpsolve.render import render_policy, render_values
from mdpsolve.value_iteration import solve


def main() -> None:
    env = GridWorld.from_file(
        os.path.join(os.path.dirname(__file__), "..", "grids", "classic.txt"),
        gamma=0.99,
        stochastic=True,
        slip_prob=0.1,
    )
    policy, V, iterations = solve(env, theta=1e-6)

    print(f"Converged after {iterations} iterations of value iteration.\n")
    print("Values:")
    print(render_values(env, V))
    print("\nGreedy policy:")
    print(render_policy(env, policy))
    print(f"\nValue at start state {env.start_state}: {V[env.start_state]:.4f}")


if __name__ == "__main__":
    main()
