#!/usr/bin/env python3
"""Solve the maze grid with policy iteration and compare it against value
iteration to show they converge to the same optimal policy.

Run from the repo root:
    python examples/policy_iteration_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdpsolve.gridworld import GridWorld
from mdpsolve.policy_iteration import policy_iteration
from mdpsolve.render import render_policy
from mdpsolve.value_iteration import extract_policy, value_iteration


def main() -> None:
    grid_path = os.path.join(os.path.dirname(__file__), "..", "grids", "maze.txt")
    env = GridWorld.from_file(grid_path, gamma=0.95, stochastic=False)

    pi_policy, pi_V, pi_iterations = policy_iteration(env, theta=1e-8, seed=0)
    vi_V, vi_iterations = value_iteration(env, theta=1e-8)
    vi_policy = extract_policy(env, vi_V)

    agree = sum(1 for s, a in vi_policy.items() if pi_policy.get(s) == a)
    print(f"Policy iteration converged in {pi_iterations} policy-improvement rounds.")
    print(f"Value iteration converged in {vi_iterations} sweeps.")
    print(f"Policies agree on {agree}/{len(vi_policy)} states.\n")
    print("Policy iteration's greedy policy:")
    print(render_policy(env, pi_policy))


if __name__ == "__main__":
    main()
