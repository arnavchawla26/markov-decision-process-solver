"""Command-line interface for mdpsolve.

Subcommands:
    mdpsolve solve   --grid PATH [--method value-iteration|policy-iteration|q-learning]
    mdpsolve animate --grid PATH [--method ...] [--start R,C]
    mdpsolve compare --grid PATH [common solver options]

Run ``mdpsolve <subcommand> --help`` for the full option list.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, Optional

from .gridworld import GridWorld, State
from .policy_iteration import policy_iteration
from .q_learning import policy_from_q, q_learning, value_from_q
from .render import animate_policy, render_policy, render_values
from .value_iteration import extract_policy, value_iteration

METHODS = ("value-iteration", "policy-iteration", "q-learning")


def _add_common_grid_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--grid", required=True, help="path to a grid layout text file")
    p.add_argument("--gamma", type=float, default=0.99, help="discount factor (default: 0.99)")
    p.add_argument(
        "--step-reward", type=float, default=-0.04, help="reward per non-terminal step (default: -0.04)"
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="disable action slippage (default: stochastic R&N dynamics)",
    )
    p.add_argument(
        "--slip-prob",
        type=float,
        default=0.1,
        help="probability of slipping to each perpendicular direction (default: 0.1)",
    )
    p.add_argument("--seed", type=int, default=None, help="random seed")


def _add_solver_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--method", choices=METHODS, default="value-iteration")
    p.add_argument("--theta", type=float, default=1e-6, help="convergence threshold for DP methods")
    p.add_argument("--episodes", type=int, default=6000, help="Q-learning training episodes")
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Q-learning learning-rate scale (actual step size decays as alpha/(1+visits))",
    )
    p.add_argument("--epsilon", type=float, default=0.3, help="Q-learning initial exploration rate")
    p.add_argument("--min-epsilon", type=float, default=0.05, help="Q-learning exploration-rate floor")
    p.add_argument("--max-steps", type=int, default=200, help="max steps per Q-learning episode")
    p.add_argument(
        "--no-exploring-starts",
        action="store_true",
        help="train Q-learning only from the grid's own start state(s), instead of every state",
    )


def _build_env(args: argparse.Namespace) -> GridWorld:
    return GridWorld.from_file(
        args.grid,
        step_reward=args.step_reward,
        gamma=args.gamma,
        stochastic=not args.deterministic,
        slip_prob=args.slip_prob,
    )


def _solve(env: GridWorld, args: argparse.Namespace):
    """Run the chosen method. Returns (policy, V, info_dict)."""
    if args.method == "value-iteration":
        V, iterations = value_iteration(env, gamma=args.gamma, theta=args.theta)
        policy = extract_policy(env, V, gamma=args.gamma)
        return policy, V, {"method": args.method, "iterations": iterations}
    if args.method == "policy-iteration":
        policy, V, iterations = policy_iteration(
            env, gamma=args.gamma, theta=args.theta, seed=args.seed
        )
        return policy, V, {"method": args.method, "iterations": iterations}
    if args.method == "q-learning":
        Q, returns = q_learning(
            env,
            gamma=args.gamma,
            episodes=args.episodes,
            alpha=args.alpha,
            epsilon=args.epsilon,
            min_epsilon=args.min_epsilon,
            max_steps=args.max_steps,
            seed=args.seed,
            exploring_starts=not args.no_exploring_starts,
        )
        policy = policy_from_q(env, Q)
        V = value_from_q(Q)
        mean_last_100 = sum(returns[-100:]) / len(returns[-100:]) if returns else 0.0
        return policy, V, {
            "method": args.method,
            "episodes": args.episodes,
            "mean_return_last_100_episodes": mean_last_100,
        }
    raise ValueError(f"unknown method {args.method!r}")


def _state_key(state: State) -> str:
    return f"{state[0]},{state[1]}"


def _parse_state(text: str) -> State:
    r, c = text.split(",")
    return (int(r), int(c))


def policy_to_json(policy: Dict[State, str]) -> Dict[str, str]:
    return {_state_key(s): a for s, a in policy.items()}


def policy_from_json(data: Dict[str, str]) -> Dict[State, str]:
    return {_parse_state(k): a for k, a in data.items()}


def cmd_solve(args: argparse.Namespace) -> int:
    env = _build_env(args)
    policy, V, info = _solve(env, args)

    print(f"# method: {info['method']}")
    for key, value in info.items():
        if key == "method":
            continue
        print(f"# {key}: {value}")
    print()
    print("Policy:")
    print(render_policy(env, policy))
    print()
    print("Values:")
    print(render_values(env, V))
    print()
    print(f"Value at start state {env.start_state}: {V.get(env.start_state, 0.0):.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "grid": args.grid,
                    "method": info["method"],
                    "policy": policy_to_json(policy),
                    "values": {_state_key(s): v for s, v in V.items()},
                },
                fh,
                indent=2,
            )
        print(f"\nWrote policy/values to {args.out}")
    return 0


def cmd_animate(args: argparse.Namespace) -> int:
    env = _build_env(args)
    policy, _V, info = _solve(env, args)
    start = _parse_state(args.start) if args.start else None

    frames = animate_policy(
        env, policy, start=start, max_steps=args.animate_max_steps, seed=args.seed
    )
    for i, frame in enumerate(frames):
        label = "start" if i == 0 else f"step {i}"
        print(f"-- {label} ({info['method']}) --")
        print(frame)
        print()
    print(f"Finished after {len(frames) - 1} step(s).")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    env = _build_env(args)

    results = {}
    t0 = time.perf_counter()
    vi_V, vi_iters = value_iteration(env, gamma=args.gamma, theta=args.theta)
    vi_policy = extract_policy(env, vi_V, gamma=args.gamma)
    vi_time = time.perf_counter() - t0
    results["value-iteration"] = (vi_policy, vi_V, vi_iters, vi_time)

    t0 = time.perf_counter()
    pi_policy, pi_V, pi_iters = policy_iteration(env, gamma=args.gamma, theta=args.theta, seed=args.seed)
    pi_time = time.perf_counter() - t0
    results["policy-iteration"] = (pi_policy, pi_V, pi_iters, pi_time)

    t0 = time.perf_counter()
    Q, returns = q_learning(
        env,
        gamma=args.gamma,
        episodes=args.episodes,
        alpha=args.alpha,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        max_steps=args.max_steps,
        seed=args.seed,
        exploring_starts=not args.no_exploring_starts,
    )
    ql_policy = policy_from_q(env, Q)
    ql_V = value_from_q(Q)
    ql_time = time.perf_counter() - t0
    results["q-learning"] = (ql_policy, ql_V, args.episodes, ql_time)

    print(f"{'method':<18}{'iterations/episodes':<22}{'time (s)':<12}{'agreement vs VI':<18}{'V(start)':<10}")
    n_states = len(vi_policy)
    for name, (policy, V, iters, elapsed) in results.items():
        agree = sum(1 for s, a in vi_policy.items() if policy.get(s) == a)
        pct = 100.0 * agree / n_states if n_states else 100.0
        v_start = V.get(env.start_state, 0.0)
        print(f"{name:<18}{iters:<22}{elapsed:<12.4f}{pct:<18.1f}{v_start:<10.4f}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdpsolve",
        description="Solve gridworld MDPs with value iteration, policy iteration, or tabular Q-learning.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_solve = sub.add_parser("solve", help="solve a grid and print the resulting policy/values")
    _add_common_grid_args(p_solve)
    _add_solver_args(p_solve)
    p_solve.add_argument("--out", default=None, help="optional path to write policy+values as JSON")
    p_solve.set_defaults(func=cmd_solve)

    p_animate = sub.add_parser("animate", help="solve a grid, then animate the greedy rollout")
    _add_common_grid_args(p_animate)
    _add_solver_args(p_animate)
    p_animate.add_argument("--start", default=None, help="start state as 'row,col' (default: grid's S)")
    p_animate.add_argument("--animate-max-steps", type=int, default=100)
    p_animate.set_defaults(func=cmd_animate)

    p_compare = sub.add_parser("compare", help="run all three methods and compare policies/timing")
    _add_common_grid_args(p_compare)
    _add_solver_args(p_compare)  # --method is accepted but unused by compare
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
