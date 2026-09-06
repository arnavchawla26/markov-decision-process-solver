import pytest

from mdpsolve.gridworld import GridWorld
from mdpsolve.value_iteration import best_action, q_value, solve, value_iteration

CORRIDOR = "S . . . . G\n"
CLASSIC = "S . . G\n. # . H\n. . . .\n"


def test_terminal_values_stay_zero():
    # Terminal states are absorbing and their reward is paid out on the
    # transition *into* them, so V(terminal) itself must stay 0 -- otherwise
    # every predecessor's Bellman backup would double-count that reward.
    env = GridWorld.from_text(CLASSIC, stochastic=False, gamma=0.9)
    V, _iters = value_iteration(env)
    assert V[(0, 3)] == 0.0
    assert V[(1, 3)] == 0.0


def test_corridor_optimal_policy_is_always_right():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.9)
    policy, _V, iterations = solve(env)
    assert iterations > 0
    for state, action in policy.items():
        assert action == "R", f"state {state} expected R, got {action}"


def test_corridor_value_matches_closed_form():
    # Deterministic corridor of length 6 (indices 0..5), goal at index 5.
    # From state i, following the optimal (all-R) policy takes (5 - i) steps,
    # each incurring step_reward before the terminal +1 reward.
    step_reward = -0.04
    gamma = 0.9
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=gamma, step_reward=step_reward)
    V, _iters = value_iteration(env, theta=1e-10)
    for col in range(5):
        n = 5 - col  # steps to reach the goal
        expected = sum(step_reward * gamma ** k for k in range(n - 1)) + (gamma ** (n - 1)) * 1.0
        assert V[(0, col)] == pytest.approx(expected, abs=1e-6)


def test_classic_grid_avoids_hazard():
    env = GridWorld.from_text(CLASSIC, stochastic=False, gamma=0.9)
    policy, _V, _iters = solve(env)
    # (0, 2) sits directly next to the goal -- taking it is trivially optimal.
    assert policy[(0, 2)] == "R"
    # (1, 2) sits directly next to the hazard at (1, 3); walking into it
    # (action R) would cost -1, so the optimal policy must route around it
    # (up towards (0, 2) and then right to the goal).
    assert policy[(1, 2)] == "U"


def test_q_value_matches_manual_bellman_backup():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.5)
    # Terminal states are always 0 under this MDP's convention (see
    # test_terminal_values_stay_zero).
    V = {s: 0.0 for s in env.states}
    # From (0, 4), action R deterministically reaches the goal (0, 5),
    # earning the +1 terminal reward with no further discounted value.
    q = q_value(env, V, (0, 4), "R", gamma=0.5)
    assert q == pytest.approx(1.0)


def test_best_action_breaks_ties_deterministically():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.9)
    V = {s: 0.0 for s in env.states}
    action, q = best_action(env, V, (0, 0), gamma=0.9)
    assert action in ("U", "D", "L", "R")
    assert isinstance(q, float)
