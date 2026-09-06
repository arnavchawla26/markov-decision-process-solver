import pytest

from mdpsolve.gridworld import GridWorld
from mdpsolve.policy_iteration import policy_evaluation, policy_improvement, policy_iteration
from mdpsolve.value_iteration import extract_policy, value_iteration

CORRIDOR = "S . . . . G\n"
CLASSIC = "S . . G\n. # . H\n. . . .\n"


def test_policy_evaluation_matches_value_iteration_for_optimal_policy():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.9)
    vi_V, _iters = value_iteration(env, theta=1e-10)
    vi_policy = extract_policy(env, vi_V)

    pe_V, _eval_iters = policy_evaluation(env, vi_policy, theta=1e-10)
    for state in env.states:
        assert pe_V[state] == pytest.approx(vi_V[state], abs=1e-6)


def test_policy_improvement_is_stable_on_optimal_policy():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.9)
    vi_V, _iters = value_iteration(env, theta=1e-10)
    vi_policy = extract_policy(env, vi_V)

    improved = policy_improvement(env, vi_V)
    assert improved == vi_policy


def test_policy_iteration_converges_to_same_policy_as_value_iteration_classic():
    env = GridWorld.from_text(CLASSIC, stochastic=False, gamma=0.9)
    vi_V, _vi_iters = value_iteration(env, theta=1e-10)
    vi_policy = extract_policy(env, vi_V)
    pi_policy, pi_V, iterations = policy_iteration(env, theta=1e-10, seed=7)

    assert iterations > 0
    assert pi_policy == vi_policy
    for state in env.states:
        assert pi_V[state] == pytest.approx(vi_V[state], abs=1e-4)


def test_policy_iteration_converges_on_stochastic_grid():
    env = GridWorld.from_text(CLASSIC, stochastic=True, slip_prob=0.1, gamma=0.9)
    policy, V, iterations = policy_iteration(env, theta=1e-8, seed=3)
    assert iterations > 0
    # every non-terminal state has a defined action
    for state in env.states:
        if not env.is_terminal(state):
            assert policy[state] in ("U", "D", "L", "R")
    # sanity: the state right next to the goal should still prefer heading there
    assert policy[(0, 2)] == "R"
