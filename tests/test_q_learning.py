from mdpsolve.gridworld import GridWorld
from mdpsolve.q_learning import policy_from_q, q_learning, value_from_q
from mdpsolve.value_iteration import extract_policy, value_iteration

CORRIDOR = "S . . . . G\n"
CLASSIC = "S . . G\n. # . H\n. . . .\n"


def test_q_learning_learns_optimal_policy_on_corridor():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.9)
    Q, _returns = q_learning(env, episodes=800, alpha=0.5, epsilon=0.3, seed=0)
    policy = policy_from_q(env, Q)
    for state, action in policy.items():
        assert action == "R", f"state {state} expected R, got {action}"


def test_q_learning_matches_value_iteration_policy_on_classic_stochastic():
    # Stochastic dynamics (the grid's default): a couple of states have two
    # actions within ~0.05 of true Q-value of each other (see README), so
    # perfect agreement isn't guaranteed run to run -- 85%+ is a reliable
    # bar across seeds with these hyperparameters (verified over 10 seeds
    # while tuning; see the "exploring starts" note on q_learning).
    env = GridWorld.from_text(CLASSIC, gamma=0.9)
    vi_V, _iters = value_iteration(env, theta=1e-10)
    vi_policy = extract_policy(env, vi_V)

    Q, _returns = q_learning(
        env, episodes=6000, alpha=0.5, epsilon=0.3, epsilon_decay=0.999, min_epsilon=0.05, seed=1
    )
    ql_policy = policy_from_q(env, Q)

    agree = sum(1 for s, a in vi_policy.items() if ql_policy.get(s) == a)
    agreement_rate = agree / len(vi_policy)
    assert agreement_rate >= 0.85, f"only {agreement_rate:.2%} of states agree with the DP-optimal policy"


def test_episode_returns_improve_with_training():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.9)
    _Q, returns = q_learning(env, episodes=500, alpha=0.5, epsilon=0.5, seed=2)
    first_chunk = returns[:50]
    last_chunk = returns[-50:]
    assert sum(last_chunk) / len(last_chunk) > sum(first_chunk) / len(first_chunk)


def test_policy_from_q_covers_all_nonterminal_states():
    env = GridWorld.from_text(CLASSIC, stochastic=False, gamma=0.9)
    Q, _returns = q_learning(env, episodes=200, seed=3)
    policy = policy_from_q(env, Q)
    nonterminal = {s for s in env.states if not env.is_terminal(s)}
    assert set(policy.keys()) == nonterminal


def test_value_from_q_is_max_over_actions():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.9)
    Q, _returns = q_learning(env, episodes=200, seed=4)
    V = value_from_q(Q)
    for state, q_row in Q.items():
        assert V[state] == max(q_row.values())


def test_q_learning_is_reproducible_with_seed():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.9)
    Q1, returns1 = q_learning(env, episodes=100, seed=123)
    Q2, returns2 = q_learning(env, episodes=100, seed=123)
    assert returns1 == returns2
    for state in Q1:
        assert Q1[state] == Q2[state]
