import random

import pytest

from mdpsolve.gridworld import GridParseError, GridWorld

CLASSIC = "S . . G\n. # . H\n. . . .\n"
CORRIDOR = "S . . . . G\n"


def test_parses_classic_grid():
    env = GridWorld.from_text(CLASSIC, stochastic=False)
    assert env.n_rows == 3
    assert env.n_cols == 4
    assert env.start_state == (0, 0)
    assert env.terminals == {(0, 3): 1.0, (1, 3): -1.0}
    # walls are excluded from the state set
    assert (1, 1) not in env.states
    assert len(env.states) == 3 * 4 - 1  # one wall cell


def test_mismatched_row_width_raises():
    with pytest.raises(GridParseError):
        GridWorld.from_text("S . G\n. .\n")


def test_missing_start_raises():
    with pytest.raises(GridParseError):
        GridWorld.from_text(". . G\n. . .\n")


def test_missing_terminal_raises():
    with pytest.raises(GridParseError):
        GridWorld.from_text("S . .\n. . .\n")


def test_unknown_symbol_raises():
    with pytest.raises(GridParseError):
        GridWorld.from_text("S . X\n. . G\n")


def test_invalid_slip_prob_raises():
    with pytest.raises(GridParseError):
        GridWorld.from_text(CLASSIC, slip_prob=0.9)


def test_custom_terminal_symbol():
    env = GridWorld.from_text(
        "S . T\n", terminal_rewards={"T": 5.0}, stochastic=False
    )
    assert env.terminals == {(0, 2): 5.0}


def test_deterministic_transition_is_certain():
    env = GridWorld.from_text(CLASSIC, stochastic=False)
    outcomes = env.transitions((0, 0), "R")
    assert outcomes == [(1.0, (0, 1), -0.04, False)]


def test_wall_move_stays_in_place():
    env = GridWorld.from_text(CLASSIC, stochastic=False)
    # (0, 1) moving down would enter the wall at (1, 1)
    outcomes = env.transitions((0, 1), "D")
    assert outcomes == [(1.0, (0, 1), -0.04, False)]


def test_off_grid_move_stays_in_place():
    env = GridWorld.from_text(CLASSIC, stochastic=False)
    outcomes = env.transitions((0, 0), "U")
    assert outcomes == [(1.0, (0, 0), -0.04, False)]


def test_terminal_states_are_absorbing():
    env = GridWorld.from_text(CLASSIC, stochastic=False)
    for action in ("U", "D", "L", "R"):
        outcomes = env.transitions((0, 3), action)
        assert outcomes == [(1.0, (0, 3), 0.0, True)]


def test_reaching_terminal_yields_terminal_reward():
    env = GridWorld.from_text(CLASSIC, stochastic=False)
    outcomes = env.transitions((0, 2), "R")
    assert outcomes == [(1.0, (0, 3), 1.0, True)]
    outcomes = env.transitions((1, 2), "R")
    assert outcomes == [(1.0, (1, 3), -1.0, True)]


def test_stochastic_transitions_sum_to_one_and_include_slips():
    env = GridWorld.from_text(CLASSIC, stochastic=True, slip_prob=0.1)
    outcomes = env.transitions((2, 1), "R")
    total_prob = sum(p for p, *_ in outcomes)
    assert total_prob == pytest.approx(1.0)
    # main direction R -> (2,2); perpendiculars for R are U and D, which slip
    # towards (1,1) (a wall, so it stays put) and (2,1) (unchanged) respectively.
    probs = {ns: p for p, ns, _r, _d in outcomes}
    assert probs[(2, 2)] == pytest.approx(0.8)


def test_step_samples_consistently_with_transitions():
    env = GridWorld.from_text(CLASSIC, stochastic=True, slip_prob=0.1)
    rng = random.Random(42)
    outcomes = env.transitions((2, 1), "R")
    valid_next_states = {ns for _p, ns, _r, _d in outcomes}
    for _ in range(200):
        next_state, _reward, _done = env.step((2, 1), "R", rng)
        assert next_state in valid_next_states


def test_sample_grid_files_parse():
    for path in ("grids/classic.txt", "grids/corridor.txt", "grids/maze.txt"):
        env = GridWorld.from_file(path)
        assert env.start_state is not None
        assert len(env.terminals) >= 1
