from mdpsolve.gridworld import GridWorld
from mdpsolve.render import animate_policy, render_grid, render_policy, render_values
from mdpsolve.value_iteration import extract_policy, value_iteration

CLASSIC = "S . . G\n. # . H\n. . . .\n"
CORRIDOR = "S . . . . G\n"


def test_render_grid_marks_agent():
    env = GridWorld.from_text(CLASSIC, stochastic=False)
    text = render_grid(env, agent_state=(0, 0))
    lines = text.splitlines()
    assert lines[0].split()[0] == "@"


def test_render_grid_without_agent_shows_raw_symbols():
    env = GridWorld.from_text(CLASSIC, stochastic=False)
    text = render_grid(env)
    lines = text.splitlines()
    assert lines[0].split() == ["S", ".", ".", "G"]
    assert lines[1].split() == [".", "#", ".", "H"]


def test_render_policy_shows_arrows_and_walls():
    env = GridWorld.from_text(CLASSIC, stochastic=False, gamma=0.9)
    V, _iters = value_iteration(env, theta=1e-10)
    policy = extract_policy(env, V)
    text = render_policy(env, policy)
    lines = text.splitlines()
    assert lines[1].split()[1] == "#"
    assert lines[0].split()[3] == "G"
    # every non-wall, non-terminal cell should render as one of the arrow glyphs
    for r, row in enumerate(lines):
        for c, sym in enumerate(row.split()):
            state = (r, c)
            if state in env.terminals or env.symbol_at(state) == "#":
                continue
            assert sym in "^v<>"


def test_render_values_shows_zero_at_terminals():
    env = GridWorld.from_text(CLASSIC, stochastic=False, gamma=0.9)
    V, _iters = value_iteration(env, theta=1e-10)
    text = render_values(env, V)
    lines = text.splitlines()
    # (0, 3) is the goal, (1, 3) is the hazard -- both terminal, both 0.
    assert lines[0].split()[3] == "0.00"
    assert lines[1].split()[3] == "0.00"
    # the state right next to the goal should have a high positive value
    assert float(lines[0].split()[2]) > 0.9


def test_animate_policy_reaches_goal_on_corridor():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.9)
    policy = {s: "R" for s in env.states if not env.is_terminal(s)}
    frames = animate_policy(env, policy, seed=0)
    assert len(frames) == 6  # start + 5 steps to reach column 5
    assert "@" in frames[-1]


def test_animate_policy_stops_at_max_steps_if_never_terminal():
    env = GridWorld.from_text(CORRIDOR, stochastic=False, gamma=0.9)
    policy = {s: "L" for s in env.states if not env.is_terminal(s)}
    frames = animate_policy(env, policy, max_steps=5, seed=0)
    assert len(frames) == 6  # start + 5 steps, never reaching the goal
