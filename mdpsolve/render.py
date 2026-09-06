"""ASCII rendering: grids, value functions, policies, and rollout animations."""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from .gridworld import ACTION_ARROWS, WALL, GridWorld, State


def render_policy(env: GridWorld, policy: Dict[State, str]) -> str:
    """Render the grid with each non-terminal cell replaced by an arrow."""
    lines = []
    for r in range(env.n_rows):
        cells = []
        for c in range(env.n_cols):
            state = (r, c)
            sym = env.symbol_at(state)
            if sym == WALL:
                cells.append("#")
            elif env.is_terminal(state):
                cells.append(sym)
            else:
                cells.append(ACTION_ARROWS[policy[state]])
        lines.append(" ".join(cells))
    return "\n".join(lines)


def render_values(env: GridWorld, V: Dict[State, float], width: int = 6) -> str:
    """Render the grid with each cell replaced by its formatted value."""
    lines = []
    for r in range(env.n_rows):
        cells = []
        for c in range(env.n_cols):
            state = (r, c)
            sym = env.symbol_at(state)
            if sym == WALL:
                cells.append("#".rjust(width))
            else:
                cells.append(f"{V.get(state, 0.0):.2f}".rjust(width))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def render_grid(env: GridWorld, agent_state: Optional[State] = None) -> str:
    """Render the raw grid, optionally with an '@' marking the agent."""
    lines = []
    for r in range(env.n_rows):
        cells = []
        for c in range(env.n_cols):
            state = (r, c)
            if agent_state == state:
                cells.append("@")
            else:
                cells.append(env.symbol_at(state))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def animate_policy(
    env: GridWorld,
    policy: Dict[State, str],
    start: Optional[State] = None,
    max_steps: int = 100,
    seed: Optional[int] = None,
) -> List[str]:
    """Roll out ``policy`` from ``start`` (default: env.start_state).

    Returns a list of ASCII-grid frames, one per timestep (including the
    initial frame before any action is taken). Stops early on reaching a
    terminal state or after ``max_steps`` steps.
    """
    rng = random.Random(seed)
    state = start if start is not None else env.start_state
    frames = [render_grid(env, agent_state=state)]

    for _ in range(max_steps):
        if env.is_terminal(state):
            break
        action = policy[state]
        state, _reward, done = env.step(state, action, rng)
        frames.append(render_grid(env, agent_state=state))
        if done:
            break

    return frames
