"""A small, dependency-free gridworld MDP.

Grid layouts are plain text, one row per line, cells separated by
whitespace:

    S . . G
    . # . H
    . . . .

Symbols:
    S   start cell (normal, non-terminal; step reward applies)
    .   normal (non-terminal) cell
    #   wall (impassable; cannot be entered or occupied)
    G   terminal cell, default reward +1.0
    H   terminal cell, default reward -1.0

Any other single uppercase letter is accepted as an additional
terminal-state symbol as long as its reward is supplied via
``terminal_rewards`` (default terminal rewards only cover G and H).

Dynamics follow the classic Russell & Norvig gridworld: taking an
action succeeds with probability ``1 - 2 * slip_prob`` and slips to
one of the two perpendicular directions with probability
``slip_prob`` each (when ``stochastic=True``). Moving into a wall or
off the edge of the grid leaves the agent in place. Terminal states
are absorbing: every action from a terminal state loops back to the
same state with reward 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

State = Tuple[int, int]

# Action name -> (delta_row, delta_col)
ACTIONS: Dict[str, Tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
ACTION_NAMES: Tuple[str, ...] = ("U", "D", "L", "R")
ACTION_ARROWS: Dict[str, str] = {"U": "^", "D": "v", "L": "<", "R": ">"}

# For a given intended action, the two perpendicular ("slip") actions.
_PERPENDICULAR: Dict[str, Tuple[str, str]] = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}

DEFAULT_TERMINAL_REWARDS: Dict[str, float] = {"G": 1.0, "H": -1.0}
WALL = "#"
START = "S"
EMPTY = "."


class GridParseError(ValueError):
    """Raised when a grid layout text is malformed."""


@dataclass
class GridWorld:
    """A rectangular gridworld MDP.

    Attributes:
        layout: row-major list of single-character cell symbols.
        step_reward: reward received for a transition into a non-terminal
            cell (default -0.04, encouraging shorter paths).
        terminal_rewards: mapping of terminal symbol -> reward.
        gamma: discount factor used by default when a solver doesn't
            override it.
        stochastic: whether actions slip per the R&N dynamics.
        slip_prob: probability of slipping to *each* perpendicular
            direction when stochastic.
    """

    layout: List[List[str]]
    step_reward: float = -0.04
    terminal_rewards: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TERMINAL_REWARDS)
    )
    gamma: float = 0.99
    stochastic: bool = True
    slip_prob: float = 0.1

    def __post_init__(self) -> None:
        if not self.layout or not self.layout[0]:
            raise GridParseError("grid layout must be non-empty")
        width = len(self.layout[0])
        for row in self.layout:
            if len(row) != width:
                raise GridParseError("all grid rows must have the same width")
        self.n_rows = len(self.layout)
        self.n_cols = width

        starts: List[State] = []
        states: List[State] = []
        terminals: Dict[State, float] = {}
        for r, row in enumerate(self.layout):
            for c, sym in enumerate(row):
                if sym == WALL:
                    continue
                pos = (r, c)
                states.append(pos)
                if sym == START:
                    starts.append(pos)
                elif sym == EMPTY:
                    pass
                elif sym in self.terminal_rewards:
                    terminals[pos] = self.terminal_rewards[sym]
                else:
                    raise GridParseError(
                        f"unknown grid symbol {sym!r} at row {r}, col {c} "
                        f"(no reward registered in terminal_rewards)"
                    )
        if not starts:
            raise GridParseError("grid must contain at least one 'S' start cell")
        if not terminals:
            raise GridParseError(
                "grid must contain at least one terminal cell (e.g. 'G' or 'H')"
            )
        if not (0.0 <= self.slip_prob <= 0.5):
            raise GridParseError("slip_prob must be in [0, 0.5]")

        self.start_states: Tuple[State, ...] = tuple(starts)
        self.states: Tuple[State, ...] = tuple(states)
        self.terminals: Dict[State, float] = terminals

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_text(cls, text: str, **kwargs) -> "GridWorld":
        rows = [line.split() for line in text.strip("\n").splitlines() if line.strip()]
        return cls(layout=rows, **kwargs)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "GridWorld":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_text(fh.read(), **kwargs)

    # -- MDP interface ---------------------------------------------------------
    @property
    def start_state(self) -> State:
        """The canonical (first-listed) start state."""
        return self.start_states[0]

    def symbol_at(self, state: State) -> str:
        r, c = state
        return self.layout[r][c]

    def is_terminal(self, state: State) -> bool:
        return state in self.terminals

    def is_wall(self, r: int, c: int) -> bool:
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return True
        return self.layout[r][c] == WALL

    def _attempt_move(self, state: State, action: str) -> State:
        dr, dc = ACTIONS[action]
        r, c = state[0] + dr, state[1] + dc
        if self.is_wall(r, c):
            return state
        return (r, c)

    def transitions(self, state: State, action: str) -> List[Tuple[float, State, float, bool]]:
        """Return a list of (probability, next_state, reward, done) tuples.

        Terminal states are absorbing: taking any action from a terminal
        state yields probability 1.0 of staying with reward 0.0 and
        done=True.
        """
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        if self.is_terminal(state):
            return [(1.0, state, 0.0, True)]

        if not self.stochastic:
            outcomes = [(1.0, action)]
        else:
            left, right = _PERPENDICULAR[action]
            main_prob = 1.0 - 2 * self.slip_prob
            outcomes = [
                (main_prob, action),
                (self.slip_prob, left),
                (self.slip_prob, right),
            ]

        results: List[Tuple[float, State, float, bool]] = []
        for prob, act in outcomes:
            if prob <= 0.0:
                continue
            next_state = self._attempt_move(state, act)
            done = self.is_terminal(next_state)
            reward = self.terminal_rewards.get(
                self.symbol_at(next_state), self.step_reward
            ) if done else self.step_reward
            results.append((prob, next_state, reward, done))
        return results

    def step(self, state: State, action: str, rng) -> Tuple[State, float, bool]:
        """Sample a single transition using ``rng.random()`` for randomness."""
        outcomes = self.transitions(state, action)
        roll = rng.random()
        cumulative = 0.0
        for prob, next_state, reward, done in outcomes:
            cumulative += prob
            if roll <= cumulative:
                return next_state, reward, done
        # floating point fallback: return the last outcome
        return outcomes[-1][1:]

    def actions(self, state: State) -> Sequence[str]:
        if self.is_terminal(state):
            return ()
        return ACTION_NAMES

    def to_layout_text(self) -> str:
        return "\n".join(" ".join(row) for row in self.layout)
