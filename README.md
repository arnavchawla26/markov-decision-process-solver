# markov-decision-process-solver

Value iteration, policy iteration, and tabular Q-learning, all implemented
from scratch, solving small gridworld Markov Decision Processes -- plus a
CLI to define a grid, solve it with any of the three methods, and animate
the resulting policy as it walks the grid.

No numpy, no gym/gymnasium, no RL framework. Just the algorithms, in pure
Python standard library.

## What's in here

- **`mdpsolve/gridworld.py`** -- a small gridworld MDP parsed from a plain
  text layout (`S`/`.`/`#`/`G`/`H` for start/empty/wall/goal/hazard). Supports
  the classic Russell & Norvig stochastic dynamics (actions "slip" to a
  perpendicular direction with configurable probability) or fully
  deterministic transitions.
- **`mdpsolve/value_iteration.py`** -- value iteration via repeated Bellman
  optimality backups to a fixed-point, plus greedy policy extraction.
- **`mdpsolve/policy_iteration.py`** -- iterative policy evaluation +
  greedy policy improvement, alternating to a stable (optimal) policy.
- **`mdpsolve/q_learning.py`** -- model-free tabular Q-learning: no access
  to the transition model, just epsilon-greedy exploration and TD updates
  from sampled transitions.
- **`mdpsolve/render.py`** -- ASCII rendering of the grid, a value
  function, a policy (as arrows), and a step-by-step animation of an agent
  following a policy from a start state to a terminal one.
- **`mdpsolve/cli.py`** -- the `mdpsolve` command: `solve`, `animate`, and
  `compare` subcommands.

## Why three algorithms side by side

They solve the same problem with different information and guarantees,
and putting them behind one CLI (`mdpsolve compare`) makes the tradeoff
visible instead of theoretical:

- **Value iteration** and **policy iteration** are *model-based* dynamic
  programming: they require the full transition/reward model
  (`GridWorld.transitions`) and converge to the exact optimal policy.
  Policy iteration usually needs far fewer outer iterations (each one is
  more expensive -- a full policy evaluation to convergence) while value
  iteration does more, cheaper sweeps.
- **Q-learning** is *model-free*: it never calls `transitions()`, only
  `step()` (a single sampled outcome, exactly what an agent would observe
  in the world). It has to discover the dynamics through experience, so it
  converges to an *approximation* of the optimal policy, and needs many
  more samples on larger/more maze-like grids to get there.

## A correctness note worth flagging

Terminal states are absorbing, and the reward for *entering* one is paid
out on the transition into it (`GridWorld.transitions` returns
`terminal_rewards[symbol]` as the reward when the destination is
terminal). That means `V(terminal)` itself must stay pinned at `0`, not
the terminal's reward -- pinning it to the reward would double-count that
reward in every predecessor state's Bellman backup (`reward_in + gamma *
V(terminal)` would count the `+1`/`-1` twice). This tripped up the first
draft of `value_iteration` and `policy_evaluation` (verified by a test
that manually recomputes the closed-form value of a plain corridor grid
and catches the discrepancy) -- fixed by leaving terminal values at their
initialized `0.0` and simply never updating them.

## Q-learning: exploring starts, and why it matters

Tabular Q-learning has two subtleties that are easy to get wrong silently
(both were caught, not assumed away, while building this repo):

1. **Learning rate must decay.** A constant learning rate never settles;
   with a small chance of a large negative outcome (stepping into a
   hazard), the estimate keeps getting knocked around by rare samples
   forever. `q_learning` uses `alpha / (1 + visits[s][a])`, a running
   average by default (`alpha=1.0`), which satisfies the Robbins-Monro
   conditions needed for the estimate to actually converge.
2. **A single fixed start state under-explores the map.** Starting every
   training episode from the grid's own `S` cell means states the greedy
   policy learns to route *around* (a corner near a hazard, a dead end)
   get visited only a handful of times even across tens of thousands of
   episodes -- their Q-values never converge, no matter how long you
   train. `exploring_starts=True` (the default) instead samples each
   training episode's start state uniformly over *every* non-terminal
   state, guaranteeing real coverage. This is the classic "exploring
   starts" technique from Sutton & Barto's *Reinforcement Learning: An
   Introduction*.

With both fixes, Q-learning's greedy policy agrees with the DP-optimal
policy on 85-90%+ of states on the small 3x4 classic grid within a few
thousand episodes (verified across 10 random seeds while tuning). On the
larger 7x7 maze grid it's closer to 60-70% with the same episode budget --
tabular Q-learning genuinely needs more samples as the state space grows,
and `mdpsolve compare` is built to make that gap visible rather than hide
it (see the real captured output below).

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.8+. No runtime dependencies.

## Usage

### Solve a grid

```bash
$ mdpsolve solve --grid grids/classic.txt --method value-iteration
# method: value-iteration
# iterations: 26

Policy:
> > > G
^ # ^ H
^ < ^ <

Values:
  0.82   0.89   0.95   0.00
  0.76      #   0.69   0.00
  0.70   0.64   0.61   0.38

Value at start state (0, 0): 0.8244
```

`--method` accepts `value-iteration` (default), `policy-iteration`, or
`q-learning`. Add `--out policy.json` to also write the policy and values
to a JSON file.

### Animate the greedy rollout

```bash
$ mdpsolve animate --grid grids/corridor.txt --method value-iteration --deterministic --seed 0
-- start (value-iteration) --
@ . . . . G

-- step 1 (value-iteration) --
S @ . . . G

...

-- step 5 (value-iteration) --
S . . . . @

Finished after 5 step(s).
```

### Compare all three methods

```bash
$ mdpsolve compare --grid grids/classic.txt --seed 1
method            iterations/episodes   time (s)    agreement vs VI   V(start)
value-iteration   26                    0.0020      100.0             0.8244
policy-iteration  5                     0.0187      100.0             0.8244
q-learning        6000                  0.0794      88.9              0.7534
```

`agreement vs VI` is the fraction of states where that method's greedy
policy matches value iteration's (the model-based ground truth).

### Define your own grid

Grids are plain text: whitespace-separated cells, one row per line.

```
S . . . #
. # . . .
. . . # G
```

`S` = start, `.` = empty, `#` = wall, `G` = goal (reward +1 by default),
`H` = hazard (reward -1 by default). Any other uppercase letter works too
as long as you pass `terminal_rewards={"X": ...}` when constructing
`GridWorld` in Python (the CLI ships with `G`/`H` only).

## Tech stack

Pure Python 3.8+ standard library (`argparse`, `dataclasses`, `random`,
`json`). Test suite is `pytest`. No numpy, no ML/RL framework -- every
algorithm (Bellman backups, TD updates, epsilon-greedy sampling) is
written out directly.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

44 tests across parsing/dynamics, value iteration, policy iteration,
Q-learning, rendering/animation, and the CLI. Includes a closed-form value
check on a plain corridor (catches the terminal-value double-counting bug
described above), cross-checks that policy iteration and value iteration
converge to identical policies and values, and statistical checks that
Q-learning's greedy policy agrees with the DP-optimal one on 85%+ of
states.

## Current status

v1, functional and tested. All three solvers, the CLI (`solve`,
`animate`, `compare`), and ASCII rendering/animation are implemented and
working. Three sample grids are included (`grids/classic.txt` -- the R&N
3x4 textbook grid, `grids/corridor.txt` -- a trivial 1x6 sanity-check
grid, `grids/maze.txt` -- a 7x7 grid with walls and dead ends). Nothing
known to be missing for v1; possible future extensions: a `--gif`/image
renderer for the animation, SARSA alongside Q-learning for an on-policy
comparison, and a random-grid generator for stress-testing larger state
spaces.

## License

MIT
