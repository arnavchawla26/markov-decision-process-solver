import json

from mdpsolve.cli import build_parser, policy_from_json, policy_to_json


def _run(args_list):
    parser = build_parser()
    args = parser.parse_args(args_list)
    return args.func(args)


def test_solve_value_iteration_prints_policy(capsys):
    rc = _run(["solve", "--grid", "grids/classic.txt", "--method", "value-iteration", "--deterministic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Policy:" in out
    assert "Values:" in out
    assert "Value at start state" in out


def test_solve_writes_json_out(tmp_path, capsys):
    out_path = tmp_path / "policy.json"
    rc = _run(
        [
            "solve",
            "--grid",
            "grids/classic.txt",
            "--method",
            "policy-iteration",
            "--deterministic",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    data = json.loads(out_path.read_text())
    assert data["method"] == "policy-iteration"
    assert "0,0" in data["policy"]
    assert "0,3" in data["values"]


def test_solve_q_learning(capsys):
    rc = _run(
        [
            "solve",
            "--grid",
            "grids/corridor.txt",
            "--method",
            "q-learning",
            "--deterministic",
            "--episodes",
            "300",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "mean_return_last_100_episodes" in out


def test_animate_command_runs(capsys):
    rc = _run(
        [
            "animate",
            "--grid",
            "grids/corridor.txt",
            "--method",
            "value-iteration",
            "--deterministic",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "start" in out
    assert "Finished after" in out


def test_animate_with_explicit_start(capsys):
    rc = _run(
        [
            "animate",
            "--grid",
            "grids/corridor.txt",
            "--method",
            "value-iteration",
            "--deterministic",
            "--start",
            "0,2",
            "--seed",
            "0",
        ]
    )
    assert rc == 0


def test_compare_command_lists_all_methods(capsys):
    rc = _run(
        [
            "compare",
            "--grid",
            "grids/corridor.txt",
            "--deterministic",
            "--episodes",
            "300",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "value-iteration" in out
    assert "policy-iteration" in out
    assert "q-learning" in out
    assert "agreement vs VI" in out


def test_policy_json_round_trip():
    policy = {(0, 0): "R", (1, 2): "U"}
    encoded = policy_to_json(policy)
    assert encoded == {"0,0": "R", "1,2": "U"}
    decoded = policy_from_json(encoded)
    assert decoded == policy
