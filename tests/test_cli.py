"""CLI runs in subprocesses: --set must happen before the sim modules import."""

import csv
import subprocess
import sys

import pytest

SMALL = ["--set", "N_MAX=4000", "--set", "N_INIT=500"]


def alife(*args):
    return subprocess.run(
        [sys.executable, "-m", "alife", *args],
        capture_output=True,
        text=True,
        check=False,  # callers check returncode
    )


def test_headless_run_writes_csv(tmp_path):
    log = tmp_path / "run.csv"
    r = alife("--headless", "--steps", "30", "--seed", "1", "--log", str(log),
              "--log-every", "10", "--set", "N_HID=8", *SMALL)  # fmt: skip
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(log.open()))
    assert [int(row["t"]) for row in rows] == [0, 10, 20, 30]
    assert int(rows[0]["pop"]) == 500
    assert {"mean_loops", "loops_1", "loops_8"} <= rows[0].keys()


def test_seed_reproduces_csv(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    for log in (a, b):
        r = alife(
            "--headless", "--steps", "20", "--seed", "3", "--log", str(log), *SMALL
        )
        assert r.returncode == 0, r.stderr
    assert a.read_text() == b.read_text()


@pytest.mark.parametrize(
    "args, message",
    [
        (["--headless"], "--headless needs --steps"),
        (["--set", "NOPE=1"], "not an overridable"),
        (["--set", "G=5"], "not an overridable"),  # derived from N_HID etc.
        (["--set", "SENSE_ANGLES=1"], "not an overridable"),
        (["--set", "N_HID=1.5"], "expected int"),
        (["--set", "THINK_COST=abc"], "expected float"),
        (["--set", "THINK_COST"], "expected NAME=VALUE"),
        (["--log-every", "0"], "at least 1"),
    ],
)
def test_bad_arguments(args, message):
    r = alife(*args)
    assert r.returncode == 2
    assert message in r.stderr


@pytest.mark.parametrize("ch", [0, 1, 3])
def test_pheromone_channels_set_derived_sizes(tmp_path, ch):
    # N_IN / N_OUT / G are derived from PHERO_CH after --set; the brain must follow.
    log = tmp_path / "run.csv"
    r = alife("--headless", "--steps", "10", "--seed", "1", "--log", str(log),
              "--log-every", "10", "--set", f"PHERO_CH={ch}", *SMALL)  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert int(list(csv.DictReader(log.open()))[-1]["t"]) == 10
