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


@pytest.mark.parametrize("on", [0, 1])
def test_predation_toggle_sets_sizes_and_columns(tmp_path, on):
    log = tmp_path / "run.csv"
    r = alife("--headless", "--steps", "10", "--seed", "1", "--log", str(log),
              "--log-every", "10", "--set", f"PREDATION={on}", *SMALL)  # fmt: skip
    assert r.returncode == 0, r.stderr
    row = list(csv.DictReader(log.open()))[-1]
    assert ("mean_diet" in row) == bool(on)


@pytest.mark.parametrize("on", [0, 1])
def test_sex_toggle_runs_and_sets_columns(tmp_path, on):
    log = tmp_path / "run.csv"
    r = alife("--headless", "--steps", "10", "--seed", "1", "--log", str(log),
              "--log-every", "10", "--set", f"SEX={on}", *SMALL)  # fmt: skip
    assert r.returncode == 0, r.stderr
    row = list(csv.DictReader(log.open()))[-1]
    assert ("mean_pick" in row) == bool(on)


@pytest.mark.parametrize("on", [0, 1])
def test_body_and_terrain_toggles_run(tmp_path, on):
    log = tmp_path / "run.csv"
    r = alife("--headless", "--steps", "10", "--seed", "1", "--log", str(log),
              "--log-every", "10", "--set", f"BODY={on}", "--set", f"TERRAIN={on}",
              *SMALL)  # fmt: skip
    assert r.returncode == 0, r.stderr
    row = list(csv.DictReader(log.open()))[-1]
    assert ("mean_size" in row) == bool(on)


def test_save_and_load_round_trip(tmp_path):
    # Seeded run to t=20 saving at the end; a resumed run to t=40 must log the same
    # rows as one straight run to t=40. The window shows only the loaded part.
    full, part, rest = (tmp_path / n for n in ("full.csv", "part.csv", "rest.csv"))
    snap = tmp_path / "w.alife"
    common = ["--headless", "--log-every", "10", *SMALL]
    runs = [
        ["--steps", "40", "--seed", "4", "--log", str(full)],
        ["--steps", "20", "--seed", "4", "--log", str(part), "--save", str(snap)],
        ["--steps", "40", "--load", str(snap), "--log", str(rest)],
    ]
    for extra in runs:
        r = alife(*common, *extra)
        assert r.returncode == 0, r.stderr
    straight = full.read_text().splitlines()
    resumed = rest.read_text().splitlines()
    assert part.read_text().splitlines() == straight[:4]  # header, t=0, 10, 20
    assert resumed[:2] == [straight[0], straight[3]]  # header, t=20 logged on load
    assert resumed[2:] == straight[4:]  # t=30, 40 identical to the straight run


def test_save_every_writes_file(tmp_path):
    snap = tmp_path / "w.alife"
    r = alife("--headless", "--steps", "10", "--seed", "1", "--save", str(snap),
              "--save-every", "5", *SMALL)  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert snap.exists()


def test_load_shape_mismatch_fails(tmp_path):
    snap = tmp_path / "w.alife"
    assert (
        alife("--headless", "--steps", "3", "--save", str(snap), *SMALL).returncode == 0
    )
    r = alife(
        "--headless", "--steps", "5", "--load", str(snap), "--set", "N_HID=8", *SMALL
    )
    assert r.returncode != 0
    assert "N_HID" in r.stderr


def test_species_log_and_n_species_column(tmp_path):
    log, sp = tmp_path / "run.csv", tmp_path / "sp.csv"
    r = alife("--headless", "--steps", "20", "--seed", "1", "--log", str(log),
              "--species-log", str(sp), "--set", "SPECIES_EVERY=10", *SMALL)  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "n_species" in next(csv.DictReader(log.open()))
    rows = list(csv.DictReader(sp.open()))
    assert rows and rows[0]["event"] == "new" and rows[0]["t"] == "0"
    assert list(rows[0]) == ["t", "event", "species", "parent_species", "size"]


def test_species_off_has_no_column_and_still_reproducible(tmp_path):
    outs = []
    for name in "ab":
        log = tmp_path / f"{name}.csv"
        r = alife("--headless", "--steps", "20", "--seed", "3", "--log", str(log),
                  "--set", "SPECIES=0", *SMALL)  # fmt: skip
        assert r.returncode == 0, r.stderr
        outs.append(log.read_text())
    assert outs[0] == outs[1] and "n_species" not in outs[0]


def test_daynight_off_has_no_light_column(tmp_path):
    cols = {}
    for d in (0, 1):
        log = tmp_path / f"d{d}.csv"
        r = alife("--headless", "--steps", "10", "--seed", "1", "--log", str(log),
                  "--log-every", "10", "--set", f"DAYNIGHT={d}", *SMALL)  # fmt: skip
        assert r.returncode == 0, r.stderr
        cols[d] = next(iter(csv.DictReader(log.open()))).keys()
    assert "light" not in cols[0] and "light" in cols[1]
    assert [c for c in cols[1] if c != "light"] == list(cols[0])


@pytest.mark.parametrize("on", [0, 1])
def test_share_toggle_sets_columns_and_n_out(tmp_path, on):
    log = tmp_path / "run.csv"
    r = alife("--headless", "--steps", "10", "--seed", "1", "--log", str(log),
              "--log-every", "10", "--set", f"SHARE={on}", *SMALL)  # fmt: skip
    assert r.returncode == 0, r.stderr
    row = list(csv.DictReader(log.open()))[-1]
    assert ("mean_give" in row) == bool(on)
    code = (
        f"from alife import config; config.override({{'SHARE': '{on}'}}); "
        "print(config.N_OUT - 2 - config.PHERO_CH - config.PREDATION)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(on)
