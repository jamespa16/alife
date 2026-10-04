import pytest
import torch

from alife import config
from alife.config import DEV, K_MAX, N_HID, N_MAX, REPRO_E, H, W
from alife.world import World, blur


def make_world(seed=0):
    torch.manual_seed(seed)
    return World()


@pytest.fixture(scope="module")
def ran():
    w = make_world()
    for _ in range(50):
        w.step()
    return w


def test_state_stays_valid_after_running(ran):
    w, a = ran, ran.alive
    assert w.t == 50
    assert 0 < int(a.sum()) <= N_MAX
    for name in ("pos", "heading", "energy", "genome", "h", "food"):
        assert torch.isfinite(getattr(w, name)).all(), name
    assert ((w.pos >= 0) & (w.pos < torch.tensor([W, H], device=DEV))).all()
    assert ((w.heading >= 0) & (w.heading < config.TAU)).all()
    assert ((w.loops >= 1) & (w.loops <= K_MAX)).all()
    assert (w.energy[a] > 0).all()
    assert (w.food >= 0).all() and (w.food <= w.fert + 1e-6).all()
    assert (w.h[a].abs() <= 1).all()  # tanh output


def test_reproduce_splits_energy_and_initialises_child():
    w = make_world()
    w.alive[:] = False
    parents = torch.tensor([3, 7], device=DEV)
    w.alive[parents] = True
    w.energy[parents] = torch.tensor([2 * REPRO_E, REPRO_E + 1], device=DEV)
    w.h[:] = 0.5  # dead slots keep stale memory from their last owner
    w.age[parents] = 100
    total = float(w.energy[w.alive].sum())

    w.reproduce()

    a = w.alive
    assert int(a.sum()) == 4
    kids = a.nonzero().squeeze(1)
    kids = kids[~torch.isin(kids, parents)]
    assert float(w.energy[a].sum()) == pytest.approx(total)
    assert (w.age[kids] == 0).all()
    assert (w.h[kids] == 0).all()
    assert (w.h[parents] == 0.5).all()  # parents keep their memory
    assert ((w.loops[kids] >= 1) & (w.loops[kids] <= K_MAX)).all()


def test_reproduce_needs_energy_and_free_slots():
    w = make_world()
    w.alive[:] = False
    w.alive[0] = True
    w.energy[0] = REPRO_E  # threshold is strict
    w.reproduce()
    assert int(w.alive.sum()) == 1

    w.alive[:] = True  # world full: nobody can split
    w.energy[:] = 2 * REPRO_E
    w.reproduce()
    assert (w.energy == 2 * REPRO_E).all()


def test_step_saves_brain_memory():
    w = make_world()
    assert (w.h == 0).all()
    w.step()
    a = w.alive & (w.age == 1)  # thought this step, not newborn
    assert a.any() and (w.h[a] != 0).any(dim=1).all()


def test_dead_creatures_do_not_eat():
    w = make_world()
    w.alive[:] = False
    food = w.food.clone()
    w.step()
    # Only regrowth happens: food can't go down with nobody alive.
    assert (w.food >= food).all()


def test_same_seed_same_world():
    # Run one after the other: worlds share torch's global RNG.
    runs = []
    for _ in range(2):
        w = make_world(5)
        for _ in range(20):
            w.step()
        runs.append(w)
    a, b = runs
    for name in ("pos", "energy", "alive", "loops", "h", "food"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_census(ran):
    row = ran.census()
    pop = int(ran.alive.sum())
    assert row["t"] == 50 and row["pop"] == pop
    assert sum(row[f"loops_{k}"] for k in range(1, K_MAX + 1)) == pop
    assert 1 <= row["mean_loops"] <= K_MAX


def test_census_of_empty_world():
    w = make_world()
    w.alive[:] = False
    row = w.census()
    assert row["pop"] == 0 and row["mean_loops"] == 0


def test_blur_conserves_mass_and_wraps():
    g = torch.zeros(H, W, device=DEV)
    g[0, 0] = 1.0  # corner: the blur has to wrap to every edge
    b = blur(g)
    assert float(b.sum()) == pytest.approx(1.0, rel=1e-4)
    assert b[-1, -1] > 0 and b[0, -1] > 0 and b[-1, 0] > 0


def test_override_refuses_after_sim_import():
    # world is imported by this module, so constants are already copied.
    with pytest.raises(RuntimeError):
        config.override({"N_HID": "8"})
    assert config.N_HID == N_HID
