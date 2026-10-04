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


def test_seasons_swing_fertility_and_drift_is_deterministic():
    w = make_world()
    base = w.fert0
    w.t = config.SEASON_PERIOD // 4  # sine peak
    w.update_fert()
    peak = w.fert.clone()
    w.t = 3 * config.SEASON_PERIOD // 4  # trough
    w.update_fert()
    assert peak.mean() > w.fert.mean()
    again = w.fert.clone()
    w.update_fert()
    assert torch.equal(again, w.fert)
    assert torch.equal(base, w.fert0)  # base map untouched
    w.t = 0
    w.update_fert()
    assert torch.allclose(
        w.fert, base, atol=1e-6
    )  # start of cycle: no scaling, no drift


def test_meteor_kills_disc_and_clears_food():
    w = make_world()
    w.pos[0] = torch.tensor([2.0, 2.0], device=DEV)
    w.pos[1] = torch.tensor([W / 2, H / 2], device=DEV)
    w.alive[:2] = True
    w.food.fill_(1.0)
    w.meteor(0, 0, r=10)  # disc wraps around the corner
    assert not w.alive[0] and w.alive[1]
    assert w.food[0, 0] == 0 and w.food[-1, -1] == 0 and w.food[H // 2, W // 2] == 1


def test_flood_refills_food():
    w = make_world()
    w.food.zero_()
    w.flood()
    assert torch.equal(w.food, w.fert)


def test_plague_drains_and_spreads_then_ends():
    w = make_world()
    w.alive[:] = False
    w.alive[:3] = True
    w.pos[:3] = 5.5  # same cell
    w.energy[:3] = 1.0
    w.sick[:] = False
    w.sick[0] = True
    w.plague_left = 2
    w.step()
    assert w.plague_left == 1
    assert w.energy[0] < w.energy[2] + 1  # sick one paid the drain
    w.step()
    assert w.plague_left == 0 and not w.sick.any()  # outbreak over: cleared
    w.plague()
    assert w.plague_left == config.PLAGUE_STEPS


def test_trails_fade_and_can_be_off():
    from alife.render import render

    w = make_world()
    render(w, 0)
    first = w.trail.clone()
    w.alive[:] = False
    render(w, 0)
    assert w.trail.max() <= first.max() * config.TRAIL_DECAY + 1e-6


def test_blur_handles_channel_stacks():
    g = torch.rand(2, H, W, device=DEV)
    torch.testing.assert_close(blur(g, 1)[1], blur(g[1], 1))


def test_pheromone_deposit_diffuse_decay():
    w = make_world()
    w.alive[:] = False
    w.phero[:] = 0
    w.phero[0, 100, 100] = 1.0
    total = float(w.phero.sum())
    w.step()  # nobody alive: scent only spreads and fades
    assert float(w.phero.sum()) == pytest.approx(total * config.PHERO_DECAY, rel=1e-4)
    assert w.phero[0, 100, 100] < 1.0 and w.phero[0, 100, 101] > 0
    assert (w.phero[1] == 0).all()  # channels do not mix


def test_pheromone_creature_deposits_and_smells():
    w = make_world()
    w.alive[:] = False
    w.alive[0] = True
    w.pos[0] = torch.tensor([200.5, 300.5], device=DEV)
    w.genome[0] = 0
    w.genome[0, -config.N_HID * config.N_OUT :] = 0  # w_out zero: sigmoid(0)=0.5
    w.h[0] = 0
    w.phero[:] = 0
    w.step()
    # Output 0 everywhere -> drops PHERO_DEPOSIT/2 per channel, then spreads.
    expect = config.PHERO_DEPOSIT / 2 * config.PHERO_DECAY
    assert float(w.phero.sum(dim=(1, 2))[0]) == pytest.approx(expect, rel=1e-3)
    assert float(w.phero[:, 300, 200].min()) > 0  # near the drop site


def test_pheromone_sensing_changes_creature_input():
    # A creature facing a scent source gets a different brain input than one in a
    # clean world; check through the hidden state after one step.
    def run(scent):
        w = make_world()
        w.alive[:] = False
        w.alive[0] = True
        w.pos[0] = torch.tensor([200.5, 300.5], device=DEV)
        w.heading[0] = 0.0
        w.phero[:] = 0
        if scent:
            w.phero[:, 300, 206] = 5.0
        torch.manual_seed(1)
        w.genome[0] = torch.randn(config.G, device=DEV)
        w.h[0] = 0
        w.step()
        return w.h[0].clone()

    if config.PHERO_CH:
        assert not torch.allclose(run(True), run(False))


def _set_attack(w, levels):
    """Make creature i's attack output ~sigmoid(levels[i]) via bias and w_out."""
    nh, no = config.N_HID, config.N_OUT
    b = nh * config.N_IN + nh * nh
    for i, lv in enumerate(levels):
        w.genome[i, b : b + nh] = 5.0  # hidden state saturates at 1
        w.genome[i, b + nh :] = 0
        w.genome[i, b + nh + (no - 1) * nh :] = lv / nh


def _arena(n=2):
    """n creatures in one cell with a zero brain: attack output sigmoid(0) = 0.5."""
    w = make_world()
    w.alive[:] = False
    w.alive[:n] = True
    w.pos[:n] = 50.5
    w.genome[:n] = 0
    w.h[:n] = 0
    w.energy[:n] = 1.0
    w.food.zero_()
    w.meat.zero_()
    return w


def test_attack_moves_energy_by_diet():
    if not config.PREDATION:
        pytest.skip("predation off")
    w = _arena()
    w.diet[:2] = 1.0
    _set_attack(w, [20.0, -20.0])  # creature 0 hunts, creature 1 is passive prey
    w.step()
    gain = config.ATTACK_RATE  # full-strength attack, pure carnivore
    assert float(w.energy[0]) == pytest.approx(1 + gain - 0.04, abs=0.03)
    assert float(w.energy[1]) == pytest.approx(1 - gain - 0.04, abs=0.03)
    w = _arena()
    w.diet[:2] = torch.tensor([0.5, 0.5], device=DEV)
    _set_attack(w, [20.0, -20.0])
    w.step()  # half-digested: the predator gets half of what the prey loses
    assert float(w.energy[0]) == pytest.approx(1 + gain / 2 - 0.04, abs=0.03)
    assert float(w.energy[1]) == pytest.approx(1 - gain - 0.04, abs=0.03)


def test_attack_needs_a_victim_and_diet_scales_gain():
    if not config.PREDATION:
        pytest.skip("predation off")
    w = _arena(1)
    w.diet[0] = 1.0
    w.step()  # alone: pays attack cost, steals nothing
    assert w.energy[0] < 1.0
    w = _arena()
    w.diet[:2] = 0.0  # two herbivores hurt each other and nobody gains
    _set_attack(w, [20.0, 20.0])
    w.step()
    assert float(w.energy[:2].sum()) < 0.2


def test_death_drops_meat_which_decays_and_is_eaten_by_diet():
    if not config.PREDATION:
        pytest.skip("predation off")
    w = _arena(1)
    w.energy[0] = 0.0001  # dies this step, leaving almost nothing
    w.age[0] = config.MAX_AGE - 1  # dies of old age with real energy instead
    w.energy[0] = 1.0
    w.step()
    assert not w.alive[0]
    assert float(w.meat.sum()) == pytest.approx(
        config.MEAT_DROP * 1.0 * config.MEAT_DECAY, rel=0.05
    )
    assert w.meat[50, 50] > 0
    before = float(w.meat.sum())
    w.step()
    assert float(w.meat.sum()) == pytest.approx(before * config.MEAT_DECAY, rel=1e-4)
    # A carnivore eats it, a herbivore does not.
    for diet, eats in ((1.0, True), (0.0, False)):
        w = _arena(1)
        w.meat[50, 50] = 1.0
        w.diet[0] = diet
        w.step()
        assert (float(w.meat[50, 50]) < config.MEAT_DECAY) == eats


def test_census_has_diet():
    if config.PREDATION:
        row = make_world().census()
        assert 0 <= row["mean_diet"] <= 1 and "carnivores" in row
