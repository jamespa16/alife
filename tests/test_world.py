import pytest
import torch

from alife import config, world
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


def test_reproduce_splits_energy_and_initialises_child(monkeypatch):
    monkeypatch.setattr(world, "SEX", 0)  # lone-parent splitting
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


def test_reproduce_needs_energy_and_free_slots(monkeypatch):
    monkeypatch.setattr(world, "SEX", 0)  # lone-parent splitting
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
    open_ = ~w.wall
    assert torch.allclose(
        w.fert[open_], base[open_], atol=1e-6
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
        if scent:  # a patch, so the sense point hits it whatever the range gene
            w.phero[:, 296:305, 202:212] = 5.0
        torch.manual_seed(1)
        w.genome[0] = torch.randn(config.G, device=DEV)
        w.h[0] = 0
        w.step()
        return w.h[0].clone()

    if config.PHERO_CH:
        assert not torch.allclose(run(True), run(False))


def _set_attack(w, levels):
    """Make creature i's attack output ~sigmoid(levels[i]) via bias and w_out."""
    nh = config.N_HID
    b = nh * config.N_IN + nh * nh
    for i, lv in enumerate(levels):
        w.genome[i, b : b + nh] = 5.0  # hidden state saturates at 1
        w.genome[i, b + nh :] = 0
        w.genome[i, b + nh + (2 + config.PHERO_CH) * nh :][:nh] = lv / nh


def _set_give(w, levels):
    """Make creature i's give output ~levels[i] (sigmoid(level - GIVE_BIAS) * GIVE_RATE)."""
    nh = config.N_HID
    b = nh * config.N_IN + nh * nh
    for i, lv in enumerate(levels):
        w.genome[i, b : b + nh] = 5.0
        w.genome[i, b + nh :] = 0
        w.genome[i, b + nh + (config.N_OUT - 1) * nh :][:nh] = lv / nh
        if config.PREDATION:  # nobody attacks
            w.genome[i, b + nh + (2 + config.PHERO_CH) * nh :][:nh] = -50.0 / nh


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
    w.body[:] = 1.0  # average bodies
    w.wall[:] = False  # the arena is open ground
    w.poison[:] = False
    w.open.fill_(1)
    w.poison_flat.zero_()
    w.terr_s.zero_()
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


# ---- sexual reproduction ----


def _sex_world(monkeypatch, n, mut=0.0):
    """Empty world with creatures 0..n-1 alive, eligible, placed by the caller."""
    monkeypatch.setattr(world, "SEX", 1)
    monkeypatch.setattr(world, "MUT_STD", mut)
    w = make_world()
    w.alive[:] = False
    w.alive[:n] = True
    w.energy[:n] = 2 * REPRO_E
    w.pick[:] = 10.0  # anyone mates with anyone
    return w


def test_pairing_only_within_a_cell(monkeypatch):
    w = _sex_world(monkeypatch, 7)
    # Cell A holds 0-2 (one pair, one left over), cell B holds 3-4, 5 and 6 are alone.
    w.pos[:3] = 10.5
    w.pos[3:5] = 20.5
    w.pos[5] = 30.5
    w.pos[6] = 40.5
    for _ in range(5):
        a, b, ok = w.pair_up(torch.arange(7, device=DEV), 3)
        assert int(ok.sum()) == 2
        a, b = a[ok], b[ok]
        assert (w.pos[a] // 1 == w.pos[b] // 1).all()
        assert len({*a.tolist(), *b.tolist()}) == 4  # nobody mates twice
        assert not ({5, 6} & {*a.tolist(), *b.tolist()})


def test_child_genes_come_from_one_parent_or_the_other(monkeypatch):
    w = _sex_world(monkeypatch, 2)
    w.pos[:2] = 5.5
    w.genome[0] = 1.0
    w.genome[1] = -1.0
    w.loops[:2] = torch.tensor([2, 6], device=DEV)
    w.color[:2] = 0.5
    total = float(w.energy[:2].sum())
    w.reproduce()
    kids = (w.alive & (torch.arange(N_MAX, device=DEV) >= 2)).nonzero().squeeze(1)
    assert len(kids) == 1
    g = w.genome[kids[0]]
    assert ((g == 1.0) | (g == -1.0)).all()
    assert (g == 1.0).any() and (g == -1.0).any()  # mixed, not a copy
    assert int(w.loops[kids[0]]) in (2, 6)
    # Both parents paid, and the child got exactly what they paid.
    assert (w.energy[:2] < 2 * REPRO_E).all()
    assert float(w.energy[w.alive].sum()) == pytest.approx(total, rel=1e-5)
    assert int(w.age[kids[0]]) == 0 and (w.h[kids[0]] == 0).all()


def test_mate_preference_blocks_dissimilar_pairs(monkeypatch):
    w = _sex_world(monkeypatch, 2)
    w.pos[:2] = 5.5
    w.color[0] = 0.3
    w.color[1] = 1.0  # distance ~1.2
    w.pick[:2] = torch.tensor([5.0, 0.5], device=DEV)  # one of them is choosy
    w.reproduce()
    assert int(w.alive.sum()) == 2
    w.pick[:2] = 5.0  # now both are open-minded
    w.reproduce()
    assert int(w.alive.sum()) == 3


def test_sex_respects_free_slots_and_needs_a_partner(monkeypatch):
    w = _sex_world(monkeypatch, 1)  # a lone eligible creature
    w.reproduce()
    assert int(w.alive.sum()) == 1
    w = _sex_world(monkeypatch, 40)
    w.pos[:40] = 5.5  # one crowded cell: 20 pairs
    w.alive[40:] = True
    w.alive[-3:] = False  # only 3 free slots
    w.energy[40:] = 0.0
    w.reproduce()
    assert int(w.alive.sum()) == N_MAX - 3 + 3
    assert int((w.energy[:40] < 2 * REPRO_E).sum()) == 6  # 3 pairs paid


def test_sex_world_runs_and_is_deterministic(monkeypatch):
    monkeypatch.setattr(world, "SEX", 1)
    runs = []
    for _ in range(2):
        w = make_world(2)
        for _ in range(15):
            w.step()
        runs.append(w)
    for name in ("pos", "energy", "alive", "genome", "pick", "color"):
        assert torch.equal(getattr(runs[0], name), getattr(runs[1], name)), name
    assert torch.isfinite(runs[0].energy).all()


# ---- body-shape genes ----


def test_size_scales_fights():
    if not (config.PREDATION and config.BODY):
        pytest.skip("needs predation and body genes")

    def duel(size_a, size_b):
        w = _arena()
        w.diet[:2] = 1.0
        w.energy[:2] = 5.0  # enough to survive the bite
        w.body[0, 0], w.body[1, 0] = size_a, size_b
        _set_attack(w, [20.0, -20.0])  # 0 hunts, 1 is passive
        w.step()
        return float(w.energy[0]), float(w.energy[1])

    even = duel(1.0, 1.0)
    big_hunter = duel(1.5, 1.0)
    big_prey = duel(1.0, 1.5)
    assert big_hunter[1] < even[1] and big_hunter[0] > even[0]
    assert big_prey[1] > even[1] and big_prey[0] < even[0]  # a big victim loses less


def test_body_costs_and_speed_gene():
    if not config.BODY:
        pytest.skip("body off")

    def run(i, val):
        w = _arena(1)
        w.body[0] = 1.0
        w.body[0, i] = val
        w.step()
        return float(w.energy[0]), w

    for i, cost in enumerate((config.SIZE_COST, config.SPEED_COST, config.SENSE_COST)):
        lo, _ = run(i, 1.0)
        hi, _ = run(i, 2.0)
        # speed also raises move cost slightly, so only require at least the gene cost
        assert lo - hi >= cost * 0.99
    # Top speed gene scales how far a creature moves.
    moved = []
    for g in (1.0, 2.0):
        w = _arena(1)
        w.body[0] = 1.0
        w.body[0, 1] = g
        w.heading[0] = 0.0
        w.step()
        moved.append(float(w.pos[0, 0]) - 50.5)
    assert 0 < moved[0] < moved[1]


def test_sense_gene_scales_reach_and_inheritance(monkeypatch):
    if not config.BODY:
        pytest.skip("body off")
    w = _sex_world(monkeypatch, 2, mut=0.0)
    monkeypatch.setattr(world, "BODY_MUT_STD", 0.0)
    w.pos[:2] = 5.5
    w.body[:2] = torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 0.5]], device=DEV)
    w.reproduce()
    kid = (w.alive & (torch.arange(N_MAX, device=DEV) >= 2)).nonzero().squeeze(1)[0]
    assert torch.allclose(w.body[kid], torch.tensor([1.5, 1.0, 0.75], device=DEV))
    monkeypatch.setattr(world, "SEX", 0)
    w.alive[:] = False
    w.alive[0] = True
    w.energy[0] = 2 * REPRO_E
    w.reproduce()
    kid = (w.alive & (torch.arange(N_MAX, device=DEV) >= 1)).nonzero().squeeze(1)[0]
    assert torch.equal(w.body[kid], w.body[0])


def test_census_has_body_means():
    if config.BODY:
        row = make_world().census()
        assert 0.5 <= row["mean_size"] <= 2 and "mean_sense" in row


# ---- terrain ----


def _walled():
    """World with a wall column at x=61..63 and one lone creature at x=58.5."""
    w = _arena(1)
    w.wall[:] = False
    w.poison[:] = False
    w.wall[:, 61:64] = True
    w.open = (~w.wall).float()
    w.poison_flat = w.poison.float().view(-1)
    w.terr_s = world.blur(torch.stack([w.wall.float(), w.poison.float()]), 2)
    w.pos[0] = torch.tensor([58.5, 50.5], device=DEV)
    w.heading[0] = 0.0
    w.body[0] = 1.0
    w.genome[0] = 0
    return w


def test_terrain_generated_mostly_open():
    if not config.TERRAIN:
        pytest.skip("terrain off")
    w = make_world()
    assert 0.5 * config.WALL_FRAC < float(w.wall.float().mean()) < 2 * config.WALL_FRAC
    assert w.poison.any() and not (w.poison & w.wall).any()
    assert (w.fert[w.wall] == 0).all() and (w.food[w.wall] == 0).all()
    inside = w.wall[w.pos[:, 1].long() % H, w.pos[:, 0].long() % W]
    assert not inside[
        :100
    ].any()  # nobody starts in a wall (barring rare redraw misses)


def test_walls_block_movement():
    if not config.TERRAIN:
        pytest.skip("terrain off")
    w = _walled()
    w.body[0, 1] = 2.0  # fast: would reach x=62.5 within a few steps
    for _ in range(10):
        w.step()
        assert float(w.pos[0, 0]) < 61
    assert float(w.pos[0, 0]) > 58.6  # it did move up to the wall
    # A diagonal approach slides along the wall instead of sticking to it.
    w = _walled()
    w.pos[0] = torch.tensor([60.9, 50.5], device=DEV)
    w.heading[0] = 0.0
    w.body[0, 1] = 2.0
    w.step()
    assert float(w.pos[0, 0]) == pytest.approx(60.9)


def test_walls_stop_food_and_flood():
    if not config.TERRAIN:
        pytest.skip("terrain off")
    w = make_world()
    w.flood()
    assert (w.food[w.wall] == 0).all() and (w.food[~w.wall] > 0).all()


def test_poison_drains_energy():
    if not config.TERRAIN:
        pytest.skip("terrain off")
    w = _walled()
    w.step()
    clean = float(w.energy[0])
    w = _walled()
    w.poison[50, 58:61] = True
    w.poison_flat = w.poison.float().view(-1)
    w.heading[0] = 0.0
    w.pos[0] = torch.tensor([58.0, 50.5], device=DEV)
    w.genome[0] = 0
    w.step()
    assert clean - float(w.energy[0]) == pytest.approx(config.POISON_DRAIN, abs=0.005)


def test_terrain_sensors_see_walls_and_poison():
    if not config.TERRAIN:
        pytest.skip("terrain off")

    def run(wall):
        w = _walled()
        w.wall[:] = False
        if wall:
            w.wall[:, 61:64] = True
        w.terr_s = world.blur(torch.stack([w.wall.float(), w.poison.float()]), 2)
        torch.manual_seed(1)
        w.genome[0] = torch.randn(config.G, device=DEV)
        w.pos[0, 0] = 56.5  # wall is within SENSE_DIST ahead
        w.step()
        return w.h[0].clone()

    assert not torch.allclose(run(True), run(False))


def run_state(w, n):
    for _ in range(n):
        w.step()
    return {k: v.clone() for k, v in vars(w).items() if isinstance(v, torch.Tensor)}


def test_save_load_continues_identically(tmp_path):
    w = make_world(5)
    run_state(w, 15)
    w.plague()  # plague state (plague_left, sick) must survive too
    w.trail = torch.ones(H, W, 3, device=DEV)  # render-only: not saved
    path = tmp_path / "w.alife"
    w.save(path)
    want = run_state(w, 10)  # the uninterrupted run
    census = w.census()

    w2 = World.load(path)
    assert w2.t == 15 and w2.plague_left > 0 and w2.trail is None
    got = run_state(w2, 10)
    assert w2.census() == census
    want.pop("trail")  # render-only, not saved
    assert got.keys() == want.keys()
    for k in want:
        assert torch.equal(got[k], want[k]), k


def test_save_covers_every_tensor_attribute(tmp_path):
    w = make_world(1)
    w.extra = torch.arange(3, device=DEV)  # a future attribute: saved automatically
    w.save(tmp_path / "w.alife")
    w2 = World.load(tmp_path / "w.alife")
    saved = {k for k, v in vars(w).items() if isinstance(v, torch.Tensor)} - {"trail"}
    assert saved <= set(vars(w2))
    assert torch.equal(w2.extra, w.extra)


def test_load_rejects_shape_mismatch(tmp_path, monkeypatch):
    w = make_world(1)
    w.save(tmp_path / "w.alife")
    monkeypatch.setattr(config, "N_HID", N_HID + 1)
    with pytest.raises(ValueError, match="N_HID"):
        World.load(tmp_path / "w.alife")


def test_load_warns_on_other_config_change(tmp_path, monkeypatch):
    w = make_world(1)
    w.save(tmp_path / "w.alife")
    monkeypatch.setattr(config, "THINK_COST", 0.5)
    with pytest.warns(UserWarning, match="THINK_COST"):
        World.load(tmp_path / "w.alife")


# ---- family and species ----


@pytest.mark.parametrize("sex", [0, 1])
def test_births_get_unique_uids_parents_and_generations(monkeypatch, sex):
    monkeypatch.setattr(world, "SEX", sex)
    monkeypatch.setattr(world, "SPECIES", 0)
    w = make_world()
    w.alive[:] = False
    w.alive[:4] = True
    w.energy[:4] = 2 * REPRO_E
    w.pick[:] = 10.0
    w.pos[:4] = w.pos[0]  # one shared cell, so sex pairs form
    w.uid[:4] = torch.tensor([100, 101, 102, 103], device=DEV)
    w.gen[:4] = torch.tensor([0, 5, 2, 1], device=DEV)
    start = int(w.next_uid)
    w.species[:4] = torch.tensor([7, 7, 8, 8], device=DEV)
    w.reproduce()
    kids = w.alive.nonzero().squeeze(1)
    kids = kids[kids >= 4]
    assert len(kids) >= 2
    uids = w.uid[kids]
    assert len(set(uids.tolist())) == len(kids) and (uids >= start).all()
    assert int(w.next_uid) >= start + len(kids)
    assert not (w.uid[:4] == uids[:, None]).any()
    parents = w.parent_id[kids]
    assert torch.isin(parents, torch.tensor([100, 101, 102, 103], device=DEV)).all()
    first = torch.stack([(w.uid[:4] == p).nonzero()[0, 0] for p in parents])
    if sex:  # one more than the older parent
        assert (w.gen[kids] >= w.gen[first] + 1).all()
    else:
        assert (w.gen[kids] == w.gen[first] + 1).all()
    assert (w.species[kids] == w.species[first]).all()  # inherited until next pass


def test_uids_stay_unique_among_the_living(ran):
    uids = ran.uid[ran.alive]
    assert len(torch.unique(uids)) == len(uids)
    assert int(ran.next_uid) > int(uids.max())


def _clustered(centres, per=30, noise=0.01):
    """World whose creatures sit in tight genome clusters around `centres` (G-vectors)."""
    w = make_world()
    w.sp_ids, w.sp_size, w.sp_rep = w.sp_ids[:0], w.sp_size[:0], w.sp_rep[:0]
    w.species[:] = -1  # forget the species found at reset
    w.alive[:] = False
    n = len(centres) * per
    w.alive[:n] = True
    for k, c in enumerate(centres):
        w.genome[k * per : (k + 1) * per] = (
            c + torch.randn(per, config.G, device=DEV) * noise
        )
    return w


def _far(n):
    return [torch.randn(config.G, device=DEV) * 2 for _ in range(n)]


def test_species_cluster_persist_and_log_splits_and_extinction():
    a, b, c = _far(3)
    w = _clustered([a, b])
    w.take_events()
    w.update_species()
    ev = w.take_events()
    assert [e["event"] for e in ev] == ["new", "new"]
    ids = w.species[:60].clone()
    assert len(set(ids[:30].tolist())) == 1 and len(set(ids[30:60].tolist())) == 1
    assert ids[0] != ids[30]
    assert len(w.sp_ids) == 2

    # Same clusters, genomes jittered a little: ids persist, no events.
    w.genome[:60] += torch.randn(60, config.G, device=DEV) * 0.01
    w.update_species()
    assert w.take_events() == [] and torch.equal(w.species[:60], ids)

    # A third cluster appears (born from cluster a's slots) while cluster b dies out.
    w.genome[30:60] = c + torch.randn(30, config.G, device=DEV) * 0.01
    w.update_species()
    ev = w.take_events()
    new = [e for e in ev if e["event"] == "new"]
    gone = [e for e in ev if e["event"] == "extinct"]
    assert len(new) == 1 and len(gone) == 1
    assert gone[0]["species"] == int(ids[30]) and gone[0]["size"] == 30
    assert new[0]["species"] not in ids.tolist() and new[0]["size"] == 30
    assert new[0]["parent_species"] == int(ids[30])  # majority previous label
    assert torch.equal(w.species[:30], ids[:30])
    assert w.census()["n_species"] == 2


def test_species_pass_with_empty_world_and_when_off(monkeypatch):
    w = _clustered(_far(1))
    w.update_species()
    w.take_events()
    w.alive[:] = False
    w.update_species()
    assert [e["event"] for e in w.take_events()] == ["extinct"]
    monkeypatch.setattr(world, "SPECIES", 0)
    assert "n_species" not in make_world().census()


def test_species_ids_survive_save_and_load(tmp_path):
    w = make_world(2)
    for _ in range(5):
        w.step()
    w.save(tmp_path / "w.alife")
    w2 = World.load(tmp_path / "w.alife")
    for k in ("uid", "parent_id", "gen", "species", "next_uid", "sp_ids", "sp_rep"):
        assert torch.equal(getattr(w2, k), getattr(w, k)), k
    assert w2.next_species == w.next_species and w2.events == []


def test_fx_render_shape_and_attack_transient(tmp_path):
    from alife.render import heading_ticks, render

    w = make_world()
    for _ in range(3):
        w.step()
    for fx in (True, False):
        assert render(w, 0, fx).shape == (W, H, 3)
    if config.PREDATION:
        assert w.atk_vis is not None and w.atk_vis.shape == (N_MAX,)
    ticks = heading_ticks(w, (0, 0, W, H))
    assert ticks and all(len(t) == 3 and type(t[0]) is float for t in ticks)
    assert {"atk_vis", "rings", "trail"} <= set(World.TRANSIENT)
    w.save(tmp_path / "w.alife")
    saved = torch.load(tmp_path / "w.alife", weights_only=True)["state"]
    assert "atk_vis" not in saved and "rings" not in saved
    w2 = World.load(tmp_path / "w.alife")
    assert w2.atk_vis is None and w2.rings == []
    assert render(w2, 0).shape == (W, H, 3)


def test_meteor_ring_expires():
    w = make_world()
    w.meteor(10.0, 20.0)
    assert len(w.rings) == 1 and w.rings[0][:3] == [10.0, 20.0, 0]
    for _ in range(config.RING_FRAMES):
        w.tick_rings()
    assert w.rings == []


def test_light_is_a_function_of_t_in_range():
    ts = range(0, 3 * config.DAY_PERIOD, 7)
    ls = [world.light_at(t) for t in ts]
    assert all(0 <= x <= 1 for x in ls)
    assert min(ls) == 0 and max(ls) == 1  # full dark and full light both reached
    assert world.light_at(5) == pytest.approx(world.light_at(5 + config.DAY_PERIOD))


def test_night_stops_growth(monkeypatch):
    monkeypatch.setattr(world, "NIGHT_GROWTH", 0.0)
    w = make_world()
    w.alive[:] = False  # no eating, so only regrowth changes food
    night = 3 * config.DAY_PERIOD // 4 - 1  # step() advances t to the sine minimum
    day = config.DAY_PERIOD // 4 - 1
    gains = []
    for t in (night, day):
        w.t = t
        w.food[:] = 0
        w.step()
        gains.append(float(w.food.sum()))
    if config.DAYNIGHT:
        assert gains[0] == 0 and gains[1] > 0


def _first_input(monkeypatch, w):
    """Step once, returning the sensor matrix the brain saw."""
    seen = {}
    orig = world.brain.think

    def spy(g, loops, x, h):
        seen["x"] = x.clone()
        return orig(g, loops, x, h)

    monkeypatch.setattr(world.brain, "think", spy)
    w.step()
    return seen["x"]


def test_night_shortens_sensing(monkeypatch):
    if not config.DAYNIGHT:
        pytest.skip("DAYNIGHT off")
    monkeypatch.setattr(world, "NIGHT_SENSE", 1.0)  # blind in full darkness

    def seen(t):
        w = make_world()
        w.t = t
        w.alive[:] = False
        w.alive[0] = True
        w.body[0, 2] = 1.0
        w.pos[0] = torch.tensor([300.5, 300.5], device=DEV)
        w.heading[0] = 0
        w.food[:] = 0
        w.food[300, 306] = 1.0  # straight ahead at SENSE_DIST
        return float(_first_input(monkeypatch, w)[0, 1])  # centre food sensor

    noon, dark = config.DAY_PERIOD // 4 - 1, 3 * config.DAY_PERIOD // 4 - 1
    assert seen(noon) > 0 and seen(dark) == 0


def test_light_sensor_is_last_column(monkeypatch):
    if not config.DAYNIGHT:
        pytest.skip("DAYNIGHT off")
    w = make_world()
    w.t = config.DAY_PERIOD // 4 - 1  # step() runs at t, noon is a quarter in
    x = _first_input(monkeypatch, w)
    assert x.shape[1] == config.N_IN
    assert torch.allclose(x[:, -1], torch.full_like(x[:, -1], world.light_at(w.t - 1)))


def test_n_in_derivation(monkeypatch):
    saved = (config.N_IN, config.G)
    try:
        monkeypatch.setattr(config, "DAYNIGHT", 0)
        config._derive()
        off = config.N_IN
        monkeypatch.setattr(config, "DAYNIGHT", 1)
        config._derive()
        assert config.N_IN == off + 1
    finally:
        monkeypatch.undo()
        config._derive()
    assert (config.N_IN, config.G) == saved


def test_sharing_conserves_energy_and_splits_equally():
    if not config.SHARE:
        pytest.skip("sharing off")

    def run(levels):
        w = _arena(4)
        w.energy[:4] = torch.tensor([2.0, 1.0, 1.0, 1.0], device=DEV)
        _set_give(w, levels)
        w.step()
        return w

    w, ctl = run([50.0, -50.0, -50.0, -50.0]), run([-50.0] * 4)  # only 0 gives
    before = ctl.energy[:4]  # same world and costs, nobody giving
    gave = float(w.gave[0])
    assert gave == pytest.approx(config.GIVE_RATE, abs=0.01)
    assert float(w.gave[1:4].sum()) < 0.03  # the others give almost nothing
    # Cellmates 1..3 each got an equal third of the gift.
    gain = w.energy[1:4] - before[1:4]
    assert float(gain.max() - gain.min()) < 1e-3
    assert float(gain.mean()) == pytest.approx(gave / 3, abs=1e-3)
    # The giver paid the gift plus overhead (relative to the control world).
    paid = float(before[0] - w.energy[0])
    assert paid == pytest.approx(gave * (1 + config.GIVE_COST), abs=1e-3)


def test_sharing_total_given_equals_received_plus_overhead(monkeypatch):
    if not config.SHARE:
        pytest.skip("sharing off")
    # Zero every other cost so the energy ledger is only sharing.
    for name in ("BASE_COST", "MOVE_COST", "THINK_COST", "PHERO_COST", "ATTACK_COST"):
        monkeypatch.setattr(world, name, 0.0)
    for name in ("SIZE_COST", "SPEED_COST", "SENSE_COST", "POISON_DRAIN"):
        monkeypatch.setattr(world, name, 0.0)
    w = _arena(5)
    w.energy[:5] = 1.0
    _set_give(w, [50.0, 50.0, -50.0, -50.0, 0.0])
    total = float(w.energy[:5].sum())
    w.step()
    given = float(w.gave.sum())
    assert given > 0
    # What leaves the group is exactly the overhead on what was given.
    lost = total - float(w.energy[:5].sum())
    assert lost == pytest.approx(config.GIVE_COST * given, abs=1e-4)


def test_lone_creature_gives_nothing_and_pays_nothing():
    if not config.SHARE:
        pytest.skip("sharing off")
    w = _arena(1)
    _set_give(w, [50.0])
    w.step()
    assert float(w.gave[0]) == 0.0
    w2 = _arena(1)
    _set_give(w2, [-50.0])
    w2.step()
    assert float(w.energy[0]) == pytest.approx(float(w2.energy[0]), abs=1e-6)


def test_giver_keeps_its_floor():
    if not config.SHARE:
        pytest.skip("sharing off")
    w = _arena(2)
    w.energy[:2] = torch.tensor([config.GIVE_FLOOR, 1.0], device=DEV)
    _set_give(w, [50.0, -50.0])
    w.step()
    assert float(w.gave[0]) == 0.0


def test_sharing_is_reproducible_per_seed():
    if not config.SHARE:
        pytest.skip("sharing off")

    def run():
        w = make_world(3)
        for _ in range(30):
            w.step()
        return w.energy.clone(), w.gave.clone()

    (e1, g1), (e2, g2) = run(), run()
    assert torch.equal(e1, e2) and torch.equal(g1, g2)


def test_newborns_start_with_zero_gave():
    if not config.SHARE:
        pytest.skip("sharing off")
    w = make_world()
    w.gave[:] = 0.7
    w.alive[:] = False
    w.alive[:2] = True
    w.energy[:2] = 2 * REPRO_E
    w.pos[:2] = 50.5
    w.pick[:] = 10.0
    w.reproduce()
    kids = (w.alive & (w.uid >= config.N_INIT)).nonzero().squeeze(1)
    assert len(kids) > 0
    assert (w.gave[kids] == 0).all()
