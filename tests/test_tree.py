"""Tree of life: history from events, hue helper, layout, hit-testing, drawing."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest
import torch

from alife.colors import species_rgb, species_rgb_py
from alife.config import DEV, TREE_SPAN
from alife.render import render
from alife.tree import History, draw_tree, hit_test, layout, make_tree
from alife.world import World


def new(t, sp, parent, size=10):
    return {
        "t": t,
        "event": "new",
        "species": sp,
        "parent_species": parent,
        "size": size,
    }


def gone(t, sp, size=3):
    return {"t": t, "event": "extinct", "species": sp, "size": size}


def sample():
    h = History()
    h.add([new(0, 0, -1, 100), new(0, 1, -1, 20)])
    h.add([new(500, 2, 0, 30), gone(1000, 1)])
    h.add([new(1500, 3, 2, 5)])
    return h


def test_history_from_events():
    h = sample()
    assert h.now == 1500
    assert h.branches[2].parent == 0 and h.branches[3].parent == 2
    assert h.branches[1].t1 == 1000 and h.branches[1].size == 3
    assert h.branches[1].peak == 20
    assert h.branches[0].t1 is None
    h.add([gone(7, 99)])  # an extinction we never saw born is ignored
    assert 99 not in h.branches
    h.clear()
    assert not h.branches and h.now == 0


def test_history_seeds_from_world_and_tracks_sizes():
    torch.manual_seed(0)
    w = World()
    h = History()
    h.sync(w)  # as after --load: no events to learn from
    ids, sizes = w.sp_ids.tolist(), w.sp_size.tolist()
    assert ids and set(h.branches) == set(ids)
    assert all(h.branches[i].parent == -1 and h.branches[i].t1 is None for i in ids)
    assert [h.branches[i].size for i in ids] == sizes
    h2 = History()
    h2.add(w.take_events())  # a fresh world: the events carry everything
    assert set(h2.branches) == set(ids)
    h2.sync(w)
    assert [h2.branches[i].size for i in ids] == sizes
    h2.branches[ids[0]].t1 = 5  # extinct stays extinct; sync doesn't revive it
    h2.sync(w)
    assert h2.branches[ids[0]].t1 == 5


def test_species_hue_helper_matches_torch():
    ids = list(range(-1, 200))
    t = species_rgb(torch.tensor(ids, device=DEV)).cpu()
    for i, sp in enumerate(ids):
        assert species_rgb_py(sp) == pytest.approx(tuple(t[i].tolist()), abs=1e-4)


def test_layout_rows_links_and_window():
    h = sample()
    lay = layout(h, (400, 200))
    assert {s[0] for s in lay.segs} == {0, 1, 2, 3}
    assert {c for _, _, _, c in lay.links} == {2, 3}
    assert len({s[3] for s in lay.segs}) == 4  # one row each
    seg = {s[0]: s for s in lay.segs}
    assert seg[0][2] > seg[1][2] > seg[1][1]  # alive reaches now; 1 ends at 1000
    h.add([new(TREE_SPAN * 3, 4, 0, 50)])  # old extinct twigs scroll off
    assert 1 not in {s[0] for s in layout(h, (400, 200)).segs}


def test_layout_prunes_to_fit():
    h = History()
    h.add([new(0, 0, -1, 1000)])
    h.add([new(i, i, 0, i) for i in range(1, 300)])
    lay = layout(h, (400, 100))
    assert 0 < len(lay.segs) < 40
    assert 0 in {s[0] for s in lay.segs}


def test_hit_test():
    h = sample()
    size = (400, 200)
    for sp, x0, x1, y, _ in layout(h, size).segs:
        assert hit_test(h, size, ((x0 + x1) / 2, y)) == sp
    assert hit_test(h, size, (200, 0)) is None  # empty space
    assert hit_test(History(), size, (10, 10)) is None


@pytest.mark.parametrize("with_font", [True, False])
@pytest.mark.parametrize("n_events", [0, 1, 3])
def test_draw_tree_paints(with_font, n_events):
    surf, font = make_tree((400, 150))
    h = sample() if n_events == 3 else History()
    if n_events == 1:
        h.add([new(0, 0, -1)])
    draw_tree(surf, h, 2, font if with_font else None)
    colours = {surf.get_at((x, y))[:3] for x in range(0, 400, 2) for y in range(150)}
    assert len(colours) > (2 if n_events else 1)
    pygame.quit()


def test_render_highlight_is_render_only():
    torch.manual_seed(0)
    w = World()
    a = render(w, 3)
    state = w.pos.clone(), w.energy.clone()
    sp = int(w.species[w.alive.nonzero()[0]])
    w.trail = None
    b = render(w, 3, highlight=sp)
    assert (w.pos == state[0]).all() and (w.energy == state[1]).all()
    assert b.sum() < a.sum()
