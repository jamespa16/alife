"""Inspector pieces: view maths, nearest-creature pick, and panel drawing."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest
import torch

from alife.config import DEV, INSPECT_W, N_HID, ZOOM_MAX, G, H, W
from alife.panel import Inspector, View, draw_panel, fetch, make_panel, pick_nearest
from alife.world import World


def make_world(seed=0):
    torch.manual_seed(seed)
    return World()


def test_pick_nearest_wraps_and_skips_dead_and_far():
    pos = torch.tensor([[10.0, 10.0], [W - 2.0, 5.0], [300.0, 300.0]], device=DEV)
    alive = torch.tensor([True, True, False], device=DEV)
    assert pick_nearest(pos, alive, 12, 11, 5) == 0
    assert pick_nearest(pos, alive, 1, 6, 5) == 1  # across the wrap
    assert pick_nearest(pos, alive, 300, 300, 5) is None  # only a dead one is there
    assert pick_nearest(pos, alive, 100, 100, 5) is None  # too far


def test_view_zoom_keeps_point_fixed_and_clamps():
    v = View()
    assert v.rect() == (0, 0, W, H)
    wx, wy = v.to_world(400, 300)
    v.zoom_at(400, 300, 4)
    assert v.zoom == 4
    assert v.to_world(400, 300) == pytest.approx((wx, wy), abs=2)
    v.zoom_at(0, 0, 1000)
    assert v.zoom == ZOOM_MAX
    v.pan(-(10**6), -(10**6))  # can never leave the map
    x, y, w, h = v.rect()
    assert 0 <= x and x + w <= W and 0 <= y and y + h <= H
    v.center_on(W / 2, H / 2)
    sx, sy = v.to_screen(W / 2, H / 2)
    assert (sx, sy) == pytest.approx((W / 2, H / 2), abs=ZOOM_MAX)
    v.reset()
    assert v.rect() == (0, 0, W, H)


def test_inspector_selects_tracks_and_notices_death():
    w = make_world()
    i = int(w.alive.nonzero()[0])
    x, y = (float(c) for c in w.pos[i])
    insp = Inspector()
    assert not insp.select(w, (x + 200) % W, (y + 200) % H, 1)
    assert insp.data is None
    assert insp.select(w, x, y, 1.0)
    d = insp.data
    assert d is not None
    assert d["uid"] == int(w.uid[insp.slot]) and d["alive"] and len(d["genome"]) == G
    assert len(d["h"]) == N_HID
    w.h[insp.slot] = 0.25
    insp.refresh(w)
    assert insp.data is not None and (insp.data["h"] == 0.25).all() and not insp.dead
    w.alive[insp.slot] = False
    w.h[insp.slot] = 0.5
    insp.refresh(w)
    assert insp.dead and (insp.data["h"] == 0.25).all()  # last data kept
    insp.clear()
    assert insp.data is None


def test_inspector_detects_reused_slot():
    w = make_world()
    insp = Inspector()
    i = int(w.alive.nonzero()[0])
    x, y = (float(c) for c in w.pos[i])
    insp.select(w, x, y, 1.0)
    w.uid[insp.slot] = 10**9  # a newborn took the slot
    insp.refresh(w)
    assert insp.dead


@pytest.mark.parametrize("dead", [False, True])
@pytest.mark.parametrize("with_font", [True, False])
def test_draw_panel_paints_and_handles_missing_font(dead, with_font):
    w = make_world()
    data = fetch(w, int(w.alive.nonzero()[0]))
    surf, font = make_panel()
    assert surf.get_size() == (INSPECT_W, H)
    draw_panel(surf, data, dead, font if with_font else None)
    colours = {
        tuple(surf.get_at((x, y)))[:3]
        for x in range(0, INSPECT_W, 3)
        for y in range(0, H, 3)
    }
    assert len(colours) > 10  # heatmaps and bars, not a flat fill
    pygame.quit()


def test_view_resized_window_letterboxes_and_maps_back():
    v = View(W * 2, H * 3)  # taller than the world's aspect: bars top and bottom
    ox, oy, dw, dh = v.dest()
    assert (ox, dw) == (0, W * 2) and dh == H * 2 and oy == H // 2
    assert v.to_world(ox, oy) == pytest.approx((0, 0))
    assert v.to_screen(W / 2, H / 2) == pytest.approx((W, H * 1.5))
    sx, sy = W / 2, H
    wx, wy = v.to_world(sx, sy)
    v.zoom_at(sx, sy, 3)
    assert v.to_world(sx, sy) == pytest.approx((wx, wy), abs=2)
    v.resize(W, H)
    assert v.dest() == (0, 0, W, H)
