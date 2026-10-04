"""HUD drawing is a pure function of the census history: test it headlessly."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from alife.config import HUD_H, HUD_W, K_MAX
from alife.hud import draw_hud, make_hud


def row(t, extra=True):
    r = {"t": t, "pop": 100 + t, "mean_loops": 1 + t % K_MAX, "mean_energy": 1.0}
    if extra:
        r.update(mean_diet=0.3, carnivores=t)
    r.update({f"loops_{k}": k * 5 for k in range(1, K_MAX + 1)})
    return r


def painted(surf):
    # Opaque-ish pixels beyond the translucent background mean something was drawn.
    return any(
        surf.get_at((x, y))[:3] != (0, 0, 0)
        for x in range(surf.get_width())
        for y in range(surf.get_height())
    )


@pytest.mark.parametrize("n", [0, 1, 2, 50])
@pytest.mark.parametrize("extra", [True, False])
def test_draw_hud_any_history_length(n, extra):
    surf, font = make_hud()
    draw_hud(surf, [row(t, extra) for t in range(n)], font)
    assert painted(surf)  # at least the panel frames


def test_draw_hud_extinct_and_flat_history():
    surf, font = make_hud()
    flat = [
        dict(
            row(t),
            pop=0,
            mean_loops=0.0,
            **{f"loops_{k}": 0 for k in range(1, K_MAX + 1)},
        )
        for t in range(5)
    ]
    draw_hud(surf, flat, font)
    draw_hud(surf, flat, None)  # works without a font too
    pygame.quit()


@pytest.mark.parametrize("scale", [0.75, 2.5])
def test_make_hud_scales(scale):
    surf, font = make_hud(scale)
    assert surf.get_size() == (round(HUD_W * scale), round(HUD_H * scale))
    draw_hud(surf, [row(t, True) for t in range(50)], font)
    assert painted(surf)
