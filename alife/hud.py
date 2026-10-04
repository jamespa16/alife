"""Live HUD: population and trait graphs plus the loop-count histogram.

draw_hud is a pure function of the census history, so it can be tested headlessly.
It is imported only after config overrides, like the other sim modules."""

import pygame

from .config import HUD_H, HUD_W, K_MAX

BG = (0, 0, 0, 170)
GRID = (90, 90, 90)
WHITE = (235, 235, 235)
BLUE = (90, 150, 255)
RED = (255, 90, 90)
PAD = 6


def _line(surf, rect, ys, hi, color, lo=0.0):
    """Polyline of ys scaled from [lo, hi] into rect, oldest sample at the left."""
    n = len(ys)
    if n < 2 or hi <= lo:
        return
    pts = [
        (
            rect.x + rect.w * i / (n - 1),
            rect.bottom - rect.h * min(max((y - lo) / (hi - lo), 0.0), 1.0),
        )
        for i, y in enumerate(ys)
    ]
    pygame.draw.lines(surf, color, False, pts)


def _label(surf, font, text, pos, color=WHITE):
    if font is not None:
        surf.blit(font.render(text, True, color), pos)


def draw_hud(surf, rows, font=None):
    """Draw graphs from `rows` (a list of census() dicts, oldest first) onto `surf`.

    Panels: population (+ carnivores), mean loop count (+ mean diet, both scaled
    to their own ranges), and the latest loop-count histogram. Padding and label
    room scale with the surface height, so a bigger surface (make_hud) is just
    a bigger HUD."""
    surf.fill(BG)
    w, h = surf.get_size()
    k = h / HUD_H
    pad, lab = round(PAD * k), round(8 * k)  # lab: half the label strip
    panel_h = (h - 4 * pad) // 3
    area = [
        pygame.Rect(pad, pad + i * (panel_h + pad), w - 2 * pad, panel_h)
        for i in range(3)
    ]
    for r in area:
        pygame.draw.rect(surf, GRID, r, 1)
    if not rows:
        return
    last = rows[-1]
    top = area[0].inflate(-2, -2 * lab).move(0, lab)

    pop = [r["pop"] for r in rows]
    hi = max(max(pop), 1)
    _line(surf, top, pop, hi, WHITE)
    text = f"pop {last['pop']} (max {hi})"
    if "carnivores" in last:
        _line(surf, top, [r["carnivores"] for r in rows], hi, RED)
        text += f"  carn {last['carnivores']}"
    _label(surf, font, text, (area[0].x + 3, area[0].y + 1))

    mid = area[1].inflate(-2, -2 * lab).move(0, lab)
    _line(surf, mid, [r["mean_loops"] for r in rows], K_MAX, BLUE, lo=1.0)
    text = f"loops {last['mean_loops']:.2f}"
    if "mean_diet" in last:
        _line(surf, mid, [r["mean_diet"] for r in rows], 1.0, RED)
        text += f"  diet {last['mean_diet']:.2f}"
    if "mean_give" in last:
        text += f"  give {last['mean_give']:.3f}"
    _label(surf, font, text, (area[1].x + 3, area[1].y + 1), BLUE)

    bars = [last.get(f"loops_{k}", 0) for k in range(1, K_MAX + 1)]
    box = area[2].inflate(-2, -2 * lab).move(0, lab)
    top_bar = max(max(bars), 1)
    bw = box.w / K_MAX
    for k, n in enumerate(bars):
        bh = round(box.h * n / top_bar)
        pygame.draw.rect(
            surf, BLUE, (box.x + k * bw + 1, box.bottom - bh, max(bw - 2, 1), bh)
        )
    _label(surf, font, "loop-count histogram", (area[2].x + 3, area[2].y + 1))


def make_hud(scale=1.0):
    """(surface, font) for draw_hud, `scale` times the HUD_W x HUD_H size; the
    font is None if pygame.font is unavailable."""
    try:
        pygame.font.init()
        font = pygame.font.Font(None, round(16 * scale))
    except (pygame.error, NotImplementedError):
        font = None
    size = (round(HUD_W * scale), round(HUD_H * scale))
    return pygame.Surface(size, pygame.SRCALPHA), font
