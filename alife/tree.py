"""Tree of life: a live species phylogeny built from the species events.

History is host-side bookkeeping fed by World.take_events() (plus one seeding
from the world's known species, for a loaded run). layout, hit_test and draw_tree
are pure functions of a History, so they can be tested headlessly. Imported only
after config overrides, like the other sim modules."""

import math
from dataclasses import dataclass

import pygame

from .colors import species_rgb_py
from .config import TREE_SPAN

BG = (0, 0, 0, 200)
WHITE = (235, 235, 235)
GREY = (140, 140, 140)
PAD = 6
LABEL_H = 14  # the title strip above the branches
MIN_ROW = 3  # pixels per branch row; more species than fit are pruned
MAX_THICK = 7


@dataclass
class Branch:
    """One species: when it appeared and went extinct (None = alive), and its size."""

    parent: int
    t0: int
    t1: int | None = None
    size: int = 0
    peak: int = 0

    def grow(s, size):
        s.size = size
        s.peak = max(s.peak, size)


class History:
    """Species branches by id, built from events; `now` is the latest known t."""

    def __init__(s):
        s.clear()

    def clear(s):
        """Start a fresh tree (the world was reset, so t restarted)."""
        s.branches = {}
        s.now = 0

    def add(s, events):
        """Apply species events (dicts from World.take_events, oldest first)."""
        for e in events:
            sp, t = e["species"], e["t"]
            s.now = max(s.now, t)
            if e["event"] == "new":
                b = s.branches.setdefault(sp, Branch(e["parent_species"], t))
                b.grow(e["size"])
            elif sp in s.branches:
                b = s.branches[sp]
                b.t1 = t
                b.grow(e["size"])

    def sync(s, world):
        """Seed species the history never saw born (a loaded run) and refresh the
        sizes of the living ones from the last species pass. Two small host
        copies: call it about twice a second, not per step."""
        s.now = max(s.now, world.t)
        for sp, n in zip(world.sp_ids.tolist(), world.sp_size.tolist()):
            b = s.branches.setdefault(sp, Branch(-1, world.t))
            if b.t1 is None:
                b.grow(n)


@dataclass
class Layout:
    """Where each branch goes, in pixels of the tree's surface."""

    lo: int  # t at the left edge
    hi: int  # t at the right edge
    segs: list  # (species, x0, x1, y, thickness)
    links: list  # (parent y, child y, x, species): the vertical split strokes


def _x(t, lo, hi, x0, x1):
    return x0 + (x1 - x0) * (t - lo) / max(hi - lo, 1)


def _kept(hist, lo, cap):
    """Species ids to show: those alive in [lo, now], living and biggest first,
    each with its ancestors that are also in the window, at most `cap` (the
    ancestors may overshoot it a little)."""
    bs = hist.branches
    seen = [sp for sp, b in bs.items() if b.t1 is None or b.t1 >= lo]
    seen.sort(key=lambda sp: (bs[sp].t1 is not None, -bs[sp].peak, sp))
    kept = set()
    for sp in seen:
        chain, p = [], sp
        while p in bs and p not in kept and (bs[p].t1 is None or bs[p].t1 >= lo):
            chain.append(p)
            p = bs[p].parent
        if len(kept) + len(chain) > cap and kept:
            continue
        kept.update(chain)
    return kept


def layout(hist, size):
    """Place the branches inside a surface of `size` (w, h): time along x over the
    last TREE_SPAN steps, one row per species in depth-first order so children sit
    near their parents, thickness by size."""
    w, h = size
    hi = max(hist.now, 1)
    lo = max(hi - TREE_SPAN, 0)
    area_x0, area_x1 = PAD, w - PAD
    top, bottom = PAD + LABEL_H, h - PAD
    rows_fit = max((bottom - top) // MIN_ROW, 1)
    kept = _kept(hist, lo, rows_fit)
    bs = hist.branches
    kids = {}
    for sp in sorted(kept, key=lambda i: (bs[i].t0, i)):
        p = bs[sp].parent
        kids.setdefault(p if p in kept else -1, []).append(sp)
    order, stack = [], kids.get(-1, [])[::-1]
    while stack:
        sp = stack.pop()
        order.append(sp)
        stack.extend(kids.get(sp, [])[::-1])
    if not order:
        return Layout(lo, hi, [], [])
    row = (bottom - top) / len(order)
    ys = {sp: top + row * (i + 0.5) for i, sp in enumerate(order)}
    big = max(max(bs[sp].size for sp in order), 1)
    segs, links = [], []
    for sp in order:
        b = bs[sp]
        x0 = _x(max(b.t0, lo), lo, hi, area_x0, area_x1)
        x1 = _x(hi if b.t1 is None else b.t1, lo, hi, area_x0, area_x1)
        thick = 1 + round((MAX_THICK - 1) * math.sqrt(b.size / big))
        thick = max(1, min(thick, round(row)))
        segs.append((sp, x0, max(x1, x0 + 1), ys[sp], thick))
        if b.parent in kept and b.t0 >= lo:
            links.append((ys[b.parent], ys[sp], x0, sp))
    return Layout(lo, hi, segs, links)


def hit_test(hist, size, pos, slop=3):
    """Species id of the branch under `pos` (surface pixels), or None."""
    x, y = pos
    best, best_d = None, slop + 1
    for sp, x0, x1, sy, thick in layout(hist, size).segs:
        d = abs(y - sy) - thick / 2
        if x0 - slop <= x <= x1 + slop and d < best_d:
            best, best_d = sp, max(d, 0)
    return best


def _rgb(sp):
    return tuple(round(255 * c) for c in species_rgb_py(sp))


def draw_tree(surf, hist, selected=None, font=None):
    """Draw the phylogeny of `hist` onto `surf`: each species a horizontal branch
    from its birth to its extinction (or now), joined to its parent's by a vertical
    stroke, thick for big species. `selected` (a species id) is outlined."""
    surf.fill(BG)
    pygame.draw.rect(surf, GREY, surf.get_rect(), 1)
    lay = layout(hist, surf.get_size())
    for py, cy, x, sp in lay.links:
        pygame.draw.line(surf, GREY, (x, py), (x, cy))
    bs = hist.branches
    for sp, x0, x1, y, thick in lay.segs:
        r = pygame.Rect(round(x0), round(y - thick / 2), max(round(x1 - x0), 1), thick)
        pygame.draw.rect(surf, _rgb(sp), r)
        if sp == selected:
            pygame.draw.rect(surf, WHITE, r.inflate(2, 2), 1)
    if font is not None:
        alive = sum(b.t1 is None for b in bs.values())
        text = f"tree of life  t {lay.lo}..{lay.hi}  {alive} alive of {len(bs)}"
        if selected in bs:
            b = bs[selected]
            end = "now" if b.t1 is None else b.t1
            text += f"   species {selected}  size {b.size}  t {b.t0}..{end}"
        surf.blit(font.render(text, True, WHITE), (PAD, PAD - 2))


def make_tree(size):
    """(surface, font) for draw_tree; the font is None if pygame.font is unavailable."""
    try:
        pygame.font.init()
        font = pygame.font.Font(None, 16)
    except (pygame.error, NotImplementedError):
        font = None
    return pygame.Surface(size, pygame.SRCALPHA), font
