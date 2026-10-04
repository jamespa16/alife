"""Creature inspector: zoom/pan view, nearest-creature pick, and the side panel.

draw_panel is a pure function of a fetched creature record, so it can be tested
headlessly. Imported only after config overrides, like the other sim modules."""

import numpy as np
import pygame
import torch

from .config import (
    BODY,
    DEV,
    INSPECT_W,
    N_HID,
    N_IN,
    N_OUT,
    PICK_RADIUS,
    PREDATION,
    SEX,
    SHARE,
    SPECIES,
    ZOOM_MAX,
    G,
    H,
    W,
)

BG = (0, 0, 0, 200)
WHITE = (235, 235, 235)
GREY = (140, 140, 140)
RED = (255, 90, 90)
BLUE = (90, 150, 255)
PAD = 6
SIZE = torch.tensor([W, H], device=DEV)


class View:
    """Which part of the full frame fills the window: zoom factor and centre.

    The window can be any size: the view is scaled to fit it, keeping the world's
    aspect ratio, and centred with black bars (`dest`)."""

    def __init__(s, sw=W, sh=H):
        s.zoom, s.cx, s.cy = 1.0, W / 2, H / 2
        s.resize(sw, sh)

    def resize(s, sw, sh):
        """Fit the view to a window of sw x sh pixels."""
        s.sw, s.sh = max(sw, 1), max(sh, 1)
        s.scale = min(s.sw / W, s.sh / H)  # window pixels per frame pixel at zoom 1

    def dest(s):
        """(x, y, w, h) in window pixels where the view is drawn."""
        w, h = round(W * s.scale), round(H * s.scale)
        return (s.sw - w) // 2, (s.sh - h) // 2, w, h

    def _virt(s, sx, sy):
        """Window pixels to unscaled W x H view pixels."""
        ox, oy, _, _ = s.dest()
        return (sx - ox) / s.scale, (sy - oy) / s.scale

    def _clamp(s):
        s.zoom = min(max(s.zoom, 1.0), float(ZOOM_MAX))
        w, h = W / s.zoom, H / s.zoom
        s.cx = min(max(s.cx, w / 2), W - w / 2)
        s.cy = min(max(s.cy, h / 2), H - h / 2)

    def rect(s):
        """(x, y, w, h) in frame pixels of the visible area."""
        s._clamp()
        w, h = W / s.zoom, H / s.zoom
        return round(s.cx - w / 2), round(s.cy - h / 2), round(w), round(h)

    def to_world(s, sx, sy):
        vx, vy = s._virt(sx, sy)
        x, y, w, h = s.rect()
        return x + vx * w / W, y + vy * h / H

    def to_screen(s, wx, wy):
        x, y, w, h = s.rect()
        ox, oy, _, _ = s.dest()
        return ox + (wx - x) * W / w * s.scale, oy + (wy - y) * H / h * s.scale

    def zoom_at(s, sx, sy, factor):
        """Zoom by `factor`, keeping the world point under screen (sx, sy) fixed."""
        wx, wy = s.to_world(sx, sy)
        vx, vy = s._virt(sx, sy)
        s.zoom *= factor
        s._clamp()
        s.cx = wx - vx / s.zoom + W / s.zoom / 2
        s.cy = wy - vy / s.zoom + H / s.zoom / 2
        s._clamp()

    def pan(s, dx, dy):
        """Drag the picture by (dx, dy) screen pixels."""
        s.cx -= dx / s.zoom / s.scale
        s.cy -= dy / s.zoom / s.scale
        s._clamp()

    def center_on(s, wx, wy):
        s.cx, s.cy = wx, wy
        s._clamp()

    def reset(s):
        s.zoom, s.cx, s.cy = 1.0, W / 2, H / 2


@torch.no_grad()
def pick_nearest(pos, alive, x, y, radius=PICK_RADIUS):
    """Slot of the living creature closest to world point (x, y) on the wrapping
    map, or None if none is within `radius`. One host sync: call on a click."""
    d = (pos - torch.tensor([x, y], device=DEV) + SIZE / 2) % SIZE - SIZE / 2
    d2 = (d**2).sum(1).masked_fill(~alive, float("inf"))
    i = d2.argmin()
    return int(i) if float(d2[i]) <= radius * radius else None


@torch.no_grad()
def fetch(world, i):
    """Everything the panel shows about slot i, in two small host copies."""
    w = world
    f = torch.cat(
        [w.genome[i], w.h[i], w.energy[i : i + 1], w.diet[i : i + 1], w.body[i]]
        + [w.pick[i : i + 1], w.pos[i], w.gave[i : i + 1]]
    )
    f = f.cpu().numpy()
    uid, pid, gen, sp, age, loops, alive = torch.stack(
        [w.uid[i], w.parent_id[i], w.gen[i], w.species[i], w.age[i], w.loops[i]]
        + [w.alive[i].long()]
    ).tolist()
    return {
        "slot": i,
        "genome": f[:G],
        "h": f[G : G + N_HID],
        "energy": float(f[G + N_HID]),
        "diet": float(f[G + N_HID + 1]),
        "body": f[G + N_HID + 2 : G + N_HID + 5],
        "pick": float(f[G + N_HID + 5]),
        "pos": (float(f[G + N_HID + 6]), float(f[G + N_HID + 7])),
        "gave": float(f[G + N_HID + 8]),
        "uid": uid,
        "parent_id": pid,
        "gen": gen,
        "species": sp,
        "age": age,
        "loops": loops,
        "alive": bool(alive),
    }


class Inspector:
    """The selected creature: its slot, uid and last fetched record."""

    def __init__(s):
        s.clear()

    def clear(s):
        s.slot = s.uid = s.data = None
        s.dead = False

    @property
    def active(s):
        return s.data is not None

    def select(s, world, x, y, radius=PICK_RADIUS):
        """Select the creature nearest world point (x, y); False if none is near."""
        i = pick_nearest(world.pos, world.alive, x, y, radius)
        if i is None:
            return False
        s.slot, s.dead = i, False
        s.data = fetch(world, i)
        s.uid = s.data["uid"]
        return True

    def refresh(s, world):
        """Re-fetch the selected creature (call once per frame). If it died or its
        slot was reused by a newborn, keep the last record and mark it dead."""
        if s.data is None or s.dead:
            return
        d = fetch(world, s.slot)
        if d["alive"] and d["uid"] == s.uid:
            s.data = d
        else:
            s.dead = True


def _heat(block, cell):
    """Diverging heatmap of a 2D array (red positive, blue negative) as a Surface."""
    m = float(np.abs(block).max()) or 1.0
    v = np.clip(block / m, -1, 1)
    rgb = np.zeros(v.shape + (3,), np.uint8)
    pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
    rgb[..., 0] = 30 + 225 * pos
    rgb[..., 2] = 30 + 225 * neg
    rgb[..., 1] = 30 + 60 * (pos + neg)
    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (v.shape[1] * cell, v.shape[0] * cell))


def _label(surf, font, text, pos, color=WHITE):
    if font is not None:
        surf.blit(font.render(text, True, color), pos)
    return pos[1] + 13


def draw_panel(surf, data, dead=False, font=None):
    """Draw creature record `data` (see fetch) onto `surf`: genome heatmaps in the
    brain's layout, the live hidden state h, and the scalar traits."""
    surf.fill(BG)
    inner = surf.get_width() - 2 * PAD
    cell = max(1, min(inner // max(N_IN, N_HID + 2), 12))
    g = data["genome"]
    a = N_HID * N_IN
    b = a + N_HID * N_HID
    c = b + N_HID
    y = PAD
    title = f"uid {data['uid']}" + ("  DEAD" if dead else "")
    y = _label(surf, font, title, (PAD, y), RED if dead else WHITE)
    y = _label(surf, font, "w_in (hidden x inputs)", (PAD, y), GREY)
    surf.blit(_heat(g[:a].reshape(N_HID, N_IN), cell), (PAD, y))
    y += N_HID * cell + 3
    y = _label(surf, font, "w_rec  |  bias", (PAD, y), GREY)
    surf.blit(_heat(g[a:b].reshape(N_HID, N_HID), cell), (PAD, y))
    surf.blit(_heat(g[b:c].reshape(N_HID, 1), cell), (PAD + (N_HID + 1) * cell, y))
    y += N_HID * cell + 3
    y = _label(surf, font, "w_out (outputs x hidden)", (PAD, y), GREY)
    surf.blit(_heat(g[c:].reshape(N_OUT, N_HID), cell), (PAD, y))
    y += N_OUT * cell + 3

    y = _label(surf, font, "hidden state h", (PAD, y), GREY)
    bw, mid, half = max(inner // N_HID, 1), y + 20, 20
    pygame.draw.line(surf, GREY, (PAD, mid), (PAD + bw * N_HID, mid))
    for k, v in enumerate(data["h"]):
        v = max(-1.0, min(1.0, float(v)))
        top = mid - round(half * max(v, 0))
        bot = mid + round(half * max(-v, 0))
        pygame.draw.rect(
            surf,
            RED if v > 0 else BLUE,
            (PAD + k * bw + 1, top, max(bw - 2, 1), bot - top + 1),
        )
    y = mid + half + 8

    lines = [
        f"gen {data['gen']}   parent {data['parent_id']}",
        f"loops {data['loops']}   age {data['age']}   energy {data['energy']:.2f}",
    ]
    if SPECIES:
        lines.insert(1, f"species {data['species']}")
    if PREDATION:
        lines.append(f"diet {data['diet']:.2f}  (0 herbivore, 1 carnivore)")
    if BODY:
        sz, sp, se = data["body"]
        lines.append(f"size {sz:.2f}  speed {sp:.2f}  sense {se:.2f}")
    if SEX:
        lines.append(f"mate pick {data['pick']:.2f}")
    if SHARE:
        lines.append(f"gave {data['gave']:.3f} energy last step")
    for text in lines:
        y = _label(surf, font, text, (PAD, y))


def make_panel():
    """(surface, font) for draw_panel; the font is None if pygame.font is unavailable."""
    try:
        pygame.font.init()
        font = pygame.font.Font(None, 17)
    except (pygame.error, NotImplementedError):
        font = None
    return pygame.Surface((INSPECT_W, H), pygame.SRCALPHA), font
