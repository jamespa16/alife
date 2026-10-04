"""Turn a World into an RGB frame for display."""

import torch

from .colors import species_rgb
from .config import (
    BIG_SIZE,
    BIG_SIZE2,
    BLOOM,
    BLOOM_RADIUS,
    BODY,
    DAYNIGHT,
    DEV,
    FLASH_MIN,
    PHERO_CH,
    PHERO_GLOW,
    PREDATION,
    TERRAIN,
    TICK_MAX,
    TRAIL_DECAY,
    H,
    W,
)
from .world import blur, light_at

# Glow colour per pheromone channel (cycled if there are more channels).
PHERO_COLORS = torch.tensor(
    [[1.0, 0.2, 0.8], [0.2, 0.9, 1.0], [1.0, 0.9, 0.2]], device=DEV
)[torch.arange(PHERO_CH) % 3]


PURPLE = torch.tensor([0.35, 0.0, 0.5], device=DEV)
FLASH = torch.tensor([1.0, 0.1, 0.1], device=DEV)
NIGHT_TINT = torch.tensor([0.2, 0.35, 1.0], device=DEV)
WALL_COLOR = torch.tensor([0.45, 0.45, 0.5], device=DEV)


@torch.no_grad()
def render(world, mode, fx=True, highlight=None):
    """mode 0: lineage colour, mode 1: loop count, mode 2: diet (blue herbivore, red carnivore), mode 3: species. fx adds attack flashes and bloom. `highlight` (a species id) dims every other creature.
    Returns a [W, H, 3] uint8 array."""
    # Creatures leave glowing trails: the creature layer fades instead of clearing.
    if world.trail is None:
        world.trail = torch.zeros(H, W, 3, device=DEV)
    trail = world.trail.mul_(TRAIL_DECAY)
    img = torch.zeros(H, W, 3, device=DEV)
    img[..., 1] = world.food * 0.35
    if PHERO_CH:  # each scent channel as a coloured glow, [C, H, W] -> [H, W, 3]
        glow = (world.phero * PHERO_GLOW).tanh()
        img += torch.einsum("chw,cr->hwr", glow, PHERO_COLORS) * 0.6
    if PREDATION:  # carrion as a dull red glow
        img[..., 0] += (world.meat * 0.5).tanh() * 0.5
    if TERRAIN:  # poison as a purple stain, walls as solid grey
        img += world.poison[..., None] * PURPLE
        img = torch.where(world.wall[..., None], WALL_COLOR, img)
    idx = world.alive.nonzero().squeeze(1)
    cx = world.pos[idx, 0].long() % W
    cy = world.pos[idx, 1].long() % H
    if mode == 0:
        col = world.color[idx]
    elif mode == 1:
        col = world.palette[world.loops[idx]]
    elif mode == 3:  # species id -> hue by golden-ratio steps (-1 = unassigned: grey)
        col = species_rgb(world.species[idx])
    else:
        d = world.diet[idx]
        col = torch.stack([d, torch.full_like(d, 0.25), 1 - d], 1)
    if fx and PREDATION and world.atk_vis is not None:  # attackers flash red
        a = world.atk_vis[idx]
        a = ((a - FLASH_MIN) / (1 - FLASH_MIN)).clamp(0, 1)[:, None]
        col = col + (FLASH - col) * a
    if highlight is not None:  # render-only: the chosen species bright, the rest dim
        mine = (world.species[idx] == highlight)[:, None]
        col = torch.where(mine, col * 0.5 + 0.5, col * 0.2)
    trail[cy, cx] = col
    if BODY:  # big bodies cover more pixels: 2x2 above BIG_SIZE, 3x3 above BIG_SIZE2
        size = world.body[idx, 0]
        for lo, hi, thr in ((0, 2, BIG_SIZE), (-1, 2, BIG_SIZE2)):
            big = size > thr
            for dy in range(lo, hi):
                for dx in range(lo, hi):
                    trail[(cy[big] + dy) % H, (cx[big] + dx) % W] = col[big]
    if DAYNIGHT:  # night dims and blues the world, but creatures stay visible
        dark = 1 - light_at(world.t)
        img = img * (1 - 0.7 * dark) + NIGHT_TINT * dark * 0.12
    img = img + trail
    if fx and BLOOM:  # soft glow: a narrow and a wide blur of the creature layer
        lay = trail.permute(2, 0, 1)
        glow = blur(lay, BLOOM_RADIUS) + blur(lay, 3 * BLOOM_RADIUS)
        img += (BLOOM * glow).permute(1, 2, 0)
    img = img.clamp(max=1)
    return (img * 255).byte().permute(1, 0, 2).contiguous().cpu().numpy()


@torch.no_grad()
def heading_ticks(world, rect):
    """Creatures inside rect (x, y, w, h in frame pixels) as a list of (x, y, heading)
    Python floats (pygame.draw rejects numpy float32), at most TICK_MAX of them: the
    only host copy of the zoomed view."""
    x, y, w, h = rect
    p = world.pos
    m = world.alive & (p[:, 0] >= x) & (p[:, 0] < x + w) & (p[:, 1] >= y)
    idx = (m & (p[:, 1] < y + h)).nonzero().squeeze(1)[:TICK_MAX]
    return torch.cat([p[idx], world.heading[idx, None]], 1).tolist()
