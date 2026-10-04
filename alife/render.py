"""Turn a World into an RGB frame for display."""

import torch

from .config import DEV, PHERO_CH, PHERO_GLOW, PREDATION, TRAIL_DECAY, H, W

# Glow colour per pheromone channel (cycled if there are more channels).
PHERO_COLORS = torch.tensor(
    [[1.0, 0.2, 0.8], [0.2, 0.9, 1.0], [1.0, 0.9, 0.2]], device=DEV
)[torch.arange(PHERO_CH) % 3]


@torch.no_grad()
def render(world, mode):
    """mode 0: lineage colour, mode 1: loop count, mode 2: diet (blue herbivore, red carnivore). Returns a [W, H, 3] uint8 array."""
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
    idx = world.alive.nonzero().squeeze(1)
    cx = world.pos[idx, 0].long() % W
    cy = world.pos[idx, 1].long() % H
    if mode == 0:
        col = world.color[idx]
    elif mode == 1:
        col = world.palette[world.loops[idx]]
    else:
        d = world.diet[idx]
        col = torch.stack([d, torch.full_like(d, 0.25), 1 - d], 1)
    trail[cy, cx] = col
    img = (img + trail).clamp(max=1)
    return (img * 255).byte().permute(1, 0, 2).contiguous().cpu().numpy()
