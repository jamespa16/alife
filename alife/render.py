"""Turn a World into an RGB frame for display."""

import torch

from .config import DEV, H, W


@torch.no_grad()
def render(world, mode):
    """mode 0: lineage colour, mode 1: loop count. Returns a [W, H, 3] uint8 array."""
    img = torch.zeros(H, W, 3, device=DEV)
    img[..., 1] = world.food * 0.35
    idx = world.alive.nonzero().squeeze(1)
    cx = world.pos[idx, 0].long() % W
    cy = world.pos[idx, 1].long() % H
    img[cy, cx] = world.color[idx] if mode == 0 else world.palette[world.loops[idx]]
    return (img * 255).byte().permute(1, 0, 2).contiguous().cpu().numpy()
