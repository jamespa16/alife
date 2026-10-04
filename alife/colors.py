"""Species colours, shared by the map (render) and the tree of life (tree).

species_rgb works on a tensor of species ids and species_rgb_py on one id; they
must agree (tests/test_tree.py checks it). Imported only after config overrides."""

import torch

from .config import DEV

GOLDEN = 0.618034  # hue step per species id, so neighbouring ids look different
_OFFSETS = torch.tensor([0.0, 2 / 3, 1 / 3], device=DEV)


def species_rgb(sp):
    """[n, 3] floats in 0..1 for a tensor of species ids (-1 = unassigned: grey)."""
    hue = (sp * GOLDEN) % 1
    rgb = ((hue[:, None] + _OFFSETS) % 1 * 6 - 3).abs() - 1
    return torch.where((sp < 0)[:, None], 0.5, rgb.clamp(0.15, 1))


def species_rgb_py(sp):
    """(r, g, b) floats in 0..1 for one species id, the same colour as species_rgb."""
    if sp < 0:
        return (0.5, 0.5, 0.5)
    hue = (sp * GOLDEN) % 1
    return tuple(
        min(max(abs((hue + o) % 1 * 6 - 3) - 1, 0.15), 1.0) for o in (0.0, 2 / 3, 1 / 3)
    )
