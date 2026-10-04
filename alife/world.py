"""Simulation state and the eat / sense / act / reproduce step."""

import torch
import torch.nn.functional as F

from . import brain
from .config import (
    BASE_COST,
    DEV,
    EAT_RATE,
    FOOD_GROWTH,
    K_MAX,
    LOOP_MUT_P,
    MAX_AGE,
    MAX_SPEED,
    MAX_TURN,
    MOVE_COST,
    MUT_STD,
    N_HID,
    N_INIT,
    N_MAX,
    N_OUT,
    REPRO_E,
    SENSE_ANGLES,
    SENSE_DIST,
    TAU,
    THINK_COST,
    G,
    H,
    W,
)


def blur(g, r=4):
    """Box blur with wrap-around, so sensors see a neighbourhood, not one cell."""
    return F.avg_pool2d(F.pad(g[None, None], (r,) * 4, mode="circular"), 2 * r + 1, 1)[
        0, 0
    ]


class World:
    def __init__(s):
        # Smooth fertility map: rich and poor regions.
        noise = torch.rand(1, 1, 8, 8, device=DEV)
        s.fert = F.interpolate(noise, size=(H, W), mode="bicubic")[0, 0].clamp(0.05, 1)
        s.sense_a = torch.tensor(SENSE_ANGLES, device=DEV)
        s.size = torch.tensor([W, H], device=DEV)
        t = torch.linspace(0, 1, K_MAX + 1, device=DEV)
        s.palette = torch.stack(
            [t, torch.full_like(t, 0.25), 1 - t], 1
        )  # blue=few loops, red=many
        s.reset()

    def reset(s):
        s.food = s.fert.clone()
        s.pos = torch.rand(N_MAX, 2, device=DEV) * s.size
        s.heading = torch.rand(N_MAX, device=DEV) * TAU
        s.energy = torch.ones(N_MAX, device=DEV)
        s.age = torch.zeros(N_MAX, dtype=torch.long, device=DEV)
        s.alive = torch.zeros(N_MAX, dtype=torch.bool, device=DEV)
        s.alive[:N_INIT] = True
        s.genome = torch.randn(N_MAX, G, device=DEV) * 0.5
        s.loops = torch.randint(1, 5, (N_MAX,), device=DEV)
        s.h = torch.zeros(N_MAX, N_HID, device=DEV)  # brain memory, kept across steps
        s.color = (
            torch.rand(N_MAX, 3, device=DEV) * 0.7 + 0.3
        )  # lineage colour, drifts on mutation
        s.t = 0

    @torch.no_grad()
    def step(s):
        alive_f = s.alive.float()
        cx = s.pos[:, 0].long() % W
        cy = s.pos[:, 1].long() % H
        cell = cy * W + cx
        count = torch.zeros(H * W, device=DEV).scatter_add_(0, cell, alive_f)

        # Eat: creatures sharing a cell split what is there.
        food = s.food.view(-1)
        bite = (food[cell] / count[cell].clamp(min=1)).clamp(max=EAT_RATE) * alive_f
        food.scatter_add_(0, cell, -bite)
        food.clamp_(min=0)

        # Sense: food and crowding at three points ahead, plus own energy.
        food_s = blur(s.food)
        dens_s = blur(count.view(H, W))
        ang = s.heading[:, None] + s.sense_a[None]
        sx = (s.pos[:, 0:1] + SENSE_DIST * ang.cos()).floor().long() % W
        sy = (s.pos[:, 1:2] + SENSE_DIST * ang.sin()).floor().long() % H
        x = torch.cat(
            [food_s[sy, sx], dens_s[sy, sx], (s.energy / REPRO_E)[:, None]], 1
        )

        # Act.
        # Only living creatures think: the brain dominates step time, and most
        # slots are usually empty. Dead slots get zero output (they're inert).
        idx = s.alive.nonzero().squeeze(1)
        out = torch.zeros(N_MAX, N_OUT, device=DEV)
        out[idx], s.h[idx] = brain.think(s.genome[idx], s.loops[idx], x[idx], s.h[idx])
        turn = out[:, 0].tanh() * MAX_TURN
        speed = out[:, 1].sigmoid() * MAX_SPEED
        s.heading = (s.heading + turn) % TAU
        s.pos[:, 0] = (s.pos[:, 0] + speed * s.heading.cos()) % W
        s.pos[:, 1] = (s.pos[:, 1] + speed * s.heading.sin()) % H

        # Metabolism and death.
        s.energy += bite - (BASE_COST + MOVE_COST * speed + THINK_COST * s.loops)
        s.age += 1
        s.alive &= (s.energy > 0) & (s.age < MAX_AGE)

        s.reproduce()
        s.food = torch.minimum(s.food + FOOD_GROWTH * s.fert, s.fert)
        s.t += 1

    def reproduce(s):
        parents = (s.alive & (s.energy > REPRO_E)).nonzero().squeeze(1)
        free = (~s.alive).nonzero().squeeze(1)
        n = min(len(parents), len(free))
        if n == 0:
            return
        parents = parents[torch.randperm(len(parents), device=DEV)[:n]]
        kids = free[:n]

        s.energy[parents] *= 0.5
        s.energy[kids] = s.energy[parents]
        s.pos[kids] = (s.pos[parents] + torch.randn(n, 2, device=DEV) * 2) % s.size
        s.heading[kids] = torch.rand(n, device=DEV) * TAU
        s.age[kids] = 0
        s.h[kids] = 0  # newborns start with a blank memory

        s.genome[kids] = s.genome[parents] + torch.randn(n, G, device=DEV) * MUT_STD
        flip = (torch.rand(n, device=DEV) < LOOP_MUT_P).long()
        sign = torch.randint(0, 2, (n,), device=DEV) * 2 - 1
        s.loops[kids] = (s.loops[parents] + flip * sign).clamp(1, K_MAX)
        s.color[kids] = (s.color[parents] + torch.randn(n, 3, device=DEV) * 0.02).clamp(
            0.2, 1
        )
        s.alive[kids] = True

    def stats(s):
        pop = int(s.alive.sum())
        loops = float(s.loops[s.alive].float().mean()) if pop else 0.0
        return pop, loops

    def census(s):
        """Population summary for logging: means plus a loop-count histogram."""
        a = s.alive
        hist = torch.bincount(s.loops[a], minlength=K_MAX + 1)[1:]
        means = torch.stack(
            [
                s.loops[a].float().mean(),
                s.energy[a].mean(),
                s.age[a].float().mean(),
                s.food.mean(),
            ]
        ).nan_to_num()  # empty population -> 0
        row = {"t": s.t, "pop": int(hist.sum())}
        row.update(
            zip(("mean_loops", "mean_energy", "mean_age", "food"), means.tolist())
        )
        row.update({f"loops_{k}": n for k, n in enumerate(hist.tolist(), 1)})
        return row
