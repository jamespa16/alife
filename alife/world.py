"""Simulation state and the eat / sense / act / reproduce step."""

import math

import torch
import torch.nn.functional as F

from . import brain
from .config import (
    BASE_COST,
    DEV,
    EAT_RATE,
    FERT_DRIFT,
    FOOD_GROWTH,
    K_MAX,
    LOOP_MUT_P,
    MAX_AGE,
    MAX_SPEED,
    MAX_TURN,
    METEOR_R,
    MOVE_COST,
    MUT_STD,
    N_HID,
    N_INIT,
    N_MAX,
    N_OUT,
    PLAGUE_DRAIN,
    PLAGUE_RECOVER,
    PLAGUE_SPREAD,
    PLAGUE_START,
    PLAGUE_STEPS,
    REPRO_E,
    SEASON_AMP,
    SEASON_PERIOD,
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
        s.fert0 = F.interpolate(noise, size=(H, W), mode="bicubic")[0, 0].clamp(0.05, 1)
        s.sense_a = torch.tensor(SENSE_ANGLES, device=DEV)
        s.size = torch.tensor([W, H], device=DEV)
        t = torch.linspace(0, 1, K_MAX + 1, device=DEV)
        s.palette = torch.stack(
            [t, torch.full_like(t, 0.25), 1 - t], 1
        )  # blue=few loops, red=many
        s.reset()

    def update_fert(s):
        """Current fertility: the base map slid sideways, scaled by the season.

        Depends only on t, so it is deterministic and needs no host sync."""
        base = s.fert0
        shift = int(s.t * FERT_DRIFT)
        if shift:
            base = base.roll(shift, 1)
        season = 1 + SEASON_AMP * math.sin(TAU * s.t / SEASON_PERIOD)
        s.fert = (base * season).clamp(min=0.01)

    def reset(s):
        s.t = 0
        s.update_fert()
        s.food = s.fert.clone()
        s.sick = torch.zeros(N_MAX, dtype=torch.bool, device=DEV)
        s.plague_left = 0  # steps of outbreak remaining
        s.trail = None  # render's fading creature layer, allocated on first frame
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

    @torch.no_grad()
    def step(s):
        alive_f = s.alive.float()
        cx = s.pos[:, 0].long() % W
        cy = s.pos[:, 1].long() % H
        cell = cy * W + cx
        count = torch.zeros(H * W, device=DEV).scatter_add_(0, cell, alive_f)

        # Plague: sick creatures infect others in their cell, and may recover.
        if s.plague_left > 0:
            sick_n = torch.zeros(H * W, device=DEV).scatter_add_(
                0, cell, (s.sick & s.alive).float()
            )
            r = torch.rand(N_MAX, device=DEV)
            catch = (sick_n[cell] > 0) & (r < PLAGUE_SPREAD)
            s.sick = (s.sick | catch) & (torch.rand(N_MAX, device=DEV) > PLAGUE_RECOVER)
            s.plague_left -= 1
            if s.plague_left == 0:
                s.sick[:] = False

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
        s.energy -= PLAGUE_DRAIN * s.sick
        s.age += 1
        s.alive &= (s.energy > 0) & (s.age < MAX_AGE)

        s.reproduce()
        s.t += 1
        s.update_fert()
        s.food = torch.minimum(s.food + FOOD_GROWTH * s.fert, s.fert)

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
        s.sick[kids] = False

        s.genome[kids] = s.genome[parents] + torch.randn(n, G, device=DEV) * MUT_STD
        flip = (torch.rand(n, device=DEV) < LOOP_MUT_P).long()
        sign = torch.randint(0, 2, (n,), device=DEV) * 2 - 1
        s.loops[kids] = (s.loops[parents] + flip * sign).clamp(1, K_MAX)
        s.color[kids] = (s.color[parents] + torch.randn(n, 3, device=DEV) * 0.02).clamp(
            0.2, 1
        )
        s.alive[kids] = True

    def meteor(s, x=None, y=None, r=METEOR_R):
        """Wipe creatures and food in a disc (wrapping) at (x, y); random spot if omitted."""
        c = (
            torch.tensor([x, y], device=DEV)
            if x is not None
            else s.size * torch.rand(2, device=DEV)
        )
        half = s.size / 2
        d = (s.pos - c + half) % s.size - half
        s.alive &= (d**2).sum(1) > r * r
        gy = torch.arange(H, device=DEV)[:, None]
        gx = torch.arange(W, device=DEV)[None]
        dy = (gy - c[1] + H / 2) % H - H / 2
        dx = (gx - c[0] + W / 2) % W - W / 2
        s.food[dx**2 + dy**2 <= r * r] = 0

    def flood(s):
        """Refill all food to the current fertility."""
        s.food.copy_(s.fert)

    def plague(s):
        """Infect a random fraction of creatures; it spreads for PLAGUE_STEPS steps."""
        s.sick |= s.alive & (torch.rand(N_MAX, device=DEV) < PLAGUE_START)
        s.plague_left = PLAGUE_STEPS

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
