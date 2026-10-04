"""Simulation state and the eat / sense / act / reproduce step."""

import math

import torch
import torch.nn.functional as F

from . import brain
from .config import (
    ATTACK_BIAS,
    ATTACK_COST,
    ATTACK_RATE,
    BASE_COST,
    BODY,
    BODY_INIT,
    BODY_MAX,
    BODY_MIN,
    BODY_MUT_STD,
    DEV,
    DIET_INIT,
    DIET_MUT_STD,
    EAT_RATE,
    FERT_DRIFT,
    FOOD_GROWTH,
    K_MAX,
    LOOP_MUT_P,
    MAX_AGE,
    MAX_SPEED,
    MAX_TURN,
    MEAT_DECAY,
    MEAT_DROP,
    MEAT_EAT_RATE,
    METEOR_R,
    MOVE_COST,
    MUT_STD,
    N_HID,
    N_INIT,
    N_MAX,
    N_OUT,
    PHERO_CH,
    PHERO_COST,
    PHERO_DECAY,
    PHERO_DEPOSIT,
    PHERO_DIFFUSE,
    PHERO_RADIUS,
    PICK_INIT,
    PICK_MUT_STD,
    PLAGUE_DRAIN,
    PLAGUE_RECOVER,
    PLAGUE_SPREAD,
    PLAGUE_START,
    PLAGUE_STEPS,
    PREDATION,
    REPRO_E,
    SEASON_AMP,
    SEASON_PERIOD,
    SENSE_ANGLES,
    SENSE_COST,
    SENSE_DIST,
    SEX,
    SEX_COST,
    SIZE_COST,
    SPEED_COST,
    TAU,
    THINK_COST,
    G,
    H,
    W,
)


def blur(g, r=4):
    """Box blur with wrap-around, so sensors see a neighbourhood, not one cell.

    Works on [H, W] or on a stack [C, H, W] (each channel blurred separately)."""
    flat = g.reshape(-1, 1, *g.shape[-2:])
    out = F.avg_pool2d(F.pad(flat, (r,) * 4, mode="circular"), 2 * r + 1, 1)
    return out.reshape(g.shape)


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
        s.phero = torch.zeros(PHERO_CH, H, W, device=DEV)  # scent, [channel, y, x]
        s.meat = torch.zeros(H, W, device=DEV)  # carrion, decays; eaten by diet
        s.aggr = torch.zeros(
            H, W, device=DEV
        )  # attack output summed per cell, last step
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
        # Diet gene: 0 = herbivore, 1 = carnivore. No RNG is drawn when predation is off.
        s.diet = (
            torch.rand(N_MAX, device=DEV) ** DIET_INIT
            if PREDATION
            else torch.zeros(N_MAX, device=DEV)
        )
        # Body genes (size, top speed, sensing range) as multipliers; all 1 and no
        # RNG drawn when BODY is off.
        s.body = (
            1 + (torch.rand(N_MAX, 3, device=DEV) - 0.5) * BODY_INIT
            if BODY
            else torch.ones(N_MAX, 3, device=DEV)
        )
        # Mate preference gene: partners must be closer in colour than both picks.
        # No RNG is drawn when sex is off.
        s.pick = (
            torch.rand(N_MAX, device=DEV) * PICK_INIT
            if SEX
            else torch.zeros(N_MAX, device=DEV)
        )
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
        if PREDATION:
            # Plants feed herbivores (1 - diet), carrion feeds carnivores (diet); an
            # eater only takes what it can digest. Meat bites are quantised and summed
            # into an empty grid first so the unordered scatter_add_ is reproducible.
            bite = (bite * (1 - s.diet) * 1024).floor() / 1024
            meat = s.meat.view(-1)
            mbite = (meat[cell] / count[cell].clamp(min=1)).clamp(max=MEAT_EAT_RATE)
            mbite = (mbite * s.diet * alive_f * 1024).floor() / 1024
            taken = torch.zeros(H * W, device=DEV).scatter_add_(0, cell, mbite)
            meat.sub_(taken).clamp_(min=0)
            eaten = torch.zeros(H * W, device=DEV).scatter_add_(0, cell, bite)
            food.sub_(eaten).clamp_(min=0)
            bite = bite + mbite
        else:
            food.scatter_add_(0, cell, -bite)
            food.clamp_(min=0)

        # Sense: food and crowding at three points ahead, plus own energy.
        food_s = blur(s.food)
        dens_s = blur(count.view(H, W))
        ang = s.heading[:, None] + s.sense_a[None]
        reach = SENSE_DIST * s.body[:, 2:3]  # sensing range gene
        sx = (s.pos[:, 0:1] + reach * ang.cos()).floor().long() % W
        sy = (s.pos[:, 1:2] + reach * ang.sin()).floor().long() % H
        # To add a sensor group: append [N, k] here and bump N_IN in config.
        sensors = [food_s[sy, sx], dens_s[sy, sx], (s.energy / REPRO_E)[:, None]]
        if PHERO_CH:  # scent per channel at each sense point -> [N, 3 * PHERO_CH]
            smell = s.phero[:, sy, sx].tanh()  # [C, N, 3]
            sensors.append(smell.permute(1, 0, 2).reshape(N_MAX, -1))
        if PREDATION:  # carrion and last step's aggression at each sense point
            sensors.append(blur(s.meat)[sy, sx].tanh())
            sensors.append(blur(s.aggr)[sy, sx].tanh())
        x = torch.cat(sensors, 1)

        # Act.
        # Only living creatures think: the brain dominates step time, and most
        # slots are usually empty. Dead slots get zero output (they're inert).
        idx = s.alive.nonzero().squeeze(1)
        out = torch.zeros(N_MAX, N_OUT, device=DEV)
        out[idx], s.h[idx] = brain.think(s.genome[idx], s.loops[idx], x[idx], s.h[idx])
        turn = out[:, 0].tanh() * MAX_TURN
        speed = out[:, 1].sigmoid() * MAX_SPEED * s.body[:, 1]
        # To add an output: it is column 2 + PHERO_CH + ...; bump N_OUT in config.
        s.heading = (s.heading + turn) % TAU
        s.pos[:, 0] = (s.pos[:, 0] + speed * s.heading.cos()) % W
        s.pos[:, 1] = (s.pos[:, 1] + speed * s.heading.sin()) % H

        # Predation: attackers take energy from the other creatures in their cell
        # (the cell they sensed in) and digest it by diet. Amounts are multiples of
        # 1/1024, so the unordered scatter_add_ sums exactly.
        atk = None
        if PREDATION:
            atk = (out[:, -1] - ATTACK_BIAS).sigmoid() * alive_f
            # Size is muscle: attack power is strength x size, and a big victim takes
            # proportionally less of it (sizes are all 1 when BODY is off).
            atk = (atk * s.body[:, 0] * 1024).round() / 1024
            a_cell = torch.zeros(H * W, device=DEV).scatter_add_(0, cell, atk)
            loss = ATTACK_RATE * (a_cell[cell] - atk) / s.body[:, 0] * alive_f
            loss = (loss.minimum(s.energy.clamp(min=0)) * 1024).floor() / 1024
            stolen = torch.zeros(H * W, device=DEV).scatter_add_(0, cell, loss)
            share = atk / a_cell[cell].clamp(min=1e-6)
            s.energy += stolen[cell] * share * s.diet - loss
            s.aggr = a_cell.view(H, W)

        # Pheromones: drop scent where the creature stood when it sensed.
        drop = None
        if PHERO_CH:
            drop = out[:, 2 : 2 + PHERO_CH].sigmoid().T * (PHERO_DEPOSIT * alive_f)
            # Fixed-point amounts add exactly in any order, so the GPU's
            # unordered scatter_add_ stays reproducible for a given seed.
            drop = (drop * 1024).round() / 1024
            # Sum into an empty grid first: adding to the decayed scent directly
            # would round differently depending on the order.
            new = torch.zeros(PHERO_CH, H * W, device=DEV)
            new.scatter_add_(1, cell.expand(PHERO_CH, -1), drop)
            s.phero += new.view(PHERO_CH, H, W)

        # Metabolism and death.
        s.energy += bite - (BASE_COST + MOVE_COST * speed + THINK_COST * s.loops)
        if drop is not None:
            s.energy -= PHERO_COST * drop.sum(0)
        if atk is not None:
            s.energy -= ATTACK_COST * atk
        s.energy -= PLAGUE_DRAIN * s.sick
        if BODY:  # bigger, faster, sharper-eyed bodies cost more to run
            b = s.body
            s.energy -= (
                SIZE_COST * b[:, 0] + SPEED_COST * b[:, 1] + SENSE_COST * b[:, 2]
            )
        s.age += 1
        was = s.alive
        s.alive = was & (s.energy > 0) & (s.age < MAX_AGE)
        if PREDATION:  # the dead leave what they had as carrion
            left = (MEAT_DROP * s.energy.clamp(min=0) * (was & ~s.alive) * 1024).floor()
            s.meat += (
                torch.zeros(H * W, device=DEV)
                .scatter_add_(0, cell, left / 1024)
                .view(H, W)
            )

        s.reproduce()
        s.t += 1
        s.update_fert()
        s.food = torch.minimum(s.food + FOOD_GROWTH * s.fert, s.fert)
        if PREDATION:
            s.meat *= MEAT_DECAY
        if PHERO_CH:  # scent spreads, then fades
            mixed = s.phero + PHERO_DIFFUSE * (blur(s.phero, PHERO_RADIUS) - s.phero)
            s.phero = mixed * PHERO_DECAY

    def pair_up(s, elig, k):
        """Pick up to k mating pairs among the creatures `elig` (slot indices).

        Shuffle, stable-sort by grid cell, then pair 1st+2nd, 3rd+4th... of each
        cell's run; a pair needs colour distance below both partners' `pick`. Pairs
        are taken in random order, so a full world gets a random subset. Returns
        (a, b, ok) of length k: ok marks real pairs, the rest is padding.
        No host syncs: invalid pairs are masked instead of compacted."""
        m = len(elig)
        e = elig[torch.randperm(m, device=DEV)]
        cell = (s.pos[e, 1].long() % H) * W + s.pos[e, 0].long() % W
        order = cell.sort(stable=True)[1]
        e, cell = e[order], cell[order]
        i = torch.arange(m, device=DEV)
        new_run = torch.ones(m, dtype=torch.bool, device=DEV)
        new_run[1:] = cell[1:] != cell[:-1]
        rank = i - torch.where(new_run, i, 0).cummax(0)[0]  # place within its cell
        nxt = e.roll(-1)
        same = torch.zeros(m, dtype=torch.bool, device=DEV)
        same[:-1] = cell[1:] == cell[:-1]
        close = (s.color[e] - s.color[nxt]).norm(dim=1)
        ok = (rank % 2 == 0) & same & (close < s.pick[e].minimum(s.pick[nxt]))
        pick = torch.where(ok, torch.rand(m, device=DEV), 2.0).argsort()[:k]
        return e[pick], nxt[pick], ok[pick]

    def reproduce_sex(s):
        elig = (s.alive & (s.energy > REPRO_E)).nonzero().squeeze(1)
        free = (~s.alive).nonzero().squeeze(1)
        n = min(len(elig) // 2, len(free))
        if n == 0:
            return
        a, b, ok = s.pair_up(elig, n)
        kids = free[:n]
        # Both parents pay; the child gets the payments. Index_add of zeros for
        # padding pairs, since those may repeat a real parent's slot.
        pay_a = SEX_COST * s.energy[a] * ok
        pay_b = SEX_COST * s.energy[b] * ok
        s.energy.index_add_(0, a, -pay_a).index_add_(0, b, -pay_b)
        s.energy[kids] = pay_a + pay_b
        s.pos[kids] = (s.pos[a] + torch.randn(n, 2, device=DEV) * 2) % s.size
        s.heading[kids] = torch.rand(n, device=DEV) * TAU
        s.age[kids] = 0
        s.h[kids] = 0
        s.sick[kids] = False
        # Uniform crossover per gene, then mutation; scalar genes mix or pick one.
        from_a = torch.rand(n, G, device=DEV) < 0.5
        child = torch.where(from_a, s.genome[a], s.genome[b])
        s.genome[kids] = child + torch.randn(n, G, device=DEV) * MUT_STD
        if PREDATION:
            mix = (s.diet[a] + s.diet[b]) / 2
            s.diet[kids] = (mix + torch.randn(n, device=DEV) * DIET_MUT_STD).clamp(0, 1)
        if BODY:
            mix = (s.body[a] + s.body[b]) / 2
            s.body[kids] = (mix + torch.randn(n, 3, device=DEV) * BODY_MUT_STD).clamp(
                BODY_MIN, BODY_MAX
            )
        mix = (s.pick[a] + s.pick[b]) / 2
        s.pick[kids] = (mix + torch.randn(n, device=DEV) * PICK_MUT_STD).clamp(min=0)
        flip = (torch.rand(n, device=DEV) < LOOP_MUT_P).long()
        sign = torch.randint(0, 2, (n,), device=DEV) * 2 - 1
        loops = torch.where(torch.rand(n, device=DEV) < 0.5, s.loops[a], s.loops[b])
        s.loops[kids] = (loops + flip * sign).clamp(1, K_MAX)
        mix = (s.color[a] + s.color[b]) / 2
        s.color[kids] = (mix + torch.randn(n, 3, device=DEV) * 0.02).clamp(0.2, 1)
        s.alive[kids] = ok

    def reproduce(s):
        if SEX:
            return s.reproduce_sex()
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
        if PREDATION:
            s.diet[kids] = (
                s.diet[parents] + torch.randn(n, device=DEV) * DIET_MUT_STD
            ).clamp(0, 1)

        s.genome[kids] = s.genome[parents] + torch.randn(n, G, device=DEV) * MUT_STD
        if BODY:
            s.body[kids] = (
                s.body[parents] + torch.randn(n, 3, device=DEV) * BODY_MUT_STD
            ).clamp(BODY_MIN, BODY_MAX)
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
        if PREDATION:
            row["mean_diet"] = float(s.diet[a].mean().nan_to_num())
            row["carnivores"] = int((s.diet[a] > 0.5).sum())
            row["meat"] = float(s.meat.sum())
        if SEX:
            row["mean_pick"] = float(s.pick[a].mean().nan_to_num())
        if BODY:
            for i, name in enumerate(("size", "speed", "sense")):
                row[f"mean_{name}"] = float(s.body[a, i].mean().nan_to_num())
        row.update({f"loops_{k}": n for k, n in enumerate(hist.tolist(), 1)})
        return row
