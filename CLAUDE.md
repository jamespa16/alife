# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An artificial-life sandbox: creatures with evolving recurrent neural-net brains eat, move, split and die on a toroidal food grid. There is no fitness function; selection comes only from survival and reproduction. Everything runs as batched PyTorch tensor ops on the GPU (CUDA, then Apple MPS, then CPU, chosen in `config.py`), and pygame displays it.

## Commands

```sh
uv sync                # install deps, incl. dev group (ruff, ty)
uv run alife           # run the sim (entry point: alife.app:main)
python -m alife        # same thing, without uv
uv run alife --headless --steps 20000 --seed 1 --log run.csv --set THINK_COST=0
                       # experiment run: no window, seeded, CSV every --log-every (100) steps
uv run ruff check      # lint (--fix to autofix)
uv run ruff format     # format
uv run ty check        # type-check
uv run pytest          # all tests (~10s; runs on the GPU device like the sim)
uv run pytest tests/test_brain.py::test_empty_batch   # a single test
```

There's no CI.

Testing notes:
- `tests/test_brain.py` checks `think()` against a per-creature reference built from the genome layout.
- `tests/test_cli.py` runs `python -m alife` in a subprocess, because `--set` must run before the sim modules are imported, and pytest has already imported them.
- Worlds share torch's global RNG. To compare two seeded worlds, run them one after the other, not interleaved.

Runtime keys: SPACE pauses, C switches the colour mode (lineage / loop count), and UP/DOWN doubles or halves the sim steps per frame (1–64), M drops a meteor at the mouse (`World.meteor`), F floods the map with food (`World.flood`), P starts a plague (`World.plague`). The window title shows t, population, mean loop count and steps/s. The world resets automatically when the population hits zero.

Each CSV row comes from `World.census()`: `t`, `pop`, means of loops/energy/age/food, and `loops_1..loops_K_MAX` histogram counts. The same seed gives an identical CSV on the same device. `--set NAME=VALUE` overrides any int/float constant in `config.py`, except the derived ones (`N_IN`, `G`, `TAU`). In windowed mode, extinction resets the world, so `t` restarts in the log.

## Architecture

**Constants are copied at import.** `world`, `brain` and `render` use `from .config import X`, so `config.override()` must run before they're imported. It raises if they already have been. That's why `app.main()` imports `World`, `render` and pygame inside the function. Keep `app.py` and `__init__.py` free of top-level imports of the sim modules.

Module flow: `app` (CLI, logging, headless and pygame loops) → `world.World.step()` → `brain.think()`, with `render` turning a `World` into a frame. All tunable constants live in `config.py`.

**Fixed-size structure-of-arrays state.** Each per-creature attribute (`pos`, `heading`, `energy`, `age`, `genome`, `loops`, `h`, `color`) is a tensor of length `N_MAX`, and the `alive` bool mask marks which slots are in use. Dead slots still go through every computation; their effects are masked out with `alive_f`. Birth means writing into free (`~alive`) slots, and death just clears `alive`. To add a new per-creature attribute, you must:
1. allocate it in `World.reset()`
2. copy or mutate it from parent to child in `World.reproduce()`

**Genome ↔ brain layout coupling.** The genome is a flat vector of length `G`, defined in `config.py`. `brain.think()` slices it, in order, into `w_in [N_HID×N_IN]`, `w_rec [N_HID×N_HID]`, `bias [N_HID]` and `w_out [N_OUT×N_HID]`. If you change the brain shape, `G` and those slices must stay consistent. `N_IN` must also match the number of sensor columns built in `World.step()` (currently food×3, density×3 and own energy). `N_OUT` must match how `step()` reads the outputs (turn, speed).

**Loop count is a gene.** `loops[i]` (1..`K_MAX`) sets how many recurrent passes creature i's brain gets. Each pass costs `THINK_COST` energy, and the value mutates by ±1 with probability `LOOP_MUT_P`. For batching, `think()` runs up to the largest loop count present and uses `torch.where` to freeze `h` once each creature has used up its loops.

**Performance.** `think()` dominates step time; the batched `bmm` over tiny per-creature matrices is the cost. `World.step()` therefore runs the brain only on living creatures, gathered by index, and dead slots get zero output. Elementwise multiply-sum was tried instead of `bmm`: it's 2× slower on MPS. Each step has a few unavoidable GPU syncs (`nonzero`, `loops.max()`), so don't add more in the hot path. To profile, time sections with `torch.mps.synchronize()` (or the CUDA equivalent) around them. Without the sync, GPU time shows up in whichever op syncs next.

**Brains have memory across steps.** Each creature's hidden state `World.h [N_MAX, N_HID]` persists: `think(genome, loops, x, h)` starts from the previous step's `h` and returns `(out, new_h)`. Newborns start with `h = 0`.

**Step order** (`World.step`): bin creatures into grid cells → eat (food split among creatures sharing a cell, capped at `EAT_RATE`) → sense (box-blurred food and density, sampled at `SENSE_ANGLES` and `SENSE_DIST` ahead) → think → move → metabolism, ageing and death → reproduce (energy > `REPRO_E` splits it in half; the child gets a mutated genome, loop count and lineage colour) → `t` advances and `fert` is recomputed (`update_fert`: the base map `fert0` slid sideways by `FERT_DRIFT` cells/step, scaled by `1 + SEASON_AMP*sin(TAU*t/SEASON_PERIOD)`; a function of `t` only) → food regrows toward `fert`. Plague runs at the start of the step while `plague_left > 0` (per-creature `sick` mask: spreads within a cell, drains `PLAGUE_DRAIN` energy). Newborns are healthy. `render` keeps a fading creature layer in `world.trail` (decay `TRAIL_DECAY` per frame, 0 disables).

**Conventions and gotchas**
- The world wraps in both axes: positions use `% W` / `% H`, and `blur` uses circular padding.
- Grids (`food`, `fert`) are `[H, W]` and indexed `[y, x]`. `render` builds `[H, W, 3]`, then permutes to `[W, H, 3]` because `pygame.surfarray` expects that.
- `World` methods name the instance `s`, not `self`.
- Sim and render code run under `@torch.no_grad()`. Keep tensors on `DEV` and avoid host syncs (`.item()`, `int()`, `.cpu()`) inside `step()`; `stats()` is only called every 0.5s for this reason.
- `reproduce()` is limited by the number of free slots. When the world is full, a random subset of eligible parents reproduces.
