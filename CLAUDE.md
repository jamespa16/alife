# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An artificial-life sandbox: creatures with evolving recurrent neural-net brains eat, move, split and die on a toroidal food grid. There is no fitness function; selection comes only from survival and reproduction. Everything runs as batched PyTorch tensor ops on the GPU (CUDA, then Apple MPS, then CPU, chosen in `config.py`), and pygame displays it.

## Commands

```sh
uv sync                # install deps, incl. dev group (ruff, ty)
uv run alife           # run the sim (entry point: alife.app:main)
python -m alife        # same thing, without uv
uv run ruff check      # lint (--fix to autofix)
uv run ruff format     # format
uv run ty check        # type-check
```

There's no test suite or CI.

Runtime keys: SPACE pauses, C switches the colour mode (lineage / loop count), and UP/DOWN doubles or halves the sim steps per frame (1–64). The window title shows t, population, mean loop count and steps/s. The world resets automatically when the population hits zero.

## Architecture

Module flow: `app` (pygame loop) → `world.World.step()` → `brain.think()`, with `render` turning a `World` into a frame. All tunable constants live in `config.py`.

**Fixed-size structure-of-arrays state.** Each per-creature attribute (`pos`, `heading`, `energy`, `age`, `genome`, `loops`, `h`, `color`) is a tensor of length `N_MAX`, and the `alive` bool mask marks which slots are in use. Dead slots still go through every computation; their effects are masked out with `alive_f`. Birth means writing into free (`~alive`) slots, and death just clears `alive`. To add a new per-creature attribute, you must:
1. allocate it in `World.reset()`
2. copy or mutate it from parent to child in `World.reproduce()`

**Genome ↔ brain layout coupling.** The genome is a flat vector of length `G`, defined in `config.py`. `brain.think()` slices it, in order, into `w_in [N_HID×N_IN]`, `w_rec [N_HID×N_HID]`, `bias [N_HID]` and `w_out [N_OUT×N_HID]`. If you change the brain shape, `G` and those slices must stay consistent. `N_IN` must also match the number of sensor columns built in `World.step()` (currently food×3, density×3 and own energy). `N_OUT` must match how `step()` reads the outputs (turn, speed).

**Loop count is a gene.** `loops[i]` (1..`K_MAX`) sets how many recurrent passes creature i's brain gets. Each pass costs `THINK_COST` energy, and the value mutates by ±1 with probability `LOOP_MUT_P`. For batching, `think()` always runs `K_MAX` iterations and uses `torch.where` to freeze `h` once a creature has used up its loops. Compute cost therefore doesn't depend on the loop gene; only the energy cost and the output do.

**Brains have memory across steps.** Each creature's hidden state `World.h [N_MAX, N_HID]` persists: `think(genome, loops, x, h)` starts from the previous step's `h` and returns `(out, new_h)`. Newborns start with `h = 0`.

**Step order** (`World.step`): bin creatures into grid cells → eat (food split among creatures sharing a cell, capped at `EAT_RATE`) → sense (box-blurred food and density, sampled at `SENSE_ANGLES` and `SENSE_DIST` ahead) → think → move → metabolism, ageing and death → reproduce (energy > `REPRO_E` splits it in half; the child gets a mutated genome, loop count and lineage colour) → food regrows toward the fertility map `fert`.

**Conventions and gotchas**
- The world wraps in both axes: positions use `% W` / `% H`, and `blur` uses circular padding.
- Grids (`food`, `fert`) are `[H, W]` and indexed `[y, x]`. `render` builds `[H, W, 3]`, then permutes to `[W, H, 3]` because `pygame.surfarray` expects that.
- `World` methods name the instance `s`, not `self`.
- Sim and render code run under `@torch.no_grad()`. Keep tensors on `DEV` and avoid host syncs (`.item()`, `int()`, `.cpu()`) inside `step()`; `stats()` is only called every 0.5s for this reason.
- `reproduce()` is limited by the number of free slots. When the world is full, a random subset of eligible parents reproduces.
