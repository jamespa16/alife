# Roadmap

Ideas for making the sim more epic. The first roadmap (trails, seasons, disasters, pheromones, predation, sex, bodies, terrain, save/load, HUD, inspector, species) is done; see the git history. Effort is a rough guess: **S** is an evening, **M** a weekend, **L** longer.

Changes that touch the brain must keep `N_IN`, `N_OUT`, `G` and the slices in `brain.think()` consistent (see CLAUDE.md). Any new per-creature attribute needs allocating in `World.reset()` and inheriting in both `World.reproduce()` and `World.reproduce_sex()`. Put each new mechanic behind a `config.py` constant so `--set` can turn it off, and check that turning it off reproduces the previous run and CSV exactly (no extra RNG draws when off). Avoid new host syncs in `step()`.

## Tree of life view (M) ✅
Key T opens a live phylogeny. Time runs along x; each species is a horizontal branch from its `new` event to its `extinct` event, joined to its `parent_species` by a vertical stroke at the split. Line thickness follows species size, and colour matches species colour mode (C, mode 3), so the tree and the map can be read together. Click a branch to highlight that species' creatures on the map.
- Touches: `app.py` (feed the events that `SpeciesLog.drain` already takes into a host-side history), a new pure draw function in the style of `hud.draw_hud` / `panel.draw_panel` (testable with `SDL_VIDEODRIVER=dummy`), and the species hue helper from `render.py` pulled out so both use it.
- Watch: species present before the window opened (after `--load`) have no `new` event, so seed the history from `sp_ids` / `sp_size`. Windowed extinction resets the world and `t`, so start a fresh tree then. Long runs need branch pruning or a scrolling time axis.
- Doesn't change the sim, so seeded runs stay identical.

## Day/night cycle (M) ✅
A global light level swings with `DAY_PERIOD` (separate from seasons). Plants only grow by day, sensing range shrinks at night by `NIGHT_SENSE`, and the render darkens with a blue tint. Add a light sensor so brains can tell the time.
- Touches: a `light` value computed from `t` only (like `update_fert`), food regrowth, `reach` in the sense step, `N_IN` += 1 in `_derive()`, `render.py` shading.
- Watch: does `mean_loops` rise, since `h` can now carry "it will be dark soon"? Do carnivores become nocturnal, with prey bunching up at dusk?
- Depends on: nothing, but it's much richer with predation on.

## Altruism and food sharing (M) ✅
A new "give" output passes energy to the others in the same cell, split among them, reusing the `cell` / `scatter_add_` binning from eating and predation. Giving costs what is given plus a small `GIVE_COST` overhead. With sex, mate colour already works as a kin marker, so kin selection has something to latch onto.
- Touches: `N_OUT` += 1, sharing between predation and metabolism in `World.step()`, quantised to 1/1024 and summed into an empty grid like the other transfers, a new `census()` column for mean giving.
- Watch: do givers cluster by lineage colour? Do cheaters (who never give) invade, and does that cycle? Compare `SEX=0` (clonal kin) with `SEX=1`.
- Depends on: nothing; interacts with predation (sharing with your attacker is a bad idea).

## Bloom and action visuals (S–M) ✅
Rendering only. A soft bloom around creatures (blur a bright copy of the creature layer and add it back), short heading ticks when zoomed in, a red flash on creatures that attacked this step, and an animated expanding shockwave ring when a meteor lands.
- Touches: `render.py` (the existing `blur` gives the bloom cheaply on the GPU), a render-only attack mask and meteor animation state added to `World.TRANSIENT` so save files ignore them.
- Watch: frame time at full population; bloom should be one or two blurs of the `[H, W, 3]` layer, not per-creature work. Keep it all toggleable (e.g. key B) and leave `TRAIL_DECAY` behaviour unchanged.

## Suggested order

1. Bloom and action visuals: quick, and makes everything after it look better.
2. Tree of life: the data already exists, and it shows off the species tracking.
3. Day/night, then altruism: new selection pressures, best judged with the tree and HUD in place.
