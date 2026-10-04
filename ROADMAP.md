# Roadmap

Ideas for making the sim more epic, grouped into phases. Earlier phases are cheap and unlock later ones. Effort is a rough guess: **S** is an evening, **M** a weekend, **L** longer.

Changes that touch the brain must keep `N_IN`, `N_OUT`, `G` and the slices in `brain.think()` consistent (see CLAUDE.md). Any new per-creature attribute needs allocating in `World.reset()` and inheriting in `World.reproduce()`. Put each new mechanic behind a `config.py` constant so `--set` can turn it off for headless comparison runs.

## Phase 1: Quick wins

### Glowing trails (S) ✅
Fade the previous frame instead of clearing it, use additive blending, and optionally draw headings.
- Touches: `render.py` (keep a persistent frame buffer, e.g. `img = img * DECAY` each frame).

### Seasons and drifting land (S) ✅
`fert` is fixed forever at the moment. Rotate or drift it, swing it with a seasonal sine wave, and add dry spells that dim whole regions. When the world changes over time, the memory in `h` becomes worth having.
- Touches: `World.step()` food regrowth; new constants `SEASON_PERIOD`, `SEASON_AMP`, `FERT_DRIFT`.
- Watch: in the CSV, does `mean_loops` rise compared with a static world?

### Disasters on a key press (S) ✅
A god game: M drops a meteor (wipes a disc of creatures and food), F floods the map with food, P adds a plague (energy drain that spreads through crowded cells).
- Touches: `app.py` key handling, plus a few `World` methods.

## Phase 2: Creatures that affect each other

### Pheromones (M) ✅
Add 1–3 outputs that drop scent onto a grid that spreads (the existing `blur`) and fades each step. Sensors read it at `SENSE_ANGLES`. It's all grid ops, so it adds no extra GPU syncs. Expect trails, swarms, and maybe a "danger" signal once predators exist.
- Touches: `N_OUT` += channels, `N_IN` += 3 × channels, `World.step()` sense and act, `render.py` overlay showing each channel as a coloured glow.

### Predators and prey (M–L) ✅
The biggest single change. Add an "attack" output: a creature with high attack takes energy from others in its cell, reusing the `cell` / `scatter_add_` binning that eating already uses. Add a diet gene that trades off plant digestion against meat digestion. Dead creatures drop their energy onto a `meat` grid that scavengers can sense.
- Touches: `N_OUT` += 1, a new `diet` attribute, a new `meat` grid, sensor columns for meat and for the aggression of nearby creatures.
- Watch: boom-and-bust cycles, herds running from hunters, and whether more loops now pay for their `THINK_COST`.
- Depends on: nothing, but it gets much better with pheromones.

### Sexual reproduction (M) ✅
Splitting needs a partner in the same cell, and the child's genome mixes both parents. A "mate preference" gene that compares colour or genome should let species form and split apart.
- Touches: `World.reproduce()` (finding a partner per cell is the tricky part to batch).

## Phase 3: Bodies and terrain

### Body-shape genes (M) ✅
Size, top speed, sensing range and number of sensors become genes, each with an energy cost. Bigger creatures beat smaller ones in fights. Expect body types to differ by region. Draw creatures at different sizes.
- Touches: new per-creature attributes, metabolism costs, `render.py`.
- Depends on: predators, for size to matter in fights. A variable number of sensors is hard to batch, so a cheaper version is to mask sensors off.

### Terrain (M) ✅
Walls or water cells that block movement and sensing, and poison zones. Mazes reward creatures that plan.
- Touches: a new `terrain` grid, movement collision in `World.step()`, `render.py`.

## Phase 4: Tools for watching and experimenting

### Save and load worlds (S–M) ✅
Save every tensor to disk, reload it, and fork a run from any point. That makes "what if I add predators to *this* world" experiments possible.
- Touches: `World.save()` / `World.load()`, CLI flags `--save` / `--load`.

### Live HUD (M) ✅
Graphs of population and the loop-count histogram over time, drawn in the window. The data already comes from `census()`.
- Touches: `app.py` / `render.py`. Sample every 0.5s, the same way `stats()` is called now, to avoid GPU syncs.

### Inspect a creature (M) ✅
Click one to see its genome as a heatmap, its hidden state `h` updating live, its loop count, age and how far back its family line goes. Add zoom and pan, and a camera that follows the selected creature.
- Touches: `app.py` mouse handling, `render.py` side panel.

### Species tracking and family tree (L) ✅
Track a real species ID instead of slowly drifting colours, and log when lines split or die out.
- Touches: a new `species` / `parent_id` attribute, periodic genome clustering, a new CSV.

## Suggested order

1. Glowing trails, then seasons: cheap, and they change the look and the selection pressure straight away.
2. Pheromones, then predators: together they turn a grazing simulator into an ecosystem.
3. Save and load, then the live HUD: needed once runs become interesting enough to keep.
4. Everything else, as you like.
