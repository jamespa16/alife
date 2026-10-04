"""Device selection and every tunable constant."""

import ast
import math
import sys

import torch

if torch.cuda.is_available():
    DEV = "cuda"
elif torch.backends.mps.is_available():  # Apple Silicon GPU
    DEV = "mps"
else:
    DEV = "cpu"
TAU = 2 * math.pi

# ---- world ----
W = H = 768  # food grid size == window size
N_MAX = 200_000  # creature slots
N_INIT = 5_000
FOOD_GROWTH = 0.003  # per step, scaled by local fertility
SEASON_PERIOD = 2000  # steps per seasonal cycle
SEASON_AMP = 0.4  # fertility swings by +-this fraction (0 = no seasons)
FERT_DRIFT = 0.05  # cells per step the fertility map slides in x (0 = static land)

# ---- display ----
TRAIL_DECAY = 0.9  # per frame brightness kept by creature trails (0 = no trails)

# ---- HUD (key H) ----
HUD_W = 260  # overlay size in pixels (render only)
HUD_H = 240
HUD_HISTORY = 240  # census samples kept (one per 0.5 s)

# ---- disasters (keys M, F, P) ----
METEOR_R = 60.0  # radius of the meteor crater
PLAGUE_STEPS = 400  # how long an outbreak keeps spreading
PLAGUE_START = 0.05  # fraction of creatures infected at the start
PLAGUE_SPREAD = (
    0.2  # per step, chance of catching it from a sick creature in the same cell
)
PLAGUE_RECOVER = 0.01  # per step, chance a sick creature recovers
PLAGUE_DRAIN = 0.05  # energy lost per step while sick

# ---- body ----
MAX_SPEED = 2.0  # cells per step
MAX_TURN = 0.5  # radians per step
SENSE_DIST = 6.0
SENSE_ANGLES = (-0.6, 0.0, 0.6)
EAT_RATE = 0.2
BASE_COST = 0.01
MOVE_COST = 0.01  # per unit of speed
THINK_COST = 0.002  # per brain loop
REPRO_E = 2.0  # energy needed to split
MAX_AGE = 3000

# ---- pheromones ----
PHERO_CH = 2  # scent channels creatures can drop and smell (0 = no pheromones)
PHERO_DEPOSIT = 0.5  # max scent a creature drops per channel per step
PHERO_DECAY = 0.95  # per step, fraction of scent kept
PHERO_DIFFUSE = 0.5  # per step, how far scent is mixed toward its blurred self
PHERO_RADIUS = 1  # blur radius used for diffusion
PHERO_COST = 0.002  # energy per unit of scent dropped
PHERO_GLOW = 1.0  # display brightness of each channel (render only)

# ---- predation ----
PREDATION = 1  # attack output, diet gene and meat grid (0 = off)
ATTACK_BIAS = 3.0  # attack = sigmoid(output - this): newborn brains rarely attack
ATTACK_RATE = 1.0  # energy a full-strength attack takes from each victim per step
ATTACK_COST = 0.005  # energy per unit of attack output, hit or miss
MEAT_DROP = 0.8  # fraction of a dead creature's energy left as meat
MEAT_DECAY = 0.98  # per step, fraction of meat kept
MEAT_EAT_RATE = 0.4  # max meat a pure carnivore eats per step
DIET_MUT_STD = 0.05  # diet gene mutation per birth
DIET_INIT = 1.0  # initial diet = rand ** DIET_INIT (higher = more herbivores)

# ---- sexual reproduction ----
SEX = 1  # 1 = two parents in the same cell, genes mixed; 0 = a lone parent splits
SEX_COST = 0.4  # fraction of its energy each parent pays; the child gets both payments
PICK_INIT = 1.5  # initial mate-choosiness = rand * this (colour distance, max ~1.2)
PICK_MUT_STD = 0.02  # mate-choosiness mutation per birth

# ---- body shape ----
BODY = 1  # size, speed and sensing-range genes with energy costs (0 = off)
BODY_MIN = 0.5  # each gene is a multiplier, clamped to [BODY_MIN, BODY_MAX]
BODY_MAX = 2.0
BODY_INIT = 0.3  # initial genes = 1 + (rand - 0.5) * this
BODY_MUT_STD = 0.03  # mutation per birth
SIZE_COST = 0.004  # energy per step per unit of size
SPEED_COST = 0.002  # per unit of top-speed multiplier
SENSE_COST = 0.002  # per unit of sensing-range multiplier
BIG_SIZE = 1.3  # render only: bigger than this is drawn 2x2, bigger than BIG_SIZE2 3x3
BIG_SIZE2 = 1.7

# ---- terrain ----
TERRAIN = 1  # walls that block movement and growth, poison zones (0 = off)
TERRAIN_SCALE = 20  # noise cells across the map (smaller = bigger blobs)
WALL_FRAC = 0.10  # fraction of the map that is wall
POISON_FRAC = 0.08  # fraction of the map that is poison
POISON_DRAIN = 0.03  # energy lost per step in a poison cell

# ---- brain ----
N_HID = 16
K_MAX = 8  # max loop count
MUT_STD = 0.03
LOOP_MUT_P = 0.05


N_IN: int
N_OUT: int
G: int


def _derive():
    global N_IN, N_OUT, G
    # food x3, density x3, own energy, then each pheromone channel x3
    # then, if PREDATION, meat x3 and nearby aggression x3
    N_IN = 2 * len(SENSE_ANGLES) + 1 + len(SENSE_ANGLES) * (PHERO_CH + 2 * PREDATION)
    # then, if TERRAIN, wall x3 and poison x3
    N_IN += 2 * len(SENSE_ANGLES) * TERRAIN
    # turn, speed, one deposit per pheromone channel, then attack if PREDATION
    N_OUT = 2 + PHERO_CH + PREDATION
    G = N_HID * N_IN + N_HID * N_HID + N_HID + N_OUT * N_HID  # genome length


_derive()
_DERIVED = {"N_IN", "N_OUT", "G", "TAU"}


def override(settings):
    """Set numeric constants from {"NAME": "value"} strings, e.g. {"THINK_COST": "0"}.

    Must run before world/brain/render are imported: they copy constants at import.
    """
    if any(f"{__package__}.{m}" in sys.modules for m in ("world", "brain", "render")):
        raise RuntimeError(
            "config.override() called after the sim modules were imported"
        )
    g = globals()
    for name, text in settings.items():
        old = g.get(name)
        if name in _DERIVED or not name.isupper() or not isinstance(old, (int, float)):
            raise KeyError(f"{name} is not an overridable numeric constant")
        kind = type(old)
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = None
        if not isinstance(value, (int, float)) or (kind is int and value != int(value)):
            raise ValueError(f"{name}={text!r}: expected {kind.__name__}")
        g[name] = kind(value)
    _derive()


# Constants whose change makes a saved world's tensors the wrong shape.
SHAPE_KEYS = {"N_IN", "N_OUT", "G", "N_MAX", "N_HID", "K_MAX", "W", "H", "PHERO_CH"}


def snapshot():
    """Every numeric constant (derived ones too) as {name: value}, for save files."""
    return {
        k: v
        for k, v in globals().items()
        if k.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool)
    }
