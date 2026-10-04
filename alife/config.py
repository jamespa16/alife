"""Device selection and every tunable constant."""

import math

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

# ---- brain ----
N_IN = 2 * len(SENSE_ANGLES) + 1  # food x3, density x3, own energy
N_HID = 16
N_OUT = 2  # turn, speed
K_MAX = 8  # max loop count
G = N_HID * N_IN + N_HID * N_HID + N_HID + N_OUT * N_HID  # genome length
MUT_STD = 0.03
LOOP_MUT_P = 0.05
