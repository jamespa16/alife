"""Weight-tied recurrent brain, evaluated for every creature in one batch."""

import torch

from .config import N_HID, N_IN, N_OUT


def think(genome, loops, x, h):
    """Run each creature's net `loops` times, starting from its hidden state `h`.

    genome: [N, G], loops: [N], x: [N, N_IN], h: [N, N_HID].
    Returns (out [N, N_OUT], new h), so memory carries over between steps.
    """
    a = N_HID * N_IN
    b = a + N_HID * N_HID
    c = b + N_HID
    w_in = genome[:, :a].reshape(-1, N_HID, N_IN)
    w_rec = genome[:, a:b].reshape(-1, N_HID, N_HID)
    bias = genome[:, b:c]
    w_out = genome[:, c:].reshape(-1, N_OUT, N_HID)

    drive = torch.bmm(w_in, x.unsqueeze(2)).squeeze(2) + bias
    # Stop once every creature is done (one GPU sync; loops rarely reach K_MAX).
    for i in range(int(loops.max()) if len(loops) else 0):
        h_new = torch.tanh(drive + torch.bmm(w_rec, h.unsqueeze(2)).squeeze(2))
        h = torch.where((loops > i).unsqueeze(1), h_new, h)
    return torch.bmm(w_out, h.unsqueeze(2)).squeeze(2), h
