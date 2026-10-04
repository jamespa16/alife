import torch

from alife.brain import think
from alife.config import DEV, K_MAX, N_HID, N_IN, N_OUT, G


def reference(genome, loops, x, h):
    """One creature at a time, straight from the genome layout in config.G."""
    a, b = N_HID * N_IN, N_HID * N_IN + N_HID * N_HID
    c = b + N_HID
    w_in = genome[:a].reshape(N_HID, N_IN)
    w_rec = genome[a:b].reshape(N_HID, N_HID)
    bias = genome[b:c]
    w_out = genome[c:].reshape(N_OUT, N_HID)
    for _ in range(loops):
        h = torch.tanh(w_in @ x + bias + w_rec @ h)
    return w_out @ h, h


def batch(n, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    genome = torch.randn(n, G, device=DEV, generator=g) * 0.5
    x = torch.randn(n, N_IN, device=DEV, generator=g)
    h = torch.randn(n, N_HID, device=DEV, generator=g).tanh()
    return genome, x, h


def test_genome_layout_fills_g():
    assert G == N_HID * N_IN + N_HID * N_HID + N_HID + N_OUT * N_HID


def test_matches_reference_for_mixed_loop_counts():
    # Mixed loop counts in one batch exercise the per-creature masking and the
    # early exit at loops.max(): each creature must match running alone.
    n = 2 * K_MAX
    genome, x, h = batch(n)
    loops = torch.arange(n, device=DEV) % K_MAX + 1
    out, h_new = think(genome, loops, x, h)
    for i in range(n):
        ref_out, ref_h = reference(genome[i], int(loops[i]), x[i], h[i])
        torch.testing.assert_close(out[i], ref_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(h_new[i], ref_h, rtol=1e-4, atol=1e-5)


def test_result_does_not_depend_on_batch_neighbours():
    # World.step thinks for a gathered subset; a creature's output must not
    # change with who else is in the batch.
    genome, x, h = batch(6)
    loops = torch.tensor([1, 3, 8, 2, 8, 1], device=DEV)
    out_all, h_all = think(genome, loops, x, h)
    sub = torch.tensor([0, 3, 5], device=DEV)  # all low loop counts
    out_sub, h_sub = think(genome[sub], loops[sub], x[sub], h[sub])
    torch.testing.assert_close(out_sub, out_all[sub])
    torch.testing.assert_close(h_sub, h_all[sub])


def test_memory_carries_over():
    genome, x, h = batch(4)
    loops = torch.ones(4, dtype=torch.long, device=DEV)
    out_a, _ = think(genome, loops, x, h)
    out_b, _ = think(genome, loops, x, torch.zeros_like(h))
    assert not torch.allclose(out_a, out_b)


def test_empty_batch():
    genome, x, h = batch(0)
    out, h_new = think(genome, torch.zeros(0, dtype=torch.long, device=DEV), x, h)
    assert out.shape == (0, N_OUT) and h_new.shape == (0, N_HID)
