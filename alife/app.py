"""Command line, CSV logging, and the headless and pygame main loops.

The sim modules (world, render) and pygame are imported inside main(), after
config overrides are applied, because they copy constants at import time.
"""

import argparse
import contextlib
import csv
import sys
import time

import torch

from . import config


def parse_args(argv):
    p = argparse.ArgumentParser(prog="alife", description="Evolving creature world.")
    p.add_argument(
        "--headless", action="store_true", help="run without a window (needs --steps)"
    )
    p.add_argument("--steps", type=int, help="stop after this many steps")
    p.add_argument("--seed", type=int, help="random seed")
    p.add_argument("--log", metavar="PATH", help="append population stats to a CSV")
    p.add_argument(
        "--log-every", type=int, default=100, metavar="K", help="log every K steps"
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a numeric constant in config.py, e.g. --set THINK_COST=0",
    )
    args = p.parse_args(argv)
    if args.headless and args.steps is None:
        p.error("--headless needs --steps")
    if args.log_every < 1:
        p.error("--log-every must be at least 1")
    for s in args.set:
        if "=" not in s:
            p.error(f"--set {s}: expected NAME=VALUE")
    try:
        config.override(dict(s.split("=", 1) for s in args.set))
    except (ValueError, KeyError) as e:
        p.error(f"--set: {e.args[0]}")
    return args


class Logger:
    """Writes World.census() rows to a CSV file every `every` steps (no-op if file is None)."""

    def __init__(s, file, every):
        s.every = every
        s.file = file
        s.writer = None

    def maybe_log(s, world):
        if s.file is None or world.t % s.every:
            return
        row = world.census()
        if s.writer is None:
            s.writer = csv.DictWriter(s.file, row.keys())
            s.writer.writeheader()
        s.writer.writerow(row)
        s.file.flush()  # a killed run still leaves usable data


def run_headless(world, steps, log):
    log.maybe_log(world)
    last = time.time()
    while world.t < steps:
        world.step()
        log.maybe_log(world)
        now = time.time()
        if now - last > 5 or world.t == steps:
            pop, loops = world.stats()
            print(f"t={world.t}  pop={pop}  mean loops={loops:.2f}", file=sys.stderr)
            if pop == 0:
                print("extinct", file=sys.stderr)
                return
            last = now


def run_window(world, steps, log):
    import pygame

    from .config import H, W
    from .render import render

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    mode, spf, paused = 0, 1, False
    last, n = time.time(), 0
    log.maybe_log(world)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_c:
                    mode = 1 - mode
                elif e.key == pygame.K_UP:
                    spf = min(spf * 2, 64)
                elif e.key == pygame.K_DOWN:
                    spf = max(spf // 2, 1)

        if not paused:
            for _ in range(spf):
                world.step()
                log.maybe_log(world)
                n += 1
                if world.t == steps:
                    running = False
                    break

        pygame.surfarray.blit_array(screen, render(world, mode))
        pygame.display.flip()

        now = time.time()
        if now - last > 0.5:
            pop, loops = world.stats()
            if pop == 0:
                world.reset()
            pygame.display.set_caption(
                f"t={world.t}  pop={pop}  mean loops={loops:.2f}  "
                f"{n / (now - last):.0f} steps/s  x{spf}  "
                f"[{'lineage' if mode == 0 else 'loops'}]"
            )
            last, n = now, 0

    pygame.quit()


def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    from .world import World

    world = World()
    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(args.log, "w", newline="")) if args.log else None
        log = Logger(f, args.log_every)
        try:
            if args.headless:
                run_headless(world, args.steps, log)
            else:
                run_window(world, args.steps, log)
        except KeyboardInterrupt:
            pass
