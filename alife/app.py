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
        "--species-log",
        metavar="PATH",
        help="write species births and extinctions to a CSV (needs SPECIES=1)",
    )
    p.add_argument(
        "--save", metavar="PATH", help="save the world here when the run ends"
    )
    p.add_argument(
        "--save-every", type=int, metavar="N", help="also save to --save every N steps"
    )
    p.add_argument(
        "--load", metavar="PATH", help="start from a saved world, not a new one"
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
    if args.save_every is not None and (args.save is None or args.save_every < 1):
        p.error("--save-every needs --save and a value of at least 1")
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


class SpeciesLog:
    """Writes World.take_events() rows (t, event, species, parent_species, size) to a CSV.

    Without a file it still drains the events, so they never pile up."""

    FIELDS = ("t", "event", "species", "parent_species", "size")

    def __init__(s, file=None):
        s.file = file
        if file is not None:
            s.writer = csv.DictWriter(file, s.FIELDS, restval="")
            s.writer.writeheader()

    def drain(s, world):
        events = world.take_events()
        if s.file is not None and events:
            s.writer.writerows(events)
            s.file.flush()


def snapshot_name():
    return time.strftime("alife-%Y%m%d-%H%M%S.alife")


def maybe_save(world, path, every):
    if path and every and world.t % every == 0:
        world.save(path)


def run_headless(world, steps, log, save=None, save_every=None, slog=None):
    slog = slog or SpeciesLog()
    log.maybe_log(world)
    slog.drain(world)
    last = time.time()
    while world.t < steps:
        world.step()
        log.maybe_log(world)
        slog.drain(world)
        maybe_save(world, save, save_every)
        now = time.time()
        if now - last > 5 or world.t == steps:
            pop, loops = world.stats()
            print(f"t={world.t}  pop={pop}  mean loops={loops:.2f}", file=sys.stderr)
            if pop == 0:
                print("extinct", file=sys.stderr)
                return
            last = now


def run_window(world, steps, log, save=None, save_every=None, slog=None):
    import pygame

    from .config import HUD_HISTORY, INSPECT_W, PICK_RADIUS, SPECIES, H, W
    from .hud import draw_hud, make_hud
    from .panel import Inspector, View, draw_panel, make_panel
    from .render import render

    slog = slog or SpeciesLog()
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    frame = pygame.Surface((W, H))  # the full-world picture; the view crops it
    modes = ("lineage", "loops", "diet") + (("species",) if SPECIES else ())
    mode, spf, paused = 0, 1, False
    last, n = time.time(), 0
    log.maybe_log(world)
    slog.drain(world)
    show_hud, rows = True, []
    hud, font = make_hud()
    panel, pfont = make_panel()
    view, insp, follow = View(), Inspector(), False

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.VIDEORESIZE:
                screen = pygame.display.get_surface()
                view.resize(*screen.get_size())
                hud, font = make_hud(max(view.scale, 0.75))  # stays readable
                draw_hud(hud, rows, font)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if not (
                    insp.data is not None and e.pos[0] >= screen.get_width() - INSPECT_W
                ):  # not on the panel
                    wx, wy = view.to_world(*e.pos)
                    insp.select(world, wx, wy, PICK_RADIUS / view.zoom)
            elif e.type == pygame.MOUSEWHEEL:
                view.zoom_at(*pygame.mouse.get_pos(), 1.25**e.y)
            elif e.type == pygame.MOUSEMOTION and e.buttons[2]:
                view.pan(*e.rel)
                follow = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_c:
                    mode = (mode + 1) % len(modes)
                elif e.key == pygame.K_UP and e.mod & pygame.KMOD_SHIFT:
                    view.pan(0, 40)
                elif e.key == pygame.K_DOWN and e.mod & pygame.KMOD_SHIFT:
                    view.pan(0, -40)
                elif e.key == pygame.K_LEFT and e.mod & pygame.KMOD_SHIFT:
                    view.pan(40, 0)
                elif e.key == pygame.K_RIGHT and e.mod & pygame.KMOD_SHIFT:
                    view.pan(-40, 0)
                elif e.key == pygame.K_UP:
                    spf = min(spf * 2, 64)
                elif e.key == pygame.K_DOWN:
                    spf = max(spf // 2, 1)
                elif e.key == pygame.K_m:
                    world.meteor(*view.to_world(*pygame.mouse.get_pos()))
                elif e.key == pygame.K_f:
                    world.flood()
                elif e.key == pygame.K_p:
                    world.plague()
                elif e.key == pygame.K_h:
                    show_hud = not show_hud
                elif e.key == pygame.K_v:
                    follow = not follow
                elif e.key == pygame.K_0:
                    view.reset()
                    follow = False
                elif e.key == pygame.K_ESCAPE:
                    insp.clear()
                    follow = False
                elif e.key == pygame.K_s:
                    name = snapshot_name()
                    world.save(name)
                    print(f"saved {name}", file=sys.stderr)

        if not paused:
            for _ in range(spf):
                world.step()
                log.maybe_log(world)
                slog.drain(world)
                maybe_save(world, save, save_every)
                n += 1
                if world.t == steps:
                    running = False
                    break

        if not paused or not insp.dead:
            insp.refresh(world)  # one small fetch per frame, for the selection only
        if follow and insp.data is not None:
            view.center_on(*insp.data["pos"])

        pygame.surfarray.blit_array(frame, render(world, mode))
        ox, oy, dw, dh = view.dest()
        if (dw, dh) != screen.get_size():
            screen.fill((0, 0, 0))  # letterbox bars
        if view.zoom == 1 and (dw, dh) == (W, H):
            screen.blit(frame, (ox, oy))
        else:
            sub = frame.subsurface(view.rect())
            screen.blit(pygame.transform.scale(sub, (dw, dh)), (ox, oy))
        if show_hud:
            screen.blit(hud, (0, 0))
        if insp.data is not None:
            sx, sy = view.to_screen(*insp.data["pos"])
            color = (255, 90, 90) if insp.dead else (255, 255, 255)
            pygame.draw.circle(screen, color, (round(sx), round(sy)), 10, 1)
            draw_panel(panel, insp.data, insp.dead, pfont)
            screen.blit(panel, (screen.get_width() - INSPECT_W, 0))
        pygame.display.flip()

        now = time.time()
        if now - last > 0.5:
            c = world.census()  # one sample per 0.5 s, like stats(): no per-step syncs
            pop, loops = c["pop"], c["mean_loops"]
            if pop == 0:
                world.reset()
                slog.drain(world)
                insp.clear()
                rows.clear()
            rows.append(c)
            del rows[:-HUD_HISTORY]
            draw_hud(hud, rows, font)
            extra = f"  species={c['n_species']}" if "n_species" in c else ""
            pygame.display.set_caption(
                f"t={world.t}  pop={pop}  mean loops={loops:.2f}{extra}  "
                f"{n / (now - last):.0f} steps/s  x{spf}  "
                f"[{modes[mode]}]"
            )
            last, n = now, 0

    pygame.quit()


def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    from .world import World

    world = World.load(args.load) if args.load else World()
    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(args.log, "w", newline="")) if args.log else None
        log = Logger(f, args.log_every)
        sf = (
            stack.enter_context(open(args.species_log, "w", newline=""))
            if args.species_log
            else None
        )
        slog = SpeciesLog(sf)
        try:
            if args.headless:
                run_headless(world, args.steps, log, args.save, args.save_every, slog)
            else:
                run_window(world, args.steps, log, args.save, args.save_every, slog)
        except KeyboardInterrupt:
            pass
        if args.save:
            world.save(args.save)
