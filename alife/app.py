"""pygame window, input handling and main loop."""

import time

import pygame

from .config import H, W
from .render import render
from .world import World


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    world = World()
    mode, spf, paused = 0, 1, False
    last, steps = time.time(), 0

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
            steps += spf

        pygame.surfarray.blit_array(screen, render(world, mode))
        pygame.display.flip()

        now = time.time()
        if now - last > 0.5:
            pop, loops = world.stats()
            if pop == 0:
                world.reset()
            pygame.display.set_caption(
                f"t={world.t}  pop={pop}  mean loops={loops:.2f}  "
                f"{steps / (now - last):.0f} steps/s  x{spf}  "
                f"[{'lineage' if mode == 0 else 'loops'}]"
            )
            last, steps = now, 0

    pygame.quit()
