from pathlib import Path

import pygame

from game.track import Track


class MainWindow:
    """Главное окно и игровой цикл MARS."""

    WIDTH = 1000
    HEIGHT = 700
    FPS = 60

    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT)
        )
        pygame.display.set_caption("MARS")

        self.clock = pygame.time.Clock()
        self.running = True

        map_path = Path("maps") / "track_01.png"
        self.track = Track(map_path)

    def run(self):
        """Главный цикл приложения."""

        while self.running:
            self.handle_events()
            self.update()
            self.draw()

            self.clock.tick(self.FPS)

        pygame.quit()

    def handle_events(self):
        """Обработка событий."""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        """Обновление состояния игры."""

        pass

    def draw(self):
        """Отрисовка игры."""

        self.screen.fill((20, 20, 20))

        self.track.draw(self.screen)

        pygame.display.flip()