from pathlib import Path

import pygame

from game.car import Car
from game.track import Track
from game.collision import CollisionDetector
from game.sensors import SensorSystem

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

        # Загрузка трассы
        map_path = Path("maps") / "track_01.png"
        self.track = Track(map_path)
        self.collision = CollisionDetector(self.track)
        # Создание автомобиля
        self.car = Car(450, 465)
        self.sensors = SensorSystem(self.track)

    def run(self):
        """Главный игровой цикл."""

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
        
        keys = pygame.key.get_pressed()

        turn = 0

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            turn = -1

        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            turn = 1

        accelerate = (
            keys[pygame.K_w]
            or keys[pygame.K_UP]
        )

        old_x = self.car.x
        old_y = self.car.y

        self.car.update(
            turn=turn,
            accelerate=accelerate
        )

        if self.collision.is_collision(self.car):
            self.car.x = old_x
            self.car.y = old_y
            self.car.speed = 0

        self.sensors.update(self.car)

        

    def draw(self):
        """Отрисовка игры."""

        self.screen.fill((20, 20, 20))

        self.track.draw(self.screen)

        self.sensors.draw(
            self.screen,
            self.car
        )

        self.car.draw(self.screen)

        pygame.display.flip()