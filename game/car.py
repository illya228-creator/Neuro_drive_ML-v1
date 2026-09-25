import math

import pygame


class Car:
    """Автомобиль MARS."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

        self.angle = 0.0
        self.speed = 0.0

        self.max_speed = 4.0
        self.acceleration = 0.1
        self.friction = 0.05
        self.rotation_speed = 3.0

        self.width = 24
        self.height = 14

    def update(self, turn: int = 0, accelerate: bool = False):
        """Обновляет положение автомобиля."""

        if accelerate:
            self.speed += self.acceleration
        else:
            self.speed -= self.friction

        self.speed = max(
            0.0,
            min(self.speed, self.max_speed)
        )

        self.angle += turn * self.rotation_speed

        angle_rad = math.radians(self.angle)

        self.x += math.cos(angle_rad) * self.speed
        self.y += math.sin(angle_rad) * self.speed

    def draw(self, screen: pygame.Surface):
        """Отрисовывает автомобиль."""

        car_surface = pygame.Surface(
            (self.width, self.height),
            pygame.SRCALPHA
        )

        pygame.draw.rect(
            car_surface,
            (220, 40, 40),
            (0, 0, self.width, self.height),
            border_radius=4
        )

        rotated = pygame.transform.rotate(
            car_surface,
            -self.angle
        )

        rect = rotated.get_rect(
            center=(self.x, self.y)
        )

        screen.blit(rotated, rect)