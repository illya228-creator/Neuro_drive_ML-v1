from pathlib import Path

import pygame


class Track:
    """Загружает и отображает карту гоночной трассы."""

    def __init__(self, map_path: str):
        self.map_path = Path(map_path)

        if not self.map_path.exists():
            raise FileNotFoundError(
                f"Карта не найдена: {self.map_path}"
            )

        self.image = pygame.image.load(
            self.map_path
        ).convert()

        self.rect = self.image.get_rect()

    def is_wall(self, x: float, y: float) -> bool:
        """Проверяет, является ли точка стеной."""

        x = int(x)
        y = int(y)

        if not self.rect.collidepoint(x, y):
            return True

        pixel = self.image.get_at((x, y))

        # Чёрный цвет — стена.
        return (
            pixel.r < 50
            and pixel.g < 50
            and pixel.b < 50
        )

    def draw(self, screen: pygame.Surface):
        """Отображает карту."""

        screen.blit(self.image, self.rect)