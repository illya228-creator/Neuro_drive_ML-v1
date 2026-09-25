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

    def draw(self, screen: pygame.Surface):
        """Отображает карту на экране."""

        screen.blit(self.image, self.rect)