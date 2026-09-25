import math

import pygame


class SensorSystem:
    """Система виртуальных датчиков автомобиля."""

    ANGLES = [-60, -40, -20, 0, 20, 40, 60]
    MAX_DISTANCE = 200
    STEP = 2

    def __init__(self, track):
        self.track = track
        self.distances = [self.MAX_DISTANCE] * len(self.ANGLES)
        self.points = []

    def update(self, car):
        """Измеряет расстояния от автомобиля до стен."""

        self.distances = []
        self.points = []

        for relative_angle in self.ANGLES:
            angle = car.angle + relative_angle

            distance, point = self._cast_ray(
                car.x,
                car.y,
                angle
            )

            self.distances.append(distance)
            self.points.append(point)

    def _cast_ray(self, start_x, start_y, angle):
        """Пускает один луч и ищет ближайшую стену."""

        angle_rad = math.radians(angle)

        for distance in range(
            0,
            self.MAX_DISTANCE,
            self.STEP
        ):
            x = start_x + math.cos(angle_rad) * distance
            y = start_y + math.sin(angle_rad) * distance

            if self.track.is_wall(x, y):
                return distance, (x, y)

        end_x = (
            start_x
            + math.cos(angle_rad) * self.MAX_DISTANCE
        )

        end_y = (
            start_y
            + math.sin(angle_rad) * self.MAX_DISTANCE
        )

        return self.MAX_DISTANCE, (end_x, end_y)

    def draw(self, screen, car):
        """Отображает лучи датчиков."""

        for distance, point in zip(
            self.distances,
            self.points
        ):
            pygame.draw.line(
                screen,
                (255, 220, 50),
                (car.x, car.y),
                point,
                2
            )

            pygame.draw.circle(
                screen,
                (255, 80, 80),
                (int(point[0]), int(point[1])),
                4
            )