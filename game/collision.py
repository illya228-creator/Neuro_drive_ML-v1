class CollisionDetector:
    """Проверяет столкновение автомобиля со стенами трассы."""

    def __init__(self, track):
        self.track = track

    def is_collision(self, car):
        """Возвращает True, если автомобиль находится в стене."""

        points = [
            (car.x - car.width / 2, car.y - car.height / 2),
            (car.x + car.width / 2, car.y - car.height / 2),
            (car.x - car.width / 2, car.y + car.height / 2),
            (car.x + car.width / 2, car.y + car.height / 2),
        ]

        for x, y in points:
            if self.track.is_wall(x, y):
                return True

        return False