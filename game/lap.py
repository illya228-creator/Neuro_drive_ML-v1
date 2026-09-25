class LapCounter:
    """Отслеживает прохождение стартовой линии."""

    START_X_MIN = 420
    START_X_MAX = 480

    START_Y_MIN = 430
    START_Y_MAX = 500

    def __init__(self):
        self.laps = 0
        self.has_left_start = False

    def is_in_start_zone(self, car):
        """Проверяет нахождение автомобиля в зоне старта."""

        return (
            self.START_X_MIN <= car.x <= self.START_X_MAX
            and
            self.START_Y_MIN <= car.y <= self.START_Y_MAX
        )

    def update(self, car):
        """
        Обновляет состояние счётчика.

        Круг засчитывается только после того,
        как автомобиль сначала покинул стартовую зону,
        а затем вернулся в неё.
        """

        in_start_zone = self.is_in_start_zone(car)

        if not in_start_zone:
            self.has_left_start = True

        elif in_start_zone and self.has_left_start:
            self.laps += 1
            self.has_left_start = False

            return True

        return False

    def reset(self):
        """Сбрасывает счётчик кругов."""

        self.laps = 0
        self.has_left_start = False