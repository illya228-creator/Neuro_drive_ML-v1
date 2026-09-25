from pathlib import Path

import pygame

from game.car import Car
from game.track import Track
from game.collision import CollisionDetector
from game.sensors import SensorSystem
from ml.agent import AIAgent


class MainWindow:
    """Главное окно и игровой цикл MARS."""

    WIDTH = 1000
    HEIGHT = 700
    FPS = 60

    START_X = 450
    START_Y = 465

    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT)
        )

        pygame.display.set_caption("MARS")

        self.clock = pygame.time.Clock()
        self.running = True

        # Трасса
        map_path = Path("maps") / "track_01.png"
        self.track = Track(map_path)

        # Столкновения
        self.collision = CollisionDetector(
            self.track
        )

        # Автомобиль
        self.car = Car(
            self.START_X,
            self.START_Y
        )

        # Датчики
        self.sensors = SensorSystem(
            self.track
        )

        # AI
        self.agent = AIAgent()

        # Предыдущее состояние
        self.previous_state = None

        # Предыдущее действие
        self.previous_action = None

        # Счётчик шагов
        self.steps = 0

        # Количество столкновений
        self.collisions = 0

    def run(self):
        """Главный игровой цикл."""

        while self.running:
            self.handle_events()
            self.update()
            self.draw()

            self.clock.tick(
                self.FPS
            )

        pygame.quit()

    def handle_events(self):
        """Обработка событий."""

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

    def reset_car(self):
        """Возвращает автомобиль на старт."""

        self.car.x = self.START_X
        self.car.y = self.START_Y

        self.car.angle = 0.0
        self.car.speed = 0.0

        self.previous_state = None
        self.previous_action = None

    def update(self):
        """Обновление состояния игры."""

        # Обновляем датчики
        self.sensors.update(
            self.car
        )

        # Получаем состояние
        state = self.agent.get_state(
            self.sensors.distances,
            self.car.speed
        )

        # Выбираем действие
        action = self.agent.predict(
            state
        )

        # Преобразуем действие в поворот
        if action == 0:
            turn = -1

        elif action == 1:
            turn = 0

        else:
            turn = 1

        # AI постоянно движется
        accelerate = True

        # Сохраняем старую позицию
        old_x = self.car.x
        old_y = self.car.y

        # Двигаем машину
        self.car.update(
            turn=turn,
            accelerate=accelerate
        )

        # Проверяем столкновение
        collision = self.collision.is_collision(
            self.car
        )

        # Получаем новое состояние
        self.sensors.update(
            self.car
        )

        next_state = self.agent.get_state(
            self.sensors.distances,
            self.car.speed
        )

        # Reward
        if collision:

            reward = -10.0
            done = True

            self.collisions += 1

            # Сохраняем опыт столкновения
            self.agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            self.agent.train_step()

            # Возвращаем машину на старт
            self.reset_car()

        else:

            # Небольшая награда за движение
            reward = 0.1
            done = False

            self.agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            self.agent.train_step()

        # Обновляем target network
        self.steps += 1

        if self.steps % 100 == 0:
            self.agent.update_target()

        # Уменьшаем epsilon
        if self.steps % 10 == 0:
            self.agent.decay_epsilon()

    def draw(self):
        """Отрисовка игры."""

        self.screen.fill(
            (20, 20, 20)
        )

        # Трасса
        self.track.draw(
            self.screen
        )

        # Датчики
        self.sensors.draw(
            self.screen,
            self.car
        )

        # Автомобиль
        self.car.draw(
            self.screen
        )

        pygame.display.flip()