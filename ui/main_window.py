from pathlib import Path
from turtle import done

import pygame
import torch

from game import collision
from game.car import Car
from game.track import Track
from game.collision import CollisionDetector
from game.sensors import SensorSystem
from game.lap import LapCounter

from ml.agent import AIAgent


class MainWindow:
    """Главное окно и игровой цикл MARS."""

    WIDTH = 1150
    HEIGHT = 700
    FPS = 60
    PANEL_WIDTH = 250

    START_X = 500
    START_Y = 465

    MAX_LAPS = 2

    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT)
        )

        pygame.display.set_caption("MARS")

        self.clock = pygame.time.Clock()
        self.running = True

        self.font = pygame.font.Font(
            None,
            26
        )

        self.small_font = pygame.font.Font(
            None,
            22
        )

        self.big_font = pygame.font.Font(
            None,
            64
        )

        # Трасса
        map_path = Path("maps") / "track_02.png"

        self.track = Track(
            map_path
        )

        # Столкновения
        self.collision = CollisionDetector(
            self.track
        )

        # Автомобиль
        self.car = Car(
            self.START_X,
            self.START_Y
        )

        # Виртуальные датчики
        self.sensors = SensorSystem(
            self.track
        )

        # AI
        self.agent = AIAgent()

        # Круги
        self.lap_counter = LapCounter()

        # Состояние победы
        self.won = False

        # Статистика
        self.steps = 0
        self.collisions = 0
        self.total_distance = 0.0
        self.last_reward = 0.0

    def run(self):
        """Главный игровой цикл."""

        while self.running:

            self.handle_events()

            if not self.won:
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

            # R — начать заново
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_r
            ):
                self.reset_game()

    def reset_car(self):
        """Возвращает автомобиль на старт."""

        self.car.x = self.START_X
        self.car.y = self.START_Y

        self.car.angle = 0.0
        self.car.speed = 0.0

    def reset_game(self):
        """Полностью перезапускает прохождение."""

        self.reset_car()

        self.lap_counter.reset()

        self.won = False

        self.steps = 0
        self.collisions = 0
        self.total_distance = 0.0
        self.last_reward = 0.0

    def update(self):
        """Обновление состояния игры и AI."""

        # --------------------------------
        # 1. Получаем состояние
        # --------------------------------

        self.sensors.update(
            self.car
        )

        state = self.agent.get_state(
            self.sensors.distances,
            self.car.speed
        )

        # --------------------------------
        # 2. AI выбирает действие
        # --------------------------------

        action = self.agent.predict(
            state
        )

        if action == 0:
            turn = -1

        elif action == 1:
            turn = 0

        else:
            turn = 1

        accelerate = True

        # --------------------------------
        # 3. Сохраняем позицию
        # --------------------------------

        old_x = self.car.x
        old_y = self.car.y

        # --------------------------------
        # 4. Двигаем автомобиль
        # --------------------------------

        self.car.update(
            turn=turn,
            accelerate=accelerate
        )

        # --------------------------------
        # 5. Проверяем столкновение
        # --------------------------------

        collision = self.collision.is_collision(
            self.car
        )

        # --------------------------------
        # 6. Расстояние
        # --------------------------------

        movement = (
            (self.car.x - old_x) ** 2
            + (self.car.y - old_y) ** 2
        ) ** 0.5

        self.total_distance += movement

        # --------------------------------
        # 7. Проверяем круг
        # --------------------------------

        lap_completed = self.lap_counter.update(
            self.car
        )

        if lap_completed:

            print(
                f"LAP {self.lap_counter.laps} COMPLETE"
            )

        # --------------------------------
        # 8. Проверяем победу
        # --------------------------------

        if self.lap_counter.laps >= self.MAX_LAPS:

            self.won = True

            self.car.speed = 0

            self.save_model()

            self.agent.save_checkpoint()

            print("YOU WIN")
            print(
                f"Laps: {self.lap_counter.laps}"
            )
            print(
                f"Distance: "
                f"{self.total_distance:.1f}"
            )

            return

        # --------------------------------
        # 9. Получаем новое состояние
        # --------------------------------

        self.sensors.update(
            self.car
        )

        next_state = self.agent.get_state(
            self.sensors.distances,
            self.car.speed
        )

        # --------------------------------
        # 10. Reward
        # --------------------------------

        if collision:

                # Большой штраф за столкновение
                reward = -10.0
                done = True

                self.collisions += 1

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

            done = False

            # Базовая награда за движение
            reward = 0.05

            # Расстояние до ближайшей стены
            min_distance = min(
                self.sensors.distances
            )

    #    Безопасное положение
            if min_distance > 70:
                reward += 0.10

    # Слишком близко к стене
            elif min_distance < 30:
                reward -= 0.15

    # Прохождение круга
            if lap_completed:

                if self.lap_counter.laps == 1:
                    reward += 20.0

                elif self.lap_counter.laps == 2:
                    reward += 50.0

        self.agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        self.agent.train_step()

        self.last_reward = reward

        # --------------------------------
        # 11. Обучение
        # --------------------------------

        self.steps += 1

        if self.steps % 100 == 0:
            self.agent.update_target()

        if self.steps % 10 == 0:
            self.agent.decay_epsilon()

    def save_model(self):
        """Сохраняет обученную модель."""

        model_path = Path(
            "models"
        ) / "mars_model.pth"

        torch.save(
            self.agent.model.state_dict(),
            model_path
        )

        print(
            f"Model saved: {model_path}"
        )

    def draw_text(
        self,
        text,
        x,
        y,
        font=None
    ):
        """Выводит текст на экран."""

        if font is None:
            font = self.font

        surface = font.render(
            text,
            True,
            (255, 255, 255)
        )

        self.screen.blit(
            surface,
            (x, y)
        )

    def draw_ui(self):
        """Отрисовка панели интерфейса."""

        panel_x = self.WIDTH - self.PANEL_WIDTH

    # Фон панели
        pygame.draw.rect(
            self.screen,
            (25, 25, 25),
            (
            panel_x,
                0,
                self.PANEL_WIDTH,
                self.HEIGHT
            )
        )

        # Разделитель
        pygame.draw.line(
            self.screen,
            (80, 80, 80),
            (panel_x, 0),
            (panel_x, self.HEIGHT),
            2
        )

        x = panel_x + 25

        self.draw_text(
            "MARS AI",
            x,
            30,
            self.font
        )

        self.draw_text(
            f"LAPS: {self.lap_counter.laps} / {self.MAX_LAPS}",
            x,
            90,
            self.small_font
     )   

        self.draw_text(
            f"SPEED: {self.car.speed:.2f}",
            x,
            130,
            self.small_font
        )

        self.draw_text(
            f"REWARD: {self.last_reward:.2f}",
            x,
            170,
            self.small_font
        )

        self.draw_text(
            f"DISTANCE: {self.total_distance:.0f}",
            x,
            210,
            self.small_font
        )

        self.draw_text(
            f"COLLISIONS: {self.collisions}",
            x,
            250,
            self.small_font
        )

        self.draw_text(
            f"EPSILON: {self.agent.epsilon:.3f}",
            x,
            290,
            self.small_font
        )

        self.draw_text(
            f"MEMORY: {len(self.agent.memory)}",
            x,
            330,
            self.small_font
        )

        self.draw_text(
            f"STEPS: {self.agent.steps}",
            x,
            370,
            self.small_font
        )

    def draw_win_screen(self):
        """Экран победы."""

        overlay = pygame.Surface(
            (
                self.WIDTH,
                self.HEIGHT
            ),
            pygame.SRCALPHA
        )

        overlay.fill(
            (0, 0, 0, 170)
        )

        self.screen.blit(
            overlay,
            (0, 0)
        )

        text = self.big_font.render(
            "YOU WIN",
            True,
            (255, 255, 255)
        )

        text_rect = text.get_rect(
            center=(
                self.WIDTH // 2,
                self.HEIGHT // 2 - 60
            )
        )

        self.screen.blit(
            text,
            text_rect
        )

        self.draw_text(
            "2 LAPS COMPLETE",
            360,
            370
        )

        self.draw_text(
            f"DISTANCE: "
            f"{self.total_distance:.0f}",
            390,
            410
        )

        self.draw_text(
            "PRESS R TO RESTART",
            355,
            470
        )

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

        # Машина
        self.car.draw(
            self.screen
        )

        # UI
        self.draw_ui()

        # Победа
        if self.won:
            self.draw_win_screen()

        pygame.display.flip()