import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ml.model import DrivingModel
from ml.replay_buffer import ReplayBuffer


class AIAgent:
    """AI-агент с обучением DQN."""

    def __init__(self):
        self.model = DrivingModel()
        self.target_model = DrivingModel()

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        self.memory = ReplayBuffer()

        self.gamma = 0.95

        # Вероятность случайного действия.
        self.epsilon = 1.0

        # Минимальная вероятность случайного действия.
        self.epsilon_min = 0.05

        # Скорость уменьшения epsilon.
        self.epsilon_decay = 0.995

        self.batch_size = 64

    def get_state(self, distances, speed):
        """Создаёт состояние автомобиля."""

        state = [
            distance / 200.0
            for distance in distances
        ]

        state.append(
            speed / 4.0
        )

        return state

    def predict(self, state):
        """Выбирает действие."""

        if random.random() < self.epsilon:
            return random.randint(0, 2)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():
            output = self.model(
                state_tensor
            )

        return torch.argmax(output).item()

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """Сохраняет опыт."""

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    def train_step(self):
        """Обучает модель на одном batch."""

        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(
            self.batch_size
        )

        states = torch.tensor(
            [item[0] for item in batch],
            dtype=torch.float32
        )

        actions = torch.tensor(
            [item[1] for item in batch],
            dtype=torch.long
        )

        rewards = torch.tensor(
            [item[2] for item in batch],
            dtype=torch.float32
        )

        next_states = torch.tensor(
            [item[3] for item in batch],
            dtype=torch.float32
        )

        dones = torch.tensor(
            [item[4] for item in batch],
            dtype=torch.float32
        )

        current_q = self.model(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target_model(
                next_states
            ).max(1)[0]

            target_q = (
                rewards
                + self.gamma
                * next_q
                * (1 - dones)
            )

        loss = F.mse_loss(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """Синхронизирует target-модель."""

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

    def decay_epsilon(self):
        """Уменьшает количество случайных действий."""

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )