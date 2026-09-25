import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ml.model import DrivingModel
from ml.replay_buffer import ReplayBuffer


class AIAgent:
    """AI-агент с обучением DQN и сохранением памяти."""

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

        self.memory = ReplayBuffer(
            capacity=10000
        )

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.batch_size = 64

        self.steps = 0

        self.load_checkpoint()

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
        """Обучает модель."""

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
        """Обновляет target-модель."""

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

    def decay_epsilon(self):
        """Уменьшает случайность действий."""

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def save_checkpoint(self):
        """Безопасно сохраняет состояние обучения."""

        checkpoint = {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "memory": list(self.memory.buffer),
        }

        temp_path = "models/mars_checkpoint.tmp"
        final_path = "models/mars_checkpoint.pth"

        torch.save(
            checkpoint,
            temp_path
        )

        # Только после успешной записи
        # заменяем старый checkpoint новым.
        import os

        os.replace(
            temp_path,
            final_path
        )

        print(
            f"Checkpoint saved: {final_path}"
        )

    def load_checkpoint(self):
        """Загружает предыдущее состояние обучения."""

        checkpoint_path = "models/mars_checkpoint.pth"
        model_path = "models/mars_model.pth"

        # Сначала пытаемся загрузить полный checkpoint
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False
            )

            self.model.load_state_dict(
                checkpoint["model"]
            )

            self.target_model.load_state_dict(
                checkpoint["target_model"]
            )   

            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

            self.epsilon = checkpoint["epsilon"]
            self.steps = checkpoint["steps"]

            self.memory.buffer.extend(
                checkpoint["memory"]
            )

            print("AI checkpoint loaded")
            print(f"Steps: {self.steps}")
            print(f"Epsilon: {self.epsilon:.3f}")
            print(f"Memory: {len(self.memory)}")

            return

        except FileNotFoundError:
            pass

        # Если полного checkpoint нет,
        # загружаем хотя бы обученную модель
        try:
            state_dict = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True
            )

            self.model.load_state_dict(
                state_dict
            )

            self.target_model.load_state_dict(
                state_dict
            )

            # Модель уже обучена,
            # поэтому не начинаем с полностью
            # случайного поведения.
            self.epsilon = 0.10

            print("Trained model loaded")
            print(
                "Checkpoint memory is unavailable."
            )
            print(
                f"Epsilon: {self.epsilon:.3f}"
            )

        except FileNotFoundError:
            print(
                "No trained model found. "
                "Starting new training."
            )