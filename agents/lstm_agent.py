import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTMAgent:
    def __init__(self, obs_shape=(5, 5, 5), n_actions=5, comm_size=0):
        self.device = "cpu"
        self.policy = LSTMPolicy(obs_shape, n_actions, comm_size=comm_size).to(self.device)
        self.hx = None
        self.cx = None
        self.has_resource = False
        self.target = None  # может быть 'resource' или 'goal'
        self.comm_vector = torch.zeros((1, comm_size))  # сообщение другим агентам
        self.last_action = None
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def act(self, observation, comm_messages=None):
        with torch.no_grad():
            # Преобразуем наблюдение в тензор
            if isinstance(observation, np.ndarray):
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            else:
                obs_tensor = observation.clone().detach().to(self.device)

            # Приводим к нужной форме (batch_size, seq_len, C, H, W)
            if obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.unsqueeze(0)  # [C, H, W] → [1, C, H, W]
            if obs_tensor.dim() == 4 and obs_tensor.shape[0] == 1:
                obs_tensor = obs_tensor.unsqueeze(0)  # [1, C, H, W] → [1, 1, C, H, W]

            # Анализируем наблюдение
            if isinstance(observation, torch.Tensor):
                obs_np = observation.cpu().numpy()
            else:
                obs_np = observation.copy()

            while obs_np.ndim < 3:
                obs_np = obs_np[np.newaxis]

            has_resource_in_view = False
            has_goal_in_view = False

            if obs_np.shape[0] >= 2:
                has_resource_in_view = np.any(obs_np[1] > 0)
            if obs_np.shape[0] >= 3:
                has_goal_in_view = np.any(obs_np[2] > 0)

            # Обновляем цель
            if not self.has_resource and has_resource_in_view:
                self.target = "resource"
            elif self.has_resource and has_goal_in_view:
                self.target = "goal"
            elif self.has_resource:
                self.target = "goal"
            else:
                self.target = "resource"

            # Теперь НЕ передаём коммуникацию
            logits, _, (hx, cx) = self.policy(obs_tensor, self.hx, self.cx, None)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()

            # Сохраняем внутреннее состояние LSTM
            self.hx, self.cx = hx, cx
            return action
    def reset_memory(self):
        """Сброс внутреннего состояния"""
        self.hx = None
        self.cx = None
        self.has_resource = False
        self.target = None

    def get_comm_message(self):
        msg = self.comm_vector.clone()
        if self.has_resource:
            msg[0, 0] = 1.0
        if self.target == "goal":
            msg[0, 1] = 1.0
        return msg  # shape == (1, 5)

    def clone(self):
        new_agent = LSTMAgent(obs_shape=(5, 5, 5), n_actions=5, comm_size=self.comm_vector.shape[-1])
        new_agent.policy.load_state_dict(self.policy.state_dict())
        new_agent.comm_vector = self.comm_vector.clone().detach()

        # Клонирование оптимизатора
        new_agent.optimizer = torch.optim.Adam(new_agent.policy.parameters(), lr=1e-3)
        old_state = self.optimizer.state_dict()
        try:
            new_agent.optimizer.load_state_dict(old_state)
        except Exception as e:
            print("⚠️ Optimizer state couldn't be loaded:", e)

        return new_agent


class LSTMPolicy(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_size=64, comm_size=0):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.comm_size = comm_size

        # CNN энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 3 * 3, hidden_size),
            nn.ReLU()
        )

        # LSTM
        self.lstm = nn.LSTM(hidden_size + comm_size, hidden_size, batch_first=True)

        # Выходные головы
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hx=None, cx=None, communication=None):
        """
        :param x: shape=(batch_size, seq_len, C, H, W)
        :param communication: Tensor или None
        :return: logits, value, (hx, cx)
        """
        # Убедимся, что x имеет нужную форму
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 4:
            x = x.unsqueeze(0)

        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.encoder(x)
        x = x.view(batch_size, seq_len, -1)

        # Если есть коммуникация — добавляем её
        #if communication is not None:
         #   while communication.dim() < 2:
          #      communication = communication.unsqueeze(0)
           # while communication.dim() > 2:
            #    communication = communication.squeeze(0)
            #communication = communication.expand(x.size(0), -1).unsqueeze(1).expand(-1, seq_len, -1)
            #x = torch.cat([x, communication], dim=-1)

        # Вызываем LSTM
        if hx is None:
            hx = torch.zeros(1, x.size(0), self.hidden_size)
            cx = torch.zeros(1, x.size(0), self.hidden_size)

        out, (hx, cx) = self.lstm(x, (hx, cx))

        # Головы действий и ценности
        logits = self.actor(out[:, -1])
        value = self.critic(out[:, -1])

        return logits, value, (hx.detach(), cx.detach())