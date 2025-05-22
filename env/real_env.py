from random import random

import numpy as np
from typing import List, Tuple, Dict


class DeliveryMultiAgentEnv:
    def __init__(self, num_agents=5, grid_size=20, max_steps=300, n_resources=10, n_goals=5, seed=None):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_resources = n_resources
        self.n_goals = n_goals

        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Состояние агентов
        self.has_resource = {i: False for i in range(self.num_agents)}

        # Цели и время их использования
        self.goal_use_step = {}  # { (x,y): last_used_step }

        # Запускаем сброс среды
        self._reset()

    def _reset(self):
        """Создаем новую карту"""
        self.steps = 0
        positions = []

        while len(positions) < self.num_agents + self.n_resources + self.n_goals + 10:
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if pos not in positions:
                positions.append(pos)

        # Позиции агентов
        self.agents_pos = {
            i: np.array(positions[i]) for i in range(self.num_agents)
        }
        start = self.num_agents

        # Ресурсы
        self.resource_positions = [tuple(positions[start + r]) for r in range(self.n_resources)]
        start += self.n_resources

        # Цели
        self.goal_positions = [tuple(positions[start + g]) for g in range(self.n_goals)]
        start += self.n_goals

        # Стены
        self.walls = [tuple(positions[start + w]) for w in range(10)]

        # Какие ресурсы уже собраны
        self.collected_resources = []

        # Когда была занята цель
        self.goal_use_step = {goal: None for goal in self.goal_positions}

    def reset(self):
        """Полный перезапуск среды"""
        self._reset()
        return [self._get_obs(i) for i in range(self.num_agents)]

    def get_obs(self):
        return [self._get_obs(i) for i in range(self.num_agents)]

    def step(self, actions):
        rewards = {}
        dones = {}

        for i in range(self.num_agents):
            x, y = self.agents_pos[i]
            dx, dy = 0, 0
            reward = -0.01  # базовый штраф за шаг

            action = actions.get(i, 0)
            if action == 1:
                dx = -1  # вверх
            elif action == 2:
                dx = 1  # вниз
            elif action == 3:
                dy = -1  # влево
            elif action == 4:
                dy = 1  # вправо

            nx, ny = x + dx, y + dy
            if (nx, ny) in self.walls or not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                reward -= 0.5  # штраф за столкновение со стеной
            else:
                self.agents_pos[i] = np.array([nx, ny])

            new_pos = tuple(map(int, self.agents_pos[i]))
            done = False

            # Собрал ли агент ресурс?
            if not self.has_resource[
                i] and new_pos in self.resource_positions and new_pos not in self.collected_resources:
                #print(f"✅ Агент {i} собрал ресурс!")
                self.collected_resources.append(new_pos)
                self.has_resource[i] = True
                reward += 100

            # Доставил ли агент ресурс к цели?
            if self.has_resource[i] and new_pos in self.goal_positions:
                goal_idx = self.goal_positions.index(new_pos)
                last_used = self.goal_use_step[new_pos]

                if last_used is None or (self.steps - last_used) > 10:
                    #print(f"🎯 Агент {i} доставил ресурс к цели!")
                    self.goal_use_step[new_pos] = self.steps
                    self.has_resource[i] = False
                    reward += 300
                    done = True
                else:
                    reward -= 1.0  # штраф за попытку использовать занятую цель

            rewards[i] = reward
            dones[i] = done

        self.steps += 1
        done = self.steps >= self.max_steps or all(dones.values())
        return [self._get_obs(i) for i in range(self.num_agents)], rewards, dones, {}

    def _get_obs(self, agent_id):
        """
        Возвращает локальное наблюдение: 5x5 вокруг агента
        Каналы:
        - 0: стены
        - 1: доступные ресурсы
        - 2: доступные цели
        - 3: другие агенты
        """
        x, y = self.agents_pos[agent_id]
        obs = np.zeros((5, 5, 5), dtype=np.float32)

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                abs_dx, abs_dy = dx + 2, dy + 2

                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # Стены
                    if (nx, ny) in self.walls:
                        obs[0, abs_dx, abs_dy] = 1.0

                    # Ресурсы
                    if (nx, ny) in self.resource_positions and (nx, ny) not in self.collected_resources:
                        obs[1, abs_dx, abs_dy] = 1.0

                    # Цели (доступные)
                    if (nx, ny) in self.goal_positions:
                        last_used = self.goal_use_step[(nx, ny)]
                        if last_used is None or (self.steps - last_used) > 10:
                            obs[2, abs_dx, abs_dy] = 1.0

                    # Другие агенты
                    for j in range(self.num_agents):
                        if j != agent_id:
                            other_pos = tuple(self.agents_pos[j])
                            if other_pos == (nx, ny):
                                obs[3, abs_dx, abs_dy] = 1.0
                else:
                    # За пределами карты → граница
                    obs[0, abs_dx, abs_dy] = 1.0

        if self.has_resource[agent_id]:
            obs[4, 2, 2] = 1.0
        return obs