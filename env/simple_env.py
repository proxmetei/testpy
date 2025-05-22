import numpy as np
from gymnasium import Env, spaces


class SimpleMultiAgentEnv(Env):
    def __init__(self, num_agents=5, grid_size=30):
        super().__init__()
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.agents_pos = {}
        self.resources = []
        self.goals = []
        self.walls = []

        self._reset_positions()

        # Наблюдение: (3, 5, 5) — каналы [стена, ресурс, цель]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 5, 5), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    def _reset_positions(self):
        num_resources = 5
        num_goals = 2
        num_walls = 10  # можешь менять по желанию

        total_objects = self.num_agents + num_resources + num_goals + num_walls
        all_positions = []

        # Генерируем уникальные позиции
        while len(all_positions) < total_objects:
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if pos not in all_positions:
                all_positions.append(pos)

        # Распределяем объекты
        self.agents_pos = {i: np.array(all_positions[i]) for i in range(self.num_agents)}

        start = self.num_agents
        self.resources = all_positions[start:start + num_resources]
        start += num_resources
        self.goals = all_positions[start:start + num_goals]
        start += num_goals
        self.walls = all_positions[start:start + num_walls]

    def reset(self):
        self._reset_positions()
        return {i: self._get_obs(i) for i in range(self.num_agents)}

    def step(self, actions):
        rewards = {}
        dones = {i: False for i in range(self.num_agents)}

        for i, action in actions.items():
            self._move_agent(i, action)

        observations = {i: self._get_obs(i) for i in range(self.num_agents)}
        rewards = {i: self._calculate_reward(i) for i in range(self.num_agents)}
        return observations, rewards, dones, {}

    def _calculate_reward(self, agent_id):
        pos = tuple(self.agents_pos[agent_id])
        reward = 0

        if pos in self.resources:
            reward += 100  # усиленная награда
            self.resources.remove(pos)
        elif pos in self.goals:
            reward += 200  # большая награда за цель
        else:
            reward -= 0.1  # штраф за бездействие

        return reward

    def _move_agent(self, i, action):
        x, y = self.agents_pos[i]
        dx, dy = 0, 0
        if action == 1: dx = -1
        elif action == 2: dx = 1
        elif action == 3: dy = -1
        elif action == 4: dy = 1

        new_pos = np.array([x + dx, y + dy])
        if (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size and
            tuple(new_pos) not in self.walls):
            self.agents_pos[i] = new_pos

    def _get_obs(self, agent_id):
        x, y = self.agents_pos[agent_id]
        obs = np.zeros((3, 5, 5), dtype=np.float32)

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                abs_dx, abs_dy = dx + 2, dy + 2
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) in self.walls:
                        obs[0, abs_dx, abs_dy] = 1.0
                    if (nx, ny) in self.resources:
                        obs[1, abs_dx, abs_dy] = 1.0
                    if (nx, ny) in self.goals:
                        obs[2, abs_dx, abs_dy] = 1.0
                else:
                    obs[0, abs_dx, abs_dy] = 1.0  # стена за пределами

        return obs

    def get_avail_agent_actions(self, agent_id):
        x, y = self.agents_pos[agent_id]
        actions = [1] * 5  # все действия доступны по умолчанию
        directions = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

        for i, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                actions[i] = 0
            elif (nx, ny) in self.walls:
                actions[i] = 0
        return np.array(actions)