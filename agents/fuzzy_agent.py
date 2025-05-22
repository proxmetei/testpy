import numpy as np
from utils.fuzzy_utils import fuzzify_distance
import random

class FuzzyAgent:
    def __init__(self, genome):
        self.genome = genome  # словарь с весами/порогами правил

    def act(self, observation):
        action_directions = {
            0: (0, 0),
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1)
        }

        # Проверяем допустимые действия
        valid_actions = []
        for action, (dx, dy) in action_directions.items():
            cell = observation[dx + 1, dy + 1]
            if cell[0] == 0:  # нет стены
                valid_actions.append(action)

        if not valid_actions:
            return 0  # нельзя двигаться

        # Получаем ближайшее расстояние до цели/ресурса
        dist = self._get_closest_distance(observation)
        fuzzy_dist = fuzzify_distance(dist)
        # print(fuzzy_dist, self.genome)

        move_weights = [
            fuzzy_dist[0] * max(0, self.genome[0]),  # close
            fuzzy_dist[1] * max(0, self.genome[1]),  # medium
            fuzzy_dist[2] * max(0, self.genome[2])  # far
        ]

        best_action_idx = np.argmax(move_weights)
        action_map = {0: 1, 1: 2, 2: 4}
        action = action_map.get(best_action_idx, 0)

        if action in valid_actions:
            return action
        else:
            return random.choice(valid_actions)

    def _get_closest_distance(self, observation):
        """
        Возвращает минимальное расстояние до ближайшего ресурса или цели.
        :param observation: np.array shape=(5,5,3), наблюдение агента
        :return: float — нормализованное расстояние от 0 до 1
        """
        distances = []

        for dx in range(observation.shape[0]):
            for dy in range(observation.shape[1]):
                # Проверяем, есть ли ресурс или цель в этой ячейке
                if observation[dx, dy, 1] > 0 or observation[dx, dy, 2] > 0:
                    # Расстояние от центральной клетки (2,2)
                    distance = abs(dx - 2) + abs(dy - 2)  # манхэттенское расстояние
                    distances.append(distance)

        if distances:
            # Нормализуем расстояние от 0 до max_dist (например, до 4 в 5x5 окружении)
            max_possible_distance = 4  # манхэттенское расстояние от центра до угла в 5x5 окне
            normalized_distance = min(max(distances), max_possible_distance) / max_possible_distance
            return normalized_distance
        else:
            # Если нет целей/ресурсов — считаем, что объект очень далеко
            return 1.0  # 1.0 = максимально "далеко"