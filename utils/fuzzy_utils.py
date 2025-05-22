import skfuzzy as fuzz
import numpy as np


GRID_SIZE = 20
def fuzzify_distance(dist):
    """
    Возвращает [close, medium, far] на основе расстояния и размера сетки
    :param dist: float — расстояние до цели/ресурса
    :param grid_size: int — размер стороны сетки (например, 30)
    :return: list[float] — значения принадлежности к нечетким множествам
    """
    # Диапазон расстояний: от 0 до max_dist
    max_dist = np.sqrt(2) * GRID_SIZE  # диагональ сетки
    x = np.arange(0, max_dist + 1, 0.1)

    # close: от 0 до ~grid_size / 5
    close_start, close_peak, close_end = 0, 0, GRID_SIZE / 5

    # medium: от close_end до grid_size / 2
    medium_start, medium_peak, medium_end = close_end, GRID_SIZE / 4, GRID_SIZE / 2

    # far: от medium_end до max_dist
    far_start, far_peak, far_end = medium_end, max_dist / 2, max_dist

    # Функции принадлежности
    close_mf = fuzz.trimf(x, [close_start, close_peak, close_end])
    medium_mf = fuzz.trimf(x, [medium_start, medium_peak, medium_end])
    far_mf = fuzz.trimf(x, [far_start, far_peak, far_end])

    # Интерполяция значений
    close = fuzz.interp_membership(x, close_mf, dist)
    medium = fuzz.interp_membership(x, medium_mf, dist)
    far = fuzz.interp_membership(x, far_mf, dist)

    return [close, medium, far]