import heapq


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_path(start, goal, env):
    """
    Возвращает путь от start до goal через A*
    :param start: tuple (x, y)
    :param goal: tuple (x, y)
    :param env: SimpleMultiAgentEnv или DeliveryMultiAgentEnv
    """
    grid = [[0 for _ in range(env.grid_size)] for _ in range(env.grid_size)]

    # Стены
    for wall in env.walls:
        x, y = wall
        grid[x][y] = 1

    # Цель
    def neighbors(node):
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        result = []
        x, y = node
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                if (nx, ny) not in env.walls:
                    result.append((nx, ny))
        return result

    # A* поиск
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next_node in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    # Восстанавливаем путь
    path = []
    if goal in came_from:
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
    return path