import pygame
import sys
import numpy as np
from env.real_env import DeliveryMultiAgentEnv
import torch
from viz.path_finding import get_path


CELL_SIZE = 20
GRID_SIZE = 30
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 2

COLORS = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (200, 0, 200), (255, 165, 0)]


def draw_grid(screen):
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (0, y), (WINDOW_SIZE, y))


def draw_static_objects(screen, env):
    # Стены
    for wall in env.walls:
        pygame.draw.rect(
            screen, (0, 0, 0),
            pygame.Rect(wall[1] * CELL_SIZE, wall[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )

    # Ресурсы
    for res in env.resource_positions:
        if res not in env.collected_resources:
            pygame.draw.circle(
                screen, (0, 255, 0),
                (res[1] * CELL_SIZE + CELL_SIZE // 2, res[0] * CELL_SIZE + CELL_SIZE // 2),
                10
            )

    # Цели
    for goal in env.goal_positions:
        pygame.draw.circle(
            screen, (0, 100, 255),
            (goal[1] * CELL_SIZE + CELL_SIZE // 2, goal[0] * CELL_SIZE + CELL_SIZE // 2),
            10
        )


def run_simulation(agent_nets, env):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    #env = DeliveryMultiAgentEnv(num_agents=len(agent_nets))
    observations = env.reset()

    running = True
    step = 0
    while running and step < 300:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        actions = {}
        for i, net in enumerate(agent_nets):
            with torch.no_grad():
                obs = observations[i]
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to("cpu")

                # Получаем коммуникацию от других агентов (как в evaluate_agent())
                #other_messages = [
                #    agent.get_comm_message() for j, agent in enumerate(agent_nets) if j != i
                #]

                # Фильтруем сообщения
                #valid_messages = []
                #for msg in other_messages:
                    #if isinstance(msg, torch.Tensor):
                        #while msg.dim() < 2:
                            #msg = msg.unsqueeze(0)
                        #while msg.dim() > 2:
                            #msg = msg.squeeze(0)
                    #valid_messages.append(msg)

                # Вызываем act(), передавая список сообщений, НЕ combined_message!
                action = net.act(obs_tensor, None)
                actions[i] = action

                # Вручную обновляем has_resource по позиции
                pos_tuple = tuple(map(int, env.agents_pos[i]))
                if pos_tuple in env.resource_positions and pos_tuple not in env.collected_resources:
                    net.has_resource = True
                if pos_tuple in env.goal_positions and net.has_resource:
                    net.has_resource = False

                print(f"Agent {i} | Action: {action}, Has resource: {net.has_resource}, Target: {net.target}")

        # Делаем шаг в среде
        next_obs, _, dones, _ = env.step(actions)
        observations = next_obs

        # Отрисовка
        screen.fill((255, 255, 255))
        draw_grid(screen)
        draw_static_objects(screen, env)

        for i in env.agents_pos:
            x, y = env.agents_pos[i]
            color = COLORS[i % len(COLORS)]
            if net.has_resource:
                color = (255, 0, 255)  # фиолетовый — несёт ресурс

            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(y * CELL_SIZE + 5, x * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10)
            )
            label = font.render(f"{i}", True, (0, 0, 0))
            screen.blit(label, (y * CELL_SIZE + 10, x * CELL_SIZE + 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

def select_action(q_values, avail_actions):
    q_values = q_values.copy()
    q_values[avail_actions == 0] = -np.inf
    return np.argmax(q_values)