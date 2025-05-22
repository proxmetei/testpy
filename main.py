from evolution.evolutionary_loop import toolbox
from viz.simulation_viewer import run_simulation
import random
import torch
import numpy as np
from env.simple_env import SimpleMultiAgentEnv
from env.real_env import DeliveryMultiAgentEnv
from agents.lstm_agent import LSTMAgent, LSTMPolicy
import os
import torch.nn.functional as F

device = "cpu"
obs_shape = (3, 5, 5)
n_actions = 5
n_agents = 5
population_amount = 100
noise_level = 0.06
elite_amount = 50
mutant_amount = 10
NGEN = 10


def create_population(pop_size):
    return [LSTMAgent(obs_shape, n_actions) for _ in range(pop_size)]


def evaluate_agent(agent_nets, env, n_episodes=6, gamma=0.99):
    total_reward = 0.0
    for ep in range(n_episodes):
        env.reset()
        episode_reward = [0.0 for _ in range(n_agents)]
        done = False
        step = 0

        # Сброс LSTM
        for net in agent_nets:
            net.reset_memory()

        transitions = [[] for _ in range(len(agent_nets))]

        while not done and step < 100:
            obs = env.get_obs()
            actions = {}
            hx_cx_per_agent = [(net.hx, net.cx) for net in agent_nets]

            for i, net in enumerate(agent_nets):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                    if obs_tensor.dim() == 3:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    if obs_tensor.dim() == 4 and obs_tensor.shape[0] == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)

                    other_messages = [
                        agent.get_comm_message() for j, agent in enumerate(agent_nets) if j != i
                    ]
                    if other_messages:
                        valid_messages = torch.stack(other_messages).mean(dim=0, keepdim=True)
                    else:
                        valid_messages = None

                    logits, value, (hx, cx) = net.policy(
                        obs_tensor, net.hx, net.cx, valid_messages
                    )
                    probs = F.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()

                    transitions[i].append({
                        "state": obs_tensor,
                        "action": action,
                        "logits": logits,
                        "value": value,
                        "hx": net.hx,
                        "cx": net.cx,
                        "comm": valid_messages
                    })

                    net.hx, net.cx = hx, cx
                    actions[i] = action

            next_obs, rewards, dones, _ = env.step(actions)
            for i in range(n_agents):
                episode_reward[i] += rewards[i]

            for i in range(len(agent_nets)):
                transitions[i][-1]["reward"] = rewards[i]
                transitions[i][-1]["done"] = dones[i]

            obs = next_obs
            step += 1
            done = all(dones.values())

        # Обучение по переходам
        for i in range(len(agent_nets)):
            optimizer = agent_nets[i].optimizer
            optimizer.zero_grad()
            loss_total = 0.0

            for t in transitions[i]:
                state = t["state"]
                action = t["action"]
                reward = t["reward"]
                done = t["done"]
                comm = None

                logits, value, (hx, cx) = agent_nets[i].policy(state, t["hx"], t["cx"], comm)

                with torch.no_grad():
                    next_state = torch.FloatTensor(next_obs[i]).unsqueeze(0)
                    if next_state.dim() == 3:
                        next_state = next_state.unsqueeze(0)
                    if next_state.dim() == 4 and next_state.shape[0] == 1:
                        next_state = next_state.unsqueeze(0)
                    _, next_value, _ = agent_nets[i].policy(next_state, hx, cx, comm)

                target_value = reward + (1 - done) * gamma * next_value
                advantage = target_value - value

                log_probs = F.log_softmax(logits, dim=-1)[0, action]
                actor_loss = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss
                loss.backward()
                loss_total += loss.item()

            optimizer.step()

        #print(f"Episode {ep+1}: Fitness = {episode_reward}")
        total_reward += sum(episode_reward)

    return [r / n_episodes for r in episode_reward]





def mutate(agent, noise_level=0.02):
    with torch.no_grad():
        for param in agent.policy.parameters():
            if len(param.shape) > 1:
                noise = torch.randn_like(param) * noise_level
                param.add_(noise)


def evolve_population(population, fitnesses, elite_amount, mutant_amount, noise_level):
    # Сортировка и отбор элит
    ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    elites = [net.clone() for net, _ in ranked[:elite_amount]]

    # Скрещивание: родитель + мутация
    mutants = []
    for _ in range(mutant_amount):
        parent = random.choice(elites)
        child = parent.clone()
        mutate(child, noise_level)
        mutants.append(child)

    # Остальные — от элитных родителей с мутацией
    rest_needed = len(population) - len(elites) - len(mutants)
    rest = []
    for _ in range(rest_needed):
        parent = random.choice(elites)
        child = parent.clone()
        mutate(child, noise_level * 0.5)  # чуть слабее
        rest.append(child)

    return elites + mutants + rest




def run(load_weights=False):
    os.makedirs("checkpoints", exist_ok=True)

    # Загрузка или инициализация популяции
    populations = []
    for a in range(n_agents):
        if load_weights:
            try:
                net = LSTMAgent(obs_shape = (5, 5, 5), n_actions=5)  # <-- укажи размеры
                net.policy.load_state_dict(torch.load(f"checkpoints/agent_{a}_best.pt"))
                populations.append([net.clone() for _ in range(population_amount)])
                print(f"✅ Загружены веса для агента {a}")
            except FileNotFoundError:
                print(f"⚠️ Нет сохранённых весов для агента {a}, создаю новую популяцию")
                populations.append(create_population(population_amount))
        else:
            populations.append(create_population(population_amount))

    fitness_history = []

    for gen in range(NGEN):
        print(f"\nПоколение {gen + 1}/{NGEN}")
        fitness_per_agent = [[] for _ in range(n_agents)]

        for pop_idx in range(population_amount):
            env = DeliveryMultiAgentEnv(num_agents=n_agents, seed=42)  # фиксированная карта
            nets = [populations[a][pop_idx] for a in range(n_agents)]
            fitnesses = evaluate_agent(nets, env, n_episodes=3)

            for a in range(n_agents):
                fitness_per_agent[a].append(fitnesses[a])

        mean_fitness = [np.mean(fits) for fits in fitness_per_agent]
        max_fitness = [np.max(fits) for fits in fitness_per_agent]
        fitness_history.append({"mean": mean_fitness, "max": max_fitness})

        print("-" * 40)
        print(f"{'Агент':<6} | {'Средний фитнес':<17} | {'Макс. фитнес'}")
        print("-" * 40)
        for a in range(n_agents):
            print(f"{a:<6} | {mean_fitness[a]:<17.2f} | {max_fitness[a]:.2f}")

        # Эволюция
        for a in range(n_agents):
            populations[a] = evolve_population(
                populations[a],
                fitness_per_agent[a],
                elite_amount,
                mutant_amount,
                noise_level
            )

        # Сохраняем лучших особей после поколения
        for a in range(n_agents):
            best_idx = int(np.argmax(fitness_per_agent[a]))
            best_net = populations[a][best_idx]
            torch.save(best_net.policy.state_dict(), f"checkpoints/agent_{a}_best.pt")

    print("\nЛучшие особи:")
    best_nets = []
    for a in range(n_agents):
        best_idx = int(np.argmax(fitness_per_agent[a]))
        #best_idx = 0
        best_net = populations[a][best_idx]
        best_nets.append(best_net)
        print(f"Агент {a}: Фитнес = {fitness_per_agent[a][best_idx]:.2f}")

    env = DeliveryMultiAgentEnv(num_agents=n_agents, seed=42)
    run_simulation(best_nets, env)



if __name__ == "__main__":
    run()