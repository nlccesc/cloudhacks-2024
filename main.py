import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from bandit import NeuralBandit
from si import RSO, enhanced_cost_func
from ec import differential_evolution_optimization
from ddos import simulate_ddos_attack
from sklearn.preprocessing import StandardScaler

def monitor_server_performance():
    simulated_response_time = np.random.uniform(1, 5)  # Simulate response time between 1 and 5 seconds
    return simulated_response_time

def simulate_ddos_mitigation():
    n_arms = 2  # Swarm Intelligence and Evolutionary Computation
    context_dim = 3  # Number of context features
    bandit = NeuralBandit(n_arms, context_dim)
    rewards = []
    chosen_arms = []
    epsilon = 1.0  # Initial epsilon for exploration
    epsilon_decay = 0.99  # Decay rate for epsilon
    epsilon_min = 0.01  # Minimum value of epsilon
    scaler = StandardScaler()

    for iteration in range(100):  # Increase the number of iterations
        context = np.random.rand(context_dim)  # Simulate random context
        context = scaler.fit_transform(context.reshape(-1, 1)).flatten()  # Normalize context

        chosen_arm = bandit.select_arm(context, epsilon)
        chosen_arms.append(chosen_arm)

        if chosen_arm == 0:
            rso = RSO(enhanced_cost_func, bounds=[(0, 10)] * 2, num_particles=50, max_iter=200)
            rso.optimize()
            params, score = rso.get_best_solution()
        else:
            params, score = differential_evolution_optimization(enhanced_cost_func, [(0, 10)] * 2)

        if np.isnan(score) or np.isinf(score):
            print(f"Invalid score detected at iteration {iteration} for arm {chosen_arm}.")
            score = 10  # Assign a high score for invalid results

        reward = -score  # In a real scenario, this would be based on actual server response
        rewards.append(reward)

        bandit.update(context, chosen_arm, reward)
        print(f'Chosen Arm: {chosen_arm}, Reward: {reward}')

        # Monitor server performance after applying the mitigation strategy
        response_time = monitor_server_performance()
        print(f"Iteration {iteration}: Server response time = {response_time:.2f} seconds")

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Ensure plotting runs in the main thread
    plot_results(rewards, chosen_arms)
    evaluate_performance(rewards, chosen_arms, n_arms)

def plot_results(rewards, chosen_arms):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Rewards')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.title('Rewards Over Iterations')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(chosen_arms, label='Chosen Arm', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('Chosen Arm')
    plt.title('Chosen Arms Over Iterations')
    plt.legend()
    plt.show()

def evaluate_performance(rewards, chosen_arms, n_arms):
    avg_reward = sum(rewards) / len(rewards)
    cumulative_reward = sum(rewards)
    selection_count = [chosen_arms.count(i) for i in range(n_arms)]
    std_dev_reward = np.std(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)

    print(f'Average Reward: {avg_reward}')
    print(f'Cumulative Reward: {cumulative_reward}')
    print(f'Selection Count: {selection_count}')
    print(f'Standard Deviation of Rewards: {std_dev_reward}')
    print(f'Maximum Reward: {max_reward}')
    print(f'Minimum Reward: {min_reward}')

if __name__ == "__main__":
    target_urls = [
        'http://example.com/api/v1/resource1',
        'http://example.com/api/v1/resource2',
        'http://example.com/api/v1/resource3'
    ]
    total_requests = 10000  # Total number of requests
    num_threads = 100  # Number of concurrent threads

    # Start DDoS attack simulation in a separate thread
    ddos_thread = threading.Thread(target=simulate_ddos_attack, args=(target_urls, total_requests, num_threads))
    ddos_thread.start()

    # Start DDoS mitigation simulation
    simulate_ddos_mitigation()

    # Wait for the DDoS attack simulation to finish
    ddos_thread.join()