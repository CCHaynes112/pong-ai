# train_ai.py

import os
import numpy as np
import pickle
from game import PongGame

# Q-Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPISODES = 100000

# Discrete state buckets
STATE_BUCKETS = [40, 40, 3, 3, 40]  # Number of buckets for each state dimension
STATE_BOUNDS = [
    [0, 800],  # Ball x-position
    [0, 600],  # Ball y-position
    [-5, 5],  # Ball x-velocity
    [-5, 5],  # Ball y-velocity
    [0, 500],  # Paddle y-position
]


def discretize_state(state):
    discrete_state = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = STATE_BUCKETS[i] - 1
        else:
            ratio = (state[i] - STATE_BOUNDS[i][0]) / (STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0])
            bucket_index = int(ratio * (STATE_BUCKETS[i] - 1))
        discrete_state.append(bucket_index)
    return tuple(discrete_state)


def train():
    env = PongGame(render=False)
    q_table = np.random.uniform(low=-1, high=1, size=(STATE_BUCKETS + [3]))  # Actions: Up, Stay, Down

    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.9995
    min_epsilon = 0.01
    all_rewards = []

    for episode in range(EPISODES):
        state = discretize_state(env.reset())
        done = False
        total_reward = 0

        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(0, 3) - 1  # Random action: -1, 0, 1
            else:
                action = np.argmax(q_table[state]) - 1  # Best action from Q-table

            next_state, reward, done = env.step(action)
            next_state_discrete = discretize_state(next_state)

            # Update Q-value
            current_q = q_table[state + (action + 1,)]
            max_future_q = np.max(q_table[next_state_discrete])
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - current_q)
            q_table[state + (action + 1,)] = new_q

            state = next_state_discrete
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    # Save Q-table
    models_dir = "../models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    q_table_path = os.path.join(models_dir, "q_table.pkl")
    with open(q_table_path, "wb") as f:
        pickle.dump(q_table, f)
    print("Training completed and Q-table saved.")

    env.close()


if __name__ == "__main__":
    train()
