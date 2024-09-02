# test_ai.py

import numpy as np
import pickle
import time
from game import PongGame
from train_ai import STATE_BOUNDS, STATE_BUCKETS

# Load Q-table
with open("../models/q_table.pkl", "rb") as f:
    q_table = pickle.load(f)


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


def test(episodes=10):
    env = PongGame(render=True)
    total_rewards = []

    for episode in range(episodes):
        state = discretize_state(env.reset())
        done = False
        total_reward = 0

        while not done:
            action = np.argmax(q_table[state]) - 1  # Best action from Q-table
            next_state, reward, done = env.step(action)
            state = discretize_state(next_state)
            total_reward += reward
            time.sleep(0.01)  # Slow down the game for visualization

        total_rewards.append(total_reward)
        print(f"Episode: {episode + 1}, Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")

    env.close()


if __name__ == "__main__":
    test()
