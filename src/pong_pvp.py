import pickle
import pygame
import numpy as np

from game import PongGame
from test_ai import discretize_state

with open("../models/q_table.pkl", "rb") as f:
    q_table = pickle.load(f)


class PongGameWithPlayer(PongGame):
    def __init__(self, render=True, is_pvp=False):
        super().__init__(render, is_pvp)

    def handle_player_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and self.enemy_paddle.top > 0:
            self.enemy_paddle.y -= self.PADDLE_SPEED
        if keys[pygame.K_DOWN] and self.enemy_paddle.bottom < self.HEIGHT:
            self.enemy_paddle.y += self.PADDLE_SPEED

    def play(self):
        while True:
            self.handle_player_input()

            # AI action
            state = self.get_state()  # Get the current state for the AI
            action = self.ai_choose_action(state)  # Choose the AI's action
            self.step_ai(action)  # AI moves based on chosen action

            if self.render:
                self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            pygame.time.Clock().tick(60)

    def ai_choose_action(self, state):
        # Use the Q-table to choose the best action based on the state
        # This assumes you have loaded the Q-table into the variable q_table
        # and you have discretized the state space appropriately
        state_discrete = discretize_state(state)
        action = np.argmax(q_table[state_discrete]) - 1  # Best action from Q-table
        return action

    def step_ai(self, action):
        # This is the existing AI step function, where the AI paddle moves
        if action == -1:
            self.ai_paddle.y -= self.PADDLE_SPEED
        elif action == 1:
            self.ai_paddle.y += self.PADDLE_SPEED

        # Call the existing step function to update the game state
        next_state, reward, done = self.step(action)


# Main game loop to play against the AI
if __name__ == "__main__":
    game = PongGameWithPlayer(render=True, is_pvp=True)
    game.play()
