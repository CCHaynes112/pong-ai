import pygame
import random
import numpy as np


class PongGame:
    def __init__(self, render=True, is_pvp=False):
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.BALL_RADIUS = 10
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 10, 100
        self.PADDLE_SPEED = 6
        self.render = render
        self.is_pvp = is_pvp

        if self.render:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Pong AI")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.ball = pygame.Rect(
            self.WIDTH // 2,
            self.HEIGHT // 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )
        self.enemy_paddle = pygame.Rect(
            self.WIDTH - 20,
            self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ai_paddle = pygame.Rect(
            10,
            self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_speed = [-5, random.choice((5, -5))]
        self.done = False
        return self.get_state()

    def get_state(self):
        state = (
            self.ball.x,
            self.ball.y,
            self.ball_speed[0],
            self.ball_speed[1],
            self.ai_paddle.y,
        )
        return np.array(state, dtype=float)

    def step(self, action):
        if not self.is_pvp:
            # Move the enemy paddle to follow the ball
            self.enemy_paddle.y = self.ball.y - self.PADDLE_HEIGHT // 2

        # Action: -1 (move up), 0 (stay), 1 (move down)
        if action == -1 and self.ai_paddle.top > 0:
            self.ai_paddle.y -= self.PADDLE_SPEED
        elif action == 1 and self.ai_paddle.bottom < self.HEIGHT:
            self.ai_paddle.y += self.PADDLE_SPEED

        # Move the ball
        self.ball.x += self.ball_speed[0]
        self.ball.y += self.ball_speed[1]

        # Ball collision with top and bottom
        if self.ball.top <= 0 or self.ball.bottom >= self.HEIGHT:
            self.ball_speed[1] = -self.ball_speed[1]

        # Ball collision with paddles
        if self.ball.colliderect(self.enemy_paddle) or self.ball.colliderect(self.ai_paddle):
            self.ball_speed[0] = -self.ball_speed[0]

        reward = self.calculate_reward()

        if self.render:
            self.draw()

        return self.get_state(), reward, self.done

    def calculate_reward(self):
        # Get the distance between the ball and the paddle
        distance_y = abs(self.ai_paddle.y - self.ball.y)  # Use absolute value for distance

        # Reward is inversely proportional to the distance, but avoid very large rewards
        reward = 1 / (distance_y + 1)  # Add 1 to avoid division by zero and to keep rewards manageable

        # Penalty for missing the ball
        if self.ball.left <= 0:
            reward = -10
            self.done = True
        # Reward for scoring
        elif self.ball.right >= self.WIDTH:
            reward = 10
            self.done = True

        return reward

    def draw(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
        pygame.draw.rect(self.screen, (255, 0, 0), self.enemy_paddle)
        pygame.draw.rect(self.screen, (0, 0, 255), self.ai_paddle)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
