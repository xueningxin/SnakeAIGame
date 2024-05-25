import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from REINFORCE import PolicyNetwork, REINFORCETrainer
from helper import plot
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

MAX_MEMORY = 100_000
BATCH_SIZE = 500
LR = 0.01
EPSILON = 0.2
class BaselineNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaselineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return mini_batch  # This should already be a list of tuples
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9
        self.memory = ReplayBuffer(MAX_MEMORY)
        self.model = PolicyNetwork(11, 256, 3)
        self.trainer = REINFORCETrainer(self.model, lr=LR)
        self.baseline_model = BaselineNetwork(11, 256, 1)
        self.baseline_optimizer = optim.Adam(self.baseline_model.parameters(), lr=LR)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_long_memory(self):
        mini_sample = []
        if len(self.memory) > BATCH_SIZE:
            mini_sample = self.memory.sample(BATCH_SIZE)
        else:
            mini_sample = list(self.memory.buffer)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.rewards.extend(rewards)
        self.trainer.train_step()

    def get_action(self, state):
        if random.random() < EPSILON:
            return [random.randint(0, 2) for _ in range(3)]
        else:
            action = self.trainer.select_action(state)
            final_move = [0, 0, 0]
            final_move[action] = 1
            return final_move

    def calculate_distance_reward(self, game):
        head = game.snake[0]
        food = game.food
        distance = abs(head.x - food.x) + abs(head.y - food.y)
        return -distance

    def calculate_food_approach_reward(self, game, prev_distance, current_distance):
        if current_distance < prev_distance:
            return (prev_distance - current_distance) * 100
        return 0

    def calculate_wall_penalty(self, game):
        head = game.snake[0]
        min_distance = min(head.x, game.w - head.x, head.y, game.h - head.y)
        if min_distance < 20:
            return -20 * (20 - min_distance)
        return 0

    def calculate_reward(self, game, done, prev_distance, current_distance):
        if done:
            return -100
        reward = 10
        if game.head == game.food:
            reward += 100
        reward += self.calculate_food_approach_reward(game, prev_distance, current_distance)
        reward += self.calculate_distance_reward(game)
        reward += self.calculate_wall_penalty(game)
        return reward

    def train_baseline(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = self.memory.sample(BATCH_SIZE)
        else:
            mini_sample = list(self.memory.buffer)  # Make sure this converts the buffer to a list of tuples

        if mini_sample:  # Ensure that mini_sample is not empty
            states, actions, rewards, next_states, dones = zip(*mini_sample)  # Correct the unpacking
            states = torch.tensor(states, dtype=torch.float)
            rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1)
            predicted_values = self.baseline_model(states)
            baseline_loss = F.mse_loss(predicted_values, rewards)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
        else:
            print("No samples to train on")


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    prev_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        current_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
        reward = agent.calculate_reward(game, done, prev_distance, current_distance)
        prev_distance = current_distance

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            agent.train_long_memory()
            agent.train_baseline()
            agent.trainer.rewards = []
            game.reset()
            agent.n_games += 1
            total_score += score
            mean_score = total_score / agent.n_games
            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            if score > record:
                record = score


if __name__ == '__main__':
    train()