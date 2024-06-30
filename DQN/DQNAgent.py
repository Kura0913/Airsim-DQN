from DQN.DQNNet import DQNNet
from DQN.ReplayBuffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
    def __init__(self, state_dim, action_dim, buffer_capacity = 10000, bacth_size = 64, gamma = 0.99, lr = 1e-3):
        self.policy_net = DQNNet(state_dim, action_dim)
        self.target_net = DQNNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = 1e-3)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_capacity)
        self.batch_size = bacth_size
        self.gamma = gamma
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            action = self.policy_net(state).numpy()

        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        state = torch.tensor(batch.state, dtype=torch.float32)
        action = torch.tensor(batch.action, dtype=torch.float32)
        reward = torch.tensor(batch.reward, dtype=torch.float32)
        next_state = torch.tensor(batch.next_state, dtype=torch.float32)
        done = torch.tensor(batch.done, dtype=torch.float32)

        q_values = self.policy_net(state).gather(1, action.long().unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, name):
        self.policy_net.load_state_dict(torch.load(name))
        self.update_target_model()