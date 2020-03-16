import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
device = torch.device("cpu")


class dualingArchitecture(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64):
        super(dualingArchitecture, self).__init__()
        self.num_actions = action_size
        fc3_1_size = fc3_2_size = 32
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3_1 = nn.Linear(fc2_size, fc3_1_size)
        self.fc4_1 = nn.Linear(fc3_1_size, 1)
        self.fc3_2 = nn.Linear(fc2_size, fc3_2_size)
        self.fc4_2 = nn.Linear(fc3_2_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        val = F.relu(self.fc3_1(x))
        val = self.fc4_1(val)
        adv = F.relu(self.fc3_2(x))
        adv = self.fc4_2(adv)
        action = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.num_actions)
        return action


class MemoryReplay:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        samples = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in samples if e])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class Solver:

    def __init__(self, state_size, action_size, seed, buffer_size, batch_size,
                 gamma_val, update_freq, learning_rate, tau_val):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma_val
        self.update_freq = update_freq
        self.tau = tau_val

        self.q_current = dualingArchitecture(self.state_size, self.action_size, seed).to(device)
        self.q_target = dualingArchitecture(self.state_size, self.action_size, seed).to(device)
        self.optimizer = optim.Adam(self.q_current.parameters(), lr=self.learning_rate)
        self.memory = MemoryReplay(self.action_size, self.buffer_size, self.batch_size, seed)
        self.delta_t = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.delta_t = (self.delta_t + 1) % self.update_freq
        if self.delta_t == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.dual_learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_current.eval()
        with torch.no_grad():
            action_values = self.q_current(state)
        self.q_current.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def dual_learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        q_argmax = self.q_current(next_states).detach()
        _, a_prime = q_argmax.max(1)
        q_targets_next = self.q_target(next_states).detach().gather(1, a_prime.unsqueeze(1))
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.q_current(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for target_param, local_param in zip(self.q_target.parameters(), self.q_current.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

