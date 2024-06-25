import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import BertModel, BertTokenizer
from collections import namedtuple, deque
import numpy as np

# Experience named tuple for clarity
Experience = namedtuple('Experience', ('context', 'action', 'reward', 'next_context'))

class DuelingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class PrioritizedExperienceReplay:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, experience, error):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
        
        self.memory[self.position] = experience
        self.priorities[self.position] = (abs(error) + 1e-5) ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
        else:
            probabilities = np.ones(len(self.memory)) / len(self.memory)
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[i] for i in indices]
        
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        
        return samples, weights, indices

    def update_priorities(self, batch_indices, batch_errors):
        for idx, error in zip(batch_indices, batch_errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.memory)

class NeuralBandit:
    def __init__(self, n_arms, context_dim, replay_buffer_size=1000, batch_size=32, gamma=0.99, lr=0.001):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.replay_buffer = PrioritizedExperienceReplay(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.model = DuelingNetwork(768, n_arms)  # BERT embeddings have 768 dimensions
        self.target_model = DuelingNetwork(768, n_arms)  # Target network for Double Q-Learning
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.update_target_network()  # Initialize target network

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_context_embedding(self, context):
        context_str = " ".join(map(str, context))
        tokens = self.tokenizer(context_str, return_tensors='pt')
        with torch.no_grad():
            embeddings = self.bert_model(**tokens).last_hidden_state[:, 0, :]
        return embeddings

    def select_arm(self, context, epsilon):
        context_embedding = self.get_context_embedding(context)
        if random.random() < epsilon:
            return random.randrange(self.n_arms)
        else:
            with torch.no_grad():
                q_values = self.model(context_embedding)
                return q_values.argmax().item()

    def update(self, context, action, reward, next_context):
        context_embedding = self.get_context_embedding(context)
        next_context_embedding = self.get_context_embedding(next_context)
        q_values = self.model(context_embedding).detach().squeeze()
        next_q_values = self.target_model(next_context_embedding).detach().squeeze()
        target = reward + self.gamma * next_q_values.max()

        td_error = abs(target - q_values[action])
        self.replay_buffer.push(Experience(context_embedding, action, reward, next_context_embedding), td_error)
        if len(self.replay_buffer) >= self.batch_size:
            self.train()

    def train(self, beta=0.4):
        batch, weights, indices = self.replay_buffer.sample(self.batch_size, beta)
        contexts, actions, rewards, next_contexts = zip(*batch)

        contexts = torch.cat(contexts, dim=0)
        next_contexts = torch.cat(next_contexts, dim=0)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        q_values = self.model(contexts)
        next_q_values = self.target_model(next_contexts).detach()
        
        target_q_values = rewards + self.gamma * next_q_values.max(dim=1)[0]
        td_errors = (q_values.gather(1, actions.unsqueeze(1)).squeeze() - target_q_values).detach().cpu().numpy()

        loss = (weights * (q_values.gather(1, actions.unsqueeze(1)).squeeze() - target_q_values) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors)

        # Update target network periodically
        self.update_target_network()

# Example usage
if __name__ == "__main__":
    bandit = NeuralBandit(n_arms=2, context_dim=3)
    context = [1, 2, 3]
    next_context = [2, 3, 4]
    action = bandit.select_arm(context, epsilon=0.1)
    bandit.update(context, action, reward=1.0, next_context=next_context)
    bandit.train()
