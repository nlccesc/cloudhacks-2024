import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import BertModel, BertTokenizer
from collections import namedtuple

# Experience named tuple for clarity
Experience = namedtuple('Experience', ('context', 'action', 'reward'))

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class NeuralBandit:
    def __init__(self, n_arms, context_dim, replay_buffer_size=1000, batch_size=32, gamma=0.99, lr=0.001):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.replay_buffer = ExperienceReplay(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.model = NeuralNetwork(768, n_arms)  # BERT embeddings have 768 dimensions
        self.target_model = NeuralNetwork(768, n_arms)  # Target network for Double Q-Learning
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

    def update(self, context, action, reward):
        context_embedding = self.get_context_embedding(context)
        self.replay_buffer.push(Experience(context_embedding, action, reward))
        if len(self.replay_buffer) >= self.batch_size:
            self.train()

    def train(self):
        batch = self.replay_buffer.sample(self.batch_size)
        contexts, actions, rewards = zip(*batch)

        contexts = torch.cat(contexts, dim=0)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        q_values = self.model(contexts)
        target_q_values = self.target_model(contexts).detach()
        for i, action in enumerate(actions):
            target_q_values[i, action] = rewards[i] + self.gamma * target_q_values[i].max()

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.update_target_network()

# Example usage
if __name__ == "__main__":
    bandit = NeuralBandit(n_arms=2, context_dim=3)
    context = [1, 2, 3]
    action = bandit.select_arm(context, epsilon=0.1)
    bandit.update(context, action, reward=1.0)
    bandit.train()
