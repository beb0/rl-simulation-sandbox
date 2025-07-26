import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Parameters
NUM_AGENTS = 20
NUM_STEPS = 10000
SIMILARITY_THRESHOLD = 0.6
AFFINITY_THRESHOLD = 2.0
EXPLORATION_RATE = 0.2  # 20% of time agents will explore randomly

class Agent:
    def __init__(self, id):
        self.id = id
        self.traits = np.random.rand(5)  # OCEAN traits
        self.memory = defaultdict(float)  # Trust built through similarity
        self.q_table = defaultdict(float)  # Q(similarity_level) → expected reward
        self.group = None

    def similarity(self, other):
        return 1 - np.mean(np.abs(self.traits - other.traits))

    def interact(self, other):
        sim = self.similarity(other)
        reward = 0
        if sim > SIMILARITY_THRESHOLD:
            self.memory[other.id] += sim
            other.memory[self.id] += sim
            reward = sim

        # Q-learning update
        state = (round(sim, 1),)
        old_q = self.q_table[state]
        alpha = 0.1
        gamma = 0.0  # No future steps here
        self.q_table[state] = old_q + alpha * (reward + gamma * 0 - old_q)

    def choose_who_to_interact_with(self, population):
        # Exploration: choose a random partner
        if random.random() < EXPLORATION_RATE:
            partner = random.choice([a for a in population if a.id != self.id])
            return partner

        # Exploitation: choose agent with similarity closest to best Q
        similarities = [(other, self.similarity(other)) for other in population if other.id != self.id]
        if not similarities:
            return None

        # Choose best similarity level from Q-table
        if not self.q_table:
            return random.choice([a for a in population if a.id != self.id])
        
        best_level = max(self.q_table.items(), key=lambda x: x[1])[0][0]  # best similarity level
        # Pick agent whose similarity is closest to best_level
        best_partner, _ = min(similarities, key=lambda x: abs(x[1] - best_level))
        return best_partner

    def choose_group(self):
        affinities = [(aid, score) for aid, score in self.memory.items() if score > AFFINITY_THRESHOLD]
        if affinities:
            best_friend = max(affinities, key=lambda x: x[1])[0]
            self.group = best_friend


# --- Simulation
agents = [Agent(i) for i in range(NUM_AGENTS)]

for step in range(NUM_STEPS):
    for agent in agents:
        partner = agent.choose_who_to_interact_with(agents)
        if partner:
            agent.interact(partner)

    # Agents choose groups after enough steps
    if step > NUM_STEPS // 2:
        for agent in agents:
            agent.choose_group()

# --- Visualization
groups = {}
for agent in agents:
    if agent.group is not None:
        groups.setdefault(agent.group, []).append(agent.id)

plt.figure(figsize=(10, 6))
for group_leader, members in groups.items():
    x = [agents[i].traits[0] for i in members]
    y = [agents[i].traits[1] for i in members]
    plt.scatter(x, y, label=f'Group {group_leader}')

plt.title("Agents grouped by affinity (via Q-learned preferences)")
plt.xlabel("Trait 1 (Openness)")
plt.ylabel("Trait 2 (Conscientiousness)")
plt.legend()
plt.grid(True)
plt.show()
