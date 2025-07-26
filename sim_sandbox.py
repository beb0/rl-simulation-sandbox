import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from collections import defaultdict

# --- Parameters ---
NUM_AGENTS = 30
NUM_TRAITS = 5
NUM_EPISODES = 100
SIMILARITY_THRESHOLD = 0.7
AFFINITY_THRESHOLD = 1.5
EXPLORATION_RATE = 0.2


# --- Agent Class ---
class Agent:
    def __init__(self, id):
        self.id = id
        self.traits = np.random.rand(NUM_TRAITS)
        self.memory = defaultdict(float)
        self.q_table = defaultdict(float)
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
        gamma = 0.0
        self.q_table[state] = old_q + alpha * (reward - old_q)

    def choose_who_to_interact_with(self, population):
        others = [a for a in population if a.id != self.id]

        if random.random() < EXPLORATION_RATE or not self.q_table:
            return random.choice(others)

        similarities = [(other, self.similarity(other)) for other in others]
        choices = []
        for other, sim in similarities:
            sim_rounded = round(sim, 1)
            q_value = self.q_table.get((sim_rounded,), 0)
            choices.append((other, q_value))

        q_vals = [q for _, q in choices]
        max_q = max(q_vals)
        exp_qs = [np.exp(q - max_q) for q in q_vals]
        sum_exp = sum(exp_qs)
        probs = [eq / sum_exp for eq in exp_qs]
        partner = random.choices([c[0] for c in choices], weights=probs, k=1)[0]
        return partner

    def choose_group(self):
        affinities = [(aid, score) for aid, score in self.memory.items() if score > AFFINITY_THRESHOLD]
        if affinities:
            best_friend = max(affinities, key=lambda x: x[1])[0]
            self.group = best_friend
        else:
            self.group = self.id  # stays alone if no strong ties


# --- Simulation ---
agents = [Agent(i) for i in range(NUM_AGENTS)]

for _ in range(NUM_EPISODES):
    for agent in agents:
        partner = agent.choose_who_to_interact_with(agents)
        agent.interact(partner)

for agent in agents:
    agent.choose_group()

# --- Visualization ---
traits = np.array([agent.traits for agent in agents])
pca = PCA(n_components=2)
reduced = pca.fit_transform(traits)

# Assign colors to groups
colors = {}
group_ids = sorted(set(a.group for a in agents if a.group is not None))
color_palette = plt.cm.get_cmap("tab10", len(group_ids))
for idx, gid in enumerate(group_ids):
    colors[gid] = color_palette(idx)

plt.figure(figsize=(10, 6))
for i, agent in enumerate(agents):
    x, y = reduced[i]
    group_color = colors.get(agent.group, "grey")
    plt.scatter(x, y, color=group_color, s=100, edgecolor="black")
    plt.text(x + 0.01, y + 0.01, str(agent.id), fontsize=9)

# Optional: Arrows showing who follows whom
for i, agent in enumerate(agents):
    if agent.group is not None and agent.group != agent.id:
        leader_index = agent.group
        x1, y1 = reduced[i]
        x2, y2 = reduced[leader_index]
        plt.arrow(x1, y1, x2 - x1, y2 - y1, color='black', alpha=0.3,
                  head_width=0.02, length_includes_head=True)

plt.title("Agents grouped by affinity — visualized in personality space")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True)
plt.show()
