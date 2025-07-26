import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from collections import defaultdict

# --- Parameters ---
NUM_AGENTS = 30
NUM_TRAITS = 5
NUM_EPISODES = 10000
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
import networkx as nx

# Build graph
G = nx.DiGraph()

# Add nodes
for agent in agents:
    G.add_node(agent.id, group=agent.group)

# Add edges (who follows whom)
for agent in agents:
    if agent.group != agent.id:
        G.add_edge(agent.id, agent.group)

# Define group colors
group_ids = sorted(set(a.group for a in agents))
color_map = plt.cm.get_cmap("tab10", len(group_ids))
group_color_dict = {gid: color_map(i) for i, gid in enumerate(group_ids)}
node_colors = [group_color_dict[agent.group] for agent in agents]

# Define node sizes: bigger if they have more followers
follower_counts = {agent.id: 0 for agent in agents}
for agent in agents:
    if agent.group in follower_counts and agent.group != agent.id:
        follower_counts[agent.group] += 1
node_sizes = [300 + follower_counts[agent.id] * 150 for agent in agents]

# Layout using spring layout (for social network feel)
pos = nx.spring_layout(G, seed=42)

# Draw
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="black")
nx.draw_networkx_labels(G, pos, labels={agent.id: str(agent.id) for agent in agents}, font_size=8)
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", edge_color='gray', alpha=0.5)

plt.title("Agent Group Formation — Network View")
plt.axis("off")
plt.show()
