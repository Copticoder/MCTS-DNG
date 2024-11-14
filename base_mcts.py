from abc import ABC, abstractmethod
import numpy as np
import os
import pickle

class MCTSBase(ABC):
    def __init__(self, env, max_episodes, checkpoint_dir, Node):
        self.checkpoint_dir = checkpoint_dir
        self.max_episodes = max_episodes
        self.env = env
        self.Node = Node
        self.root = self.Node(state=self.env.reset())
        self.action_space = self.env.nA
        self.discount_gamma = 0.95
        self.s_a_s = {}
    def save_checkpoint(self, iteration):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        filename = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self.root, f)
        print(f"Checkpoint saved: {filename}")
    def online_planning(self):
        max_horizon = 100
        for num_episodes in range(1,self.max_episodes + 1):
            self.run_mcts(self.root, max_horizon)
            self.env.reset()
            if num_episodes % 1000 == 0:
                if num_episodes % 1000 == 0:
                    self.save_checkpoint(num_episodes)
                print(f"Episode {num_episodes}: Root Average Value = {sum(self.root.q_values.values())}, Visits = {sum(self.root.visits.values())}")

    @abstractmethod
    def run_mcts(self, node, max_horizon):
        """Abstract method to run the specific MCTS algorithm."""
        pass

    def selection(self, node):
        action = node.best_child(self.action_space, exploration_constant=3)
        next_state, reward, terminated, _ = self.env.step(action)
        if action not in node.visits:
            node.visits[action] = 0
            node.q_values[action] = 0
        node.last_action = action
        if next_state not in node.children:
            node.children[next_state] = self.Node(next_state)
        child_node = node.children[next_state]
        return child_node, terminated, reward

    def expansion(self, node):
        # try actions that weren't tried before
        action = np.random.choice([i for i in range(self.action_space) if i not in node.visits])
        next_state, reward, terminated, _ = self.env.step(action)
        if action not in node.visits:
            node.visits[action] = 0
            node.q_values[action] = 0
        if next_state not in node.children:
            node.children[next_state] = self.Node(next_state)
        node.last_action = action
        return reward, terminated

    def rollout(self, max_horizon):
        total_reward = 0
        for step in range(max_horizon):
            action = np.random.randint(0, self.action_space)
            _, reward, terminated, _ = self.env.step(action)
            total_reward += (self.discount_gamma ** step) * reward
            if terminated:
                break
        return total_reward