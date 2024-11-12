import numpy as np
from race_track_env.racetrack import RaceTrack
import pickle, os
from state_node import UCTNode
from copy import deepcopy

class UCT_MCTS:
    def __init__(self, env_name, env, checkpoint_dir) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.max_episodes = int(5e5)
        self.env = env
        self.root = UCTNode(state = self.env.reset())
        self.action_space = self.env.nA
        self.discount_gamma = 0.95

    def save_checkpoint(self, iteration):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        filename = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self.root, f)
        print(f"Checkpoint saved: {filename}")

    def online_planning(self):
        max_horizon = 100
        for num_episodes in range(self.max_episodes+1):
            self.uct_mcts(self.root, max_horizon, False)
            self.env.reset()
            if num_episodes % 1000 == 0:
                if num_episodes%50000 == 0:
                    self.save_checkpoint(num_episodes)
                print(f"Episode {num_episodes}: Root Average Value = {self.root.value/self.root.visits}, Visits = {self.root.visits}")

    def uct_mcts(self, node, max_horizon, terminated=False) -> UCTNode:
        if max_horizon == 0 or terminated:
            return 0
        elif node.visits == 0:
            reward = self.rollout(node,max_horizon)
            node.visits += 1
            node.value += reward
            return reward
        else:
            child_node, terminated, R_s_a = self.selection(node)
            discounted_return = R_s_a + (self.discount_gamma*self.uct_mcts(child_node, max_horizon - 1, terminated))
            node.visits += 1
            node.value += discounted_return
            return discounted_return
    def selection(self, node) -> UCTNode:
        """Select the most promising node using the UCB score."""
        action = node.best_child(self.action_space)
        state, reward, terminated, _ = self.env.step(action)
        if action not in node.children:
            node.children[action] = UCTNode(state=state, parent=node)
        child_node = node.children[action]
        return child_node, terminated, reward
    
    def expansion(self, node):
        """Expand the current node by adding a new child node."""
        untried_actions = [a for a in range(self.action_space) if a not in node.children]
        action = np.random.choice(untried_actions)
        next_state, reward, terminated, truncated = self.env.step(action)
        child_node = UCTNode(state=next_state, parent=node)
        node.children[action] = child_node
        return child_node, reward 
    def rollout(self, child_node, max_horizon) -> float:
        """Simulate the game using a random policy until a terminal state is reached."""
        total_reward = 0
        copy_env = deepcopy(self.env)
        for step in range(max_horizon):
            action = np.random.randint(0, self.action_space)
            state, reward, terminated, truncated = copy_env.step(action)
            total_reward = total_reward + ((self.discount_gamma**step) * reward)
            if terminated:
                break
        return total_reward