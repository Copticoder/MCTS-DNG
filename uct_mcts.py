import numpy as np
from race_track_env.racetrack import RaceTrack
import pickle, os
from state_node import UCTNode

class UCT_MCTS:
    def __init__(self,env_name, env) -> None:
        self.checkpoint_dir = f"{env_name}_uct_checkpoints"
        self.cur_node = None
        self.max_episodes = int(5e5)
        self.env = env
        self.root = UCTNode(state = self.env.reset()) 
        self.cur_node = self.root 
        self.max_horizon = 100
        self.action_space = self.env.nA

    def save_checkpoint(self, iteration, checkpoint_dir="checkpoints"):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filename = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self.root, f)
        print(f"Checkpoint saved: {filename}")

    def start_mcts(self) -> UCTNode:
        for num_episodes in range(self.max_episodes):
            self.cur_node = self.root
            self.selection()
            if not self.cur_node.is_fully_expanded(self.action_space):
                self.cur_node = self.expansion()
            reward = self.rollout()

            self.backpropagation(reward)
            self.env.reset()

            if num_episodes % 1000 == 0:
                self.save_checkpoint(num_episodes)
                print(f"Episode {num_episodes}: Root value = {self.root.value}, Visits = {self.root.visits}")

    def selection(self) -> UCTNode:
        """Select the most promising node using the UCB score."""
        terminated = False
        while not terminated and self.cur_node.is_fully_expanded(self.action_space):
            action = self.cur_node.best_child()
            next_state, reward, terminated, _ = self.env.step(action)
            self.cur_node = self.cur_node.children[action]
        return self.cur_node
    
    def expansion(self) -> UCTNode:
        """Expand the current node by adding a new child node."""
        untried_actions = [a for a in range(self.action_space) if a not in self.cur_node.children]
        action = np.random.choice(untried_actions)
        next_state, _, terminated, truncated = self.env.step(action)
        child_node = UCTNode(state=next_state, parent=self.cur_node)
        self.cur_node.children[action] = child_node
        return child_node
        
    def rollout(self):
        """Simulate the game using a random policy until a terminal state is reached."""
        total_reward = 0
        for _ in range(self.max_horizon):
            action = np.random.randint(0, self.action_space - 1)
            state, reward, terminated, truncated = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward
    
    def backpropagation(self, value: float) -> None:
        node = self.cur_node
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent 