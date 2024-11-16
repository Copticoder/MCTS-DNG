from abc import ABC, abstractmethod
import numpy as np
import os
import pickle
class MCTSBase(ABC):
    def __init__(self, env, max_episodes, checkpoint_dir, Node) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.max_episodes = max_episodes
        self.env = env
        self.Node = Node
        self.root = self.Node(observation=self.env.reset())
        self.action_space = self.env.nA
        self.discount_gamma = 0.95

    def save_checkpoint(self, iteration):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_file_name = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration}.pkl")
        with open(checkpoint_file_name, "wb") as f:
            pickle.dump(self.root, f)
        print(f"Checkpoint saved: {checkpoint_file_name}")

    def online_planning(self):
        max_horizon = 100
        for num_episodes in range(1,self.max_episodes + 1):
            self.run_mcts(self.root, max_horizon)
            self.env.reset()
            if num_episodes % 1000 == 0:
                self.save_checkpoint(num_episodes)
                total_episode_rewards, terminated = self.test_episode(max_horizon)
                print(f"Episode {num_episodes}: Episode Rewards = {total_episode_rewards}, Episode Terminated = {terminated}")
    @abstractmethod
    def test_episode(self, max_horizon):
        pass
    @abstractmethod
    def run_mcts(self, node, max_horizon):
        """Abstract method to run the specific MCTS algorithm."""
        pass
    @abstractmethod
    def selection(self, node):
        pass
    @abstractmethod
    def expansion(self, node):
        pass
    
    def rollout(self, max_horizon):
        if max_horizon == 0:
            return 0
        action = np.random.randint(0, self.action_space)
        observation, reward, terminated, _ = self.env.step(action)
        if terminated:
            return reward
        return reward+self.discount_gamma*self.rollout(max_horizon-1)