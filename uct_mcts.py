from base_mcts import MCTSBase
from node import UCTNode
import numpy as np
class UCT_MCTS(MCTSBase):
    def __init__(self, env, max_episodes, checkpoint_dir, checkpoint_interval) -> None:
        super().__init__(env, max_episodes, checkpoint_dir, UCTNode, checkpoint_interval)

    def run_mcts(self, node, max_horizon, terminated=False) -> float:
        if max_horizon == 0 or terminated:
            return 0
        elif node.visits == 0:
            R_s_a, terminated = self.expansion(node)
            if terminated:
                node.visits += 1
                node.value += R_s_a
                return R_s_a
            discounted_return = R_s_a + self.rollout(max_horizon-1)
            node.visits += 1
            node.value += discounted_return
            return discounted_return
        else:
            child_node, terminated, R_s_a = self.selection(node)
            discounted_return = R_s_a + (self.discount_gamma*self.run_mcts(child_node, max_horizon - 1, terminated))
            node.visits += 1
            node.value += discounted_return
            return discounted_return
        
    def selection(self, node):
        action = node.best_child(self.action_space, env = self.env, exploration_constant=1.41)
        next_observation, reward, terminated, _ = self.env.step(action)
        if next_observation not in node.children:
            node.children[next_observation] = self.Node(next_observation)
        child_node = node.children[next_observation]
        return child_node, terminated, reward

    def expansion(self, node):
        # try actions that weren't tried before
        for action in range(self.action_space):
            next_observation = self.env.get_next_observation(node.observation, action)
            # Ensure the next_observation is in node.children
            if next_observation not in node.children:
                node.children[next_observation] = self.Node(next_observation)
        action = np.random.choice([a for a in range(self.action_space)])
        next_observation, reward, terminated, _ = self.env.step(action)
        if next_observation not in node.children:
            node.children[next_observation] = self.Node(next_observation)

        return reward, terminated