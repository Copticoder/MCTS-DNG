from base_mcts import MCTSBase
from state_node import DNGNode

class DNG_MCTS(MCTSBase):
    def __init__(self, env, max_episodes, checkpoint_dir) -> None:
        super().__init__(env, max_episodes, checkpoint_dir, DNGNode)
    def run_mcts(self, node, max_horizon) -> float:
        if max_horizon == 0 or terminated:
            return 0
        elif node.visits == 0:
            reward = self.rollout(max_horizon)
            node.visits += 1
            node.value += reward
            return reward
        else:
            child_node, terminated, R_s_a = self.selection(node)
            discounted_return = R_s_a + (self.discount_gamma*self.run_mcts(child_node, max_horizon - 1, terminated))
            node.visits += 1
            node.value += discounted_return
            return discounted_return
