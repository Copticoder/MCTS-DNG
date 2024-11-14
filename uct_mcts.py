import numpy as np
from race_track_env.racetrack import RaceTrack
import pickle, os
from base_mcts import MCTSBase
from state_node import UCTNode

class UCT_MCTS(MCTSBase):
    def __init__(self, env, max_episodes, checkpoint_dir) -> None:
        super().__init__(env, max_episodes, checkpoint_dir, UCTNode)

    def run_mcts(self, node, max_horizon, terminated=False) -> float:
        if max_horizon == 0 or terminated:
            return 0
        elif sum(node.visits.values()) == 0 or not node.is_fully_expanded(self.action_space):
            R_s_a, terminated = self.expansion(node)
            if terminated:
                node.visits[node.last_action] += 1
                node.q_values[node.last_action] += R_s_a
                return R_s_a
            discounted_return = R_s_a + self.rollout(max_horizon)
            node.visits[node.last_action] += 1
            node.q_values[node.last_action] += (discounted_return - node.q_values[node.last_action]) / node.visits[node.last_action]
            return discounted_return
        else:
            child_node, terminated, R_s_a = self.selection(node)
            if terminated:
                node.visits[node.last_action] += 1
                node.q_values[node.last_action] += R_s_a
                return R_s_a
            discounted_return = R_s_a + (self.discount_gamma*self.run_mcts(child_node, max_horizon - 1, terminated))
            node.visits[node.last_action] += 1
            node.q_values[node.last_action] += (discounted_return - node.q_values[node.last_action]) / node.visits[node.last_action]
            return discounted_return