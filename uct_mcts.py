import numpy as np
from base_mcts import MCTSBase
from state_node import UCTNode
from copy import deepcopy

class UCT_MCTS(MCTSBase):
    def __init__(self, env, max_episodes, checkpoint_dir) -> None:
        super().__init__(env, max_episodes, checkpoint_dir, UCTNode)
    
    def selection(self, node):
        action = node.best_child(self.action_space, exploration_constant=1.41)
        next_state, reward, terminated, _ = self.env.step(action)
        if action not in node.visits:
            node.visits[action] = 0
            node.q_values[action] = 0
        node.last_action = action
        if next_state not in node.children:
            node.children[next_state] = self.Node(next_state)
        child_node = node.children[next_state]
        return child_node, terminated, reward
    def test_episode(self, max_horizon):
        total_reward = 0
        node = self.root
        for step in range(max_horizon):
            action = node.best_child(self.action_space, exploration_constant=0, sample=False)
            next_state, reward, terminated, _ = self.env.step(action)
            total_reward += reward
            if terminated:
                break
            if next_state not in node.children:
                node.children[next_state] = self.Node(next_state)
            node = node.children[next_state]
        self.env.reset()
        return total_reward, terminated

    def expansion(self, node):
        # try actions that weren't tried before
        for action in range(self.action_space):
            deepcopy_env = deepcopy(self.env)
            deepcopy_env.env_dynamics = False
            next_state, reward, terminated, _ = deepcopy_env.step(action)
            if action not in node.visits:
                node.visits[action] = 0
                node.q_values[action] = 0
            if next_state not in node.children:
                node.children[next_state] = self.Node(next_state)
        # now choose a random action
        action = np.random.choice([a for a in range(self.action_space)])
        _, reward, terminated, _ = self.env.step(action)
        node.last_action = action
        return reward, terminated
    
    def run_mcts(self, node, max_horizon, terminated=False) -> float:
        if max_horizon == 0 or terminated:
            return 0
        elif sum(node.visits.values()) == 0:
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