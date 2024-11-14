from base_mcts import MCTSBase
from state_node import DNGNode
from copy import deepcopy
import numpy as np
class DNG_MCTS(MCTSBase):
    def __init__(self, env, max_episodes, checkpoint_dir) -> None:
        super().__init__(env, max_episodes, checkpoint_dir, DNGNode)
        self.state_space = {self.root.state:self.root}

    def run_mcts(self, node, max_horizon, terminated=False) -> float:
        if max_horizon == 0 or terminated:
            return 0
        elif node.visits == 0 or len(node.rho_a_s) < self.action_space:
            R_s_a, terminated = self.expansion(node)
            if terminated:
                node.visits += 1
                return R_s_a
            discounted_return = self.rollout(max_horizon)
            node.visits += 1
            return discounted_return
        else:
            child_node, terminated, R_s_a = self.selection(node)
            discounted_return = R_s_a + (self.discount_gamma*self.run_mcts(child_node, max_horizon - 1, terminated))
            node.alpha_s += 0.5
            node.beta_s += (node.lambda_s * (discounted_return - node.mu_s)**2 / (node.lambda_s + 1)) / 2
            node.mu_s = (node.lambda_s * node.mu_s + discounted_return) / (node.lambda_s + 1)
            node.lambda_s += 1
            if child_node not in node.rho_a_s[node.last_action]:
                node.rho_a_s[node.last_action][child_node] = 0
            node.rho_a_s[node.last_action][child_node] += 1
            node.visits += 1
            return discounted_return
    def test_episode(self, max_horizon):
        total_reward = 0
        node = self.root
        for step in range(max_horizon):
            try:
                action = node.best_child(self.action_space, exploration_constant=0, sampling = False)
                next_state, reward, terminated, _ = self.env.step(action)
                total_reward += reward
                if terminated:
                    break
                if self.state_space[next_state] not in node.rho_a_s[action]:
                    node.rho_a_s[action][self.state_space[next_state]] = 0
                node = node.children[next_state]
            except:
                break
        self.env.reset()
        return total_reward, terminated

    def expansion(self, node):
        for action in range(self.action_space):
            deepcopy_env = deepcopy(self.env)
            deepcopy_env.env_dynamics = False
            next_state, reward, terminated, _ = deepcopy_env.step(action)
            # Ensure the next_state is in state_space
            self.state_space.setdefault(next_state, self.Node(next_state))
            node.children[next_state] = self.state_space[next_state]
            # Ensure the action key exists in rho_a_s and initialize it as an empty dictionary if missing
            node.rho_a_s.setdefault(action, {})

            # Ensure the state key exists in rho_a_s[action] and initialize it with 0 if missing
            node.rho_a_s[action].setdefault(self.state_space[next_state], 0)
        action = np.random.choice([a for a in range(self.action_space)])
        next_state, reward, terminated, _ = self.env.step(action)
        node.last_action = action
        node.children[next_state] = self.state_space[next_state]
        if self.state_space[next_state] not in node.rho_a_s[action]:
            node.rho_a_s[action][self.state_space[next_state]] = 0
        node.rho_a_s[action][self.state_space[next_state]] +=1
        return reward, terminated
    
    def selection(self, node):
        action = node.best_child(self.action_space, exploration_constant=1.41)
        next_state, reward, terminated, _ = self.env.step(action)
        node.children[next_state] = self.state_space[next_state]
        self.state_space.setdefault(next_state, self.Node(next_state))
        node.rho_a_s.setdefault(action, {}).setdefault(self.state_space[next_state], 0)
        node.last_action = action
        return self.state_space[next_state], terminated, reward