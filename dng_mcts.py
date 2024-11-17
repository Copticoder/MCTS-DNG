from base_mcts import MCTSBase
from node import DNGNode
import numpy as np
class DNG_MCTS(MCTSBase):
    def __init__(self, env, max_episodes, checkpoint_dir) -> None:
        super().__init__(env, max_episodes, checkpoint_dir, DNGNode)
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
            if child_node.observation not in node.rho_a_s[node.last_action]:
                node.rho_a_s[node.last_action][child_node.observation] = 0
            node.rho_a_s[node.last_action][child_node.observation] += 1
            node.visits += 1
            return discounted_return
        
    def test_episode(self, max_horizon):
        total_reward = 0
        node = self.root
        terminated = False
        self.env.render_mode = "human"
        for step in range(max_horizon):
            try:
                action = node.best_child(self.action_space, exploration_constant=0, sampling = False)
                next_observation, reward, terminated, _ = self.env.step(action)
                total_reward += reward
                if terminated:
                    break
                node = node.children[next_observation]
            except Exception as e:
                print(e)
                break
        self.env.render_mode = None
        self.env.reset()
        return total_reward, terminated

    def expansion(self, node):
        for action in range(self.action_space):
            next_observation = self.env.get_next_observation(node.observation, action)
            # Ensure the next_observation is in node.children
            if next_observation not in node.children:
                node.children[next_observation] = self.Node(next_observation)
            child_node = node.children[next_observation]
            node.rho_a_s.setdefault(action, {})
            node.rho_a_s[action].setdefault(child_node.observation, 0)

        action = np.random.choice([a for a in range(self.action_space)])
        next_observation, reward, terminated, _ = self.env.step(action)
        if next_observation not in node.children:
            node.children[next_observation] = self.Node(next_observation)
        child_node = node.children[next_observation]
        node.last_action = action
        node.rho_a_s[action][child_node.observation] = node.rho_a_s[action].get(child_node.observation, 0) + 1

        return reward, terminated

    
    def selection(self, node):
        action = node.best_child(self.action_space, exploration_constant=1.41, sampling=True)
        next_observation, reward, terminated, _ = self.env.step(action)
        # Ensure the next_observation is in node.children
        if next_observation not in node.children:
            node.children[next_observation] = self.Node(next_observation)
        child_node = node.children[next_observation]
        node.rho_a_s.setdefault(action, {}).setdefault(child_node.observation, 0)
        node.last_action = action
        return child_node, terminated, reward