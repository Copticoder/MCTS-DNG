import numpy as np
from abc import ABC, abstractmethod
class BaseNode(ABC):
    def __init__(self, observation) -> None:
        self.observation = observation
        self.visits = {}
        self.q_values = {}
        self.last_action = -1
        self.children = {}
    @abstractmethod
    def best_child(self):
        pass
    
class UCTNode(BaseNode):
    def __init__(self, state) -> None:
        super().__init__(state)

    def best_child(self, action_space, **kwargs):
        exploration_constant = kwargs.get('exploration_constant', 1.41)
        best_score = -float('inf')
        best_actions = []
        for action in range(action_space):
            child = self.children[action]
            if child.visits == 0:
                if exploration_constant == 0:
                    ucb_score = -float('inf')
                else:
                    ucb_score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * np.sqrt(np.log(self.visits) / child.visits)
                ucb_score = exploitation + exploration

            # If we find a higher score, update best_score and reset best_actions
            if ucb_score > best_score:
                best_score = ucb_score
                best_actions = [action]
            # If the score is equal to the current best, add to best_actions
            elif ucb_score == best_score:
                best_actions.append(action)

        # Randomly select one of the best actions
        return np.random.choice(best_actions)
    
class DNGNode(BaseNode):
    def __init__(self, observation) -> None:
        super().__init__(observation)
        self.mu_s = 0
        self.lambda_s = 0.01
        self.alpha_s = 1
        self.beta_s = 100
        self.rho_a_s = {}
        self.visits = 0

    def value_sampling(self, s_bar, sampling=True):
        if sampling:
            tao = np.random.gamma(s_bar.alpha_s, 1 / s_bar.beta_s)
            std_dev = np.sqrt(1 / (tao * s_bar.lambda_s))
            mu = np.random.normal(s_bar.mu_s, std_dev)
            return mu
        return s_bar.mu_s
    
    def q_value(self, action, discount_factor=0.95, sampling = True):   
        if action not in self.rho_a_s:
            self.rho_a_s[action] = {}
                     
        # get the alphas for each observation given action from rho 
        rhos = [rho+1 for rho in self.rho_a_s[action].values()]
        reward = 0
        R_s_a = -1
        # create the dirichlet distribution
        dirichlet = np.random.dirichlet(rhos)
        for s_bar_idx, s_bar in enumerate(self.rho_a_s.get(action, {})):
            if sampling == True:
                # get the mixing coefficient of the next observation s'
                w_s_bar = dirichlet[s_bar_idx]
            else:
                # if not sampling, use the sample mean
                w_s_bar = rhos[s_bar_idx]/sum(rhos)
            reward += discount_factor * w_s_bar * self.value_sampling(self.children[s_bar], sampling)
        return R_s_a+reward
        
    def best_child(self, action_space, **kwargs):
        sampling = kwargs.get('sampling', True)
        discount_factor = kwargs.get('discount_factor', 0.95)
        action_values = []
        for action in range(action_space):
            q_value = self.q_value(action, discount_factor, sampling)
            action_values.append(q_value)
        selected_action = np.argmax(action_values)
        return selected_action