import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
class BaseNode(ABC):
    def __init__(self, state) -> None:
        self.state = state
        self.visits = {}
        self.q_values = {}
        self.last_action = -1
        self.children = {}
    # def is_fully_expanded(self, action_space):
    #     return len(self.visits) == action_space
    @abstractmethod
    def best_child(self):
        pass

class UCTNode(BaseNode):
    def __init__(self, state) -> None:
        super().__init__(state)

    def best_child(self, action_space, **kwargs):
        exploration_constant = kwargs.get('exploration_constant', 1.41)
        best_score = -float('inf')
        total_visits = sum(self.visits.values())  # Total number of visits across all actions
        best_actions = []
        for action in range(action_space):
            if self.visits.get(action,0) == 0:   
                if exploration_constant == 0:
                    ucb_score = -float('inf')
                else:
                    ucb_score = float('inf')
            else:
                exploitation = self.q_values[action]
                # sum over the self.visits values 
                exploration = exploration_constant * np.sqrt(np.log(total_visits) / self.visits[action])
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
    def __init__(self, state) -> None:
        super().__init__(state)
        self.mu_s = 0
        self.lambda_s = 0.01
        self.alpha_s = 1
        self.beta_s = 100
    def value_sampling(self, child_node, sampling=True):
        if sampling: 
            tao = np.random.gamma(child_node.alpha_s, child_node.beta_s)
            mu = np.random.normal(child_node.mu_s, 1/(tao * child_node.lambda_s))
            return mu
        return child_node.mu_s
    
    def q_value(self, action, sampling = True):
            # get the alphas for each state given action from rho 
            alphas = [self.rho_s_a_s[action][state] for state in self.rho_s_a_s[action]]
            reward = 0
            # create the dirichlet distribution
            dirichlet = np.random.dirichlet(alphas)
            for s_bar_idx,s_bar in enumerate(self.rho_s_a_s[action]):
                if sampling == True:
                    # get the mixing coefficient of the next state s'
                    w_s_bar = dirichlet[s_bar_idx]
                    # check which s_bar in children 
                    if s_bar not in self.children[action]:
                        self.children[action][s_bar] = DNGNode(state = s_bar, parent = self)    
                else:
                    # if not sampling, use the sample mean
                    w_s_bar = alphas[s_bar_idx]/sum(alphas)
                reward = reward + w_s_bar * self.value_sampling(self.children[action][s_bar], sampling)
            ## TODO: Check if need to add immediate reward R_s_a and step environment.
            return reward
    
    def populate_rho_s_a_s(self, starting_action, action_space, env):
        if starting_action not in self.rho_s_a_s:
            self.rho_s_a_s[starting_action] = {}
            for action in range(action_space):
                copy_env = deepcopy(env)
                new_state, _, terminated,_ = copy_env.step(action)
                if action not in self.children:
                    self.children[action] = {}
                if new_state not in self.children[action]:
                    self.children[action][new_state] = DNGNode(state = new_state)                    
                self.rho_s_a_s[starting_action][new_state] = 1
                del copy_env
        
    def best_child(self, action_space, **kwargs):
        env = kwargs.get('env')
        sampling = kwargs.get('sampling', True)
        """Sample best child using thompson sampling"""
        action_values = []
        for action in range(action_space):
            self.populate_rho_s_a_s(action,action_space, env)
            copy_env = deepcopy(env)
            action_values.append(self.q_value(copy_env, action, sampling))
            del copy_env
        return np.argmax(action_values)