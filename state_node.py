import numpy as np
class UCTNode:
    def __init__(self, state, parent = None) -> None:
        self.state = state
        self.children = {}
        self.parent = parent
        self.visits = 0
        self.value = 0
    def is_fully_expanded(self, action_space):
        """Check if all possible actions have been expanded."""
        return len(self.children) == action_space
    def best_child(self, action_space, exploration_constant=1.41):
        best_score = -float('inf')
        best_actions = []

        for action in range(action_space):
            child = self.children.get(action, UCTNode(state=self.state))
            if child.visits == 0:
                if not exploration_constant:
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
    

class DNGNode(UCTNode):
    def __init__(self, state, parent = None) -> None:
        super().__init__(state, parent)
        if hasattr(self, 'value'):
            del self.value
        self.mu_s = 0
        self.lambda_s = 0.01
        self.alpha_s = 1
        self.beta_s = 100
        self.rho_s_a_s = {a: np.random.dirichlet(np.ones(4)) for a in range(9)}
    def value_sampling(self, sampling=True):
        if sampling:
            tao = np.random.gamma(self.alpha_s, self.beta_s)
            mu = np.random.normal(self.mu_s, 1/(tao * self.lambda_s))
            return mu
        return self.mu_s
    def q_value(self, action, sampling = True):
        if sampling == True:
            if action in self.rho_s_a_s:
                return self.rho_s_a_s[action]
            return 0
    def best_child_thompson_sampling(self):
        for action, child in self.children.items():
            child.value = child.value_sampling()