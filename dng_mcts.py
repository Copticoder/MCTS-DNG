from uct_mcts import UCT_MCTS
from state_node import DNGNode
class DNG_MCTS(UCT_MCTS):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.root = DNGNode(state = self.env.reset())
        self.rollout_node = None
        
    def online_planning(self):
        for num_episodes in range(self.max_episodes):
             self.dng_mcts()
        if num_episodes % 1000 == 0:
            self.save_checkpoint(num_episodes)
            print(f"Episode {num_episodes}: Root value = {self.root.value}, Visits = {self.root.visits}")

    def dng_mcts(self):
        self.selection()
        if not self.cur_node.is_fully_expanded(self.action_space):
            self.cur_node = self.expansion()
        reward = self.rollout()

        self.backpropagation(reward)
        self.env.reset()

    def selection(self):
        terminated = False
        while not terminated and self.cur_node.is_fully_expanded(self.action_space):
            action = self.cur_node.best_child_thompson_sampling()
            next_state, reward, terminated, _ = self.env.step(action)
            self.cur_node = self.cur_node.children[action]
        return self.cur_node