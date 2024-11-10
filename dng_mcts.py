from uct_mcts import UCT_MCTS

class DNG_MCTS(UCT_MCTS):
    def __init__(self, env) -> None:
        super().__init__(env)
        