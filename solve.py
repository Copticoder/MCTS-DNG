from race_track_env.racetrack import RaceTrack
from uct_mcts import UCT_MCTS
import pickle
algorithm = 'uct'
env_name = 'b'
train = False
env = RaceTrack(env_name, render_mode=None, size=20, render_fps=20)
uct_mcts = UCT_MCTS(env_name,env)
if train: 
    # play the best actions
    uct_mcts.start_mcts()
else:
    # load every checkpoint and play the best action
    env.render_mode = 'human'
    for i in range(0, 300000, 20000):
        filename = f"{env_name}_{algorithm}_checkpoints/checkpoint_{i}.pkl"
        root = pickle.load(open(filename, "rb"))
        env.reset()
        uct_mcts.root = root
        terminated = False
        node = uct_mcts.root
        for _ in range(50):
            action = node.best_child(exploration_constant=0)
            if action is None:
                break
            observation, reward, terminated, truncated = env.step(action)
            if terminated:
                break
            node = node.children[action]
        env.reset()
        print("finished checkpoint")