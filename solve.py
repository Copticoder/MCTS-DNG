from race_track_env.racetrack import RaceTrack
from uct_mcts import UCT_MCTS
import pickle
algorithm = 'uct'
env_name = 'b'
train = True
env = RaceTrack(env_name, render_mode=None, size=20, render_fps=20)
uct_mcts = UCT_MCTS(env_name,env, f"{env_name}_{algorithm}_checkpoints")
if train: 
    # play the best actions
    uct_mcts.online_planning()
else:
    # load every checkpoint and play the best action
    env.render_mode = 'human'
    for i in range(0, 500000, 50000):
        filename = f"{env_name}_{algorithm}_checkpoints/checkpoint_{i}.pkl"
        root = pickle.load(open(filename, "rb"))
        env.reset()
        uct_mcts.root = root
        terminated = False
        node = uct_mcts.root
        for _ in range(50):
            action = node.best_child(env.nA,exploration_constant=0)
            if action is None:
                break
            observation, reward, terminated, truncated = env.step(action)
            if terminated:
                break
            if action not in node.children:
                break
            node = node.children[action]
        env.reset()
        print(f"finished checkpoint-{i}")