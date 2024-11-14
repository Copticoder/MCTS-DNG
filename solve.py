from race_track_env.racetrack import RaceTrack
from uct_mcts import UCT_MCTS
import pickle
algorithm = 'uct'
env_name = 'a'
stochastic = False
train = False
max_episodes = int(5e5)
env = RaceTrack(env_name, render_mode=None, size=20, render_fps=20)
uct_mcts = UCT_MCTS(env, max_episodes, f"{env_name}_{algorithm}_{'stochastic' if stochastic else ''}_checkpoints")
if train: 
    # play the best actions
    uct_mcts.online_planning()
else:
    # load every checkpoint and play the best action
    env.render_mode = 'human'
    for i in range(1000, max_episodes+1,1000):
        filename = f"{env_name}_{algorithm}_{'stochastic' if stochastic else ''}_checkpoints/checkpoint_{i}.pkl"
        root = pickle.load(open(filename, "rb"))
        env.reset()
        uct_mcts.root = root
        terminated = False
        node = uct_mcts.root
        for _ in range(100):
            action = node.best_child(uct_mcts.action_space,exploration_constant=0)
            print(action)
            if action is None:
                break
            observation, reward, terminated, truncated = env.step(action)
            if terminated:
                break
            if observation not in node.children:
                break
            node = node.children[observation]
        env.reset()
        print(f"finished checkpoint: {i}")