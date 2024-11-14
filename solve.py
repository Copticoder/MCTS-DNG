from race_track_env.racetrack import RaceTrack
from uct_mcts import UCT_MCTS
import pickle
algorithm = 'uct'
env_name = 'b'
stochastic = True
train = False
max_episodes = int(3e5)
env = RaceTrack(env_name, render_mode=None, size=20, render_fps=20)
uct_mcts = UCT_MCTS(env, max_episodes, f"{env_name}_{algorithm}_{'stochastic' if stochastic else ''}_checkpoints")
if train: 
    # play the best actions
    uct_mcts.online_planning()
else:
    # load every checkpoint and play the best action
    env.render_mode = 'human'
    for i in range(40000, max_episodes+1,20000):
        filename = f"{env_name}_{algorithm}_{'stochastic' if stochastic else ''}_checkpoints/checkpoint_{i}.pkl"
        root = pickle.load(open(filename, "rb"))
        env.reset()
        uct_mcts.root = root
        terminated = False
        node = uct_mcts.root
        for step in range(200):
            try:
                action = node.best_child(uct_mcts.action_space,exploration_constant=0)
                if action is None:
                    break
                next_state, reward, terminated, truncated = env.step(action)
                if terminated:
                    break
                if next_state not in node.children:
                    break 
                node = node.children[next_state]
            except:
                break    
        env.reset()
        print(f"finished checkpoint: {i}")