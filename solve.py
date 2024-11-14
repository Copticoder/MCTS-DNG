from race_track_env.racetrack import RaceTrack
from uct_mcts import UCT_MCTS
from dng_mcts import DNG_MCTS
import pickle
algorithm = 'dng'
env_name = 'a'
env_dynamics = False
train = True
max_episodes = int(5e5)
env = RaceTrack(env_name, render_mode=None, size=20, render_fps=20, env_dynamics=env_dynamics)
mcts = DNG_MCTS(env, max_episodes, f"{env_name}_{algorithm}_{'stochastic' if env_dynamics else 'nonstochastic'}_checkpoints")
if train: 
    # play the best actions
    mcts.online_planning()
else:
    # load every checkpoint and play the best action
    env.render_mode = 'human'
    for i in range(1000, max_episodes+1,1000):
        filename = f"{env_name}_{algorithm}_{'stochastic' if env_dynamics else 'nonstochastic'}_checkpoints/checkpoint_{i}.pkl"
        root = pickle.load(open(filename, "rb"))
        env.reset()
        mcts.root = root
        terminated = False
        node = mcts.root
        total_rewards = 0
        for step in range(50):
            try:
                action = node.best_child(mcts.action_space,exploration_constant=0, sampling = False)
                next_state, reward, terminated, truncated = env.step(action)
                total_rewards += reward
                if terminated:
                    break
                node = node.children[next_state]
            except:
                break
        env.reset()
        print(f"finished checkpoint: {i}, Total Rewards: ", total_rewards, "Terminated: ", terminated)