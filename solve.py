from race_track_env.racetrack import RaceTrack
from uct_mcts import UCT_MCTS
import pickle
algorithm = 'uct'
env_name = 'a'
env_dynamics = False
train = True
max_episodes = int(5e5)
env = RaceTrack(env_name, render_mode=None, size=20, render_fps=20, env_dynamics=env_dynamics)
uct_mcts = UCT_MCTS(env, max_episodes, f"{env_name}_{algorithm}_{'stochastic' if env_dynamics else 'nonstochastic'}_checkpoints")
if train: 
    # play the best actions
    uct_mcts.online_planning()
else:
    # load every checkpoint and play the best action
    env.render_mode = 'human'
    for i in range(1000, max_episodes+1,1000):
        filename = f"{env_name}_{algorithm}_{'stochastic' if env_dynamics else 'nonstochastic'}_checkpoints/checkpoint_{i}.pkl"
        root = pickle.load(open(filename, "rb"))
        env.reset()
        uct_mcts.root = root
        terminated = False
        node = uct_mcts.root
        total_rewards = 0
        for step in range(50):
            action = node.best_child(uct_mcts.action_space,exploration_constant=0)
            next_state, reward, terminated, truncated = env.step(action)
            if next_state not in node.children:
                break
            total_rewards += reward
            if terminated:
                break
            node = node.children[next_state]
        env.reset()
        print(f"finished checkpoint: {i}, Total Rewards: ", total_rewards)