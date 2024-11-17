import argparse
import pickle
from race_track_env.racetrack import RaceTrack
from dng_mcts import DNG_MCTS
from uct_mcts import UCT_MCTS

def create_environment(env_name, env_dynamics):
    """Initialize the RaceTrack environment."""
    return RaceTrack(env_name, render_mode=None, size=20, render_fps=60, env_dynamics=env_dynamics)


def initialize_mcts(env, algorithm, max_episodes):
    """Initialize the MCTS algorithm based on the specified algorithm."""
    checkpoint_dir = f"{env.env_name}_{algorithm}_{'stochastic' if env.env_dynamics else 'nonstochastic'}_checkpoints"
    if algorithm == "dng":
        return DNG_MCTS(env, max_episodes, checkpoint_dir)
    elif algorithm == "uct":
        return UCT_MCTS(env, max_episodes, checkpoint_dir)
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")

def train_mcts(mcts):
    """Train the MCTS algorithm."""
    print("Starting training...")
    mcts.online_planning()
    print("Training completed.")


def evaluate_checkpoints(mcts, env, max_episodes, episode_number = 1000, step_limit=50):
    """Evaluate MCTS checkpoints by replaying the best actions."""
    print("Starting evaluation...")
    env.render_mode = 'human'

    for i in range(episode_number, max_episodes + 1, 50000):
        filename = f"{mcts.checkpoint_dir}/checkpoint_{i}.pkl"
        try:
            with open(filename, "rb") as f:
                root = pickle.load(f)
        except FileNotFoundError:
            print(f"Checkpoint file not found: {filename}")
            continue

        node = root
        env.start_state = (node.observation[0], node.observation[1])
        total_rewards = 0
        terminated = False

        for step in range(step_limit):
            try:
                action = node.best_child(mcts.action_space, exploration_constant=0, sampling=False)
                print(f"Step {step}, Action: {action}")

                next_observation, reward, terminated, _ = env.step(action)
                total_rewards += reward

                if terminated:
                    print(f"Episode terminated after {step + 1} steps.")
                    break

                node = node.children.get(next_observation)
                if node is None:
                    print("Node for next observation not found.")
                    break

            except Exception as e:
                print(f"Error during evaluation: {e}")
                break

        env.reset()
        print(f"Finished checkpoint: {i}, Total Rewards: {total_rewards}, Terminated: {terminated}")

    print("Evaluation completed.")


def main(args):
    # Initialize environment and MCTS
    env = create_environment(args.env_name, args.env_dynamics)
    mcts = initialize_mcts(env , args.algorithm, args.max_episodes)

    # Train or evaluate based on user input
    if args.train:
        train_mcts(mcts)
    else:
        evaluate_checkpoints(mcts, env, args.max_episodes, args.episode_number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS for RaceTrack Environment")
    parser.add_argument("--algorithm", type=str, default="dng", choices=["dng", "uct"], help="Algorithm to use (dng)")
    parser.add_argument("--env_name", type=str, default="b", choices=["b","a"], help="Environment name")
    parser.add_argument("--env_dynamics", action="store_true", help="Enable stochastic environment dynamics")
    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--max_episodes", type=int, default=int(5e5), help="Maximum number of episodes")
    parser.add_argument("--eval_episode_number", type=int, default=50, help="Number of episodes to evaluate")
    args = parser.parse_args()
 
    main(args)
