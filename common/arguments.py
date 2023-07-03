import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Bilevel Reinforcement Learning for highway environments")
    # Environment
    parser.add_argument("--file-path", type=str, default="./roundabout_env_result/exp1", help="file path for reading config and saving result")
    parser.add_argument("--scenario-name", type=str, default="roundabout-v0", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000, help="number of time steps")
    parser.add_argument("--action-type", type=str, default="continuous", help="action type")
    parser.add_argument("--version", type=str, default="c_bilevel", help="version of algorithm")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")

    # Core training parameters
    parser.add_argument("--lagrangian_multiplier", type=float, default=1, help="initial value of lagrangian multiplier")
    parser.add_argument("--lagrangian_max_bound", type=float, default=20, help="max bound of lagrangian multiplier")
    parser.add_argument("--cost_threshold", type=float, default=2, help="threshold of cost")
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--lr-lagrangian", type=float, default=1e-3, help="learning rate of lagrangian multiplier")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="min epsilon greedy")
    parser.add_argument("--min_noise_rate", type=float, default=0.05, help="min noise rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--sample-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--update-rate", type=int, default=10, help="target network update rate")
    parser.add_argument("--enable-cost", type=bool, default=False, help="enable cost constraint")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./roundabout_env_result/exp1", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=300, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    parser.add_argument("--record-video", type=bool, default=False, help="record video")
    args = parser.parse_args()

    return args
