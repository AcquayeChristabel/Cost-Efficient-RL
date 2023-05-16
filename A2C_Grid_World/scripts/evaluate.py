import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils
from utils import device
import matplotlib.pyplot as plt

args = {
# "env": "MiniGrid-DoorKey-6x6-v0",
"env": "MiniGrid-Dynamic-Obstacles-16x16-v0", 
"model": "DynamicObstacles",
# "model": "DoorKey",
"seed": 0,
"episodes": 100,
"procs": 16,
"argmax": False,
"worst_episodes_to_show": 10,
"memory": False,
"text": False
}


if __name__ == "__main__":
   
    # Set seed for all randomness sources

    utils.seed(args['seed'])

    # Set device

    print(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args['procs']):
        env = utils.make_env(args['env'], args['seed'] + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent

    model_dir = utils.get_model_dir(args['model'])
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args['argmax'], num_envs=args['procs'],
                        use_memory=args['memory'], use_text=args['text'])
    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args['procs'], device=device)
    log_episode_num_frames = torch.zeros(args['procs'], device=device)

    while log_done_counter < args['episodes']:
        actions = agent.get_actions(obss)
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args['procs'], device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))

    # Print worst episodes

    n = args['worst_episodes_to_show']
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))



    

    # Get return per episode data
    returns = logs["return_per_episode"]

    # Calculate rolling mean of returns
    window_size = 10
    rolling_returns = [sum(returns[i:i+window_size])/window_size for i in range(len(returns)-window_size+1)]

    # Create the plot
    x = range(len(logs["return_per_episode"]))
    plt.plot(x, logs["return_per_episode"])

# Add labels and title
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Return per episode")

plt.savefig("plot.png")