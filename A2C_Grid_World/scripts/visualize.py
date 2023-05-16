import argparse
import numpy

import utils
from utils import device


args = {
# "env": "MiniGrid-DoorKey-6x6-v0",
"env": "MiniGrid-Dynamic-Obstacles-16x16-v0", 
"model": "DynamicObstacles",
# "model": "DoorKey",
"seed": 1,
"shift": 0,
"argmax": False,
"pause": 0.1,
"gif": None,
"episodes": 1000000,
"memory": False,
"text": False
}





# Set seed for all randomness sources

utils.seed(args['seed'])

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args['env'], args['seed'], render_mode="human")
for _ in range(args['shift']):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args['model'])
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args['argmax'], use_memory=args['memory'], use_text=args['text'])
print("Agent loaded\n")

# Run the agent

if args['gif']:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render()

for episode in range(args['episodes']):
    obs, _ = env.reset()

    while True:
        env.render()
        if args['gif']:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))

        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)

        if done:
            break

    if env.window.closed:
        break
env.close()
# if args.gif:
#     print("Saving gif... ", end="")
#     write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
#     print("Done.")

# 