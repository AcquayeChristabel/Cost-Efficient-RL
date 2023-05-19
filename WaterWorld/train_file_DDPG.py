import argparse

from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.apex_ddpg import ApexDDPGConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v4

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for training.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: Only one episode will be "
    "sampled.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    def env_creator(args):
        return PettingZooEnv(waterworld_v4.env())
# n_pursuers=3, n_evaders=3, n_poisons=6, n_coop=2, n_sensors=10,
# sensor_range=0.2,radius=0.015, obstacle_radius=0.2,
# obstacle_coord=(0.5, 0.5), pursuer_max_accel=0.01, evader_speed=0.01,
# poison_speed=0.01, poison_reward=-1.0, food_reward=10.0, encounter_reward=0.01,
# thrust_penalty=-0.5, local_ratio=1.0, speed_features=True, max_cycles=500
    env = env_creator({})
    register_env("waterworld", env_creator)

    config = (
        ApexDDPGConfig()
        .environment("waterworld")
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=2)
        .multi_agent(
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )

    if args.as_test:
        # Only a compilation test of running waterworld / independent learning.
        stop = {"training_iteration": 1}
    else:
        stop = {"episodes_total": 200}

    tune.Tuner(
        "APEX_DDPG",
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=100,
            ),
        ),
        param_space=config,
    ).fit()