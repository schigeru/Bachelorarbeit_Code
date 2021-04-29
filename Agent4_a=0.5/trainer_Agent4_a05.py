#!/usr/bin/env python3

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from forward_kin_Agent4_a05 import PandaRobotGymEnv
import numpy as np

policy_name = "reaching_policy"


def main():

    robot = PandaRobotGymEnv()
    robot = DummyVecEnv([lambda: robot])
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    model = PPO2(MlpPolicy, robot, n_steps=512, nminibatches=32, learning_rate=0.0002, policy_kwargs=policy_kwargs,
                 verbose=1, tensorboard_log="/home/valentin/BA_Logs/Szenario3_Agent4_a=0.5")
    model.learn(total_timesteps=1000000)
    model.save("/home/valentin/BA_Logs/Szenario3_Agent4_a=0.5/reaching_policy/1mio_timesteps")
if __name__ == '__main__':
    main()



