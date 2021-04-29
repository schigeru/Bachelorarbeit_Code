#!/usr/bin/env python3

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from forward_kin_Agent5_d_002_a_05_512 import PandaRobotGymEnv
import numpy as np

policy_name = "reaching_policy"


def main():

    robot = PandaRobotGymEnv()
    robot = DummyVecEnv([lambda: robot])
    policy_kwargs = dict(net_arch=[dict(pi=[512, 512], vf=[512, 512])])
    model = PPO2(MlpPolicy, robot, n_steps=512, nminibatches=32, learning_rate=0.0001, policy_kwargs=policy_kwargs,
                 verbose=1, tensorboard_log="/home/valentin/BA_Logs/Agent5_a_0_5_knots_512_d_0_02")
    model.learn(total_timesteps=5000000)
    model.save("/home/valentin/BA_Logs/Agent5_a_0_5_knots_512_d_0_02/reaching_policy/5mio_timesteps")
if __name__ == '__main__':
    main()



