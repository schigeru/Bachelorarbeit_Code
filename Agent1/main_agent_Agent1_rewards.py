#!/usr/bin/env python3

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym_Agent1_rewards import PandaRobotGymEnv
import rospy
import numpy as np
import os


def main():
    robot = PandaRobotGymEnv()
    robot = DummyVecEnv([lambda: robot])
    model = PPO2.load("/home/valentin/BA_Logs/Agent1/reaching_policy/200k_timesteps", env=robot)

    obs = robot.reset()

    for i in range(0, 500):
        if i == 499:
            print("Last Step")
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = robot.step(action)


if __name__ == '__main__':
    main()



