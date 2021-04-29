#!/usr/bin/env python3
'''Implementiert zufällige Robotersteuerung'''
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym_Agent1_rewards import PandaRobotGymEnv
import rospy
import numpy as np
import os
from stable_baselines.common.env_checker import check_env

def main():
    testEnv = PandaRobotGymEnv()
    testEnv = DummyVecEnv([lambda: testEnv])
    obs = testEnv.reset()

    n_steps = 100
    for _ in range(n_steps):
        if _ == n_steps-1:
            print("Last Step")
        action = testEnv.action_space.sample()
        action = [action]
        testEnv.step(action)




if __name__ == '__main__':
    main()



