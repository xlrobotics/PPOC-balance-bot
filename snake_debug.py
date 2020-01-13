import gym
# from stable_baselines import DQN as deepq
from stable_baselines import A2C as ac
from stable_baselines.common.policies import MlpLnLstmPolicy
import snake_bot

if __name__ == '__main__':
    env = gym.make("snakebot-v0")
    env.debug_mode()
    exit(0)