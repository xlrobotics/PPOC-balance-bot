import gym
# from stable_baselines import DQN as deepq
from stable_baselines import A2C as ac
from stable_baselines.common.policies import MlpLnLstmPolicy
import snake_bot

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\projects\\python\\PPOC-balance-bot', 'D:/projects/python/PPOC-balance-bot'])

def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['log_interval'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return not is_solved


def main(mode="train"):

    env = gym.make("snakebot-v0")
    if mode == "train":
        model = ac(policy=MlpLnLstmPolicy,
                      env=env,
                      verbose=0,
                      tensorboard_log="a2c_snakebot_tensorboard")
        model.learn(
            total_timesteps=2000,
            callback=callback
        )
        print("Saving model to snake_dqn.pkl...")
        model.save("snake_a2c.pkl")
        print("done.")

        del model  # remove to demonstrate saving and loading

    if mode == "test":
        model = ac.load("snake_a2c.pkl")

        obs = env.reset()
        done = False
        env.set_done(5000)
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            # env.render()
            print(obs)


if __name__ == '__main__':
    main(mode="train")
    exit(0)
