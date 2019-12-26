import gym
from stable_baselines import DQN as deepq
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import LnMlpPolicy
import balance_bot


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['log_interval'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return not is_solved


def main(mode="train"):

    env = gym.make("balancebot-v0")
    if mode == "train":
        model = deepq(policy=LnMlpPolicy,
                      env=env,
                      double_q=True,
                      prioritized_replay=True,
                      learning_rate=1e-3,
                      buffer_size=10000,
                      verbose=0,
                      tensorboard_log="./dqn_balancebot_tensorboard")
        model.learn(
            total_timesteps=200000,
            callback=callback
        )
        print("Saving model to balance_dqn.pkl")
        model.save("balance_dqn.pkl")

        del model  # remove to demonstrate saving and loading

    if mode == "test":
        model = deepq.load("balance_dqn.pkl")

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

