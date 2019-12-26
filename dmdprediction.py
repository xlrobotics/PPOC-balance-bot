import gym
from stable_baselines import DQN as deepq
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import LnMlpPolicy
import balance_bot

from pydmd import DMDc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
import numpy as np


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['log_interval'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return not is_solved


def main(mode="train", sizes=[50], initflag=True):

    if initflag:
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
            total_timesteps=100000,
            callback=callback
        )
        print("Saving model to balance_dqn.pkl")
        model.save("balance_dqn.pkl")

        del model  # remove to demonstrate saving and loading

    if mode == "test":
        model = deepq.load("balance_dqn.pkl")

        for size in sizes:
            dmdc = DMDc(svd_rank=-1)
            obs = env.reset(testmode=True)
            done = False
            env.set_done(2000)
            error = []
            fitflag = 0

            while not done:
                action, _states = model.predict(obs)
                action = 7 if action > 4 else 1
                obs, rewards, done, info = env.step(action)
                # env.render()
                # print(obs)

                if len(env.state_queue) > size:
                    snapshots = env.get_states(size=size)
                    u = env.get_inputs(size=size)

                    if fitflag % 50 == 0:
                        dmdc.fit(snapshots, u)
                        # fitflag = False
                        # print(fitflag)
                    else:
                        dmdc._snapshots = snapshots
                        dmdc._controlin = u

                    fitflag += 1

                    diff = np.linalg.norm(dmdc.reconstructed_data(u)[:, 2].real-snapshots[:, 2].real)
                    error.append(diff)

                    if np.isnan(diff):
                        print(dmdc.reconstructed_data().real[0], dmdc.eigs, np.log(dmdc.eigs))

                    # plt.figure(figsize=(16, 6))
                    # plt.figure()
                    #
                    # plt.subplot(311)
                    # plt.title('1')
                    # # plt.pcolor(snapshots.real[0, :])
                    # plt.plot(snapshots.real[:, 0])
                    # plt.plot(dmdc.reconstructed_data().real[:, 0])
                    # # plt.colorbar()
                    #
                    # plt.subplot(312)
                    # plt.title('2')
                    # plt.plot(snapshots.real[:, 1])
                    # plt.plot(dmdc.reconstructed_data().real[:, 1])
                    # # plt.pcolor(dmdc.reconstructed_data().real)
                    # # plt.colorbar()
                    #
                    # plt.subplot(313)
                    # plt.title('3')
                    # plt.plot(snapshots.real[:, 2])
                    # plt.plot(dmdc.reconstructed_data().real[:, 2])
                    #
                    # plt.show()

            plt.plot(error)
            print(error)




if __name__ == '__main__':
    main(mode="test", sizes=[100])
    # main(mode="test", size=500, initflag=False)
    # main(mode="test", size=1000, initflag=False)
    # main(mode="test", size=1500, initflag=False)
    plt.show()

