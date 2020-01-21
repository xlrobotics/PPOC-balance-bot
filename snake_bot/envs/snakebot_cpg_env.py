import logging
import numpy as np

import pybullet as p
from gym import spaces

from balance_bot.envs.balancebot_env_continuum import BalancebotEnv
from balance_bot.envs.cpg_net import Matsuoka

logger = logging.getLogger(__name__)


class BalancebotEnvCPG(BalancebotEnv):

    def __init__(self, render=True, discrete=True, cpg=True, plot_summary=False):
        super().__init__(render=render, discrete=discrete, cpg=cpg)
        # self.action_space = spaces.Box(np.array([0.0, -1.0]), np.array([1.0, 1.0]))
        # self.action_space = spaces.multi_discrete([(0, 1), (0, 1)])
        self.robot = None
        self.dt = 0.01
        self.plot_summary = plot_summary
        self.trial_count = 0
        self.maxV = 24.6
        self.oscillator_setup()

    def oscillator_setup(self):
        config = [self.maxV, 2.0, 1.0, 12.0, 2.5,
                  0.1]

        a = config[0]
        alpha = config[1]
        tr = config[2]
        ta = config[3]
        beta = config[4]
        fk = config[5]

        self.robot = Matsuoka(id=0, tr=tr, ta=ta, beta=beta, w12=alpha, w21=alpha, A=a, w_prev=0,  w_next=0, fk=fk,
                              dt=self.dt)

    def step(self, action):
        # activation = 1/(1+np.exp(action[0]))
        # dv = 0.1
        # activation1 = list(np.array(range(10))*0.1)[action]
        activation = [list(np.array(range(10)) * 0.1)[action[0]], list(np.array(range(10)) * 0.1)[action[1]]]
        # activation = 1.0
        # self.robot.activation(np.array([activation, activation]))
        self.robot.activation(np.array(activation))
        # target_dfreq = self.clamp(self.dt * action[1], -0.1, 0.1)
        # target_dfreq = 0.0

        # self.robot.int_freq(target_dfreq)
        # target_freq = action[1]
        # self.robot.set_freq(target_freq)

        self.robot.step(f_next1=0.0, f_next2=0.0)
        self._assign_throttle(self.robot.theta[-1])

        p.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()

        self._envStepCounter += 1

        if done:
            print("trial:", self.trial_count)
            print("steps:", len(self.robot.theta))
            print("total step:", self._envStepCounter)
            self.trial_count += 1
            if self.plot_summary:
                self.robot.plot_traj()
            self.robot.reset_traj()

        return np.array(self._observation), reward, done, {}

    def _assign_throttle(self, target_v):

        vt = self.clamp(target_v, -self.maxV, self.maxV)
        self.vt = vt
        print(vt)

        p.setJointMotorControl2(bodyUniqueId=self.botId,
                                jointIndex=0,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=vt)
        p.setJointMotorControl2(bodyUniqueId=self.botId,
                                jointIndex=1,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=-vt)
