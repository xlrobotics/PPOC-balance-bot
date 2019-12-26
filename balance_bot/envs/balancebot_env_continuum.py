import logging
import numpy as np
import math

from gym import spaces
import pybullet as p

from balance_bot.envs.balancebot_env import BalancebotEnv

logger = logging.getLogger(__name__)


class BalancebotEnvContinuum(BalancebotEnv):

    def __init__(self, render=True, discrete=False):
        super().__init__(render=render, discrete=discrete)

    def _assign_throttle(self, action):
        dv = 0.1
        # A = np.tanh(action)
        # deltav = [-10.*dv, -5.*dv, -2.*dv, -0.1*dv, 0, 0.1*dv, 2.*dv, 5.*dv, 10.*dv][action]
        deltav = (action[0]+action[1])/2*dv
        vt = self.clamp(self.vt + deltav, -self.maxV, self.maxV)
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
