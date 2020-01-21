import os
import math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data
import numpy as np

# from time import time
import time

class SnakebotEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render=True, discrete=True, cpg=False, manual_mode=False, body_size=6, division=5, snake_urdf=None):
        self._observation = []
        self._manual_mode = manual_mode  # for manual control testing
        self._obs_buffer = [] #history poses for calculating averaged speed and heading angles
        self._botId = None
        self.discrete_division = division

        if discrete:
            action_space = list(np.ones(body_size)*division)
            if cpg:
                self.action_space = spaces.MultiDiscrete(action_space)
            else:
                self.action_space = spaces.MultiDiscrete(action_space)
        else:
            action_space_n = list(-np.ones(body_size))
            action_space_p = list(np.ones(body_size))
            #TODO: not correctly implemented, will need modification
            self.action_space = spaces.Box(np.array(action_space_n), np.array(action_space_p))

        self.observation_space = spaces.Box(np.array([-math.pi, -math.pi, -5]),
                                            np.array([math.pi, math.pi, 5]))  # pitch, gyro, com.sp.

        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

        self._seed()
        self.default_done = 1500
        self.snake_urdf = snake_urdf
        self.body_size = body_size
        # paramId = p.addUserDebugParameter("My Param", 0, 100, 50)

    @property
    def botId(self):
        return self._botId

    @property
    def observation(self):
        return self._observation

    @property
    def obs_buffer(self):
        return self._obs_buffer

    @property
    def manual_mode(self):
        return self._manual_mode

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # embedded method to create the snake
    def create_robot(self) -> int:
        sphereRadius = 0.05
        # colBoxId = p.createCollisionShapeArray([p.GEOM_BOX, p.GEOM_SPHERE],radii=[sphereRadius+0.03,sphereRadius+0.03],
        #                                        halfExtents=[[sphereRadius,sphereRadius,sphereRadius],[sphereRadius,sphereRadius,sphereRadius]])
        colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[sphereRadius, 2 * sphereRadius, sphereRadius])

        useMaximalCoordinates = True
        mass = 1
        visualShapeId = -1

        link_Masses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        indices = []
        jointTypes = []
        axis = []

        for i in range(self.body_size):  #
            link_Masses.append(1)
            linkCollisionShapeIndices.append(colBoxId)
            linkVisualShapeIndices.append(-1)
            linkPositions.append([0, sphereRadius * 4.0 + 0.01, 0])
            linkOrientations.append([0, 0, 0, 1])
            linkInertialFramePositions.append([0, 0, 0])
            linkInertialFrameOrientations.append([0, 0, 0, 1])
            indices.append(i)
            jointTypes.append(p.JOINT_REVOLUTE)
            axis.append([0, 0, 1])

        basePosition = [0, 0, 0]
        baseOrientation = [0, 0, 0, 1]
        botId = p.createMultiBody(mass,
                                      colBoxId,
                                      visualShapeId,
                                      basePosition,
                                      baseOrientation,
                                      linkMasses=link_Masses,
                                      linkCollisionShapeIndices=linkCollisionShapeIndices,
                                      linkVisualShapeIndices=linkVisualShapeIndices,
                                      linkPositions=linkPositions,
                                      linkOrientations=linkOrientations,
                                      linkInertialFramePositions=linkInertialFramePositions,
                                      linkInertialFrameOrientations=linkInertialFrameOrientations,
                                      linkParentIndices=indices,
                                      linkJointTypes=jointTypes,
                                      linkJointAxis=axis,
                                      useMaximalCoordinates=useMaximalCoordinates)

        return botId

    def step(self, action):

        if self.manual_mode:
            keys = p.getKeyboardEvents()
            m_steering = 0.0
            for k, v in keys.items():
                if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
                    m_steering = -.2
                if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
                    m_steering = 0.0
                if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
                    m_steering = .2
                if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
                    m_steering = 0.0
        else:
            m_steering = 0.0

        self._assign_throttle(action, m_steering)

        p.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()

        self._envStepCounter += 1

        return np.array(self._observation), reward, done, {}

    def reset(self):
        # reset is called once at initialization of simulation
        self.vt = 0
        self.vd = 0
        self.maxV = 24.6  # 235RPM = 24,609142453 rad/sec
        self._envStepCounter = 0

        p.resetSimulation(0)

        self.gravId = p.setGravity(0, 0, -9.8) # m/s^2
        p.setTimeStep(0.01)  # sec

        # p.setGravity(0, 0, -9.8)
        # p.setRealTimeSimulation(0)

        # initialize ground
        plane = p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0, plane)

        # initialize snake position config
        cubeStartPos = [0, 0, 0.001]
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        path = os.path.abspath(os.path.dirname(__file__))

        if self.snake_urdf is not None:
            self._botId = p.loadURDF(os.path.join(path, self.snake_urdf),  # "snakebot_simple.xml"
                                    cubeStartPos,
                                    cubeStartOrientation)
        else:
            self._botId = self.create_robot()

        anistropicFriction = [1, 0.01, 0.01]
        p.changeDynamics(self.botId, -1, lateralFriction=2, anisotropicFriction=anistropicFriction)

        self.jointIds = []
        self.paramIds = []
        # joint dynamics initialization
        for i in range(p.getNumJoints(self.botId)):
            p.getJointInfo(self.botId, i)
            p.changeDynamics(self.botId, i, lateralFriction=2, anisotropicFriction=anistropicFriction)

            info = p.getJointInfo(self.botId, i)
            # print(info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(i)
                self.paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -1, 1, 0.))
                p.resetJointState(self.botId, i, 0.)

        p.addUserDebugParameter(b"amplitude".decode("utf-8"), -1, 1, 0.)
        p.addUserDebugParameter(b"frequency".decode("utf-8"), 0, 1, 0.)
        p.addUserDebugParameter(b"alpha".decode("utf-8"), 0, 1, 0.)
        p.addUserDebugParameter(b"offset".decode("utf-8"), -1, 1, 0.)

        # you *have* to compute and return the observation from reset()
        self._observation = self._compute_observation()
        self._obs_buffer.append(self._observation)

        return np.array(self._observation)

    def debug_mode(self):
        self.reset()

        p.setRealTimeSimulation(1)
        while(1):
            keys = p.getKeyboardEvents()
            p.getCameraImage(320, 200)
            # p.setGravity(0, 0, p.readUserDebugParameter(self.gravId))
            for i in range(len(self.paramIds)):
                c = self.paramIds[i]
                targetPos = p.readUserDebugParameter(c)
                p.setJointMotorControl2(self.botId, self.jointIds[i], p.POSITION_CONTROL, targetPos, force=30.)

            targetAmp = p.readUserDebugParameter(c)

            time.sleep(0.01)

            for k, v in keys.items():
                if (k == p.B3G_BACKSPACE and (v & p.KEY_WAS_TRIGGERED)):
                    return

    def _assign_throttle(self, action, m_steering):
        '''
        :param action:
        :param m_steering:
        :return:
        '''

        target = np.linspace(-1, 1, self.discrete_division)
        m_waveAmplitude = 0.4
        scaleStart = 1.0

        for joint in range(p.getNumJoints(self.botId)):
            # segment = joint  # numMuscles-1-joint
            # map segment to phase
            # phase = (m_waveFront - (segment + 1) * m_segmentLength) / m_waveLength
            # phase -= math.floor(phase)
            # phase *= math.pi * 2.0

            # map phase to curvature
            # targetPos = math.sin(action[i]) * scaleStart * m_waveAmplitude
            targetPos = target[action[joint]] * scaleStart * m_waveAmplitude # direct input command

            # // steer snake by squashing +ve or -ve side of sin curve
            if (m_steering > 0 and targetPos < 0):
                targetPos *= 1.0 / (1.0 + m_steering)

            if (m_steering < 0 and targetPos > 0):
                targetPos *= 1.0 / (1.0 - m_steering)

            # set our motor
            p.setJointMotorControl2(self.botId,
                                    joint,
                                    p.POSITION_CONTROL,
                                    targetPosition=targetPos + m_steering,
                                    force=30)

    def set_done(self, val):
        self.default_done = val

    def _compute_observation(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.botId)
        return [cubeEuler[0], angular[0], self.vt]

    def _compute_reward(self, window_size=10): #TODO: needs to be modified for goal tracking, also needs to consider the curriculum
        return 0.1 - abs(self.vt - self.vd) * 0.005

    def _compute_done(self):
        cubePos, _ = p.getBasePositionAndOrientation(self.botId)
        return cubePos[2] < 0.15 or self._envStepCounter >= self.default_done

    def _render(self, mode='human', close=False):
        pass

    @staticmethod
    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)
