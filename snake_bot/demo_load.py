import pybullet as p
import time
import math

p.connect(p.GUI)
plane = p.createCollisionShape(p.GEOM_PLANE)

p.createMultiBody(0, plane)

sphereUid = p.loadBullet("snakebot_simple.bullet")

sphereRadius = 0.05
dt = 1. / 240.
SNAKE_NORMAL_PERIOD = 0.1  # 1.5
m_wavePeriod = SNAKE_NORMAL_PERIOD

m_waveLength = 4
m_wavePeriod = 1.5
m_waveAmplitude = 0.4
m_waveFront = 0.0
# our steering value
m_steering = 0.0
m_segmentLength = sphereRadius * 2.0
forward = 0

while(1):
    keys = p.getKeyboardEvents()
    for k, v in keys.items():

        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            m_steering = -.2
        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
            m_steering = 0
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            m_steering = .2
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
            m_steering = 0

    amp = 0.2
    offset = 0.6
    numMuscles = p.getNumJoints(sphereUid)
    scaleStart = 1.0

    # start of the snake with smaller waves.
    # I think starting the wave at the tail would work better ( while it still goes from head to tail )
    if (m_waveFront < m_segmentLength * 4.0):
        scaleStart = m_waveFront / (m_segmentLength * 4.0)

    segment = numMuscles - 1

    # we simply move a sin wave down the body of the snake.
    # this snake may be going backwards, but who can tell ;)
    for joint in range(p.getNumJoints(sphereUid)):
        segment = joint  # numMuscles-1-joint
        # map segment to phase
        phase = (m_waveFront - (segment + 1) * m_segmentLength) / m_waveLength
        phase -= math.floor(phase)
        phase *= math.pi * 2.0

        # map phase to curvature
        targetPos = math.sin(phase) * scaleStart * m_waveAmplitude

        # // steer snake by squashing +ve or -ve side of sin curve
        if (m_steering > 0 and targetPos < 0):
            targetPos *= 1.0 / (1.0 + m_steering)

        if (m_steering < 0 and targetPos > 0):
            targetPos *= 1.0 / (1.0 - m_steering)

        # set our motor
        p.setJointMotorControl2(sphereUid,
                                joint,
                                p.POSITION_CONTROL,
                                targetPosition=targetPos + m_steering,
                                force=30)

    # wave keeps track of where the wave is in time
    m_waveFront += dt / m_wavePeriod * m_waveLength
    p.stepSimulation()

    time.sleep(dt)