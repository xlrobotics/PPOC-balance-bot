import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# from scipy.integrate import odeint

class Matsuoka:

    def __init__(self, id, tr, ta, beta, w12, w21, A,
                 w_prev=1, w_next=1, q=1, dt=0.016, s1=1, s2=1, fk=1):  # dt = 0.016
        # all zero will be equilibrium point, need slightly different initial value to trigger the oscillation
        # inner states of the oscillator
        self.u1 = 0.0
        self.u2 = 0.0
        self.v1 = 0.0
        self.v2 = 0.0

        self.y1 = 0.0
        self.y2 = 0.0

        self.id = id
        self.fk = fk  # tuning frequency for tau_r and tau_a
        self.origin_fk = fk
        self.tau_r = tr  #* self.fk
        self.tau_a = ta  #* self.fk
        self.s1 = s1
        self.s2 = s2

        self.beta = beta
        self.w12 = w12
        self.w21 = w21

        self.w_prev = w_prev
        self.w_next = w_next

        self.q = q
        self.dt = dt
        self.A = A

        # self.kp = kp # feedback coefficient for position

        # joint position input command memory, extract current theta command from self.theta[-1]
        self.theta = [0]

    def step(self, f_prev1=0.0, f_prev2=0.0, f_next1=0.0, f_next2=0.0):

        # computing feedback terms from the sensor
        # self.fb1 = self.kp * fb
        # self.fb2 = -self.kp * fb

        # Euler integration
        in_outer1 = self.w_prev * f_prev1 + self.w_next * f_next1
        in_outer2 = self.w_prev * f_prev2 + self.w_next * f_next2

        du1 = 1/(self.tau_r*self.fk) * (-self.u1 - self.beta*self.v1 - self.w12*self.y2 - in_outer1 + self.s1)  # + self.fb1)
        dv1 = 1/(self.tau_a*self.fk) * (-self.v1 + self.y1**self.q)

        self.u1 += du1 * self.dt
        self.v1 += dv1 * self.dt
        self.y1 = np.maximum(0.0, self.u1)

        du2 = 1/(self.tau_r*self.fk) * (-self.u2 - self.beta*self.v2 - self.w21*self.y1 - in_outer2 + self.s2)  # + self.fb2)
        dv2 = 1/(self.tau_a*self.fk) * (-self.v2 + self.y2**self.q)

        self.u2 += du2 * self.dt
        self.v2 += dv2 * self.dt
        self.y2 = np.maximum(0.0, self.u2)

        self.theta.append(self.A*(self.y1 - self.y2))

    def activation(self, params):
        self.s1, self.s2 = params[0], params[1]

    def set_freq(self, param):
        self.fk = param * self.origin_fk
        print(self.fk)

    def int_freq(self, df):
        self.fk += df

    def set_amp(self, param):
        self.A = param

    def set_frame_step(self, param):
        self.dt = param

    def plot_traj(self):
        plt.plot(self.theta)
        plt.show()

    def reset_traj(self):
        self.theta = [0]
        self.u1 = 0.0
        self.u2 = 0.0
        self.v1 = 0.0
        self.v2 = 0.0

        self.y1 = 0.0
        self.y2 = 0.0