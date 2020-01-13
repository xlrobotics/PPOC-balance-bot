import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt

class Module(keras.Model):
    def __init__(self, nf):
        super(Module, self).__init__()
        self.dense_1 = Dense(nf, activation='tanh')
        self.dense_2 = Dense(nf, activation='tanh')

    def call(self, inputs, **kwargs):
        t, x = inputs
        h = self.dense_1(x)
        return self.dense_2(h) - 0.25*x

def odeint(func, y0, t, solver):
    dts = t[1:] - t[:-1]
    tk = t[0]
    yk = y0
    hist = [(tk, y0)]
    for dt in dts:
        print(tk, end="\r")
        yk = solver(dt, tk, yk, func)
        tk = tk + dt
        hist.append((tk, yk))
        # sys.stdout.write('\033[2K\033[1G')
    return hist

def midpoint_step_keras(dt, tk, hk, fun):
    k1 = fun([tk, hk])
    k2 = fun([tk + dt, hk + dt*k1])
    return hk + dt * (k1 + k2)/2

# def figure_attention(attention):
#     fig, ax = tfp.subplots(figsize=(4, 3))
#     im = ax.imshow(attention, cmap='jet')
#     fig.colorbar(im)
#     return fig

if __name__ == '__main__':
    tf.enable_eager_execution()
    t_grid = np.linspace(0, 500., 2000)
    h0 = tf.to_float([[1., -1.]])
    model = Module(2)

    hist = odeint(model, h0, t_grid, midpoint_step_keras)

    results = []
    for h in hist:
        with tf.Session() as sess:
            result = h[1].numpy()
            results.append(result[0])
    # plot_op = tfp.plot(figure_attention, [hist])
    # execute_op_as_image(plot_op)
    results = np.array(results)

    plt.plot(results.T[0, :])
    plt.plot(results.T[1, :])
    plt.show()

    exit(0)