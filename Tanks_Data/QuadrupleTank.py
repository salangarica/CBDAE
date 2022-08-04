import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
import sys
import random
import threading
from scipy.io import loadmat, savemat
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pandas as pd


class QuadrupleTank():
    def __init__(self, x0, Hmax, voltmax):
        self.x0 = x0
        self.t = 0

        # Parámetros
        self.A = [28, 32, 28, 32] # cm^2
        self.a = [0.071, 0.057, 0.071, 0.057] # cm^2
        self.g = 981 # cm/s^2
        self.rho = 1 # g/cm^3
        self.kout = 0.5
        self.kin = 2.5
        self.time_scaling = 1
        self.gamma = [0.4, 0.55] # %
        self.volt = [0, 0] # %
        self.voltmax = voltmax
        self.x = self.x0
        self.ti = 0
        self.Ts = 0
        self.Hmax = Hmax
        self.Hmin = 0.0

    # Restricciones físicas de los tanques
    def Limites(self):
        for i in range(len(self.x)):
            if self.x[i] > self.Hmax:
                self.x[i] = self.Hmax
            elif self.x[i] <1e-2:
                self.x[i] = 1e-2

        for i in range(2):
            if self.volt[i] > 1:
                self.volt[i] = 1
            elif self.volt[i] < -1:
                self.volt[i] = -1

    # Ecuaciones diferenciales de los tanques
    def xd_func(self, x, t):
        xd0 = -self.a[0]/self.A[0]*np.sqrt(2*self.g*x[0]) + self.a[2]/self.A[0]*np.sqrt(2*self.g*x[2]) + self.gamma[0]*self.kin*self.volt[0]*self.voltmax/self.A[0]
        xd1 = -self.a[1]/self.A[1]*np.sqrt(2*self.g*x[1]) + self.a[3]/self.A[1]*np.sqrt(2*self.g*x[3]) + self.gamma[1]*self.kin*self.volt[1]*self.voltmax/self.A[1]
        xd2 = -self.a[2]/self.A[2]*np.sqrt(2*self.g*x[2]) + (1 - self.gamma[1])*self.kin*self.volt[1]*self.voltmax/self.A[2]
        xd3 = -self.a[3]/self.A[3]*np.sqrt(2*self.g*x[3]) + (1 - self.gamma[0])*self.kin*self.volt[0]*self.voltmax/self.A[3]
        res = [xd0,xd1,xd2, xd3]
        for i in range(len(res)):
            if np.isnan(res[i]):
                res[i] = 0
        return np.multiply(self.time_scaling, res)

    # Integración en "tiempo real"
    def sim(self):
        self.x0 = np.array(self.x) # Estado actual se vuelve condición inicial para el nuevo estado
        self.Ts = time.time() - self.ti
        #self.Ts = 0.7
        #self.Ts = time.time() - self.ti
        self.Ts = 7
        t = np.linspace(0, self.Ts, 2)
        x = odeint(self.xd_func, self.x0, t)  # Perform integration using Fortran's LSODA (Adams & BDF methods)
        self.x = [x[-1, 0], x[-1,1], x[-1, 2], x[-1, 3]]
        self.Limites()
        #print(self.x)
        self.ti = time.time()
        return self.x


def generate_pattern(n_seqs=500, seqlen=60):
    u1_seqs = []
    u2_seqs = []
    for n in range(n_seqs):
        u1_val = max(random.random(), 0.05)
        u2_val = max(random.random(), 0.05)

        u1_seqs.append([u1_val for i in range(seqlen)])
        u2_seqs.append([u2_val for i in range(seqlen)])

    u1_seqs = np.concatenate(u1_seqs).reshape(-1, 1)
    u2_seqs = np.concatenate(u2_seqs).reshape(-1, 1)

    u_seqs = np.concatenate([u1_seqs, u2_seqs], axis=1)
    u_seqs = u_seqs + np.random.normal(loc=np.zeros_like(u_seqs), scale=np.ones_like(u_seqs)*0.01)
    u_seqs = np.array(pd.DataFrame(u_seqs).rolling(window=20, min_periods=0, win_type='hamming').mean())
    u_seqs = u_seqs + np.random.normal(loc=np.zeros_like(u_seqs), scale=np.ones_like(u_seqs)*0.0015)
    #u_seqs = savgol_filter(u_seqs, window_length=21, polyorder=3, axis=0)

    return u_seqs

if __name__ == '__main__':
    series = []
    sistema = QuadrupleTank(x0=[50,50,50,50], Hmax=50, voltmax=10)
    sistema.time_scaling = 1 # Para el tiempo

    # input1 = [[0, 0] for i in range(400)]
    # input2 = [[0, 1] for i in range(400)]
    # input3 = [[1, 0] for i in range(400)]
    # input4 = list(loadmat('prbs.mat')['u'][:20000, :])
    # x = np.linspace(0, 5000, 5000)
    # sin1 = 0.5*np.sin(x*(np.pi/180)).reshape(-1, 1) + 0.5
    # sin2 = 0.5*np.sin(2*x*(np.pi/180)).reshape(-1, 1) + 0.5
    # input5 = list(np.concatenate([sin1, sin2], axis=1))
    #
    # sin1 = 0.5*np.sin(x/2*(np.pi/180)).reshape(-1, 1) + 0.5
    # sin2 = 0.5*np.sin(x/4*(np.pi/180)).reshape(-1, 1) + 0.5
    # input6 = list(np.concatenate([sin1, sin2], axis=1))
    #
    # inputs = np.array(input1 + input2 + input3 + input4 + input5 + input6)

    inputs = generate_pattern(n_seqs=800, seqlen=60)
    plt.figure()
    plt.plot(inputs[:2000, :])
    plt.show()
    print(inputs.shape)

    #noise = np.random.normal(loc=0, scale=0.05, size=inputs.shape)
    #inputs = inputs + noise
    #print(inputs.shape)

    for i in range(len(inputs)):
        sistema.volt = list(inputs[i, :])
        series.append(sistema.sim())

    series = np.array(series)

    plt.figure()
    #plt.title('Alturas')
    plt.subplot(2, 1, 1)
    plt.plot(series[:, 0], label='Tanque1')
    plt.plot(series[:, 1], label='Tanque2')
    plt.plot(series[:, 2], label='Tanque3')
    plt.plot(series[:, 3], label='Tanque4')
    plt.legend()
    plt.grid()

    #plt.figure()
    #plt.title('Inputs')
    plt.subplot(2, 1, 2)
    plt.plot(inputs[:, 0], label='Input1')
    plt.plot(inputs[:, 1], label='Input2')
    plt.grid()
    plt.legend()

    plt.show()

    #series_noise = series + np.random.normal(loc=0, scale=1, size=series.shape)
    #inputs_noise = inputs + np.random.normal(loc=0, scale=0.1, size=inputs.shape)

    # prob_saturation = 0.05
    # length_saturation = 60
    # for i in range(len(series_noise) - length_saturation):
    #     saturation = True if random.random() < prob_saturation else False
    #     if saturation:
    #         sat_value = 0 if random.random() < 0.5 else 50
    #         tanks = np.random.choice([0, 1, 2, 3])
    #         series_noise[i: i + length_saturation, tanks] = np.ones(series_noise[i: i + length_saturation, tanks].shape)*sat_value
    #

    # plt.figure()
    # plt.title('Alturas_noise')
    # plt.plot(series_noise[:, 0], label='Tanque1')
    # plt.plot(series_noise[:, 1], label='Tanque2')
    # plt.plot(series_noise[:, 2], label='Tanque3')
    # plt.plot(series_noise[:, 3], label='Tanque4')
    # plt.legend()

    np.save('data_clean_v2.npy', np.concatenate([inputs, series], axis=1))
    #np.save('data_noise.npy', np.concatenate([inputs_noise, series_noise], axis=1))



