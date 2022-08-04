import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
import copy
import os
import colorednoise as cn


def generate_noise(data_clean, multiplier=1, noise_inputs=False, plot=False):
    if noise_inputs:
        data_clean[:, :2] = data_clean[:, :2]*30
    else:
        inputs = data_clean[:, :2]
        data_clean = data_clean[:, 2:]

    scaler = StandardScaler()
    scaler.fit(data_clean)
    scales = scaler.scale_

    beta = 0  # the exponent

    # White Noises
    samples = data_clean.shape[0]
    y_white = [cn.powerlaw_psd_gaussian(beta, samples) for i in range(data_clean.shape[1])]
    y_white = np.array(y_white).transpose() * multiplier

    mult = [0.5, 0.7, 1, 1.2, 1.5]
    p = 0.05
    data_salt_and_pepper = copy.deepcopy(data_clean + y_white)

    signs = [-1, 1]
    for j in range(data_clean.shape[0]):
        selected_mults = np.array([random.choice(mult) for j in range(data_clean.shape[1])])
        selected_probs = np.array([int(random.random() < p) for j in range(data_clean.shape[1])])
        selected_signs = np.array([random.choice(signs) for j in range(data_clean.shape[1])])
        addition = scales * selected_mults * selected_probs * selected_signs
        data_salt_and_pepper[j, :] += addition
    data_salt_and_pepper[np.where(data_salt_and_pepper < 0)] = 0

    if not noise_inputs:
        data_salt_and_pepper = np.concatenate([inputs, data_salt_and_pepper], axis=1)
    else:
        data_salt_and_pepper[:, :2] = data_salt_and_pepper[:, :2]/30

    if plot:
        plt.figure()
        plt.title(multiplier)
        plt.plot(data_salt_and_pepper[1000:1300, 2])
        plt.grid()
    plt.show()

    return data_salt_and_pepper



data_clean = np.load('data_clean.npy')


generate_noise(data_clean, multiplier=1, noise_inputs=False, plot=True)

noise_inputs = False
if noise_inputs:
    data_clean[:, :2] = data_clean[:, :2]*30

multiplier = 2.5
scaler = StandardScaler()
scaler.fit(data_clean)


if not noise_inputs:
    inputs = data_clean[:, :2]
    data_clean = data_clean[:, 2:]


# White Noise
beta = 0 # the exponent
samples = len(data_clean) # number of samples to generate
y_white = [cn.powerlaw_psd_gaussian(beta, samples) for i in range(data_clean.shape[1])]
y_white = np.array(y_white).transpose()*multiplier

# Pink Noise
beta = 1 # the exponent
samples = len(data_clean) # number of samples to generate
y_pink = [cn.powerlaw_psd_gaussian(beta, samples) for i in range(data_clean.shape[1])]
y_pink = np.array(y_pink).transpose()*multiplier


# Browmanin Noise
beta = 2 # the exponent
samples = len(data_clean) # number of samples to generate
y_brown = [cn.powerlaw_psd_gaussian(beta, samples) for i in range(data_clean.shape[1])]
y_brown = np.array(y_brown).transpose()*multiplier

if not noise_inputs:
    n1 = 1
    n2 = 1
    n3 = 2
else:
    n1 = 2
    n2 = 2
    n3 = 2

# Combined
beta = 0
samples = len(data_clean) # number of samples to generate
y1 = [cn.powerlaw_psd_gaussian(beta, samples) for i in range(n1)]
y1 = np.array(y1).transpose()*multiplier

beta = 1
samples = len(data_clean) # number of samples to generate
y2 = [cn.powerlaw_psd_gaussian(beta, samples) for i in range(n2)]
y2 = np.array(y2).transpose()*multiplier

beta = 2
samples = len(data_clean) # number of samples to generate
y3 = [cn.powerlaw_psd_gaussian(beta, samples) for i in range(n3)]
y3 = np.array(y3).transpose()*multiplier




y_all = np.concatenate([y1, y2, y3], axis=1)


# Salt and pepper
mult = [0.3, 0.5, 0.7, 1, 1.2, 1.5]
p = 0.05
data_salt_and_pepper = copy.deepcopy(data_clean + y_white)
if not noise_inputs:
    scales = scaler.scale_[2:]
else:
    scales = scaler.scale_

signs = [-1, 1]
for i in range(data_clean.shape[0]):
    selected_mults = np.array([random.choice(mult) for j in range(data_clean.shape[1])])
    selected_probs = np.array([int(random.random() < p) for j in range(data_clean.shape[1])])
    selected_signs = np.array([random.choice(signs) for j in range(data_clean.shape[1])])
    addition = scales*selected_mults*selected_probs*selected_signs
    data_salt_and_pepper[i, :] += addition
data_salt_and_pepper[np.where(data_salt_and_pepper < 0)] = 0


# for i in range(data_salt_and_pepper.shape[1]):
#     plt.figure()
#     #plt.plot(data_salt_and_pepper[:, i])
#     plt.plot(data_clean[:, i] + y_white[:, i])
#     plt.grid()
#     plt.show()

data_white_noise = data_clean + y_white
data_pink_noise = data_clean + y_pink
data_brown_noise = data_clean + y_brown
data_all_noises = data_clean + y_all


if not noise_inputs:
    data_clean = np.concatenate([inputs, data_clean], axis=1)
    data_white_noise = np.concatenate([inputs, data_white_noise], axis=1)
    data_pink_noise = np.concatenate([inputs, data_pink_noise], axis=1)
    data_brown_noise = np.concatenate([inputs, data_brown_noise], axis=1)
    data_all_noises = np.concatenate([inputs, data_all_noises], axis=1)
    data_salt_and_pepper = np.concatenate([inputs, data_salt_and_pepper], axis=1)

# datas = [data_white_noise, data_pink_noise, data_brown_noise, data_all_noises, data_salt_and_pepper]
# names = ['white', 'pink', 'brown', 'all', 'salt_pepper']
# for i in range(len(datas)):
#     plt.figure(figsize=(21, 10))
#     plt.suptitle(names[i])
#     for j in range(datas[i].shape[1]):
#         plt.subplot(3, 2, j + 1)
#         plt.plot(datas[i][:, j], label='signal: {}'.format(j + 1))
#         plt.grid()
#         plt.legend()
#     plt.show()

# if noise_inputs:
#     if not os.path.exists('Noised_Inputs/'):
#         os.makedirs('Noised_Inputs/')
#     torch.save(data_clean, 'Noised_Inputs/data_clean.pkl')
#     #torch.save(data_white_noise, 'Noised_Inputs/data_white_noise.pkl')
#     torch.save(data_pink_noise, 'Noised_Inputs/data_pink_noise.pkl')
#     torch.save(data_brown_noise, 'Noised_Inputs/data_brown_noise.pkl')
#     torch.save(data_all_noises, 'Noised_Inputs/data_all_noises.pkl')
#     torch.save(data_salt_and_pepper, 'Noised_Inputs/data_saltPepper_noises.pkl')
# else:
#     if not os.path.exists('No_Noised_Inputs_v2/'):
#         os.makedirs('No_Noised_Inputs_v2/')
#     torch.save(data_clean, 'No_Noised_Inputs_v2/data_clean.pkl')
#     torch.save(data_white_noise, 'No_Noised_Inputs_v2/data_white_noise.pkl')
#     torch.save(data_pink_noise, 'No_Noised_Inputs_v2/data_pink_noise.pkl')
#     torch.save(data_brown_noise, 'No_Noised_Inputs_v2/data_brown_noise.pkl')
#     torch.save(data_all_noises, 'No_Noised_Inputs_v2/data_all_noises.pkl')
#     torch.save(data_salt_and_pepper, 'No_Noised_Inputs_v2/data_saltPepper_noises.pkl')
