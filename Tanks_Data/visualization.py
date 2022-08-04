import torch
import pickle
import torch
import matplotlib.pyplot as plt


data = torch.load('data_white_noise.pkl')

for i in range(data.shape[1]):
    plt.figure()
    plt.plot(data[:, i], label= i + 1)
    plt.grid()
    plt.show()