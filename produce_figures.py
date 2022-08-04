import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.data_processor import DataProcessor
from sklearn.metrics import mean_squared_error
from matplotlib import cm
from collections import OrderedDict
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import os
from pylab import rcParams
import json
rcParams.update({'font.size': 30})

class Evaluation:
    def __init__(self, seqlen=30, folder_model='Nets/RNNMSE_20/', folder_tanks='../../Tanks_Data/Noised_Inputs/', num_features=6,
                 input_indices=[0, 1], output_indices=[2,3, 4, 5], use_one_layer=True, missing=False, noise='white', noise_inputs=False):
        with open(folder_model + 'meta.json', 'r') as meta:
            self.meta = json.load(meta)
        self.seqlen = seqlen
        self.processor = DataProcessor(seqlen=seqlen)
        try:
            print('Noise Power: {}'.format(self.meta['noise_power']))
            self.data_dict = self.processor.process_tanks(shuffle=False, folder=folder_tanks, type_of_noise=noise, noise_inputs=noise_inputs, noise_power=self.meta['noise_power'])
        except:
            self.data_dict = self.processor.process_tanks(shuffle=False, folder=folder_tanks, type_of_noise=noise, noise_inputs=noise_inputs)

        self.folder_model = folder_model
        self.scaler = self.data_dict['scaler']
        self.scaler_preproc = self.data_dict['scaler_preproc']
        self.clean_data = np.array(self.data_dict['test_data_preproc'][:, -1, :])
        self.clean_data = self.scaler_preproc.inverse_transform(self.clean_data)
        self.noisy_data = self.scaler.inverse_transform(self.data_dict['test_data'][:, -1, :])
        orig_array = np.array(torch.load('{}/eval_response.pkl'.format(self.folder_model)))
        orig_shape = orig_array.shape
        self.model_response = self.scaler.inverse_transform(orig_array.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        self.missing = missing
        if self.missing:
            orig_array = np.array(torch.load('{}/eval_response_no_loss.pkl'.format(self.folder_model)))
            orig_shape = orig_array.shape
            self.model_response_no_loss = self.scaler.inverse_transform(orig_array.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        [self.X_in_test_list, self.h_test_list, self.X_out_test_list, self.X_clean_test_list] = torch.load('{}/h_test_list.pkl'.format(self.folder_model))
        self.h_statistics = torch.load('{}/h_statistics.pkl'.format(self.folder_model))
        self.mse_list = []
        self.mse_list_noisy = []
        self.mse_list_no_loss = []
        self.num_features = num_features
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.use_one_layer = use_one_layer
        self.constrastive_loss = np.array(torch.load('{}/eval_losses.pkl'.format(self.folder_model)))[:, 1]
        self.total_loss = np.array(torch.load('{}/eval_losses.pkl'.format(self.folder_model)))[:, 0]


    def prediction_with_data(self, X, X_clean, n_preds=5, seqlen=10, ratio=0.85, name='X_in'):
        X_train = X[:int(len(X)*ratio)]
        X_test = X[int(len(X)*ratio):]
        X_test_clean = X_clean[int(len(X)*ratio):]

        X_train_list = []
        y_train_list = []
        for elem in X_train:
            elem_processed = elem[:, -(n_preds + seqlen):, :]
            X_train_list.append(elem_processed[:, :seqlen, :].reshape(-1, seqlen*self.num_features))
            y_train_list.append(elem_processed[:, seqlen:, self.output_indices])

        X_test_list = []
        y_test_list = []
        y_test_clean_list = []
        for i in range(len(X_test)):
            elem_processed = X_test[i][:, -(n_preds + seqlen):, :]
            elem_processed_clean = X_test_clean[i][:, -(n_preds + seqlen):, :]
            X_test_list.append(elem_processed[:, :seqlen, :].reshape(-1, seqlen * self.num_features))
            y_test_list.append(elem_processed[:, seqlen:, self.output_indices])
            y_test_clean_list.append(elem_processed_clean[:, seqlen:, self.output_indices])


        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices, :]
        y_train = y_train[indices, :, :]

        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        y_test_clean = np.concatenate(y_test_clean_list, axis=0)

        rmse_list = []
        for i in range(len(self.output_indices)):
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train[:, :, i])
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_pred, y_test_clean[:, :, i]))
            rmse_list.append(rmse)
            print('Model {} {} RMSE: {}'.format(name, i, rmse))
        return rmse_list


    def prediction_with_latent(self, h, X, X_clean, n_preds=5, ratio=0.85, name='h'):
        X_input = []
        X_output = []
        X_output_clean = []
        if self.use_one_layer:
            layer_shape = int(h[0].shape[1]/2)
        else:
            layer_shape = h[0].shape[1]

        for i in range(len(h)):
            for j in range(h[i].shape[0] - n_preds - 1):
                X_input.append(h[i][j, -layer_shape:].reshape(1, -1))
                X_output.append(X[i][j + 1: j + n_preds + 1, -1, :].reshape(1, n_preds, -1))
                X_output_clean.append(X_clean[i][j + 1: j + n_preds + 1, -1, :].reshape(1, n_preds, -1))


        X_input = np.concatenate(X_input, axis=0)
        X_output = np.concatenate(X_output, axis=0)
        X_output_clean = np.concatenate(X_output_clean, axis=0)
        length = X_input.shape[0]

        X_train = X_input[:int(length*ratio), :]
        y_train = X_output[:int(length * ratio), :, :]
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices, :]
        y_train = y_train[indices, :, :]

        X_test = X_input[int(length * ratio):, :]
        y_test = X_output[int(length * ratio):, :, :]
        y_test_clean = X_output_clean[int(length * ratio):, :, :]

        rmse_list = []
        for i in range(len(self.output_indices)):
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train[:, :, i])
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_pred, y_test_clean[:, :, i]))
            rmse_list.append(rmse)
            print('Model {} {} RMSE: {}'.format(name, i, rmse))
        return rmse_list

    def average_cosine_similarity(self, h_list):
        cosine_sum = 0
        h = np.concatenate(h_list, axis=0)
        if self.use_one_layer:
            layer_shape = int(h[0].shape[1]/2)
        else:
            layer_shape = h[0].shape[1]

        for i in range(h.shape[0] - 1):
            cosine_sum += cosine_similarity(h[i, -layer_shape:].reshape(1, -1), h[i + 1, -layer_shape:].reshape(1, -1))[0, 0]
        avg_cosine_sim = cosine_sum/(h.shape[0] - 1)
        print('Average cosine similarity: {}'.format(avg_cosine_sim))
        return avg_cosine_sim

    def run(self, plot=False, seqlen=10, n_preds=5, plot_contrastive=False):
        for i in range(len(self.model_response)):
            self.mse_list.append(np.sqrt(mean_squared_error(self.clean_data[:, 2:], self.model_response[i][:, 2:])))
            self.mse_list_noisy.append(
                np.sqrt(mean_squared_error(self.noisy_data[:, 2:], self.model_response[i][:, 2:])))
            if self.missing:
                self.mse_list_no_loss.append(
                    np.sqrt(mean_squared_error(self.clean_data[:, 2:], self.model_response_no_loss[i][:, 2:])))

        argmin = np.argmin(np.array(self.mse_list))
        min_ = self.mse_list[argmin]
        # if self.missing:
        #     min_no_loss = self.mse_list_no_loss[argmin]
        #     print('Min: {}|{}  Min no missing: {}'.format(argmin, min_, min_no_loss))
        # else:
        #     print('Min: {}|{}'.format(argmin, min_))
        lim_ord = 5

        pca_error = np.array(self.h_statistics['pca_error'])
        pca_error_argmin = np.argmin(pca_error[lim_ord:, :], axis=0)
        pca_error_argmax = np.argmax(pca_error[lim_ord:, :], axis=0)

        total_variance = np.array(self.h_statistics['total_variance'])
        total_variance_argmin = np.argmin(total_variance[lim_ord:])
        total_variance_argmax = np.argmax(total_variance[lim_ord:])

        mean_variance = np.array(self.h_statistics['mean_variance'])
        mean_variance_argmin = np.argmin(mean_variance[lim_ord:])
        mean_variance_argmax = np.argmax(mean_variance[lim_ord:])

        cosine_sim = np.array(self.h_statistics['cosine_sim'])
        cosine_sim_div = np.diff(cosine_sim)
        cosine_sim_argmin = np.argmin(cosine_sim[lim_ord:])
        cosine_sim_argmax = np.argmax(cosine_sim[lim_ord:])

        contrastive_argmin = np.argmin(self.constrastive_loss[lim_ord:])
        contrastive_argmax = np.argmax(self.constrastive_loss[lim_ord:])

        total_loss_argmin = np.argmin(self.total_loss)
        total_loss_min = self.total_loss[total_loss_argmin]

        x = [t for t in range(len(self.mse_list))]

        plt.figure(figsize=(21, 10))
        plt.subplot(4, 1, 1)
        if self.missing:
            title = 'MSE: Total loss min: {} | Clean loss min: {} | Final clean loss: {} | Total loss no missing {}|{}'.format(
                round(float(self.mse_list[total_loss_argmin]), 2),
                round(float(min_), 2),
                round(float(self.mse_list[-1]), 2),
                round(float(self.mse_list_no_loss[total_loss_argmin]), 2),
                round(float(self.mse_list_no_loss[-1]), 2))
            plt.title(title)
        else:
            title = 'MSE: Total loss min: {} | Clean loss min: {} | Final clean loss: {}'.format(
                round(float(self.mse_list[total_loss_argmin]), 2),
                round(float(min_), 2),
                round(float(self.mse_list[-1]), 2))
            plt.title(title)
        print(title)
        plt.plot(self.mse_list, label='MSE')
        plt.plot(self.mse_list_noisy, label='MSE_noisy')
        plt.axvline(total_loss_argmin, label='Total loss min', linestyle='--')
        if self.missing:
            plt.plot(self.mse_list_no_loss, label='MSE no missing')
        plt.scatter(argmin, min_, s=50)
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.title('PCA error')
        for i in range(pca_error.shape[1]):
            plt.plot(pca_error[:, i], label='N {}'.format(10 + i))
            plt.scatter(pca_error_argmin[i] + lim_ord, pca_error[pca_error_argmin[i] + lim_ord, i], s=50)
            plt.scatter(pca_error_argmax[i] + lim_ord, pca_error[pca_error_argmax[i] + lim_ord, i], s=50)

        plt.legend()
        plt.grid()

        plt.subplot(4, 1, 3)
        if plot_contrastive:
            plt.title('Constrastive loss')
            plt.plot(self.constrastive_loss, label='Contrastive loss')
            plt.scatter(contrastive_argmin + lim_ord, self.constrastive_loss[contrastive_argmin + lim_ord], s=50)
            plt.scatter(contrastive_argmax + lim_ord, self.constrastive_loss[contrastive_argmax + lim_ord], s=50)
        else:
            plt.title('Variances')
            plt.plot(total_variance, label='Total Variance')
            plt.plot(mean_variance, label='Mean Variance')
            plt.scatter(total_variance_argmin + lim_ord, total_variance[total_variance_argmin + lim_ord], s=50)
            plt.scatter(total_variance_argmax + lim_ord, total_variance[total_variance_argmax + lim_ord], s=50)
            plt.scatter(mean_variance_argmin + lim_ord, mean_variance[mean_variance_argmin + lim_ord], s=50)
            plt.scatter(mean_variance_argmax + lim_ord, mean_variance[mean_variance_argmax + lim_ord], s=50)
        plt.grid()
        plt.legend()

        ax = plt.subplot(4, 1, 4)
        ax.set_title('Cosine similarity')
        ax.plot(cosine_sim, label='Cosine Sim')
        ax.scatter(cosine_sim_argmin + lim_ord, cosine_sim[cosine_sim_argmin + lim_ord], s=50)
        ax.scatter(cosine_sim_argmax + lim_ord, cosine_sim[cosine_sim_argmax + lim_ord], s=50)
        ax.grid()
        ax.legend()
        # ax2 = ax.twinx()
        # color = 'tab:red'
        # ax2.set_ylabel('diff', color=color)  # we already handled the x-label with ax1
        # ax2.plot(cosine_sim_div, color=color)

        plt.tight_layout()
        #plt.savefig(self.folder_model + 'h_statistics.svg')
        if plot:
            plt.show()
        plt.close()

        for i in range(self.clean_data.shape[1]):
            plt.figure(figsize=(21, 10))
            plt.plot(self.noisy_data[:, i], label='Noisy', alpha=0.1)
            plt.plot(self.model_response[argmin][:, i], label='{}'.format(i))
            plt.plot(self.clean_data[:, i], label='Clean')

            plt.legend()
            plt.grid()
            if plot:
                plt.show()
            plt.close()



    def plot_results_fig2(self, plot=False, seqlen=10, n_preds=5, plot_contrastive=False, save=False):
        for i in range(len(self.model_response)):
            self.mse_list.append(np.sqrt(mean_squared_error(self.clean_data[:, 2:], self.model_response[i][:, 2:])))
            self.mse_list_noisy.append(
                np.sqrt(mean_squared_error(self.noisy_data[:, 2:], self.model_response[i][:, 2:])))
            if self.missing:
                self.mse_list_no_loss.append(
                    np.sqrt(mean_squared_error(self.clean_data[:, 2:], self.model_response_no_loss[i][:, 2:])))

        argmin = np.argmin(np.array(self.mse_list))
        min_ = self.mse_list[argmin]
        if self.missing:
            min_no_loss = self.mse_list_no_loss[argmin]
            # print('Min: {}|{}  Min no missing: {}'.format(argmin, min_, min_no_loss))
        else:
            pass
            # print('Min: {}|{}'.format(argmin, min_))
        lim_ord = 5

        pca_error = np.array(self.h_statistics['pca_error'])
        pca_error_argmin = np.argmin(pca_error[lim_ord:, :], axis=0)
        pca_error_argmax = np.argmax(pca_error[lim_ord:, :], axis=0)

        pca_error_add = np.array(self.h_statistics['pca_error_add'])
        pca_error_argmin_add = np.argmin(pca_error_add[lim_ord:, :], axis=0)
        pca_error_argmax_add = np.argmax(pca_error_add[lim_ord:, :], axis=0)

        total_variance = np.array(self.h_statistics['total_variance'])
        total_variance_argmin = np.argmin(total_variance[lim_ord:])
        total_variance_argmax = np.argmax(total_variance[lim_ord:])

        total_variance_add = np.array(self.h_statistics['total_variance_add'])
        total_variance_argmin_add = np.argmin(total_variance_add[lim_ord:])
        total_variance_argmax_add = np.argmax(total_variance_add[lim_ord:])

        mean_variance = np.array(self.h_statistics['mean_variance'])
        mean_variance_argmin = np.argmin(mean_variance[lim_ord:])
        mean_variance_argmax = np.argmax(mean_variance[lim_ord:])

        mean_variance_add = np.array(self.h_statistics['mean_variance_add'])
        mean_variance_argmin_add = np.argmin(mean_variance_add[lim_ord:])
        mean_variance_argmax_add = np.argmax(mean_variance_add[lim_ord:])

        cosine_sim = np.array(self.h_statistics['cosine_sim'])
        cosine_sim_argmin = np.argmin(cosine_sim[lim_ord:])
        cosine_sim_argmax = np.argmax(cosine_sim[lim_ord:])

        cosine_sim_add = np.array(self.h_statistics['cosine_sim_add'])
        cosine_sim_argmin_add = np.argmin(cosine_sim_add[lim_ord:])
        cosine_sim_argmax_add = np.argmax(cosine_sim_add[lim_ord:])

        contrastive_argmin = np.argmin(self.constrastive_loss[lim_ord:])
        contrastive_argmax = np.argmax(self.constrastive_loss[lim_ord:])

        total_loss_argmin = np.argmin(self.total_loss)
        total_loss_min = self.total_loss[total_loss_argmin]

        x = [t for t in range(len(self.mse_list))]

        plt.figure(figsize=(21, 10))
        plt.subplot(4, 1, 1)
        if self.missing:
            title = 'MSE: Total loss min: {} | Clean loss min: {} | Final clean loss: {} | Total loss no missing {}|{}'.format(
                round(float(self.mse_list[total_loss_argmin]), 2),
                round(float(min_), 2),
                round(float(self.mse_list[-1]), 2),
                round(float(self.mse_list_no_loss[total_loss_argmin]), 2),
                round(float(self.mse_list_no_loss[-1]), 2))
            plt.title(title)
        else:
            title = 'MSE: Total loss min: {} | Clean loss min: {} | Final clean loss: {}'.format(
                round(float(self.mse_list[total_loss_argmin]), 2),
                round(float(min_), 2),
                round(float(self.mse_list[-1]), 2))
            plt.title(title)
        print(title)

        plt.plot(self.mse_list, label='MSE')
        plt.plot(self.mse_list_noisy, label='MSE_noisy')
        plt.axvline(total_loss_argmin, label='Total loss min', linestyle='--')
        if self.missing:
            plt.plot(self.mse_list_no_loss, label='MSE no missing')
        plt.scatter(argmin, min_, s=50)
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.title('PCA error')
        for i in range(pca_error.shape[1]):
            plt.plot(pca_error[:, i], label='N {}'.format(10 + i))
            plt.scatter(pca_error_argmin[i] + lim_ord, pca_error[pca_error_argmin[i] + lim_ord, i], s=50)
            plt.scatter(pca_error_argmax[i] + lim_ord, pca_error[pca_error_argmax[i] + lim_ord, i], s=50)

            plt.plot(pca_error_add[:, i], label='Nadd {}'.format(10 + i))
            plt.scatter(pca_error_argmin_add[i] + lim_ord, pca_error_add[pca_error_argmin_add[i] + lim_ord, i], s=50)
            plt.scatter(pca_error_argmax_add[i] + lim_ord, pca_error_add[pca_error_argmax_add[i] + lim_ord, i], s=50)

        plt.legend()
        plt.grid()

        plt.subplot(4, 1, 3)
        if plot_contrastive:
            plt.title('Constrastive loss')
            plt.plot(self.constrastive_loss, label='Contrastive loss')
            plt.scatter(contrastive_argmin + lim_ord, self.constrastive_loss[contrastive_argmin + lim_ord], s=50)
            plt.scatter(contrastive_argmax + lim_ord, self.constrastive_loss[contrastive_argmax + lim_ord], s=50)
        else:
            plt.title('Variances')
            plt.plot(total_variance, label='Total Variance')
            plt.plot(mean_variance, label='Mean Variance')
            plt.scatter(total_variance_argmin + lim_ord, total_variance[total_variance_argmin + lim_ord], s=50)
            plt.scatter(total_variance_argmax + lim_ord, total_variance[total_variance_argmax + lim_ord], s=50)
            plt.scatter(mean_variance_argmin + lim_ord, mean_variance[mean_variance_argmin + lim_ord], s=50)
            plt.scatter(mean_variance_argmax + lim_ord, mean_variance[mean_variance_argmax + lim_ord], s=50)
            plt.plot(total_variance_add, label='Total Variance addd')
            plt.plot(mean_variance_add, label='Mean Variance add')

        plt.grid()
        plt.legend()

        ax = plt.subplot(4, 1, 4)
        ax.set_title('Cosine similarity')
        ax.plot(cosine_sim, label='Cosine Sim')
        ax.scatter(cosine_sim_argmin + lim_ord, cosine_sim[cosine_sim_argmin + lim_ord], s=50)
        ax.scatter(cosine_sim_argmax + lim_ord, cosine_sim[cosine_sim_argmax + lim_ord], s=50)
        ax.plot(cosine_sim_add, label='Cosine Sim add')
        ax.grid()
        ax.legend()
        # ax2 = ax.twinx()
        # color = 'tab:red'
        # ax2.set_ylabel('diff', color=color)  # we already handled the x-label with ax1
        # ax2.plot(cosine_sim_div, color=color)

        plt.tight_layout()
        #plt.savefig(self.folder_model + 'h_statistics.svg')
        if plot:
            plt.show()
        plt.close()
        x_lim = [2000, 5000]
        t = [7*i/60 for i in range(x_lim[0], x_lim[1])]
        for i in range(self.clean_data.shape[1]):
            plt.figure(figsize=(21, 7))
            plt.plot(t, self.noisy_data[x_lim[0]:x_lim[1], i], label='Noisy Signal (Input)', alpha=0.85, color='cornflowerblue', linewidth=2)
            plt.plot(t, self.model_response_no_loss[argmin][x_lim[0]:x_lim[1], i], label='BDAE output', color='tab:red', linewidth=2)
            plt.plot(t, self.clean_data[x_lim[0]:x_lim[1], i], label='Clean Signal', color='darkorange', linewidth=2)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.ylabel('Water Level [cm]')
            plt.xlabel('Time [min]')
            if save and i >= 2:
                pass
                #plt.savefig('utils/Images/BDAE_tanks_results{}.svg'.format(i))
            if plot:
                plt.show()
            plt.close()

    def plot_results_fig1(self, plot=False, seqlen=10, n_preds=5, save=False):
        for i in range(len(self.model_response)):
            self.mse_list.append(np.sqrt(mean_squared_error(self.clean_data[:, 2:], self.model_response[i][:, 2:])))
            self.mse_list_noisy.append(np.sqrt(mean_squared_error(self.noisy_data[:, 2:], self.model_response[i][:, 2:])))
            if self.missing:
                self.mse_list_no_loss.append(np.sqrt(mean_squared_error(self.clean_data[:, 2:], self.model_response_no_loss[i][:, 2:])))

        argmin = np.argmin(np.array(self.mse_list))
        min_ = self.mse_list[argmin]
        # if self.missing:
        #     min_no_loss = self.mse_list_no_loss[argmin]
        #     print('Min: {}|{}  Min no missing: {}'.format(argmin, min_, min_no_loss))
        # else:
        #     print('Min: {}|{}'.format(argmin, min_))
        lim_ord = 5
        print('Min {}'.format(min_))


        pca_error = np.array(self.h_statistics['pca_error'])
        pca_error_argmin = np.argmin(pca_error[lim_ord:, :], axis=0)
        pca_error_argmax = np.argmax(pca_error[lim_ord:, :], axis=0)

        total_variance = np.array(self.h_statistics['total_variance'])
        total_variance_argmin = np.argmin(total_variance[lim_ord:])
        total_variance_argmax = np.argmax(total_variance[lim_ord:])

        mean_variance = np.array(self.h_statistics['mean_variance'])
        mean_variance_argmin = np.argmin(mean_variance[lim_ord:])
        mean_variance_argmax = np.argmax(mean_variance[lim_ord:])

        cosine_sim = np.array(self.h_statistics['cosine_sim'])
        cosine_sim_div = np.diff(cosine_sim)
        cosine_sim_argmin = np.argmin(cosine_sim[lim_ord:])
        cosine_sim_argmax = np.argmax(cosine_sim[lim_ord:])

        contrastive_argmin = np.argmin(self.constrastive_loss[lim_ord:])
        contrastive_argmax = np.argmax(self.constrastive_loss[lim_ord:])

        total_loss_argmin = np.argmin(self.total_loss)
        total_loss_min = self.total_loss[total_loss_argmin]

        x = [t for t in range(len(self.mse_list))]

        plt.figure(figsize=(21, 7))
        plt.plot(self.mse_list_noisy, label='Loss calculated with noisy data', linewidth=2)
        plt.plot(self.mse_list, label='Loss calculated with clean data', linewidth=2)
        # plt.axvline(total_loss_argmin, label='Total loss min', linestyle='--')
        # if self.missing:
        #     plt.plot(self.mse_list_no_loss, label='MSE no missing')
        # plt.scatter(argmin, min_, s=50)
        plt.ylabel('Evaluation Loss')
        plt.xlabel('Epochs')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        if save:
            pass
            #plt.savefig('utils/Images/overfit.svg')
        plt.show()

        i = 2
        #for i in range(self.clean_data.shape[1]):
        plt.figure(figsize=(21, 7))
        plt.plot(self.model_response[argmin][:, i])
        plt.plot(self.clean_data[:, i])

        #plt.xlabel('Timesteps')
        #plt.ylabel('Cms')
        plt.xlim([0, 800])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if save:
            pass
            #plt.savefig('utils/Images/overfit_clean.svg')
        if plot:
            plt.show()
        plt.close()

        # for i in range(self.clean_data.shape[1]):
        plt.figure(figsize=(21, 7))
        plt.plot(self.model_response[-1][:, i])
        plt.plot(self.clean_data[:, i])
        #plt.xlabel('Timesteps')
        #plt.ylabel('Cms')
        plt.xlim([0, 800])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        #plt.savefig('utils/Images/overfit_noisy.svg')
        if plot:
            plt.show()
        plt.close()


if __name__ == '__main__':
    # Baseline
    folder_model = 'Results/Nets_60_saltPepper_Addlinear/'
    folder_tanks = 'Tanks_Data/No_Noised_Inputs/'
    noise = 'saltPepper'
    noise_inputs = False
    seqlen = 30
    plot = True
    plot_contrastive = True
    missing = True
    savefig = False

    list_dir = os.listdir(folder_model)
    list_dir = [elem for elem in list_dir if 'baseline' not in elem and 'figs' not in elem and 'noisePower_3' in elem and '.5' in elem]
    print(list_dir)
    print(list_dir)
    mins_list = []
    for i in range(len(list_dir)):
        print(list_dir[i], '{}|{}'.format(i + 1, len(list_dir)))
        seqlens = [60]#[15, 30, 60, 90]
        next = False
        p = 0
        while not next:
            try:
                seqlen = seqlens[p]
                print('Evaluating with sequence length : {}'.format(seqlen))
                evaluation = Evaluation(seqlen=seqlen, folder_model=folder_model + list_dir[i] + '/',
                                        folder_tanks=folder_tanks, missing=missing, noise=noise)
                min_ = evaluation.plot_results_fig2(plot=plot, plot_contrastive=plot_contrastive, save=savefig)
                next = True
                mins_list.append([list_dir[i], min_])
            except ValueError as e:
                print(e)
                p += 1

    print('Mins')
    print(mins_list)







