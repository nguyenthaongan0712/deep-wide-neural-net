import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import seaborn as sns
sns.set()

from collections import OrderedDict

import sys
sys.path.append("../")
from NTK import LinearNeuralTangentKernel, TwoLayersNet, train_net, circle_transform, variance_est, cpu_tuple, kernel_mats


def plot_nn(gamma_data, target_data):
	
	gamma_data = torch.tensor(gamma_data)
	target_data = torch.tensor(target_data).float()
	input_data = circle_transform(gamma_data)

	"""### Create Plot"""

	gamma_vec = torch.tensor(np.linspace(-np.pi, np.pi, 100))
	circle_points = circle_transform(gamma_vec)

	for i in range(10):
	# 1000 width first
		net = TwoLayersNet(1000)
		train_net(net, 1000, input_data, target_data)
		output_vec = net(circle_points).cpu()
		plt.plot(gamma_vec.numpy(), output_vec.detach().numpy(), color='red',
			linestyle='--', alpha = 0.3, label = '$n=1000$')
	# 50 width
		net = TwoLayersNet(50)
		train_net(net, 1000, input_data, target_data)
		output_vec = net(circle_points).cpu()
		plt.plot(gamma_vec.numpy(), output_vec.detach().numpy(), color='green',
		linestyle='--', alpha = 0.3, label = '$n=50$')

	#print('Completed initialisation {}'.format(i))

	plt.xlabel('$\gamma$')
	plt.ylabel('$f_{ \\theta}(sin(\gamma),cos(\gamma))$')
	net = TwoLayersNet(1000)
	K_testvtrain, K_trainvtrain = kernel_mats(net, gamma_data, gamma_vec, kernels = 'both')
	K_trainvtrain_inv = torch.inverse(K_trainvtrain)
	"""### Getting the GP process plot is harder (I think) because of a lack of a standard kernel, here is an attempt but it is probably horrifically inefficient"""

	n_pts=100

	temp_mat = torch.mm(K_testvtrain, K_trainvtrain_inv)	

	# number of points in plot

	target_data = target_data.cpu()

	mean_vec = torch.mm(temp_mat, target_data.unsqueeze(1))


	variance_vec = variance_est(10000, 100, temp_mat, 10000)

	plt.plot(gamma_vec.numpy(), mean_vec.view(-1).detach().numpy()+1.28*np.sqrt(variance_vec.detach().numpy()),
	 color='darkblue', linestyle = '--')
	plt.plot(gamma_vec.numpy(), mean_vec.view(-1).detach().numpy()-1.28*np.sqrt(variance_vec.detach().numpy()), 
	 color='darkblue', linestyle = '--', label = '$n=\infty, \{P_{10}, P_{90}\}$')
	plt.plot(gamma_vec.numpy(), mean_vec.view(-1).detach().numpy()+0*np.sqrt(variance_vec.detach().numpy()), 
	 color='darkblue', label = '$n=\infty, P_{50}$')


	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys(), loc = 'upper left')


	plt.show()
