import numpy as np
import torch
from torch.autograd import Variable
import pdb

def pairwise_distances(x, y=None):
	'''
	Input: x is a Nxd matrix
		   y is an optional Mxd matirx
	Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
			if y is not given then use 'y=x'.
	i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
	'''
	x = x.cuda()
	y = y.cuda()
	x_norm = (x ** 2).sum(1).view(-1, 1)
	if y is not None:
		y_t = torch.transpose(y, 0, 1)
		y_norm = (y ** 2).sum(1).view(1, -1)
	else:
		y_t = torch.transpose(x, 0, 1)
		y_norm = x_norm.view(1, -1)

	dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
	return torch.clamp(dist, 0.0, np.inf)

def cos_batch(x, y):
	"Returns the cosine distance batchwise"
	# x is the feature: bs * d
	# y is the feature: bt * d
	# return: bs * bt
	# print(x.size())

	bs = x.size(0)
	D = x.size(1)
	assert(x.size(1)==y.size(1))
	x = x.contiguous().view(bs, D)
	x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
	y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
	cos_dis = torch.mm(x, torch.transpose(y,0,1))

	beta = 0.1
	min_score = cos_dis.min()
	max_score = cos_dis.max()
	threshold = min_score + beta * (max_score - min_score)
	res = cos_dis - threshold

	return torch.nn.functional.relu(res)

def cos_batch_torch(x, y):
	"Returns the cosine distance batchwise"
	# x is the feature: bs * d
	# y is the feature: bt * d
	# return: bs * bt
	# print(x.size())

	bs = x.size(0)
	D = x.size(1)
	assert(x.size(1)==y.size(1))
	x = x.contiguous().view(bs, D)
	x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
	y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
	cos_dis = torch.mm(x, torch.transpose(y,0,1))
	cos_dis = 1 - cos_dis

	beta = 0.1
	min_score = cos_dis.min()
	max_score = cos_dis.max()
	threshold = min_score + beta * (max_score - min_score)
	res = cos_dis - threshold

	return torch.nn.functional.relu(res)

def IPOT_torch_batch_uniform(C, bs, bt, beta=0.5, iteration=50):
	# C is the distance matrix
	# c: bs by bt
	sigma = torch.ones(bt, 1).cuda()/float(bt)
	T = torch.ones(bs, bt).cuda()
	A = torch.exp(-C/beta).float().cuda()
	for t in range(iteration):
		Q = A * T # bs * bt
		for k in range(1):
			delta = 1 / (bs * torch.mm(Q, sigma))
			a = torch.mm(torch.transpose(Q,0,1), delta)
			sigma = 1 / (float(bt) * a)
		T = delta * Q * sigma.transpose(1,0)

	return T

def GW_torch_batch(Cs, Ct, bs, bt, p, q, beta=0.5, iteration=5, OT_iteration=20):
	one_m = torch.ones(bt, 1).float().cuda()
	one_n = torch.ones(bs, 1).float().cuda()

	Cst = torch.mm(torch.mm(Cs**2, p), torch.transpose(one_m, 0, 1)) + \
	      torch.mm(one_n, torch.mm(torch.transpose(q,0,1), torch.transpose(Ct**2, 0, 1))) # bs by bt
	gamma = torch.mm(p, q.transpose(1,0)) # outer product, init
	for i in range(iteration):
		C_gamma = Cst - 2 * torch.mm(torch.mm(Cs, gamma), torch.transpose(Ct, 0, 1))
		gamma = IPOT_torch_batch_uniform(C_gamma, bs, bt, beta=beta, iteration=OT_iteration)

	Cgamma = Cst - 2 * torch.mm(torch.mm(Cs, gamma), torch.transpose(Ct, 0, 1))
	return gamma.detach(), Cgamma

def GW_distance(X, Y, p, q, lamda=0.5, iteration=5, OT_iteration=20):
	'''
	:param X, Y: Source and target featuers , batchsize by embed_dim
	:param p, q: probability vectors
	:param lam: regularization
	:return: GW distance
	'''
	Cs = cos_batch_torch(X, X).float().cuda()
	Ct = cos_batch_torch(Y, Y).float().cuda()

	bs = Cs.size(0)
	bt = Ct.size(1)
	T, Cst = GW_torch_batch(Cs, Ct, bs, bt, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
	temp = torch.mm(torch.transpose(Cst,0,1), T)
	distance = batch_trace(temp, bs)
	return distance

def edge_alignment(X, Y, lamda=1e-1, iteration=5, OT_iteration=20):
	bs = X.size(0)
	bt = Y.size(0)
	p = (torch.ones(bs, 1)/bs).cuda()
	q = (torch.ones(bt, 1)/bt).cuda()
	return GW_distance(X, Y, p, q, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration)

def node_alignment(X, Y, iteration=50):
	bs = X.size(0)
	bt = Y.size(0)
	Cst = cos_batch(X, Y)
	Cst = Cst.transpose(0,1)
	Cst = Cst.float().cuda()
	T = IPOT_torch_batch_uniform(Cst, bs, bt, iteration=iteration)
	distance = torch.norm((X - torch.mm(T, Y)).abs(), 2, 1).sum() / float(bs)
	return distance, T

def batch_trace(input_matrix, bs):
	a = torch.eye(bs).cuda()
	b = a * input_matrix
	return torch.mean(torch.sum(b,-1))


if __name__ == '__main__':
	Fs = torch.randn(256,128)
	Ft = torch.randn(256,128)
	dis = edge_alignment(Fs, Ft) + node_alignment(Fs, Ft)[0]
	print(dis)