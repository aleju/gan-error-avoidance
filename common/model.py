from __future__ import print_function, division
import torch
import torch.nn as nn
from modules.WeightNormalizedConv import *
from modules.WeightNormalizedLinear import *
from modules.TPReLU import *
from modules.View import *
import random

def build_discriminator(w_in, h_in, f_first, num_down_layers, norm):
	net = nn.Sequential()
	if (w_in % 2 != 0) or (h_in % 2 != 0):
		raise ValueError('input width and height must be even numbers')
	f_prev = 3
	f = f_first
	w = w_in
	h = h_in
	for i in range(num_down_layers):
		if i == num_down_layers - 1:
			pad_w = 0
			pad_h = 0
		else:
			if (w % 4 == 2):
				pad_w = 1
			else:
				pad_w = 0
			if (h % 4 == 2):
				pad_h = 1
			else:
				pad_h = 0
		if (norm == 'weight') or (norm == 'weight-affine'):
			net.add_module('level.{0}.conv'.format(i),
				WeightNormalizedConv2d(f_prev, f, 4, 2, (1 + pad_h, 1 + pad_w),
					scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine')))
		else:
			net.add_module('level.{0}.conv'.format(i),
				nn.Conv2d(f_prev, f, 4, 2, (1 + pad_h, 1 + pad_w)))
		if (norm == 'batch') and (i > 0):
			net.add_module('level.{0}.batchnorm'.format(i),
				nn.BatchNorm2d(f))
		if norm == 'weight':
			net.add_module('level.{0}.tprelu'.format(i),
				TPReLU(f))
		else:
			net.add_module('level.{0}.prelu'.format(i),
				nn.PReLU(f))
		f_prev = f
		f = f * 2
		w = (w + pad_w * 2) // 2
		h = (h + pad_h * 2) // 2
	if (norm == 'weight') or (norm == 'weight-affine'):
		net.add_module('final.conv',
			WeightNormalizedConv2d(f_prev, 1, (h, w)))
	else:
		net.add_module('final.conv',
			nn.Conv2d(f_prev, 1, (h, w)))
	net.add_module('final.sigmoid', nn.Sigmoid())
	net.add_module('final.view', View(1))
	return net

def build_generator(w_out, h_out, f_last, num_up_layers, code_size, norm):
	net = nn.Sequential()
	if (w_out % 2 != 0) or (h_out % 2 != 0):
		raise ValueError('output width and height must be even numbers')
	pad_w = []
	pad_h = []
	w = w_out
	h = h_out
	f = f_last
	for i in range(num_up_layers - 1):
		if (w % 4 == 2):
			pad_w.append(1)
			w = (w + 2) // 2
		else:
			pad_w.append(0)
			w = w // 2
		if (h % 4 == 2):
			pad_h.append(1)
			h = (h + 2) // 2
		else:
			pad_h.append(0)
			h = h // 2
		f = f * 2
	w = w // 2
	h = h // 2
	pad_w.append(0)
	pad_h.append(0)

	if (norm == 'weight') or (norm == 'weight-affine'):
		net.add_module('initial.linear',
			WeightNormalizedLinear(code_size, f * h * w, init_factor = 0.01,
				scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine')))
	else:
		net.add_module('initial.linear',
			nn.Linear(code_size, f * h * w))

	net.add_module('initial.view', View(f, h, w))

	if norm == 'batch':
		net.add_module('initial.batchnorm', nn.BatchNorm2d(f))
	if norm == 'weight':
		net.add_module('initial.tprelu', TPReLU(f))
	else:
		net.add_module('initial.prelu', nn.PReLU(f))

	for i in range(num_up_layers - 1):
		level = num_up_layers - 1 - i

		if (norm == 'weight') or (norm == 'weight-affine'):
			net.add_module('level.{0}.conv'.format(level),
				WeightNormalizedConvTranspose2d(f, f // 2, 4, 2, (1 + pad_h[level], 1 + pad_w[level]),
					scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine')))
		else:
			net.add_module('level.{0}.conv'.format(level),
				nn.ConvTranspose2d(f, f // 2, 4, 2, (1 + pad_h[level], 1 + pad_w[level])))

		if norm == 'batch':
			net.add_module('level.{0}.batchnorm'.format(level),
				nn.BatchNorm2d(f // 2))
		if norm == 'weight':
			net.add_module('level.{0}.tprelu'.format(level),
				TPReLU(f // 2))
		else:
			net.add_module('level.{0}.prelu'.format(level),
				nn.PReLU(f // 2))

		f = f // 2

	if (norm == 'weight') or (norm == 'weight-affine'):
		net.add_module('level.0.conv',
			WeightNormalizedConvTranspose2d(f, 3, 4, 2, (1 + pad_h[0], 1 + pad_w[0])))
	else:
		net.add_module('level.0.conv',
			nn.ConvTranspose2d(f, 3, 4, 2, (1 + pad_h[0], 1 + pad_w[0])))
	net.add_module('level.0.sigmoid', nn.Sigmoid())
	return net

class GeneratorLearnedInputSpace(nn.Module):
	def __init__(self, w_out, h_out, f_last, num_up_layers, code_size, norm, n_lis_layers):
		super(GeneratorLearnedInputSpace, self).__init__()

		if (w_out % 2 != 0) or (h_out % 2 != 0):
			raise ValueError('output width and height must be even numbers')
		pad_w = []
		pad_h = []
		w = w_out
		h = h_out
		f = f_last
		for i in range(num_up_layers - 1):
			if (w % 4 == 2):
				pad_w.append(1)
				w = (w + 2) // 2
			else:
				pad_w.append(0)
				w = w // 2
			if (h % 4 == 2):
				pad_h.append(1)
				h = (h + 2) // 2
			else:
				pad_h.append(0)
				h = h // 2
			f = f * 2
		w = w // 2
		h = h // 2
		pad_w.append(0)
		pad_h.append(0)

		self.w = w
		self.h = h
		self.f = f

		# learning input space
		self.lis_layers = []
		for i in range(n_lis_layers):
			seq = nn.Sequential()

			if (norm == 'weight') or (norm == 'weight-affine'):
				seq.add_module("lis.{0}-1.linear".format(i), WeightNormalizedLinear(code_size, code_size, init_factor = 0.01, scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine')))
			else:
				seq.add_module("lis.{0}-1.linear".format(i), nn.Linear(code_size, code_size))

			#if norm == 'batch':
			#	seq.add_module("lis.{0}-1.norm", nn.BatchNorm2d(f))

			if norm == "weight":
				seq.add_module("lis.{0}-1.act".format(i), TPReLU(code_size))
			else:
				seq.add_module("lis.{0}-1.act".format(i), nn.PReLU(code_size))


			if (norm == 'weight') or (norm == 'weight-affine'):
				seq.add_module("lis.{0}-2.linear".format(i), WeightNormalizedLinear(code_size, code_size, init_factor = 0.01, scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine')))
			else:
				seq.add_module("lis.{0}-2.linear".format(i), nn.Linear(code_size, code_size))

			self.lis_layers.append(seq)
		self.lis_layers = nn.ModuleList(self.lis_layers)

		# initial linear
		initial_linear = []
		if (norm == 'weight') or (norm == 'weight-affine'):
			initial_linear.append(WeightNormalizedLinear(code_size, f * h * w, init_factor = 0.01, scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine')))
		else:
			initial_linear.append(nn.Linear(code_size, f * h * w))

		initial_linear.append(View(f, h, w))

		if norm == 'batch':
			initial_linear.append(nn.BatchNorm2d(f))

		if norm == 'weight':
			initial_linear.append(TPReLU(f))
		else:
			initial_linear.append(nn.PReLU(f))
		self.initial_linear = nn.ModuleList(initial_linear)

		# conv layers
		self.conv_layers = []
		for i in range(num_up_layers - 1):
			level = num_up_layers - 1 - i

			if (norm == 'weight') or (norm == 'weight-affine'):
				self.conv_layers.append(
					WeightNormalizedConvTranspose2d(f, f // 2, 4, 2, (1 + pad_h[level], 1 + pad_w[level]),
						scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine'))
				)
			else:
				self.conv_layers.append(
					nn.ConvTranspose2d(f, f // 2, 4, 2, (1 + pad_h[level], 1 + pad_w[level]))
				)

			if norm == 'batch':
				self.conv_layers.append(nn.BatchNorm2d(f // 2))
			if norm == 'weight':
				self.conv_layers.append(TPReLU(f // 2))
			else:
				self.conv_layers.append(nn.PReLU(f // 2))

			f = f // 2

		if (norm == 'weight') or (norm == 'weight-affine'):
			self.conv_layers.append(
				WeightNormalizedConvTranspose2d(f, 3, 4, 2, (1 + pad_h[0], 1 + pad_w[0]))
			)
		else:
			self.conv_layers.append(
				nn.ConvTranspose2d(f, 3, 4, 2, (1 + pad_h[0], 1 + pad_w[0]))
			)
		self.conv_layers.append(nn.Sigmoid())
		self.conv_layers = nn.ModuleList(self.conv_layers)

	def forward(self, x, n_execute_lis_layers=None):
		#print("x", type(x))
		lis_results = []
		for i, lis_layer in enumerate(self.lis_layers):
			#print("lis_layer", lis_layer)
			#p = (i + 1) / len(self.lis_layers)
			# at 3 layers, break at
			#  0 => (1/2)^3 = 1/8
			#  1 => (1/2)^2 = 1/4
			#  2 => (1/2)^1 = 1/2
			p = (1/2) ** (len(self.lis_layers) - i)
			if not self.training:
				p = 0
			if n_execute_lis_layers is not None:
				if n_execute_lis_layers == "all" or (i+1) <= n_execute_lis_layers:
					p = 0
				else:
					p = 1

			if random.random() < p:
				#if n_execute_lis_layers is not None:
				#	print("breaking at ", i, p)
				break
			#else:
				#if n_execute_lis_layers is not None:
				#	print("NOT breaking at ", i, p)

			#print("lis", type(lis_layer(x)))
			x = x + lis_layer(x)
			lis_results.append(x)

		for layer in self.initial_linear:
			x = layer(x)

		#x = x.view(self.f, self.h, self.w)

		for layer in self.conv_layers:
			x = layer(x)

		return x, lis_results

def build_reverser(w_in, h_in, f_first, num_down_layers, code_size, norm, spatial_dropout_r):
	net = nn.Sequential()
	if (w_in % 2 != 0) or (h_in % 2 != 0):
		raise ValueError('input width and height must be even numbers')
	f_prev = 3
	f = f_first
	w = w_in
	h = h_in
	for i in range(num_down_layers):
		if i == num_down_layers - 1:
			pad_w = 0
			pad_h = 0
		else:
			if (w % 4 == 2):
				pad_w = 1
			else:
				pad_w = 0
			if (h % 4 == 2):
				pad_h = 1
			else:
				pad_h = 0
		if (norm == 'weight') or (norm == 'weight-affine'):
			net.add_module('level.{0}.conv'.format(i),
				WeightNormalizedConv2d(f_prev, f, 4, 2, (1 + pad_h, 1 + pad_w),
					scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine')))
		else:
			net.add_module('level.{0}.conv'.format(i),
				nn.Conv2d(f_prev, f, 4, 2, (1 + pad_h, 1 + pad_w)))
		if i >= 1 and spatial_dropout_r > 0:
			net.add_module('level.{0}.sd'.format(i),
				nn.Dropout2d(spatial_dropout_r))
		if (norm == 'batch') and (i > 0):
			net.add_module('level.{0}.batchnorm'.format(i),
				nn.BatchNorm2d(f))
		if norm == 'weight':
			net.add_module('level.{0}.tprelu'.format(i),
				TPReLU(f))
		else:
			net.add_module('level.{0}.prelu'.format(i),
				nn.PReLU(f))
		f_prev = f
		f = f * 2
		w = (w + pad_w * 2) // 2
		h = (h + pad_h * 2) // 2
	if (norm == 'weight') or (norm == 'weight-affine'):
		net.add_module('final.conv',
			WeightNormalizedConv2d(f_prev, code_size, (h, w)))
	else:
		net.add_module('final.conv',
			nn.Conv2d(f_prev, code_size, (h, w)))
	#net.add_module('final.tanh', nn.Tanh())
	net.add_module('final.view', View(code_size))
	return net
