"""Script to train an R-separate model on a previously trained
G-LIS (does not have to contain any LIS modules).
Example:
	python train_r.py --dataset folder --dataroot /path/to/datasets/celeba \
		--crop_size 160 --image_size 80 --code_size 256 --norm weight \
		--r_iterations 1 --lr 0.00005 --niter 2500 --spatial_dropout_r 0.1 \
		--save_path_r /path/to/checkpoints/r-exp01 \
		--load_path /path/to/checkpoints/exp01

	Trains an R-separate based on a previous training of a G-LIS with 1 LIS
	modules which was saved to /path/to/checkpoints/exp01.
	R-separate is trained for 2.5k batches at learning rate 0.00005 and
	spatial dropout of 10%. It is saved to /path/to/checkpoints/r-exp01.
"""
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import os.path
import numpy as np
import imgaug as ia
from scipy import misc

from common import plotting

from common.model import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',                          required = True,
	help = 'cifar10 | lsun | imagenet | folder | lfw')

parser.add_argument('--lsun_class',                       default = 'bedroom',
	help = 'class of lsun dataset to use')

parser.add_argument('--dataroot',                         required = True,
	help = 'path to dataset')

parser.add_argument('--batch_size',         type = int,   default = 32,
	help = 'input batch size')

parser.add_argument('--image_size',         type = int,   default = -1,
	help = 'image size')

parser.add_argument('--width',              type = int,   default = -1,
	help = 'image width')

parser.add_argument('--height',             type = int,   default = -1,
	help = 'image height')

parser.add_argument('--crop_size',          type = int,   default = -1,
	help = 'crop size before scaling')

parser.add_argument('--crop_width',         type = int,   default = -1,
	help = 'crop width before scaling')

parser.add_argument('--crop_height',        type = int,   default = -1,
	help = 'crop height before scaling')

parser.add_argument('--code_size',          type = int,   default = 128,
	help = 'size of latent code')

parser.add_argument('--nfeature',           type = int,   default = 64,
	help = 'number of features of first conv layer')

parser.add_argument('--nlayer',             type = int,   default = -1,
	help = 'number of down/up conv layers')

parser.add_argument('--norm',                             default = 'none',
	help = 'type of normalization: none | batch | weight | weight-affine')

parser.add_argument('--load_path',                        required = True,
	help = 'load to continue existing experiment')

parser.add_argument('--save_path_r',                      default = None,
	help = 'path to save generated files')

parser.add_argument('--load_path_r',                      default = None,
	help = 'load to continue existing experiment')

parser.add_argument('--lr',                 type = float, default = 0.0001,
	help = 'learning rate')

parser.add_argument('--test_interval',      type = int,   default = 1000,
	help = 'how often to test reconstruction')

parser.add_argument('--test_lr',            type = float, default = 0.01,
	help = 'learning rate for reconstruction test')

parser.add_argument('--test_steps',         type = int,   default = 50,
	help = 'number of steps in running reconstruction test')

parser.add_argument('--vis_interval',       type = int,   default = 100,
	help = 'how often to save generated samples')

parser.add_argument('--vis_size',           type = int,   default = 10,
	help = 'size of visualization grid')

parser.add_argument('--vis_row',            type = int,   default = -1,
	help = 'height of visualization grid')

parser.add_argument('--vis_col',            type = int,   default = -1,
	help = 'width of visualization grid')

parser.add_argument('--save_interval',      type = int,   default = 500,
	help = 'how often to save network')

parser.add_argument('--niter',              type = int,   default = 50000,
	help = 'number of iterations to train')

parser.add_argument('--final_test',         action = 'store_true', default = False,
	help = 'do final test')

parser.add_argument('--ls',                 action = 'store_true', default = False,
	help = 'use LSGAN')

parser.add_argument('--output_scale',       action = 'store_true', default = False,
	help = 'save x*2-1 instead of x when saving image')

parser.add_argument('--net',                              default = 'last',
	help = 'network prefix to use for loading G')

parser.add_argument('--spatial_dropout_r',  type = float,   default = 0,
	help = 'spatial dropout applied to R')

parser.add_argument('--r_iterations', type = int, default = 3,
	help = 'how many LIS modules to use in G')

parser.add_argument('--g_upscaling',  default='fractional',
	help = 'upscaling method to use in G: fractional|nearest|bilinear')

opt = parser.parse_args()
print(opt)

transform_list = []

if (opt.crop_height > 0) and (opt.crop_width > 0):
	transform_list.append(transforms.CenterCrop(opt.crop_height, crop_width))
elif opt.crop_size > 0:
	transform_list.append(transforms.CenterCrop(opt.crop_size))

if (opt.height > 0) and (opt.width > 0):
	transform_list.append(transforms.Scale(opt.height, opt.width))
elif opt.image_size > 0:
	transform_list.append(transforms.Scale(opt.image_size))
	transform_list.append(transforms.CenterCrop(opt.image_size))
	opt.height = opt.image_size
	opt.width = opt.image_size
else:
	raise ValueError('must specify valid image size')

transform_list.append(transforms.ToTensor())

if (opt.vis_row <= 0) or (opt.vis_col <= 0):
	opt.vis_row = opt.vis_size
	opt.vis_col = opt.vis_size

if opt.nlayer < 0:
	opt.nlayer = 0
	s = max(opt.width, opt.height)
	while s >= 8:
		s = (s + 1) // 2
		opt.nlayer = opt.nlayer + 1

if opt.dataset == 'cifar10':
	dataset1 = datasets.CIFAR10(root = opt.dataroot, download = True,
		transform = transforms.Compose(transform_list))
	dataset2 = datasets.CIFAR10(root = opt.dataroot, train = False,
		transform = transforms.Compose(transform_list))
	def get_data(k):
		if k < len(dataset1):
			return dataset1[k][0]
		else:
			return dataset2[k - len(dataset1)][0]
else:
	if opt.dataset in ['imagenet', 'folder', 'lfw']:
		dataset = datasets.ImageFolder(root = opt.dataroot,
			transform = transforms.Compose(transform_list))
	elif opt.dataset == 'lsun':
		dataset = datasets.LSUN(db_path = opt.dataroot, classes = [opt.lsun_class + '_train'],
			transform = transforms.Compose(transform_list))
	def get_data(k):
		return dataset[k][0]

data_index = torch.load(os.path.join(opt.dataroot, 'data_index.pt'))
train_index = data_index['train']

if opt.final_test:
	test_index = data_index['final_test']
else:
	test_index = data_index['running_test']

gen = GeneratorLearnedInputSpace(opt.width, opt.height, opt.nfeature, opt.nlayer, opt.code_size, opt.norm, n_lis_layers=opt.r_iterations, upscaling=opt.g_upscaling)
print(gen)
gen.cuda()
testfunc = nn.MSELoss()

r = build_reverser(opt.width, opt.height, opt.nfeature//2, opt.nlayer, opt.code_size, opt.norm, opt.spatial_dropout_r)
print(r)
r.cuda()

if not opt.final_test:
	dis = build_discriminator(opt.width, opt.height, opt.nfeature, opt.nlayer, opt.norm)
	print(dis)
	dis.cuda()

	for param in dis.parameters():
		param.requires_grad = False

	if opt.ls:
		lossfunc = nn.MSELoss()
	else:
		lossfunc = nn.BCELoss()
	lossfunc_r = nn.MSELoss()
	r_opt = optim.RMSprop(r.parameters(), lr = opt.lr, eps = 1e-6, alpha = 0.9)

history = plotting.History()
history.add_group("loss-r", ["train"], increasing=False)
history.add_group("loss-stage1", ["train"], increasing=False)
history.add_group("loss-stage2", ["train"], increasing=True)
history.add_group("loss-stage-mix", ["train-stage1", "train-stage2"], increasing=True)

state = {}

def load_state(path, prefix, gen_only = False):
	gen.load_state_dict(torch.load(os.path.join(opt.load_path, 'net_archive', '{0}_gen.pt'.format(prefix))))

	if not gen_only:
		dis.load_state_dict(torch.load(os.path.join(opt.load_path, 'net_archive', '{0}_dis.pt'.format(prefix))))

def load_state_r(path, prefix):
	global history
	saved_state = torch.load(os.path.join(opt.load_path_r, 'net_archive', '{0}_state.pt'.format(prefix)))
	state.update(saved_state)
	history = plotting.History.from_string(saved_state['history'])

	r_fp = os.path.join(opt.load_path_r, 'net_archive', '{0}_r.pt'.format(prefix))
	r_opt_fp = os.path.join(opt.load_path_r, 'net_archive', '{0}_r_opt.pt'.format(prefix))
	if os.path.isfile(r_fp):
		r.load_state_dict(r_fp)
	if os.path.isfile(r_opt_fp):
		r_opt.load_state_dict(r_opt_fp)

def save_state(path, prefix):
	torch.save(r.state_dict(), os.path.join(opt.save_path_r, 'net_archive', '{0}_r.pt'.format(prefix)))
	torch.save(r_opt.state_dict(), os.path.join(opt.save_path_r, 'net_archive', '{0}_r_opt.pt'.format(prefix)))

	state.update({
		'index_shuffle' : index_shuffle,
		'current_iter' : current_iter,
		'best_iter' : best_iter,
		'min_loss' : min_loss,
		'current_sample' : current_sample,
		'history' : history.to_string()
	})
	torch.save(state, os.path.join(opt.save_path_r, 'net_archive', '{0}_state.pt'.format(prefix)))

loss_plotter = plotting.LossPlotter(
	history.get_group_names(),
	history.get_groups_increasing(),
	save_to_fp=os.path.join(opt.save_path_r, "loss.jpg")
)
loss_plotter.start_batch_idx = 100


def visualize(code, filename, filename_r, filename_all):
	gen.eval()
	generated = torch.Tensor(code.size(0), 3, opt.height, opt.width)
	generated_r = [torch.Tensor(code.size(0), 3, opt.height, opt.width) for _ in range(opt.r_iterations)]
	generated_all = []
	for i in range((code.size(0) - 1) // opt.batch_size + 1):
		batch_size = min(opt.batch_size, code.size(0) - i * opt.batch_size)
		batch_code = Variable(code[i * opt.batch_size : i * opt.batch_size + batch_size])
		generated_images, _ = gen(batch_code, n_execute_lis_layers=0)
		generated[i * opt.batch_size : i * opt.batch_size + batch_size].copy_(generated_images.data)

		generated_images_r_by_iter = []
		for r_iter in range(opt.r_iterations):
			generated_images_r, _ = gen(batch_code, n_execute_lis_layers=r_iter+1)
			generated_r[r_iter][i * opt.batch_size : i * opt.batch_size + batch_size].copy_(generated_images_r.data)
			generated_images_r_by_iter.append(generated_images_r)

		for imgidx in range(generated_images.size(0)):
			image = generated_images[imgidx].data.cpu().numpy()
			image = (image * 255).astype(np.uint8).transpose((1, 2, 0))
			generated_all.append(image)

			for r_iter in range(opt.r_iterations):
				image_r = generated_images_r_by_iter[r_iter][imgidx].data.cpu().numpy()
				image_r = (image_r * 255).astype(np.uint8).transpose((1, 2, 0))
				generated_all.append(image_r)

	if opt.output_scale:
		torchvision.utils.save_image(generated * 2 - 1, filename, opt.vis_row)
		for r_iter in range(opt.r_iterations):
			torchvision.utils.save_image(generated_r[r_iter] * 2 - 1, filename_r.format(r_iter), opt.vis_row)
	else:
		torchvision.utils.save_image(generated, filename, opt.vis_row)
		for r_iter in range(opt.r_iterations):
			torchvision.utils.save_image(generated_r[r_iter], filename_r.format(r_iter), opt.vis_row)
	misc.imsave(filename_all, ia.draw_grid(generated_all, cols=opt.vis_col*(1+opt.r_iterations)))
	gen.train()

def visualize(code, filename, filename_r, filename_both):
	gen.eval()
	r.eval()
	generated = torch.Tensor(code.size(0), 3, opt.height, opt.width)
	generated_r = torch.Tensor(code.size(0), 3, opt.height, opt.width)
	generated_both = []
	for i in range((code.size(0) - 1) // opt.batch_size + 1):
		batch_size = min(opt.batch_size, code.size(0) - i * opt.batch_size)
		batch_code = Variable(code[i * opt.batch_size : i * opt.batch_size + batch_size])
		generated_images, _ = gen(batch_code, n_execute_lis_layers=opt.r_iterations)
		generated[i * opt.batch_size : i * opt.batch_size + batch_size].copy_(generated_images.data)

		code_reversed = r(generated_images)
		generated_images_r, _ = gen(code_reversed, n_execute_lis_layers=opt.r_iterations)
		generated_r[i * opt.batch_size : i * opt.batch_size + batch_size].copy_(generated_images_r.data)

		for j in range(generated_images.size(0)):
			image = generated_images[j].data.cpu().numpy()
			image_r = generated_images_r[j].data.cpu().numpy()
			image = (image * 255).astype(np.uint8).transpose((1, 2, 0))
			image_r = (image_r * 255).astype(np.uint8).transpose((1, 2, 0))
			generated_both.append(image)
			generated_both.append(image_r)

	if opt.output_scale:
		torchvision.utils.save_image(generated * 2 - 1, filename, opt.vis_row)
		torchvision.utils.save_image(generated_r * 2 - 1, filename_r, opt.vis_row)
	else:
		torchvision.utils.save_image(generated, filename, opt.vis_row)
		torchvision.utils.save_image(generated_r, filename_r, opt.vis_row)
	misc.imsave(filename_both, ia.draw_grid(generated_both, cols=opt.vis_col*2))
	gen.train()
	r.train()

def makedirs():
	if not os.path.exists(opt.save_path_r):
		os.makedirs(opt.save_path_r)
	for sub_folder in ('samples', 'samples_r', 'samples_both', 'running_test', 'net_archive', 'log'):
		if not os.path.exists(os.path.join(opt.save_path_r, sub_folder)):
			os.mkdir(os.path.join(opt.save_path_r, sub_folder))

load_state(opt.load_path, opt.net)

if opt.load_path_r is not None:
	if opt.save_path_r is None:
		opt.save_path_r = opt.load_path_r
	if opt.load_path_r != opt.save_path_r:
		makedirs()
	vis_code = torch.load(os.path.join(opt.load_path_r, 'samples', 'vis_code.pt')).cuda()

	load_state_r(opt.load_path_r, 'last')
	index_shuffle = state['index_shuffle']
	current_iter = state['current_iter']
	best_iter = state['best_iter']
	min_loss = state['min_loss']
	current_sample = state['current_sample']
else:
	if opt.save_path_r is None:
		raise ValueError('must specify save path if not continue training')
	makedirs()
	vis_code = torch.randn(opt.vis_row * opt.vis_col, opt.code_size).cuda()
	torch.save(vis_code, os.path.join(opt.save_path_r, 'samples', 'vis_code.pt'))

	index_shuffle = torch.randperm(train_index.size(0))
	current_iter = 0
	best_iter = 0
	min_loss = 1e100
	current_sample = 0

	vis_target = torch.Tensor(min(test_index.size(0), opt.vis_row * opt.vis_col), 3, opt.height, opt.width)
	for i in range(vis_target.size(0)):
		vis_target[i].copy_(get_data(test_index[i]))
	if opt.output_scale:
		torchvision.utils.save_image(vis_target * 2 - 1, os.path.join(opt.save_path_r, 'running_test', 'target.jpg'), opt.vis_row)
	else:
		torchvision.utils.save_image(vis_target, os.path.join(opt.save_path_r, 'running_test', 'target.jpg'), opt.vis_row)

ones = Variable(torch.ones(opt.batch_size, 1).cuda())
zeros = Variable(torch.zeros(opt.batch_size, 1).cuda())

loss_record = torch.zeros(opt.test_interval, 3)

visualize(
	vis_code,
	filename=os.path.join(opt.save_path_r, 'samples', 'sample_{0}.jpg'.format(current_iter)),
	filename_r=os.path.join(opt.save_path_r, 'samples_r', 'sample_{0}_r.jpg'.format(current_iter)),
	filename_both=os.path.join(opt.save_path_r, 'samples_both', 'sample_{0}_both.jpg'.format(current_iter))
)

# train for --niter batches
while current_iter < opt.niter:
	current_iter = current_iter + 1
	current_loss_record = loss_record[(current_iter - 1) % opt.test_interval]

	rand_code = Variable(torch.randn(opt.batch_size, opt.code_size).cuda())
	generated, _ = gen(rand_code, n_execute_lis_layers=opt.r_iterations)

	# loss D orig samples
	loss = lossfunc(
		dis(Variable(generated.data, volatile=True)),
		zeros
	)
	current_loss_record[0] = loss.data[0]
	generated.detach()

	r.zero_grad()

	# train R
	rand_code_fixed = r(generated)
	loss = lossfunc_r(rand_code_fixed, rand_code)
	current_loss_record[1] = loss.data[0]
	loss.backward()
	r_opt.step()

	# loss D error-fixed samples
	generated_fixed, _ = gen(rand_code_fixed, n_execute_lis_layers=opt.r_iterations)
	loss = lossfunc(
		dis(Variable(generated_fixed.data, volatile=True)),
		zeros
	)
	current_loss_record[2] = loss.data[0]

	history.add_value("loss-r", "train", current_iter, current_loss_record[1], average=False)
	history.add_value("loss-stage1", "train", current_iter, current_loss_record[0], average=False)
	history.add_value("loss-stage2", "train", current_iter, current_loss_record[2], average=False)
	history.add_value("loss-stage-mix", "train-stage1", current_iter, current_loss_record[0], average=False)
	history.add_value("loss-stage-mix", "train-stage2", current_iter, current_loss_record[2], average=False)
	print('{0} | loss: r:{1} dis-stage1:{2} dis-stage2:{3}'.format(current_iter, current_loss_record[1], current_loss_record[0], current_loss_record[2]))

	if current_iter % opt.vis_interval == 0:
		visualize(
			vis_code,
			filename=os.path.join(opt.save_path_r, 'samples', 'sample_{0}.jpg'.format(current_iter)),
			filename_r=os.path.join(opt.save_path_r, 'samples_r', 'sample_{0}_r.jpg'.format(current_iter)),
			filename_both=os.path.join(opt.save_path_r, 'samples_both', 'sample_{0}_both.jpg'.format(current_iter))
		)
		loss_plotter.plot(history)

	if current_iter % opt.save_interval == 0:
		save_state(opt.save_path_r, current_iter)
