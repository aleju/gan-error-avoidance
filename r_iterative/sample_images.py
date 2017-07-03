from __future__ import print_function, division

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
import time
import random

from common import plotting
from common import util
from common.model import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',         type = int,   default = 32,
        help = 'input batch size')

    parser.add_argument('--image_size',         type = int,   default = -1,
        help = 'image size')

    parser.add_argument('--width',              type = int,   default = -1,
        help = 'image width')

    parser.add_argument('--height',             type = int,   default = -1,
        help = 'image height')

    parser.add_argument('--code_size',          type = int,   default = 128,
        help = 'size of latent code')

    parser.add_argument('--nfeature',           type = int,   default = 64,
        help = 'number of features of first conv layer')

    parser.add_argument('--nlayer',             type = int,   default = -1,
        help = 'number of down/up conv layers')

    parser.add_argument('--norm',                             default = 'none',
        help = 'type of normalization: none | batch | weight | weight-affine')

    parser.add_argument('--load_path_g', required = True,
        help = 'path of G to use')

    parser.add_argument('--load_path_r', required = True,
        help = 'path of R to use')

    parser.add_argument('--save_path', required = True,
        help = 'Path to save sampled images to')

    parser.add_argument('--output_scale',       action = 'store_true', default = False,
        help = 'save x*2-1 instead of x when saving image')

    parser.add_argument('--r_iterations', type = int, default = 3,
        help = 'How often to execute the reverse projection via R')

    parser.add_argument('--spatial_dropout_r',  type = float,   default = 0,
        help = 'Spatial dropout applied to R')

    opt = parser.parse_args()
    print(opt)

    if (opt.height > 0) and (opt.width > 0):
        pass
    elif opt.image_size > 0:
        opt.height = opt.image_size
        opt.width = opt.image_size
    else:
        raise ValueError('must specify valid image size')

    if opt.nlayer < 0:
        opt.nlayer = 0
        s = max(opt.width, opt.height)
        while s >= 8:
            s = (s + 1) // 2
            opt.nlayer = opt.nlayer + 1

    gen = build_generator(opt.width, opt.height, opt.nfeature, opt.nlayer, opt.code_size, opt.norm)
    print(gen)
    gen.cuda()
    gen.load_state_dict(torch.load(opt.load_path_g))
    gen.eval()

    r = build_reverser(opt.width, opt.height, opt.nfeature//2, opt.nlayer, opt.code_size, opt.norm, opt.spatial_dropout_r)
    print(r)
    r.cuda()
    r.load_state_dict(torch.load(opt.load_path_r))
    r.eval()

    makedirs(opt.save_path, opt.r_iterations)

    for i in range(20):
        nb_rows = 16
        nb_cols = 16
        vis_code = torch.randn(nb_rows * nb_cols, opt.code_size).cuda()
        last_images = None
        images_by_r = []
        for r_idx in range(1+opt.r_iterations):
            images, vis_code = generate_images(gen, r, vis_code, r_idx, opt.batch_size, opt.output_scale)
            images_by_r.append(images)
            #grid = ia.draw_grid(images, rows=nb_rows, cols=nb_cols)
            grid_full = util.draw_grid(images, rows=nb_rows, cols=nb_cols)
            grid_small = util.draw_grid(images[0:(nb_rows//2 * nb_cols//2)], rows=nb_rows//2, cols=nb_cols//2)
            misc.imsave(os.path.join(opt.save_path, 'sampled_images_r%d' % (r_idx,), 'r%d_full_%04d.jpg' % (r_idx, i)), grid_full)
            misc.imsave(os.path.join(opt.save_path, 'sampled_images_r%d' % (r_idx,), 'r%d_small_%04d.jpg' % (r_idx, i)), grid_small)

        images_chains = []
        for j in range(len(images_by_r[0])):
            images_one_chain = [images_by_r[r_idx][j] for r_idx in range(len(images_by_r))]
            images_chains.append(np.hstack(images_one_chain))
        #images_chains = add_border(images_chains)
        #grid = ia.draw_grid(images_chains, cols=4)
        grid_full = util.draw_grid(images_chains[0:8*8], cols=4)
        grid_small = util.draw_grid(images_chains[0:4*4], cols=4)
        misc.imsave(os.path.join(opt.save_path, 'sampled_images_chains', 'chain_full_%04d.jpg' % (i,)), grid_full)
        misc.imsave(os.path.join(opt.save_path, 'sampled_images_chains', 'chain_small_%04d.jpg' % (i,)), grid_small)

def makedirs(save_path, r_iterations):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_folder in ['sampled_images_chains']:
        if not os.path.exists(os.path.join(save_path, sub_folder)):
            os.mkdir(os.path.join(save_path, sub_folder))
    for r_iter in range(1+r_iterations):
        fp = os.path.join(save_path, "sampled_images_r%d" % (r_iter,))
        if not os.path.exists(fp):
            os.mkdir(fp)

def generate_images(gen, r, code, n_execute_lis_layers, batch_size, output_scale):
    codes_r = []
    images_all = []
    for i in range((code.size(0) - 1) // batch_size + 1):
        this_batch_size = min(batch_size, code.size(0) - i * batch_size)
        batch_code = Variable(code[i * batch_size : i * batch_size + this_batch_size])
        generated_images = gen(batch_code)
        codes_r.append(r(generated_images).data)
        #print("[generate_images]", batch_size, this_batch_size, generated_images.size())
        if output_scale:
            generated_images = generated_images * 2 - 1
        generated_images_np = (generated_images.data.cpu().numpy() * 255).astype(np.uint8).transpose((0, 2, 3, 1))
        images_all.extend(list(generated_images_np))
    return images_all, torch.cat(codes_r, dim=0)

if __name__ == "__main__":
    main()
