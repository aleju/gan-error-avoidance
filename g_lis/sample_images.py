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
from scipy import misc, ndimage
import time
import random
import glob

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

    parser.add_argument('--load_path_r', required = False,
        help = 'path of R-separate to use')

    parser.add_argument('--save_path', required = True,
        help = 'Path to save sampled images to')

    parser.add_argument('--output_scale',       action = 'store_true', default = False,
        help = 'save x*2-1 instead of x when saving image')

    parser.add_argument('--r_iterations', type = int, default = 3,
        help = 'How often to execute the reverse projection via R')

    parser.add_argument('--spatial_dropout_r',  type = float,   default = 0,
        help = 'Spatial dropout applied to R')

    parser.add_argument('--with_real_images',  action = 'store_true', default = False,
        help = 'Whether to create perturbations/interpolations of images in images/sample_real_images/')

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

    gen = GeneratorLearnedInputSpace(opt.width, opt.height, opt.nfeature, opt.nlayer, opt.code_size, opt.norm, n_lis_layers=opt.r_iterations)
    print(gen)
    gen.cuda()
    gen.load_state_dict(torch.load(opt.load_path_g))
    gen.eval()

    if opt.load_path_r is not None:
        r = build_reverser(opt.width, opt.height, opt.nfeature//2, opt.nlayer, opt.code_size, opt.norm, opt.spatial_dropout_r)
        print(r)
        r.cuda()
        r.load_state_dict(torch.load(opt.load_path_r))
        r.eval()
    else:
        r = None

    makedirs(opt.save_path, opt.r_iterations, add_rsep_folders=r is not None)

    for i in range(20):
        nb_rows = 16
        nb_cols = 16
        vis_code = torch.randn(nb_rows * nb_cols, opt.code_size).cuda()
        images_by_r = []
        for r_idx in range(1+opt.r_iterations):
            images, images_rsep = generate_images(gen, r, vis_code, r_idx, opt.batch_size, opt.output_scale)
            #images = add_border(images)
            images_by_r.append(images)
            #grid = ia.draw_grid(images, rows=nb_rows, cols=nb_cols)
            grid_full = util.draw_grid(images, rows=nb_rows, cols=nb_cols)
            grid_small = util.draw_grid(images[0:(nb_rows//2 * nb_cols//2)], rows=nb_rows//2, cols=nb_cols//2)
            misc.imsave(os.path.join(opt.save_path, 'sampled_images_r%d' % (r_idx,), 'r%d_full_%04d.jpg' % (r_idx, i)), grid_full)
            misc.imsave(os.path.join(opt.save_path, 'sampled_images_r%d' % (r_idx,), 'r%d_small_%04d.jpg' % (r_idx, i)), grid_small)

            if len(images_rsep) > 0:
                grid_full = util.draw_grid(images_rsep, rows=nb_rows, cols=nb_cols)
                grid_small = util.draw_grid(images_rsep[0:(nb_rows//2 * nb_cols//2)], rows=nb_rows//2, cols=nb_cols//2)
                misc.imsave(os.path.join(opt.save_path, 'sampled_images_rsep_r%d_after' % (r_idx,), 'rsep_r%d_full_%04d.jpg' % (r_idx, i)), grid_full)
                misc.imsave(os.path.join(opt.save_path, 'sampled_images_rsep_r%d_after' % (r_idx,), 'rsep_r%d_small_%04d.jpg' % (r_idx, i)), grid_small)

                images_rsep_both = []
                for j in xrange(len(images_rsep)):
                    images_rsep_both.append(np.hstack([images[j], images_rsep[j]]))
                grid_full = util.draw_grid(images_rsep_both, rows=nb_rows, cols=nb_cols)
                grid_small = util.draw_grid(images_rsep_both[0:(nb_rows//2 * nb_cols//2)], rows=nb_rows//2, cols=nb_cols//2)
                misc.imsave(os.path.join(opt.save_path, 'sampled_images_rsep_r%d_both' % (r_idx,), 'rsep_r%d_chain_full_%04d.jpg' % (r_idx, i)), grid_full)
                misc.imsave(os.path.join(opt.save_path, 'sampled_images_rsep_r%d_both' % (r_idx,), 'rsep_r%d_chain_small_%04d.jpg' % (r_idx, i)), grid_small)

        images_chains = []
        for j in xrange(len(images_by_r[0])):
            images_one_chain = [images_by_r[r_idx][j] for r_idx in range(len(images_by_r))]
            images_chains.append(np.hstack(images_one_chain))
        #images_chains = add_border(images_chains)
        #grid = ia.draw_grid(images_chains, cols=4)
        grid_full = util.draw_grid(images_chains[0:8*8], cols=4)
        grid_small = util.draw_grid(images_chains[0:4*4], cols=4)
        misc.imsave(os.path.join(opt.save_path, 'sampled_images_chains', 'chain_full_%04d.jpg' % (i,)), grid_full)
        misc.imsave(os.path.join(opt.save_path, 'sampled_images_chains', 'chain_small_%04d.jpg' % (i,)), grid_small)

    # interpolations
    for i in range(20):
        codes = torch.randn(2, opt.code_size).cuda()
        images_by_r = []
        for r_idx in range(1+opt.r_iterations):
            images = generate_interpolations(gen, codes[0], codes[1], 8, r_idx, opt.output_scale)
            images = np.hstack(images)
            misc.imsave(os.path.join(opt.save_path, 'sampled_images_interpolations_r%d' % (r_idx,), 'interp_r%d_%04d.jpg' % (r_idx, i,)), images)
            #if r_idx > 0:
            #    images[0] = np.zeros(images[0]) + 255
            #    images[-1] = np.zeros(images[-1]) + 255
            images_by_r.append(images)
        misc.imsave(os.path.join(opt.save_path, 'sampled_images_interpolations_all', 'interp_all_%04d.jpg' % (i,)), np.vstack(images_by_r))

    # perturbations
    for i in range(20):
        code = torch.randn(1, opt.code_size).cuda()
        seed = random.randint(0, 10**6)
        images_by_r = []
        for r_idx in range(1+opt.r_iterations):
            images = generate_perturbations(gen, code, seed, 1.0, 64, r_idx, opt.output_scale)
            grid = util.draw_grid(images, cols=8, rows=8)
            misc.imsave(os.path.join(opt.save_path, 'sampled_images_perturbations_r%d' % (r_idx,), 'pert_r%d_%04d.jpg' % (r_idx, i,)), grid)
            images_by_r.append(np.vstack(images[0:16]))
        misc.imsave(os.path.join(opt.save_path, 'sampled_images_perturbations_all', 'pert_all_%04d.jpg' % (i,)), np.hstack(images_by_r))

    if opt.with_real_images and r is not None:
        real_imgs_fps = glob.glob("../images/sample_real_images/*.jpg")
        print("Embedding %d real images and creating perturbations/interpolations..." % (len(real_imgs_fps),))
        if len(real_imgs_fps) > 0:
            real_images = [misc.imresize(ndimage.imread(fp, mode="RGB"), (opt.height, opt.width)) for fp in real_imgs_fps]
            real_images_tnsr = torch.from_numpy(
                np.array(real_images, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
            ).cuda()
            #real_images_var = Variable(real_images_tnsr)
            #codes = r(real_images_var)
            #codes = [embed_real_image(gen, r, real_images_tnsr[i], opt.code_size) for i in range(len(real_images))]
            codes = embed_real_images(gen, r, real_images_tnsr, opt.code_size)

            for i in xrange(len(codes)):
                real_image = real_images[i]
                #images = generate_perturbations(gen, codes.data[i:i+1, :], 42, 1.0, 7*7, opt.r_iterations, opt.output_scale)
                images = generate_perturbations(gen, codes[i], 42, 1.0, 7*7, opt.r_iterations, opt.output_scale)
                grid = util.draw_grid(images, cols=7, rows=7)
                misc.imsave(os.path.join(opt.save_path, 'sampled_images_real_images_perturbations', 'pert_real_%04d.jpg' % (i,)), grid)

                for j in xrange(len(codes)):
                    if i != j:
                        #images = generate_interpolations(gen, codes.data[i], codes.data[j], 8, opt.r_iterations, opt.output_scale)
                        images = generate_interpolations(gen, codes[i], codes[j], 8, opt.r_iterations, opt.output_scale)
                        images_horizontal = np.hstack(images)
                        images_vertical = np.vstack(images)
                        misc.imsave(os.path.join(opt.save_path, 'sampled_images_real_images_interpolations', 'interp_%04d_to_%04d_horizontal.jpg' % (i, j)), images_horizontal)
                        misc.imsave(os.path.join(opt.save_path, 'sampled_images_real_images_interpolations', 'interp_%04d_to_%04d_vertical.jpg' % (i, j)), images_vertical)

def makedirs(save_path, r_iterations, add_rsep_folders):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_folder in ['sampled_images_chains', 'sampled_images_interpolations_all', 'sampled_images_perturbations_all', 'sampled_images_real_images_perturbations', 'sampled_images_real_images_interpolations']:
        if not os.path.exists(os.path.join(save_path, sub_folder)):
            os.mkdir(os.path.join(save_path, sub_folder))
    for r_iter in range(1+r_iterations):
        for sub_folder in ['sampled_images_r%d', 'sampled_images_interpolations_r%d', 'sampled_images_perturbations_r%d']:
            fp = os.path.join(save_path, sub_folder % (r_iter,))
            if not os.path.exists(fp):
                os.mkdir(fp)

    if add_rsep_folders:
        for r_iter in range(1+r_iterations):
            for sub_folder in ['sampled_images_rsep_r%d_after' % (r_iter,), 'sampled_images_rsep_r%d_both' % (r_iter,)]:
                if not os.path.exists(os.path.join(save_path, sub_folder)):
                    os.mkdir(os.path.join(save_path, sub_folder))

def generate_images(gen, r, code, n_execute_lis_layers, batch_size, output_scale):
    images_all = []
    images_rsep = []
    for i in range((code.size(0) - 1) // batch_size + 1):
        this_batch_size = min(batch_size, code.size(0) - i * batch_size)
        batch_code = Variable(code[i * batch_size : i * batch_size + this_batch_size])
        generated_images, _ = gen(batch_code, n_execute_lis_layers=n_execute_lis_layers)
        if r is not None:
            generated_images_r, _ = gen(r(generated_images), n_execute_lis_layers=n_execute_lis_layers)
            if output_scale:
                generated_images_r = generated_images_r * 2 -1
            generated_images_r_np = (generated_images_r.data.cpu().numpy() * 255).astype(np.uint8).transpose((0, 2, 3, 1))
            images_rsep.extend(list(generated_images_r_np))
        #print("[generate_images]", batch_size, this_batch_size, generated_images.size())
        if output_scale:
            generated_images = generated_images * 2 - 1
        generated_images_np = (generated_images.data.cpu().numpy() * 255).astype(np.uint8).transpose((0, 2, 3, 1))
        images_all.extend(list(generated_images_np))
    return images_all, images_rsep

def generate_interpolations(gen, code_start, code_end, nb_steps, n_execute_lis_layers, output_scale):
    #codes = torch.randn(2, opt.code_size).cuda()
    result = []
    code_start = code_start.cpu().numpy()
    code_end = code_end.cpu().numpy()
    #print("interpolating from ", code_start[0:16])
    #print("to", code_end[0:16])
    code_stepsize = (code_end - code_start) / nb_steps
    #print("stepsize", code_stepsize[0:16])
    vectors = [code_start] + [code_start + i * code_stepsize for i in range(nb_steps)] + [code_end]
    #for i in range(2+nb_steps):
    for i, vec in enumerate(vectors):
        #print("i", i, "-->", vec[0:16])
        tnsr = torch.from_numpy(np.array([vec], dtype=np.float32)).cuda()
        images, _ = gen(Variable(tnsr), n_execute_lis_layers=n_execute_lis_layers)
        images_np = (images.data.cpu().numpy() * 255).astype(np.uint8).transpose((0, 2, 3, 1))
        result.append(images_np[0])
    return result

def generate_perturbations(gen, code, seed, stddev, nb_images, n_execute_lis_layers, output_scale):
    if len(code.size()) == 1:
        code = code.unsqueeze(0)
    assert code.size(0) == 1
    assert code.size(1) >= 1
    torch.manual_seed(seed)
    codes = code.expand(nb_images, code.size(1))
    rands = torch.randn(nb_images, code.size(1)).cuda() * stddev
    rands[0] = codes[0]
    codes_perturbed = codes + rands
    images, _ = gen(Variable(codes_perturbed), n_execute_lis_layers=n_execute_lis_layers)
    images_np = (images.data.cpu().numpy() * 255).astype(np.uint8).transpose((0, 2, 3, 1))
    return images_np

def embed_real_images(gen, r, images, code_size, lr=0.0001, test_steps=100000):
    testfunc = nn.MSELoss()

    for param in gen.parameters():
        param.requires_grad = False
    best_code = torch.Tensor(len(images), code_size).cuda()

    batch_size = len(images)
    batch_code = Variable(torch.zeros(batch_size, code_size).cuda())
    batch_code.requires_grad = True

    batch_target = torch.Tensor(batch_size, images[0].size(0), images[0].size(1), images[0].size(2))
    for i, image in enumerate(images):
        batch_target[i].copy_(image)
    batch_target = Variable(batch_target.cuda())
    batch_code.data.copy_(r(batch_target).data)

    #test_opt = optim.RMSprop([batch_code], lr=lr)
    test_opt = optim.Adam([batch_code], lr=lr)
    for j in range(test_steps):
        generated, _ = gen(batch_code)
        loss = testfunc(generated, batch_target)
        loss.backward()
        test_opt.step()
        batch_code.grad.data.zero_()
        if j % 100 == 0:
            #lr = lr * 0.98
            print("Embedding real images... iter %d with loss %.08f and lr %.08f" % (j,loss.data[0], lr))
            #test_opt = optim.RMSprop([batch_code], lr=lr)
    best_code = batch_code.data

    for param in gen.parameters():
        param.requires_grad = True

    return best_code

if __name__ == "__main__":
    main()
