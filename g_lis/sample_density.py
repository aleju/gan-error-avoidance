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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from common import plotting
from common.model import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',         type = int,   default = 256,
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

    parser.add_argument('--save_path', required = True,
        help = 'Path to save sampled images to')

    parser.add_argument('--r_iterations', type = int, default = 3,
        help = 'How often to execute the reverse projection via R')

    parser.add_argument('--nb_points', type = int, default = 500000,
        help = 'Number of points from which to estimate densities')

    opt = parser.parse_args()
    print(opt)

    if (opt.height > 0) and (opt.width > 0):
        pass
    elif opt.image_size > 0:
        opt.height = opt.image_size
        opt.width = opt.image_size
    else:
        raise ValueError('must specify valid image size')

    #if (opt.vis_row <= 0) or (opt.vis_col <= 0):
    #   opt.vis_row = opt.vis_size
    #   opt.vis_col = opt.vis_size

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

    makedirs(opt.save_path)

    print("Generating points...")
    codes = torch.randn(opt.nb_points, opt.code_size).cuda()
    codes_r = [generate_codes_by_r(gen, codes, r_idx, opt.batch_size) for r_idx in range(1+opt.r_iterations)]
    np.set_printoptions(precision=6, suppress=True)
    print("means before LIS", np.mean(codes_r[0], axis=0))
    print("std before LIS", np.std(codes_r[0], axis=0))

    print("Plotting...")
    for v_idx in range(50):
        lines_r = [points_to_line(codes_r[r_idx][:, v_idx]) for r_idx in range(opt.r_iterations+1)]
        save_lines(lines_r, v_idx, opt.save_path)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for r_idx in range(1+opt.r_iterations):
            xx, yy = lines_r[r_idx]
            ax.plot(xx, yy, label="after %d LIS modules" % (r_idx,))
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 0.06)
        ax.legend()
        fig.savefig(os.path.join(opt.save_path, 'density_plots', 'density_all_v%03d.jpg' % (v_idx,)), bbox_inches="tight")
        plt.close()

        for r_idx in range(1+opt.r_iterations):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            xx, yy = lines_r[0]
            ax.plot(xx, yy, label="after 0 LIS modules")
            if r_idx > 0:
                xx, yy = lines_r[r_idx]
                ax.plot(xx, yy, c="red", label="after %d LIS modules" % (r_idx,))
            ax.set_xlim(-6, 6)
            ax.set_ylim(0, 0.06)
            #ax.legend()
            fig.savefig(os.path.join(opt.save_path, 'density_plots', 'density_r%02d_v%03d.jpg' % (r_idx, v_idx,)), bbox_inches="tight")
            plt.close()

        sns_plot = None
        for r_idx in range(1+opt.r_iterations):
            sns_plot = sns.kdeplot(codes_r[r_idx][:, v_idx], bw=0.35, label="%d lis modules" % (r_idx,))
        sns_plot.set(xlim=(-6, 6))
        sns_plot.set(ylim=(-0, 0.5))
        plt.legend()
        fig = sns_plot.get_figure()
        fig.savefig(os.path.join(opt.save_path, 'density_plots', 'density_kde_all_v%03d.jpg' % (v_idx,)), bbox_inches="tight")
        plt.clf()

        for r_idx in range(1+opt.r_iterations):
            sns_plot = None
            sns_plot = sns.kdeplot(codes_r[0][v_idx], bw=0.35, label="%d lis modules" % (0,))
            if r_idx > 0:
                sns_plot = sns.kdeplot(codes_r[r_idx][:, v_idx], bw=0.35, label="%d lis modules" % (r_idx,))
            sns_plot.set(xlim=(-6, 6))
            sns_plot.set(ylim=(-0, 0.5))
            plt.legend()
            fig = sns_plot.get_figure()
            fig.savefig(os.path.join(opt.save_path, 'density_plots', 'density_kde_r%02d_v%03d.jpg' % (r_idx, v_idx,)), bbox_inches="tight")
            plt.clf()

def makedirs(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_folder in ['density_plots']:
        if not os.path.exists(os.path.join(save_path, sub_folder)):
            os.mkdir(os.path.join(save_path, sub_folder))

def generate_codes_by_r(gen, code, n_execute_lis_layers, batch_size):
    if n_execute_lis_layers == 0:
        result = code.cpu().numpy()
    else:
        codes_all = []
        for i in range((code.size(0) - 1) // batch_size + 1):
            this_batch_size = min(batch_size, code.size(0) - i * batch_size)
            batch_code = Variable(code[i * batch_size : i * batch_size + this_batch_size])
            generated_images, codes_result = gen(batch_code, n_execute_lis_layers=n_execute_lis_layers)
            codes_result = codes_result[-1]
            codes_result = codes_result.data.cpu().numpy()
            codes_all.extend(list(codes_result))
        result = np.array(codes_all, dtype=np.float32)
    return np.clip(result, -6, 6)

def points_to_line(values, nb_bins=100):
    print("[points_to_line]", values.shape)
    heights, bins = np.histogram(values, bins=nb_bins)
    #print(values.shape, heights, bins)

    # Normalize
    heights = heights / float(sum(heights))
    bin_mids = bins[:-1] + np.diff(bins) / 2.
    return bin_mids, heights

def save_lines(lines_r, v_idx, save_path):
    for r_idx in range(len(lines_r)):
        xx, yy = lines_r[r_idx]
        fp = os.path.join(save_path, 'density_plots', 'density_r%02d_v%03d.csv' % (r_idx, v_idx,))
        with open(fp, "w") as f:
            f.write("x,y\n")
            for x, y in zip(xx, yy):
                f.write("%.8f,%.8f\n" % (x, y))

if __name__ == "__main__":
    main()
