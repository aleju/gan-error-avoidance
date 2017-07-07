"""Code to generate t-SNE embeddings and plots of noise vectors, before
and after applying LIS modules.
This requires a trained G-LIS.
Note that t-SNE often stopped quite early, so you might not always have
success. Even successful plots tend to not show much change compared to
components sampled from N(0,1). Increasing the learning rate of t-SNE seems
to help sometimes.

Example:
    python sample_tsne.py --image_size 80 --code_size 256 --norm weight \
        --r_iterations 1 \
        --load_path_g /path/to/checkpoints/exp01/net_archive/last_gen.pt \
        --save_path /path/to/outputs/exp01/tsne/
"""
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
#from MulticoreTSNE import MulticoreTSNE as TSNE
#from sklearn.decomposition import PCA
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
        help = 'path to save sampled images to')

    parser.add_argument('--r_iterations', type = int, default = 3,
        help = 'how many LIS modules to use.')

    parser.add_argument('--nb_points', type = int, default = 17500,
        help = 'number of points to embed and plot via t-SNE')

    parser.add_argument('--g_upscaling',  default='fractional',
        help = 'upscaling method to use in G: fractional|nearest|bilinear')

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

    gen = GeneratorLearnedInputSpace(opt.width, opt.height, opt.nfeature, opt.nlayer, opt.code_size, opt.norm, n_lis_layers=opt.r_iterations, upscaling=opt.g_upscaling)
    print(gen)
    gen.cuda()
    gen.load_state_dict(torch.load(opt.load_path_g))
    gen.eval()

    makedirs(opt.save_path)

    codes = torch.randn(opt.nb_points, opt.code_size).cuda()
    for r_idx in range(1+opt.r_iterations):
        codes_r_2d = embed_or_load_cache(codes, gen, r_idx, opt.batch_size, opt.save_path)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.scatter(codes_r_2d[:, 0], codes_r_2d[:, 1], s=2, alpha=0.2)
        ax.set_xlim((-10, 10))
        ax.set_ylim((-10, 10))
        fig.savefig(os.path.join(opt.save_path, 'tsne_plots', 'tsne_%04d_r%02d.jpg' % (0, r_idx,)))
        plt.close()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        sns_plot = sns.kdeplot(codes_r_2d[:, 0], codes_r_2d[:, 1], shade=True, ax=ax)
        sns_plot.axes.set_xlim((-10, 10))
        sns_plot.axes.set_ylim((-10, 10))
        fig = sns_plot.get_figure()
        fig.savefig(os.path.join(opt.save_path, 'tsne_plots', 'tsne_%04d_r%02d_kde.jpg' % (0, r_idx,)))

def makedirs(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_folder in ['tsne_plots']:
        if not os.path.exists(os.path.join(save_path, sub_folder)):
            os.mkdir(os.path.join(save_path, sub_folder))

def generate_codes_by_r(gen, code, n_execute_lis_layers, batch_size):
    if n_execute_lis_layers == 0:
        return code.cpu().numpy()

    codes_all = []
    for i in range((code.size(0) - 1) // batch_size + 1):
        this_batch_size = min(batch_size, code.size(0) - i * batch_size)
        batch_code = Variable(code[i * batch_size : i * batch_size + this_batch_size])
        generated_images, codes_result = gen(batch_code, n_execute_lis_layers=n_execute_lis_layers)
        codes_result = codes_result[-1]
        codes_result = codes_result.data.cpu().numpy()
        codes_all.extend(list(codes_result))
    return np.array(codes_all, dtype=np.float32)

def embed_or_load_cache(codes, gen, r_idx, batch_size, save_path):
    cache_fp = os.path.join(save_path, 'tsne_plots', 'embedded_points_r%02d.csv' % (r_idx,))
    if os.path.isfile(cache_fp):
        lines = open(cache_fp).readlines()
        lines = [line.strip().split(",") for line in lines[1:]]
        vals = [(float(x), float(y)) for (x, y) in lines]
        return np.array(vals, dtype=np.float32)
    else:
        codes_r = generate_codes_by_r(gen, codes, r_idx, batch_size)

        print(codes_r.shape)
        print("Embedding %s via TSNE..." % (str(codes_r.shape),))
        tsne = TSNE(perplexity=40, n_iter=10000, learning_rate=4000, verbose=True)
        #tsne = TSNE(perplexity=40, n_iter=10000, n_jobs=4, verbose=True)
        #tsne = PCA(n_components=2)
        codes_r_2d = tsne.fit_transform(codes_r.astype(np.float64))
        print("shape after embedding: %s" % (str(codes_r_2d.shape),))

        with open(cache_fp, "w") as f:
            f.write("#x,y\n")
            for i in xrange(codes_r.shape[0]):
                f.write("%.6f,%.6f\n" % (codes_r_2d[i, 0], codes_r_2d[i, 1]))
        return codes_r_2d

if __name__ == "__main__":
    main()
