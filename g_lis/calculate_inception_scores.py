"""Code to calculate inception scores. Don't try to understand how the code
works. Lots of background processes to deal with tensorflow/pytorch running
side-by-side, forcing them to release GPU RAM and forcing python to release
normal RAM.

Example:
    python calculate_inception_scores.py --dataset folder \
        --dataroot /path/to/datasets/celeba --crop_size 160 --image_size 80 \
        --code_size 256 --norm weight --r_iterations 3 \
        --inception_images 50000 \
        --load_path /path/to/checkpoints/exp01
"""
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import math
import os
import os.path
import numpy as np
import imgaug as ia
from scipy import misc
import glob
import re
import multiprocessing
import time

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from imgaug import augmenters as iaa

try:
    import cPickle as pickle
except ImportError:
    import pickle

from common import plotting
from common.model import *

AUGMENTATIONS = {
    "none": None,
    "dropout": iaa.Dropout(p=(0.0, 0.1)),
    "gaussian-noise": iaa.AdditiveGaussianNoise(scale=(0, 30)),
    "piecewise-affine": iaa.PiecewiseAffine(scale=(0, 0.09), mode="symmetric"),
    "vertical-flips": iaa.Flipud(1.0)
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',                          required = True,
        help = 'cifar10 | lsun | imagenet | folder | lfw')

    parser.add_argument('--lsun_class',                       default = 'bedroom',
        help = 'class of lsun dataset to use')

    parser.add_argument('--dataroot',                         required = True,
        help = 'path to dataset')

    parser.add_argument('--crop_size',          type = int,   default = -1,
        help = 'crop size before scaling')

    parser.add_argument('--crop_width',         type = int,   default = -1,
        help = 'crop width before scaling')

    parser.add_argument('--crop_height',        type = int,   default = -1,
        help = 'crop height before scaling')


    parser.add_argument('--batch_size',         type = int,   default = 50,
        help = 'input batch size')

    parser.add_argument('--inception_batch_size',         type = int,   default = 50,
        help = 'batch size for inception model')

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

    #parser.add_argument('--save_interval',      type = int,   default = 500,
    #	help = 'how often to save network')

    parser.add_argument('--load_path',                        required = True,
        help = 'load to continue existing experiment')

    parser.add_argument('--spatial_dropout_r',  type = float,   default = 0,
        help = 'spatial dropout applied to R')

    parser.add_argument('--nb_splits', type = int,   default = 10,
        help = 'number of image chunks to calculate inception score on')

    parser.add_argument('--inception_images', type = int,   default = 50000,
        help = 'number of image to calculate inception score on')

    parser.add_argument('--inception_images_real', type = int,   default = 50000,
        help = 'number of image to calculate inception score on')

    parser.add_argument('--r_iterations', type = int, default = 3,
        help = 'how many LIS modules to use in G')

    parser.add_argument('--every_nth_checkpoint', type = int, default = 1,
        help = 'calculate score only for every nth checkpoint')

    parser.add_argument('--augment', default = 'none',
        help = 'set of augmentations to run on the input images')

    opt = parser.parse_args()
    print(opt)

    opt.split_size = int(math.ceil(opt.inception_images / opt.nb_splits))
    opt.nb_batches_per_split = int(math.ceil(opt.split_size / opt.batch_size))
    print("Split Size:", opt.split_size, "Split Batch Size:", opt.nb_batches_per_split)

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

    # ------------
    # Inception score for real images
    # ------------
    print("---------------")
    print("Evaluating real data")
    print("---------------")
    mean_by_image_size, std_by_image_size = evaluate_on_real_data(opt)
    with open(os.path.join(opt.load_path, "inception-scores-real-data.csv"), "w") as f:
        f.write("#image_size,mean,std\n")
        for i in range(len(mean_by_image_size)):
            ims = mean_by_image_size[i][0]
            mean = mean_by_image_size[i][1]
            std = std_by_image_size[i][1]
            f.write("%d,%.10f,%.10f\n" % (ims, mean, std))

    # ------------
    # Inception score for fake images
    # ------------
    print("---------------")
    print("Evaluating fake data")
    print("---------------")
    history = plotting.History()
    history.add_group("inception-score", ["train-g%d" % (r_idx,) for r_idx in range(1+opt.r_iterations)], increasing=True)
    loss_plotter = plotting.LossPlotter(
        history.get_group_names(),
        history.get_groups_increasing(),
        save_to_fp=os.path.join(opt.load_path, "inception.jpg")
    )

    first_stage1_mean = None
    first_stage1_std = None
    best_means_idx = [0] * (1 + opt.r_iterations)
    best_means = [-999999] * (1 + opt.r_iterations)
    best_means_batches = [-1] * (1 + opt.r_iterations)

    g_fps = glob.glob(os.path.join(opt.load_path, "net_archive/*_gen.pt"))
    g_fps = [fp for fp in g_fps if re.match(r".*\/[0-9]+_gen\.pt", fp)]
    g_fps = sorted(g_fps, key=lambda fp: extract_batch(fp))

    # only last files currently
    g_fps = g_fps[-1:]

    with open(os.path.join(opt.load_path, "inception-scores.csv"), "w") as f:
        f.write("#nth_file,batch,g0_mean,g0_std")
        for r_idx in range(opt.r_iterations):
            f.write(",g%d_mean,g%d_std" % (r_idx, r_idx))
        f.write(",g_fp\n")

        for i, g_fp in enumerate(g_fps):
            if opt.every_nth_checkpoint > 1 and i % opt.every_nth_checkpoint != 0:
                pass
            else:
                gbatch = extract_batch(g_fp)

                time_start = time.time()
                measurer = BackgroundMeasureForFp(g_fp, i, opt)
                gr_means, gr_stds = measurer.get_results()
                time.sleep(0.25)
                measurer.terminate()
                time_end = time.time()

                msg_means = []
                msg_stds = []
                for r_idx, (gr_mean, gr_std) in enumerate(zip(gr_means, gr_stds)):
                    history.add_value("inception-score", "train-g%d" % (r_idx,), gbatch, gr_mean)
                    if gr_mean > best_means[r_idx]:
                        best_means[r_idx] = gr_mean
                        best_means_idx[r_idx] = i
                        best_means_batches[r_idx] = gbatch
                    msg_means.append("g%d:%06.4f" % (r_idx, gr_mean))
                    msg_stds.append("g%d:%06.4f" % (r_idx, gr_std))
                print("")
                print("Checkpoint %03d means[%s] std[%s] T[%.4fs] fp: %s" % (i, " ".join(msg_means), " ".join(msg_stds), time_end - time_start, g_fp))
                f.write("%d,%d" % (i, gbatch))
                for r_idx, (gr_mean, gr_std) in enumerate(zip(gr_means, gr_stds)):
                    f.write(",%.10f,%.10f" % (gr_mean, gr_std))
                f.write(",%s\n" % (g_fp,))
    with open(os.path.join(opt.load_path, "inception-scores-best.txt"), "w") as f:
        for i, (idx, mean, batch) in enumerate(zip(best_means_idx, best_means, best_means_batches)):
            print("Best g%d mean was %06.4f at %d" % (i, mean, idx))
            f.write("g%d: %.10f at file %d / batch %d\n" % (i, mean, idx, batch))
    loss_plotter.plot(history)
    history.save_to_filepath(os.path.join(opt.load_path, "inception-history.json"))

def evaluate_on_real_data(opt):
    mean_by_image_size = []
    std_by_image_size = []

    print("Initializing dataset...")
    real = RealDataFetcher(opt)

    print("Loading data...")
    nb_loaded = 0
    means_all = []
    stds_all = []
    for split_idx in range(opt.nb_splits):
        split_size = min(opt.split_size, real.size - nb_loaded)
        if split_size < 100:
            break
        real_data = real.get_batch(split_size)
        print("loaded %d images at shape %s" % (len(real_data), str(real_data[0].shape)))
        #nb_batches = math.ceil(10000 / opt.batch_size)
        #for i in range(nb_batches):
        #    real_data.extend(real.get_batch())
        bgscorer = BackgroundInceptionScorer({"images": real_data}, opt)
        means, stds = bgscorer.get_scores()
        bgscorer.terminate()
        time.sleep(5)
        print("")
        print("means", means)
        print("stds", stds)
        means_all.append(np.average(means))
        stds_all.append(np.average(stds))
        nb_loaded += len(real_data)
    mean_by_image_size.append((opt.image_size, np.average(means_all)))
    #std_by_image_size.append((opt.image_size, np.average(stds_all)))
    std_by_image_size.append((opt.image_size, np.std(means_all)))
    return mean_by_image_size, std_by_image_size

def extract_batch(fp):
    fp = fp[fp.rindex("/")+1:]
    return int(re.sub(r"[^0-9]", "", fp))

def measure_for_fp(g_fp, i, opt):
    means_by_riter = [[] for _ in range(1 + opt.r_iterations)]
    stds_by_riter = [[] for _ in range(1 + opt.r_iterations)]

    #for chunk_idx in range(opt.nb_chunks):
    for split_idx in range(opt.nb_splits):
        for r_idx in range(1+opt.r_iterations):
            nb_batches = opt.nb_batches_per_split
            bgimggen = BackgroundSingleCheckpointImageGenerator(nb_batches, g_fp, r_idx, i == 0 and split_idx == 0, opt)
            result = bgimggen.get_images()
            bgimggen.terminate()
            time.sleep(5)

            bgscorer = BackgroundInceptionScorer(result, opt)
            scores = bgscorer.get_scores()
            means, stds = scores
            bgscorer.terminate()
            time.sleep(5)

            means_by_riter[r_idx].append(means[0])
            stds_by_riter[r_idx].append(stds[0])

    means_by_riter = [np.average(means_by_riter[r_idx]) for r_idx in range(len(means_by_riter))]
    stds_by_riter = [np.average(stds_by_riter[r_idx]) for r_idx in range(len(stds_by_riter))]

    return means_by_riter, stds_by_riter

class RealDataFetcher(object):
    def __init__(self, opt):
        transform_list = []

        if (opt.crop_height > 0) and (opt.crop_width > 0):
            transform_list.append(transforms.CenterCrop(opt.crop_height, crop_width))
        elif opt.crop_size > 0:
            transform_list.append(transforms.CenterCrop(opt.crop_size))

        transform_list.append(transforms.Scale(opt.image_size))
        transform_list.append(transforms.CenterCrop(opt.image_size))

        transform_list.append(transforms.ToTensor())

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

        self.opt = opt
        self.get_data = get_data
        self.train_index = data_index['train']
        self.counter = 0

    @property
    def size(self):
        return len(self.train_index)

    def get_batch(self, batch_size):
        print("Loading %d real images of max possible %d..." % (batch_size, self.train_index.size(0)))
        batch = torch.Tensor(batch_size, 3, self.opt.image_size, self.opt.image_size)
        for j in range(batch_size):
            img = self.get_data(self.train_index[self.counter])
            #print(img.size())
            batch[j].copy_(img)
            self.counter = (self.counter + 1) % self.train_index.size(0)
            #print("counter", self.counter)
        #print(np.average(batch.cpu().numpy()), np.min(batch.cpu().numpy()), np.max(batch.cpu().numpy()))
        batch_np = (batch.cpu().numpy() * 255).astype(np.uint8).transpose((0, 2, 3, 1))
        #return Variable(batch.cuda(), volatile=volatile)
        #from scipy import misc
        #misc.imshow(batch_np[0])
        return batch_np

#
# Background processes below
# Tensorflow and pytorch stuff is separated in different processes, because
# otherwies the two will bitch at each other
# The whole measuring of inception scores for checkpoints is in background
# processes, because otherwise python will not free the memory properly and
# run into out of memory errors sooner or later
#

class BackgroundMeasureForFp(object):
    def __init__(self, g_fp, i, opt):
        self.queue = multiprocessing.Queue(1)
        worker = multiprocessing.Process(target=self._measure, args=(g_fp, i, opt, self.queue))
        #worker.daemon = True
        worker.start()
        self.worker = worker

    def get_results(self):
        result_str = self.queue.get()
        return pickle.loads(result_str)

    def _measure(self, g_fp, i, opt, queue):
        means_by_riter, stds_by_riter = measure_for_fp(g_fp, i, opt)
        result_str = pickle.dumps(
            (means_by_riter, stds_by_riter),
            protocol=-1
        )
        queue.put(result_str)

    def terminate(self):
        self.worker.terminate()

class BackgroundSingleCheckpointImageGenerator(object):
    def __init__(self, nb_batches, g_fp, r_idx, show_info, opt):
        self.queue = multiprocessing.Queue(1)

        worker = multiprocessing.Process(target=self._generate_images, args=(nb_batches, g_fp, r_idx, opt, show_info, self.queue,))
        worker.daemon = True
        worker.start()
        self.worker = worker

    def get_images(self):
        images_str = self.queue.get()
        return pickle.loads(images_str)

    def _generate_images(self, nb_batches, g_fp, r_idx, opt, show_info, queue):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        from torch.autograd import Variable

        #np.random.seed(42)
        #random.seed(42)
        #torch.manual_seed(42)

        gen = GeneratorLearnedInputSpace(opt.width, opt.height, opt.nfeature, opt.nlayer, opt.code_size, opt.norm, n_lis_layers=opt.r_iterations)
        if show_info:
            print("G:", gen)
        gen.cuda()
        prefix = "last"
        gen.load_state_dict(torch.load(g_fp))
        gen.train()

        print("Generating images for checkpoint G'%s'..." % (g_fp,))
        #imgs_by_riter = [[] for _ in range(1+opt.r_iterations)]
        images_all = []
        for i in range(nb_batches):
            code = Variable(torch.randn(opt.batch_size, opt.code_size).cuda(), volatile=True)

            #for r_idx in range(1+opt.r_iterations):
            images, _ = gen(code, n_execute_lis_layers=r_idx)
            images_np = (images.data.cpu().numpy() * 255).astype(np.uint8).transpose((0, 2, 3, 1))

            #from scipy import misc
            #print(np.average(images[0]), np.min(images[0]), np.max(images[0]))
            #print(np.average(images_fixed[0]), np.min(images_fixed[0]), np.max(images_fixed[0]))
            #misc.imshow(list(images_np)[0])
            #misc.imshow(list(images_fixed)[0])

            #imgs_by_riter[r_idx].extend(list(images_np))
            images_all.extend(images_np)

        result_str = pickle.dumps({
            "g_fp": g_fp,
            "images": images_all
        }, protocol=-1)
        queue.put(result_str)

    def terminate(self):
        self.worker.terminate()

class BackgroundInceptionScorer(object):
    def __init__(self, checkpoint_result, opt):
        self.queue = multiprocessing.Queue(1)

        checkpoint_result = pickle.dumps(checkpoint_result, protocol=-1)
        worker = multiprocessing.Process(target=self._score_images, args=(checkpoint_result, opt, self.queue))
        worker.daemon = True
        worker.start()
        self.worker = worker

    def get_scores(self):
        scores_str = self.queue.get()
        return pickle.loads(scores_str)

    def _score_images(self, checkpoint_result, opt, queue):
        from common.inception_score import get_inception_score

        result = pickle.loads(checkpoint_result)

        images = list(result["images"])

        augseq = AUGMENTATIONS[opt.augment]
        if augseq is not None:
            images_aug = augseq.augment_images(images)
        else:
            images_aug = images

        if images_aug[0].shape != (299, 299, 3):
            images_aug_rs = [misc.imresize(image, (299, 299)) for image in images_aug]
        else:
            images_aug_rs = images_aug
        #misc.imshow(np.hstack(list(images_aug[0:32])))
        #misc.imshow(np.hstack(list(images_aug_rs[0:5])))
        nb_splits = 1
        print("Calculating inception score on %d images at shape %s and %d splits..." % (len(images_aug_rs), str(images_aug_rs[0].shape), nb_splits))
        mean, std = get_inception_score(images_aug_rs, splits=nb_splits, bs=opt.inception_batch_size)

        result_str = pickle.dumps(
            ([mean], [std]),
            protocol=-1
        )
        queue.put(result_str)

    def terminate(self):
        self.worker.terminate()

if __name__ == "__main__":
    main()
