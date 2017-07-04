"""Script to convert images to squared ones by zero-padding and then
resize them to a given size.
This script is only recommended to use if your dataset does not contain
squared images. Resizing can be done automatically by the other training
scripts.

Example:
    python resize_images.py --image_size 128 \
        --load_path /path/to/images
        --save_path /path/to/output
"""
from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from scipy import misc, ndimage
#import glob
import re

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=80,
    	help = 'Desired output size of images')

    parser.add_argument('--load_path', required=True,
    	help = 'glob-ready path to load images from. Example: "/foo/bar/"')

    parser.add_argument('--save_path', required=True,
    	help = 'Directory to save images to. Example: "/foo/bar2/"')

    args = parser.parse_args()
    print(args)

    print("Loading filepaths...")
    fps = get_images(args.load_path)

    print("Resizing...")
    for i, fp in enumerate(fps):
        fn = os.path.basename(fp)
        fn = fn[:fn.index(".")-1]
        out_fp = os.path.join(args.save_path, "%s.jpg" % (fn,))
        print("[%d] Image '%s' to '%s'" % (i+1, fp, out_fp))
        img = ndimage.imread(fp, mode="RGB")
        img_sq = make_square(img)
        img_sq_rs = misc.imresize(img_sq, (args.image_size, args.image_size))
        misc.imsave(out_fp, img_sq_rs)

def get_images(dir_path):
    for root, subFolders, files in os.walk(dir_path):
        for filename in files:
            if re.search(r"\.(jpg|jpeg|bmp|gif|png|webp)$", filename):
                file_path = os.path.join(root, filename)
                yield file_path

def make_square(img):
    h, w = img.shape[0:2]
    pad_top = pad_bottom = pad_left = pad_right = 0
    if h < w:
        pad_by = w - h
        pad_top = pad_by // 2
        pad_bottom = pad_by // 2
        if pad_top + pad_bottom < pad_by:
            pad_top += 1
    elif w < h:
        pad_by = h - w
        pad_left = pad_by // 2
        pad_right = pad_by // 2
        if pad_left + pad_right < pad_by:
            pad_left += 1
    img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=0)
    return img

if __name__ == "__main__":
    main()
