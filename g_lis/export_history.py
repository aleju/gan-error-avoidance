"""Script to export the history of loss values gathered during an experiment
to a csv file."""
from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import math
import torch
import numpy as np
from scipy import misc, signal
import time
import random
import re

from common import plotting

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path_state', required=True,
        help = 'path of to the state file containing the history')

    parser.add_argument('--save_path', required=True,
        help = 'path to save the csv file to')

    parser.add_argument('--start', type=int, default=0,
        help = 'row index to start at (inclusive)')

    parser.add_argument('--end', type=int, default=-1,
        help = 'row index to end at (exclusive)')

    parser.add_argument('--subsample', type=int, default=1,
        help = 'output only every nth row')

    parser.add_argument('--kernel_size', type=int, default=101,
        help = 'kernel size of used filter')

    opt = parser.parse_args()
    print(opt)

    state = {}
    state.update(torch.load(opt.load_path_state))
    history = plotting.History.from_string(state["history"])

    line_to_xx = dict()
    line_to_yy = dict()
    line_names = []

    for group_name, line_group in history.line_groups.items():
        group_name = line_group.group_name
        for line_name, line in line_group.lines.items():
            key = line_key(group_name, line_name)
            line_to_xx[key] = line.xs[:line.last_index+1]
            line_to_yy[key] = line.ys[:line.last_index+1]
            line_names.append(key)

    print("Found line names:", line_names)
    line_to_xx_uq = dict([(line_name, set(list(lxx))) for line_name, lxx in line_to_xx.items()])
    xx = np.concatenate(line_to_xx.values())
    xx_uq_sorted = np.unique(xx)
    #xx_s = np.sort(xx_uq)
    print("Found %d unique x values, lowest: %d, highest: %d" % (len(xx_uq_sorted), xx_uq_sorted[0], xx_uq_sorted[-1]))

    line_x_to_pos = dict()
    for line_name in line_names:
        xx = line_to_xx[line_name]
        #x_to_pos = np.zeros(xx.shape, dtype=np.uint64)
        x_to_pos = dict()
        for pos, x in enumerate(xx):
            x_to_pos[x] = pos
        line_x_to_pos[line_name] = x_to_pos

    csv_lines = []
    last_value = dict([(line_name, 0) for line_name in line_names])
    for i, x in enumerate(xx_uq_sorted):
        row = []
        for line_name in line_names:
            if x in line_to_xx_uq[line_name]:
                x_pos = line_x_to_pos[line_name][x]
                y = line_to_yy[line_name][x_pos]
                row.append(y)
                last_value[line_name] = y
            else:
                row.append(last_value[line_name])
        csv_lines.append(row)

    csv_lines = np.array(csv_lines)
    if opt.kernel_size > 1:
        kernel = np.ones(opt.kernel_size, dtype=np.float32) / opt.kernel_size
        for col_idx in range(len(line_names)):
            #csv_lines[:, col_idx] = signal.medfilt(csv_lines[:, col_idx], opt.median_kernel_size)
            csv_lines[:, col_idx] = np.convolve(csv_lines[:, col_idx], kernel, mode="same")

    if opt.end == -1:
        opt.end = len(csv_lines)

    with open(os.path.join(opt.save_path, "history.csv"), "w") as f:
        f.write("__index")
        for line_name in line_names:
            f.write(",")
            f.write(re.sub(r"[\,\s\t\"\']", "", line_name))
        for row_idx, csv_line in enumerate(csv_lines):
            if opt.start <= row_idx < opt.end and row_idx % opt.subsample == 0:
                f.write("\n")
                f.write("%d" % (xx_uq_sorted[row_idx],)) # x
                for i, col in enumerate(csv_line):
                    f.write(",")
                    f.write("%.4f" % (col,))

def line_key(group_name, line_name):
    return "%s_%s" % (group_name, line_name)

if __name__ == "__main__":
    main()
