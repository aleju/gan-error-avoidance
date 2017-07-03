from __future__ import print_function, division
import math
import numpy as np
from PIL import Image

def draw_grid(images, rows=None, cols=None, border=1):
    nb_images = len(images)
    cell_height = max([image.shape[0] for image in images])
    cell_width = max([image.shape[1] for image in images])
    channels = set([image.shape[2] for image in images])
    assert len(channels) == 1
    nb_channels = list(channels)[0]
    if rows is None and cols is None:
        rows = cols = int(math.ceil(math.sqrt(nb_images)))
    elif rows is not None:
        cols = int(math.ceil(nb_images / rows))
    elif cols is not None:
        rows = int(math.ceil(nb_images / cols))
    assert rows * cols >= nb_images

    cell_height = cell_height + 1 * border
    cell_width = cell_width + 1 * border

    width = cell_width * cols
    height = cell_height * rows
    grid = np.zeros((height, width, nb_channels), dtype=np.uint8)
    cell_idx = 0
    for row_idx in range(rows):
        for col_idx in range(cols):
            if cell_idx < nb_images:
                image = images[cell_idx]
                border_top = border_right = border_bottom = border_left = border
                #if row_idx > 1:
                border_top = 0
                #if col_idx > 1:
                border_left = 0
                image = np.pad(image, ((border_top, border_bottom), (border_left, border_right), (0, 0)), mode="constant", constant_values=0)

                cell_y1 = cell_height * row_idx
                cell_y2 = cell_y1 + image.shape[0]
                cell_x1 = cell_width * col_idx
                cell_x2 = cell_x1 + image.shape[1]
                grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image
            cell_idx += 1

    grid = np.pad(grid, ((border, 0), (border, 0), (0, 0)), mode="constant", constant_values=0)

    return grid

class ImgaugPytorchWrapper(object):
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, img):
        #img_np = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
        img_np = np.array(img)
        img_aug_np = self.seq.augment_image(img_np)
        img_aug_pil = Image.fromarray(img_aug_np)
        return img_aug_pil
