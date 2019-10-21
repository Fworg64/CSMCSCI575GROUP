#!/usr/bin/env python3
import cv2
import numpy as np


def classify_tiles_from_avgs(image, salt_avg, no_salt_avg, norm_type=2, tile_width=16):
    offset = np.int((128 - 101) / 2)
    padded_image = np.zeros((128, 128), dtype="uint8")
    padded_image[offset:(offset + 101), offset:(offset + 101)] = image

    #padded_mask = np.zeros((128, 128), dtype="uint8")
    #padded_mask[offset:(offset + 101), offset:(offset + 101)] = mask

    division_size = tile_width
    #salt_tiles = []
    tiles = []
    classification = []
    for row_dex in range(0, 128 - division_size, division_size):
        for col_dex in range(0, 128 - division_size, division_size):
            #mask_sum = np.sum(padded_mask[row_dex:(row_dex + division_size),
            #                  col_dex:(col_dex + division_size)])
            #if mask_sum > 0:
            #    salt_tiles.append(np.fft.fft2(padded_image[row_dex:(row_dex + division_size),
            #                                 col_dex:(col_dex + division_size)]))
            tiles.append(np.fft.fft2(padded_image[row_dex:(row_dex + division_size),
                                                 col_dex:(col_dex + division_size)]))
            # Take distance from either avg
            salt_dist = np.linalg.norm(tiles[-1] - salt_avg, norm_type)
            no_salt_dist = np.linalg.norm(tiles[-1] - no_salt_avg, norm_type)
            if salt_dist < no_salt_dist:
                classification.append(1)
            else:
                classification.append(0)

    return classification
