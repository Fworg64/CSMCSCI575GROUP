#!/usr/bin/env python3
import cv2
import numpy as np

def avg_spec(image, mask):
  """
    Returns the split averages of each image. Average salt mask spectrum and average 
    no salt mask spectrum. Split into size of 8 tiles.
  """
  # Zero pad image to 128x128
  # Image_size = 101
  offset = np.int((128 - 101)/2)
  padded_image = np.zeros((128, 128), dtype="uint8")
  padded_image[offset:(offset+101), offset:(offset+101)] = image

  padded_mask = np.zeros((128, 128), dtype="uint8")
  padded_mask[offset:(offset+101), offset:(offset+101)] = mask
  
  division_size = 16
  salt_tiles = []
  no_salt_tiles = []
  for row_dex in range(0, 128 - division_size, division_size):
    for col_dex in range(0, 128 - division_size, division_size):
      mask_sum = np.sum(padded_mask[row_dex:(row_dex+division_size),
                                     col_dex:(col_dex+division_size)])
      if mask_sum > 0:
        salt_tiles.append(np.fft.fft2(padded_image[row_dex:(row_dex+division_size), 
                                             col_dex:(col_dex+division_size)]))
      else:
        no_salt_tiles.append(np.fft.fft2(padded_image[row_dex:(row_dex+division_size), 
                                                col_dex:(col_dex+division_size)]))

  running_sum_mag = np.zeros((division_size, division_size), dtype="float64")
  for tile in salt_tiles:
    running_sum_mag = running_sum_mag + np.abs(tile);
  if len(salt_tiles) > 1:
    running_sum_mag = running_sum_mag / len(salt_tiles)
  salt_display = running_sum_mag.astype(np.uint8)

  running_sum_mag = np.zeros((division_size, division_size), dtype="float64")
  for tile in no_salt_tiles:
    running_sum_mag = running_sum_mag + np.abs(tile);
  if len(no_salt_tiles) > 1:
    running_sum_mag = running_sum_mag / len(no_salt_tiles)
  no_salt_display = running_sum_mag.astype(np.uint8)


  print("Found %d salty" % len(salt_tiles))
  print("Found %d non-salty" % len(no_salt_tiles))

  return (salt_display, no_salt_display)
