#!/usr/bin/env python3
import cv2
import numpy as np

print("hello")

image = cv2.imread("/home/fworg64/gradschool/csci575/salt_data/train/images/0a1742c740.png")
mask  = cv2.imread("/home/fworg64/gradschool/csci575/salt_data/train/masks/0a1742c740.png")

# Zero pad image to 128x128
offset = np.int((128 - 101)/2)
padded = np.zeros((128, 128, 3), dtype="uint8")
padded[offset:(offset+101), offset:(offset+101)] = image

padded_mask = np.zeros((128, 128, 3), dtype="uint8")
padded_mask[offset:(offset+101), offset:(offset+101)] = mask

#cv2.imshow("helllo", padded)
#cv2.waitKey(0)

# Divide into 8x8 or 16x16 tiles and take FFT
# Sort tiles into ANY salt in mask and NO salt in mask
division_size = 8
salt_tiles = []
no_salt_tiles = []
for row_dex in range(0, 128 - division_size, division_size):
  for col_dex in range(0, 128 - division_size, division_size):
    mask_sum = np.sum(padded_mask[row_dex:(row_dex+division_size),
                                   col_dex:(col_dex+division_size)])
    if mask_sum > 0:
      salt_tiles.append(np.fft.fft2(padded[row_dex:(row_dex+division_size), 
                                           col_dex:(col_dex+division_size)]))
    else:
      no_salt_tiles.append(np.fft.fft2(padded[row_dex:(row_dex+division_size), 
                                              col_dex:(col_dex+division_size)]))

print("Found %d salty" % len(salt_tiles))
print("Found %d non-salty" % len(no_salt_tiles))

# Display statistics (frequency contents of images)
# Compute average for each list
running_sum_mag = np.zeros((division_size, division_size,3), dtype="float64")
for tile in salt_tiles:
  running_sum_mag = running_sum_mag + np.abs(tile);
running_sum_mag = running_sum_mag / len(salt_tiles)
salt_display = running_sum_mag.astype(np.uint8)

running_sum_mag = np.zeros((division_size, division_size,3), dtype="float64")
for tile in no_salt_tiles:
  running_sum_mag = running_sum_mag + np.abs(tile);
running_sum_mag = running_sum_mag / len(no_salt_tiles)
no_salt_display = running_sum_mag.astype(np.uint8)


cv2.imshow("helllo", salt_display)
cv2.waitKey(0)
cv2.imshow("helllo", no_salt_display)
cv2.waitKey(0)


# Take frequency content of mask areas (train data)

# Get frequency content of test images and filter