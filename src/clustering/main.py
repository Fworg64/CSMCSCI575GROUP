#!/usr/bin/env python3

import cv2
import os
import numpy as np
import sys

import lib.avg_spectrum as avg_spectrum
import lib.classify_tiles_from_avgs as classify_tiles_from_avgs

# compute averages of sample

mypath = "/home/fworg64/gradschool/csci575/salt_data/train/images"
mypath2 = "/home/fworg64/gradschool/csci575/salt_data/train/masks"

(_, _, image_files) = next(os.walk(mypath))
(_, _, mask_files) = next(os.walk(mypath2))
division_size = 16
if len(sys.argv) < 2:
    salt_list = []
    no_salt_list = []

# Load images and run function

    for each_name in image_files:
        print(each_name)
        image = cv2.imread(mypath + '/' + each_name)
        mask = cv2.imread(mypath2 + '/' + each_name)

        [a, b] = avg_spectrum.avg_spec(image[:, :, 0], mask[:, :, 0])
        salt_list.append(a)
        no_salt_list.append(b)

# Find Average and display for each type

# Display statistics (frequency contents of images)
# Compute average for each list
    running_sum_mag = np.zeros((division_size, division_size), dtype="float64")
    for each_salt in salt_list:
        running_sum_mag = running_sum_mag + each_salt
    running_sum_mag = running_sum_mag / len(salt_list)
    salt_display = running_sum_mag.astype(np.uint8)

    running_sum_mag = np.zeros((division_size, division_size), dtype="float64")
    for each_no_salt in no_salt_list:
        running_sum_mag = running_sum_mag + each_no_salt
    running_sum_mag = running_sum_mag / len(no_salt_list)
    no_salt_display = running_sum_mag.astype(np.uint8)

    final_disp = np.concatenate((np.fft.fftshift(salt_display),
                                 np.fft.fftshift(no_salt_display)),
                                axis=1)
else:
    final_disp = cv2.imread(sys.argv[1])
    salt_display = final_disp[0:division_size, 0:division_size,0]
    no_salt_display = final_disp[0:division_size, division_size:(2*division_size),0]

cv2.imshow("helllo", final_disp)
cv2.waitKey(0)

# take distance of each tile from either average

for each_name in image_files:
    print(each_name)
    image = cv2.imread(mypath + '/' + each_name)
    classification = classify_tiles_from_avgs.classify_tiles_from_avgs(image[:,:,0], salt_display, no_salt_display)
    print(classification)
    # mask = cv2.imread(mypath2 + '/' + each_name)

# assign tile to whichever it is closest to
