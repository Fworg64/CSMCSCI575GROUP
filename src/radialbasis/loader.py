#import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from image_loader import load_image
import os
import numpy as np
import cv2

import time

import multiprocessing
from multiprocessing import Pool

model = load_model('./model_file_1573620687.335283.h5')

mypath  = "/u/st/da/aoltmanns/Pictures/saltimages/train/images"
mypath2 = "/u/st/da/aoltmanns/Pictures/saltimages/train/masks"

(_, _, image_files) = next(os.walk(mypath))
(_, _, mask_files) = next(os.walk(mypath2))


# Load images and run function
image_list = []
mask_list = []

kernel_shape = (8, 8)
# kernel_shape = np.shape(my_kernel)
offset = np.int((kernel_shape[0]) / 2)

for idx, each_name in enumerate(image_files):
  print(each_name)
  image = cv2.imread(mypath  + '/' + each_name)
  mask  = cv2.imread(mypath2 + '/' + each_name)
  # extend image by kernal_width/2

  padded_image = np.pad(image[:,:,0], offset, "symmetric")
  padded_mask  = np.pad(mask[:,:,0], offset, "symmetric")

  image_list.append(padded_image)
  mask_list.append(padded_mask)
  if idx > 10:
    break

num_cores = 10#multiprocessing.cpu_count()
print("Processing image files with %d cores" % num_cores)
t0 = time.time()
image_mask_list = list(zip(image_list, mask_list))


with Pool(num_cores) as p:
    feature_labels_data = p.starmap(load_image, image_mask_list)

input_vector_list = []
output_class_list = []

for feature_label in feature_labels_data:
    input_vector_list.extend(feature_label[0])
    output_class_list.extend(feature_label[1])
print("input and output sizes")
print(np.shape(input_vector_list[0]))
print(np.shape(output_class_list[0]))
t1 = time.time()
total = t1-t0
print("Time taken:")
print(total)
first_nonzero = -1
for idx,val in enumerate(output_class_list):
  if val > 0:
    first_nonzero = idx
    break
input_array = np.array(input_vector_list)
output_array = np.array(output_class_list)

prediction_raw = model.predict(input_array)
prediction = [1.0*(y>0.5) for y in prediction_raw]
print("abs diff per img")
for index in range(0,len(output_array)):
    print(prediction[index] - output_array[index])
print((np.sum(np.abs(prediction - output_array))))
print(np.shape(output_array))
