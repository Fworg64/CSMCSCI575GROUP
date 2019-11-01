
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Activation
from datetime import datetime
from sklearn.metrics import matthews_corrcoef

import multiprocessing
from multiprocessing import Pool

import time

#load images

#mypath = "/home/fworg64/gradschool/csci575/salt_data/train/images"
#mypath2 = "/home/fworg64/gradschool/csci575/salt_data/train/masks"

mypath  = "/u/st/da/aoltmanns/Pictures/saltimages/train/images"
mypath2 = "/u/st/da/aoltmanns/Pictures/saltimages/train/masks"

(_, _, image_files) = next(os.walk(mypath))
(_, _, mask_files) = next(os.walk(mypath2))


# Load images and run function
image_list = []
mask_list = []

#my_kernel = np.array([[1, 0, 1],
#                      [0, 1, 0],
#                      [1, 0, 1]], dtype = "uint8")

# my_kernel = np.ones((8,8),dtype = "uint8")
# DFT Tile size
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
    if idx > 20:
        break

num_cores = 10#multiprocessing.cpu_count()
print("Processing image files with %d cores" % num_cores)
t0 = time.time()


image_mask_list = list(zip(image_list, mask_list))

# Pack images and masks
def load_image(image,mask):
    # Do DFT of chunk
    #image = image_mask_tuple[0]
    #mask = image_mask_tuple[1]
    dft_size = 8 # BAD GLOBAL
    dft_size_2 = int(dft_size/2)
    dims = np.shape(image)

    features_vec = []
    labels = []

    #print(np.shape(image))
    #print(np.shape(mask))

    for row_dex in range(0, dims[0]-dft_size):
        for col_dex in range(0, dims[1] - dft_size):
            #print("head")
            #print(row_dex)
            #print(col_dex)
            #print(row_dex + dft_size)
            #print(col_dex + dft_size)
            #print(np.shape(image))
            chunk = image[row_dex:(row_dex+dft_size), col_dex:(col_dex+dft_size)]
            #chunk1 = image[row_dex:(row_dex+dft_size)]
            #print(np.shape(chunk1))
            #chunk = chunk1[:][col_dex:(col_dex+dft_size)]
            #print(np.shape(chunk))
            #print(chunk)
            dft_res = np.fft.fft2(chunk)
            scaller = np.max(dft_res)
            if scaller != 0:
                dft_res = dft_res/np.abs(scaller)
            # Put absolute val in list
            features_vec.append(np.reshape(np.absolute(dft_res), -1).tolist())
            labels.append(mask[row_dex+dft_size_2][col_dex + dft_size_2]/255.0)

    return (features_vec, labels)



with Pool(num_cores) as p:
    feature_labels_data = p.starmap(load_image, image_mask_list)
#feature_label = load_image(image_mask_list[0][0], image_mask_list[0][1])
#feature_labels_data = [feature_label]
print(type(feature_labels_data))
print(type(feature_labels_data[0][0]))
print(type(feature_labels_data[0][1]))

input_vector_list = []
output_class_list = []

for feature_label in feature_labels_data:
  input_vector_list.extend(feature_label[0])
  #  print(np.shape(feature_label[0]))
  output_class_list.extend(feature_label[1])
  #  print(np.shape(feature_label[1]))
print("input and output sizes")
print(np.shape(input_vector_list[0]))
print(np.shape(output_class_list[0]))
# for idx, each_image in enumerate(image_list):
#     print("On image %d of %d" % (idx, len(image_list)))
#     padded_size = np.shape(each_image)
#     for rowdex in range(0, padded_size[0] - 2*offset):
#        for coldex in range(0, padded_size[1] - 2*offset):
#             window = each_image[rowdex:(rowdex+kernel_shape[0]), coldex:(coldex + kernel_shape[1])]
#             #if np.shape(window) != (3, 3):
#             #    print("Uh-Oh")
#             feature_vector = []
 #            for win_rowdex in range(0,kernel_shape[0]):
 #                for win_coldex in range(0,kernel_shape[1]):
 #                    if my_kernel[win_rowdex, win_coldex] == 1:
 #                        feature_vector.append(window[win_rowdex, win_coldex]/128.0 - 1.0)
 #            input_vector_list.append(feature_vector)
 #            output_class_list.append([mask_list[idx][rowdex+offset][coldex+offset]])
t1 = time.time()
total = t1-t0
print("Time taken:")
print(total)
first_nonzero = -1
for idx,val in enumerate(output_class_list):
  if val > 0:
      first_nonzero = idx
      break
print("Example input vector")
print(input_vector_list[first_nonzero])
print("Corresponding lable")
print(output_class_list[first_nonzero])
input_array = np.array(input_vector_list)
output_array = np.array(output_class_list)

x_train, x_test, y_train, y_test = train_test_split(
                                      input_array, output_array, test_size=0.2, random_state=0)

print("Done loading")

#kernel_input_shape = np.sum(my_kernel)
kernel_input_shape = kernel_shape[0]*kernel_shape[1]
layers = [
    Dense(640,input_shape=(kernel_input_shape,), activation='relu'),
    # Dropout(0.3),
    Dense(640,input_shape=(640,), activation='relu'),
    # Dropout(0.1),
    Dense(640,input_shape=(640,), activation='relu'),
    Dense(640,input_shape=(640,), activation='relu'),
    #Dense(300,input_shape=(400,), activation='tanh'),
    #Dense(200,input_shape=(300,), activation='tanh'),
    Dense(2,input_shape=(64,), activation='softmax'),
    #Dense(1,input_shape=(10,), activation='tanh'),
]
model = keras.Sequential(layers)

model.compile(optimizer='adam', loss='mae', metrics=['mae', "mse", "accuracy"])
model.fit(x_train, y_train, epochs=5, verbose=1)
now = datetime. now()
timestamp = datetime.timestamp(now)
model.save("model_file_" + str(timestamp) + ".h5")
model.summary()
y_pred_float = model.predict(x_test)
print(y_pred_float)
y_other_pred = y_pred_float.argmax()
print(y_other_pred)
#y_pred = [int(my_num) for my_num in y_pred_float]
matthews_corr_coef = matthews_corrcoef(y_test, y_pred)
print("Matthews Correlation Coef: %.4f" % (matthews_corr_coef,))
scores = model.evaluate(x_test, y_test, verbose=0)
print("FFNN Scores, mae, mae, mse")
print(scores)


#print(y_pred)
#print(y_pred_float)

