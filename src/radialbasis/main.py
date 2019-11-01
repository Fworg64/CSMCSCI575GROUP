
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Activation
from datetime import datetime
from sklearn.metrics import matthews_corrcoef


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

my_kernel = np.ones((8,8),dtype = "uint8")

kernel_shape = np.shape(my_kernel)
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
    if idx > 1500:
        break

#x_train, x_test, y_train, y_test = train_test_split(
#                                      image_list, mask_list, test_size=0.33, random_state=0)

input_vector_list = []
output_class_list = []
for idx, each_image in enumerate(image_list):
    print("On image %d of %d" % (idx, len(image_list)))
    padded_size = np.shape(each_image)
    for rowdex in range(0, padded_size[0] - 2*offset):
        for coldex in range(0, padded_size[1] - 2*offset):
            window = each_image[rowdex:(rowdex+kernel_shape[0]), coldex:(coldex + kernel_shape[1])]
            #if np.shape(window) != (3, 3):
            #    print("Uh-Oh")
            feature_vector = []
            for win_rowdex in range(0,kernel_shape[0]):
                for win_coldex in range(0,kernel_shape[1]):
                    if my_kernel[win_rowdex, win_coldex] == 1:
                        feature_vector.append(window[win_rowdex, win_coldex]/128.0 - 1.0)
            input_vector_list.append(feature_vector)
            output_class_list.append([mask_list[idx][rowdex+offset][coldex+offset]])

print(input_vector_list[10])
input_array = np.array(input_vector_list)
output_array = np.array(output_class_list)

x_train, x_test, y_train, y_test = train_test_split(
                                      input_array, output_array, test_size=0.2, random_state=0)

print("Done loading")

kernel_input_shape = np.sum(my_kernel)

layers = [
    Dense(2000,input_shape=(kernel_input_shape,), activation='tanh'),
    # Dropout(0.3),
    Dense(1000,input_shape=(2000,), activation='tanh'),
    # Dropout(0.1),
    Dense(500,input_shape=(1000,), activation='tanh'),
    Dense(400,input_shape=(500,), activation='tanh'),
    #Dense(300,input_shape=(400,), activation='tanh'),
    #Dense(200,input_shape=(300,), activation='tanh'),
    Dense(10,input_shape=(200,), activation='tanh'),
    Dense(1,input_shape=(10,), activation='tanh'),
]
model = keras.Sequential(layers)

model.compile(optimizer='adam', loss='mae', metrics=['mae', "mse"])
model.fit(x_train, y_train, epochs=50, verbose=1)
y_pred_float = model.predict(x_test)
y_pred = [int(my_num) for my_num in y_pred_float]
matthews_corr_coef = matthews_corrcoef(y_test, y_pred)
print("Matthews Correlation Coef: %.4f" % (matthews_corr_coef,))
scores = model.evaluate(x_test, y_test, verbose=0)
print("FFNN Scores, mae, mae, mse")
print(scores)
now = datetime. now()
timestamp = datetime.timestamp(now)
model.save("model_file_" + str(timestamp) + ".h5")
model.summary()
#print(y_pred)
#print(y_pred_float)

