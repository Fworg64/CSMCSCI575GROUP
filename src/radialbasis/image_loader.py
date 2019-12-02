import numpy as np

def load_image(image, mask, dft_size=8):
    dft_size_2 = int(dft_size/2)
    dims = np.shape(image)

    features_vec = []
    labels = []

    for row_dex in range(0, dims[0] - dft_size):
        for col_dex in range(0, dims[1] - dft_size):
            chunk = image[row_dex:(row_dex+dft_size), col_dex:(col_dex+dft_size)]
            dft_res = np.fft.fft2(chunk)
            scaller = np.max(dft_res)
            if scaller != 0:
                dft_res = dft_res/np.abs(scaller)
            features_vec.append(np.reshape(np.absolute(dft_res), -1).tolist())
            labels.append(mask[row_dex+dft_size_2][col_dex + dft_size_2]/255.0)

    return(features_vec, labels)
