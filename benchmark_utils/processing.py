import numpy as np


def grey_image(colored_img):
    n, m, _ = np.shape(colored_img)
    new_img = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            new_img[i][j] = 1/3 * (colored_img[0] +
                                   colored_img[1] +
                                   colored_img[2])

    return new_img
