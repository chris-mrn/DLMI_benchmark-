import numpy as np


def flat_set_for_deep(X_img, y):
    X_img_flat = []
    y_flat = []
    for i in range(len(X_img)):
        for j in range(len(X_img[i])):
            X_img_flat.append(X_img[i][j].flatten())
            y_flat.append(y[i])
    return X_img_flat, y_flat


def flat_set_img_bio(X_img, X_bio, y):
    X_flat = []
    y_flat = []
    for i in range(len(X_img)):
        for j in range(len(X_img[i])):
            image_bio = np.concatenate((X_img[i][j].flatten(), X_bio[i]))
            X_flat.append(image_bio)
            y_flat.append(y[i])
    return X_flat, y_flat
