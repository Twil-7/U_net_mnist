import numpy as np
import cv2


def img_to_cuboid(annotation, p):
    size = annotation.shape
    cuboid = np.zeros((size[0], size[1], size[2], p))

    for i in range(size[0]):
        for j in range(p):
            annotation_img = annotation[i]
            slice_img = np.zeros((size[1], size[2]))
            slice_img[annotation_img == j] = 1
            cuboid[i, :, :, j] = slice_img

    return cuboid


def read_img():
    p = 11
    path1 = '/home/archer/CODE/PF/data/combined.npy'
    path2 = '/home/archer/CODE/PF/data/segmented.npy'
    image = np.load(path1)/255            # (5000, 64, 84)
    annotation = np.load(path2)           # (5000, 64, 84)
    cuboid = img_to_cuboid(annotation, p)

    # cv2.namedWindow("Image")
    # cv2.imshow("Image", image[0])
    # cv2.waitKey(0)

    train_x = image[0:4900, :, :]
    test_x = image[4900:5000, :, :]
    train_y = cuboid[0:4900, :, :, :]
    test_y = cuboid[4900:5000, :, :, :]

    return train_x,  train_y, test_x, test_y



