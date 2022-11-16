import os
import GetEnvVar as env
import cv2
import numpy as np


def random_crop(args, rgb_image_name, fg_image_name, centers_image_name):
    '''
    crops a random patch from the image - base on the center of the object
    :param args:
    :param rgb_image_file: original RGB image file name
    :param fg_image_file: mask of the image showing the location of the object in the image
    :param centers_image_file: image representing the 'centers' of the leaves in the image
    :return: cropped RGB image, cropped centers image
    '''
    dataset_path = args.dataset_path

    rgb_image_file = os.path.join(dataset_path, rgb_image_name)
    fg_image_file = os.path.join(dataset_path, fg_image_name)
    centers_image_file = os.path.join(dataset_path, centers_image_name)

    rgb_image = cv2.imread(rgb_image_file)
    fg_image = cv2.imread(fg_image_file, 0)
    centers_image = cv2.imread(centers_image_file)

    nonzero_inds = np.nonzero(fg_image)

    y_u = nonzero_inds[0][0]
    y_d = nonzero_inds[0][-1]

    x_l = np.min(nonzero_inds[1])
    x_r = np.max(nonzero_inds[1])

    x_middle = (x_l + x_r) // 2
    y_middle = (y_u + y_d) // 2

    quarter_crop = round(np.random.rand()*3 + 1)
    factor = 10
    # like a regular cartesian coordinates quadrants
    if quarter_crop == 1:
        rgb_image[y_u:y_middle+factor, x_middle-factor:x_r, :] = 0
        centers_image[y_u:y_middle+factor, x_middle-factor:x_r,:] = 0
    elif quarter_crop == 2:
        rgb_image[y_u:y_middle+factor, x_l:x_middle+factor, :] = 0
        centers_image[y_u:y_middle+factor, x_l:x_middle+factor,:] = 0
    elif quarter_crop == 3:
        rgb_image[y_middle-factor:y_d, x_l:x_middle+factor, :] = 0
        centers_image[y_middle-factor:y_d, x_l:x_middle+factor,:] = 0
    elif quarter_crop == 4:
        rgb_image[y_middle-factor:y_d, x_middle-factor:x_r, :] = 0
        centers_image[y_middle-factor:y_d, x_middle-factor:x_r,:] = 0

    return rgb_image, centers_image
