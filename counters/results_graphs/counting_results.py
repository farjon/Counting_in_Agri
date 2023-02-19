import os
import numpy as np


def report_mse(images_names, annotations, predictions):
    # calculate the mse of the predictions ant the annotations and return a dictionary with the mse of each image
    mse_dict = {}
    for image_name in images_names:
        mse_dict[image_name] = []
    for image_name, annotation, prediction in zip(images_names, annotations, predictions):
        mse_dict[image_name].append(np.square(np.subtract(annotation, prediction)).mean())
    return mse_dict

def report_mae(images_names, annotations, predictions):
    # calculate the mae of the predictions ant the annotations and return a dictionary with the mae of each image
    mae_dict = {}
    for image_name in images_names:
        mae_dict[image_name] = []
    for image_name, annotation, prediction in zip(images_names, annotations, predictions):
        mae_dict[image_name].append(np.abs(np.subtract(annotation, prediction)).mean())
    return mae_dict


def report_r_squared(images_names, annotations, predictions):
    # calculate the r_squared of the predictions ant the annotations and return a dictionary with the r_squared of each image
    r_squared_dict = {}
    for image_name in images_names:
        r_squared_dict[image_name] = []
    for image_name, annotation, prediction in zip(images_names, annotations, predictions):
        r_squared_dict[image_name].append(1 - np.square(np.subtract(annotation, prediction)).sum()/np.square(np.subtract(annotation, np.mean(annotation))).sum())
    return r_squared_dict


