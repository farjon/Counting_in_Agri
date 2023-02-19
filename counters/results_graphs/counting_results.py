import os
import numpy as np


def report_mse(images_names, annotations, predictions):
    # calculate the mse of the predictions ant the annotations and return a dictionary with the mse of each image
    mse_dict = {}
    for image_name, annotation, prediction in zip(images_names, annotations, predictions):
        mse_dict[image_name] = np.square(np.subtract(annotation, prediction))
    return mse_dict, np.mean(list(mse_dict.values()))

def report_mae(images_names, annotations, predictions):
    # calculate the mae of the predictions ant the annotations and return a dictionary with the mae of each image
    mae_dict = {}
    for image_name, annotation, prediction in zip(images_names, annotations, predictions):
        mae_dict[image_name] = np.abs(np.subtract(annotation, prediction))
    return mae_dict, np.mean(list(mae_dict.values()))


def report_r_squared(images_names, annotations, predictions):
    # calculate the r_squared of the predictions ant the annotations and return a dictionary with the r_squared of each image
    dividend = []
    divisor = []
    for _, annotation, prediction in zip(images_names, annotations, predictions):
        dividend.append(np.square(np.subtract(annotation, prediction)))
        divisor.append(np.square(np.subtract(annotation, np.mean(annotations))))
    return 1 - np.sum(dividend)/np.sum(divisor)

def report_agreement(images_names, annotations, predictions):
    # calculate the agreement of the predictions ant the annotations and return a dictionary with the agreement of each image
    agreement_dict = {}
    for image_name, annotation, prediction in zip(images_names, annotations, predictions):
        agreement_dict[image_name] = np.equal(annotation, prediction)
    return agreement_dict, np.mean(list(agreement_dict.values()))

def report_mrd(images_names, annotations, predictions):
    # calculate the mrd of the predictions ant the annotations and return a dictionary with the mrd of each image
    mrd_dict = {}
    for image_name, annotation, prediction in zip(images_names, annotations, predictions):
        mrd_dict[image_name] = np.abs(np.subtract(annotation, prediction))/annotation
    return mrd_dict, np.mean(list(mrd_dict.values()))
