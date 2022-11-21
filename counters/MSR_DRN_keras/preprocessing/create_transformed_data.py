import argparse
import csv
import os
import sys

import cv2
import numpy as np

import GetEnvVar as env
import create_csv_of_leaf_center

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.preprocessing"

from ..preprocessing.csv_DRN_MSR_generator import CSVLCCGenerator

from transform import random_transform_generator
from .cropping_transform import random_crop


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height


def resize_image(image, annotations):
    width = image.shape[1]
    height = image.shape[0]
    resized_width, resized_height = get_new_img_size(width, height, 600)
    image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

    num_bboxes = annotations.shape[0]
    gta = np.zeros((num_bboxes, 4))
    gta = gta.astype(int)
    for index in range(num_bboxes):
        bbox = annotations[index, :-1]
        gta[index, 0] = int(bbox[0] * (resized_width / float(width)))
        gta[index, 1] = int(bbox[2] * (resized_width / float(width)))
        gta[index, 2] = int(bbox[1] * (resized_height / float(height)))
        gta[index, 3] = int(bbox[3] * (resized_height / float(height)))
        cv2.rectangle(image, (gta[index, 0], gta[index, 2]), (gta[index, 1], gta[index, 3]), (0, 0, 255))

    return image

def parse_args(args):

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--option', help='choose an augmantation option - "rotation"/"translation"/"shear"/"scaling"/"flip"/"crop".')

    return parser.parse_args(args)


def write_to_csv( new_csv_file_name, new_csv_file_data):
    with open(new_csv_file_name, 'w', newline='') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for key in new_csv_file_data.keys():
            wr.writerow([key, new_csv_file_data[key]])

def main(dataset_name, transformation_names,  args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    args.dataset_path = os.path.join(env.GetEnvVar('DatasetsPath'), 'Phenotyping Datasets', 'Plant phenotyping',
                                  'data_2','CVPPP2017_LCC_training', 'training', dataset_name, dataset_name + '_Train')

    args.num_of_leaves_file = os.path.join(args.dataset_path, dataset_name + '_Train.csv')
    args.leaves_centers_file = os.path.join(args.dataset_path, dataset_name + '_Train_leaf_location.csv')

    args.outputs_dir = os.path.join(args.dataset_path, 'Augmented_Data')
    os.makedirs(args.outputs_dir, exist_ok=True)

    args.batch_size = 1

    args.visualise = False

    use_cropping = False #initial value, will be changed to True if transformation_name=="crop"

    original_counts_data = {}
    with open(args.num_of_leaves_file) as csvfile:
        readCSV = csv.reader(csvfile)
        for row in readCSV:
            rgbImage_name = row[0]
            count = row[1]
            original_counts_data[rgbImage_name]=count

    transformed_rgb_data = {}

    for transformation_name in transformation_names:

        if transformation_name == "rotation":
            transform_generator = random_transform_generator(
                min_rotation=-0.9,
                max_rotation=0.9,
            )

        elif transformation_name == "translation":
            transform_generator = random_transform_generator(
                min_translation=(-0.1, -0.1),
                max_translation=(0.1, 0.1),
            )
        elif transformation_name == "shear":
            transform_generator = random_transform_generator(
                min_shear=-0.1,
                max_shear=0.1,
            )
        elif transformation_name == "scaling":
            transform_generator = random_transform_generator(
                min_scaling=(0.9, 0.9),
                max_scaling=(1.1, 1.1),
            )
        elif transformation_name == "flip":
            transform_generator = random_transform_generator(
                flip_x_chance=0.5,
                flip_y_chance=0.5,
            )
        elif transformation_name == "crop":
            use_cropping = True

        else:
            print('{} does not exist as a transformation'.format(transformation_name))


        data_generator = CSVLCCGenerator(
            args.num_of_leaves_file,
            args.leaves_centers_file,
            base_dir=None,
            transform_generator=transform_generator,
            batch_size=1,
            group_method='ratio',  # one of 'none', 'random', 'ratio'
            shuffle_groups=True,
            image_min_side=800,
            image_max_side=1333
        )

        for rgb_image_name in data_generator.rbg_images_names:

            name = rgb_image_name.split("_rgb")
            name = name[0]
            centers_image_name =name+"_centers.png"

            if use_cropping:
                fg_image_name = name + "_fg.png"
                transformed_rgb_image, transformed_centers_image = random_crop(args, rgb_image_name, fg_image_name,
                                                                               centers_image_name)
            # else - not using cropping but any other transformation
            else:
                rgb_image = data_generator.load_image_byName(rgb_image_name)
                centers_image = data_generator.load_image_byName(centers_image_name)
                transformed_rgb_image, transformed_centers_image = data_generator.random_transform_rbg_centers_images(
                    rgb_image, centers_image)

            if args.visualise:
                rgb_images_horizontal = np.hstack((rgb_image, transformed_rgb_image))
                centers_images_horizontal = np.hstack((centers_image, transformed_centers_image))
                results_to_vis = np.vstack((rgb_images_horizontal, centers_images_horizontal))
                cv2.imshow('transformed image', results_to_vis)
                cv2.waitKey()

            transformed_rgb_image_file_name = args.outputs_dir + '\\' + name+"_" + transformation_name + "_rgb.png"
            transformed_centers_image_file_name = args.outputs_dir + '\\' + name + '_' + transformation_name + "_centers.png"

            cv2.imwrite(transformed_rgb_image_file_name, transformed_rgb_image)
            cv2.imwrite(transformed_centers_image_file_name, transformed_centers_image)

            transformed_rgb_data[name+"_" + transformation_name + "_rgb.png"] = original_counts_data[rgb_image_name]

    #create the xxx.csv file which has the leaf count data for the new augmented images
    new_num_of_leaves_file = os.path.join(args.outputs_dir,dataset_name+"_Train_Augmented.csv")
    write_to_csv(os.path.join(args.outputs_dir,new_num_of_leaves_file), transformed_rgb_data)

    #create the center points csv file for the augmented images
    create_csv_of_leaf_center.main(args.outputs_dir, dataset_name + "_Train_Augmented")


if __name__ == '__main__':

    dataset_name = 'A2' #"A1A2A3A4" #'
    transformation_names = ["rotation", "translation", "shear", "scaling", "flip", "crop"]
    main(dataset_name, transformation_names)
