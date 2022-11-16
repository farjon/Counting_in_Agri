
from __future__ import print_function

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import cv2
from PIL import Image
from GetEnvVar import GetEnvVar
import sys
import argparse
import random
import keras
import csv


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import keras_retinanet.bin
__package__ = "keras_retinanet.bin"



from keras_retinanet import models
from evaluate_LCC import get_session
from evaluate_LCC import create_generator
from torch_version import check_keras_version
from ..preprocessing.csv_LCC_generator import CSVLCCGenerator



def parse_args(args):
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')

    return parser.parse_args(args)



def _get_GTandPredictions_for_sub(option, generator, model, save_path):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_GT_counts[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        option          : The option number of the model
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """

    all_predicted_counts = []

    with open(save_path, 'w', newline='') as csvfile:
        for index in range(generator.size()):

            image = generator.next()
            #image, _ = generator.next()

            image_index = generator.groups[generator.group_index-1][0]
            full_rgbImage_name = generator.rbg_images_names[image_index]


            # get predictions - run network
            if option == 0 or option == 1:
                count = model.predict_on_batch(image)[0]

            elif option == 2 or option == 3 or option == 10 or option == 20:
                count = model.predict_on_batch(image)[0][0][0]

            count = round(count)
            all_predicted_counts.append(count)

            #print predictions to the Bar-Hillel file
            writer = csv.writer(csvfile)#,  quoting=csv.QUOTE_ALL)
            writer.writerow([full_rgbImage_name, count])



if __name__ == '__main__':
    args = None

    # parse arguments
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

    args.pipe = 'keyPfinder' #'reg' #


    if args.pipe == 'reg':
        args.option = 0

    elif args.pipe == 'keyPfinder':
        args.option = 20

    else:
        print("Choose a relevant pipe - keyPfinder or reg")
        sys.exit


    args.gpu = '1'


    args.dataset_type = 'csv'
    random.seed(10)

    args.snapshot = None
    args.imagenet_weights = True
    args.weights = None

    args.backbone = 'resnet50'
    args.batch_size = 1

    args.multi_gpu = False
    args.multi_gpu_force = False
    args.epochs = 1000
    args.freeze_backbone = False
    args.random_transform = True
    args.evaluation = True
    # TODO - choose min and max image size
    args.image_min_side = 800
    args.image_max_side = 1333


    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())



    for ds in ['A1', 'A2', 'A3', 'A4', 'A5']:

        #choose cv fold you would like to check base on the validation results
        cv_fold = 1

        args.dataset = ds

        args.exp_num = 4020

        args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets',
                                     'CVPPP2017_LCC_training', 'testing', args.dataset)

        model_path_name = os.path.join('exp_' + str(args.exp_num), 'cv_' + str(cv_fold))

        args.model = os.path.join(GetEnvVar('ModelsPath'), 'Counting_Models_snapshots', args.pipe,
                                          model_path_name, 'resnet50_csv.h5')

        results_dir = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe,
                                    str(args.exp_num)+'_predictions_on_eval_Ac',"results_" + args.pipe)


        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        args.bar_hillel_results_path = os.path.join(results_dir, args.dataset + ".csv")

        args.val_csv_leaf_number_file = None
        args.val_csv_leaf_location_file = None

        # create the generator
        generator =  CSVLCCGenerator(
            args.val_csv_leaf_number_file,
            args.val_csv_leaf_location_file,
            args.option,
            base_dir=args.data_path,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )

        # load the model
        print('Loading model, this may take a second...')
        model = models.load_model(args.model, backbone_name=args.backbone)
        # print model summary
        print(model.summary())

        # make prediction model
        if args.pipe == 'reg':
            from gyf_net_reg import gyf_net_LCC
        elif args.pipe == 'keyPfinder':
            from gyf_net_keyPfinder import gyf_net_LCC
        else:
            print('Wrong pipe name - should be reg or keyPfinder')


        model = gyf_net_LCC(model=model, option=args.option)

        _get_GTandPredictions_for_sub(args.option, generator, model, args.bar_hillel_results_path)

