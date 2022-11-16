#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys

import keras
import tensorflow as tf
import pandas as pd
import random
from GetEnvVar import GetEnvVar
import numpy as np
import cv2


# Allow relative imports when being executed as script.
# if __name__ == "__main__" and __package__ is None:
#if __package__ is None:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# import keras_retinanet.bin
__package__ = "keras_retinanet.bin"


# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..utils.eval_LCC import _get_prediction
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.eval_LCC import evaluate
from ..utils.keras_version import check_keras_version
from ..preprocessing.csv_LCC_generator import CSVLCCGenerator

from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
    resize_image_320,
    read_image_bgr,
    read_image_gray_scale
)
from ..utils.transform import transform_ab


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )
    elif args.dataset_type == 'pascal':
        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )
    elif args.dataset_type == 'csv':
        validation_generator = CSVLCCGenerator(
            args.val_csv_leaf_number_file,
            args.val_csv_leaf_location_file,
            args.option,
            base_dir=args.data_path,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')

    return parser.parse_args(args)


def main(args=None):

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)
    # print model summary
    print(model.summary())

    # make prediction model
    if args.pipe == 'reg':
        from ..models.gyf_net_reg import gyf_net_LCC
    elif args.pipe == 'keyPfinder':
        from ..models.gyf_net_keyPfinder import gyf_net_LCC
    else:
        print('Wrong pipe name - should be reg or keyPfinder')
        return

    model = gyf_net_LCC(model=model, option=args.option)

    # start evaluation

    if args.calc_det_performance:
        CountDiff, AbsCountDiff, CountAgreement, MSE, R_2, ap = evaluate(
            args.option,
            args.val_csv_leaf_number_file,
            generator,
            model,
            save_path=args.save_path,
            calc_det_performance=args.calc_det_performance
        )
    else:
        CountDiff, AbsCountDiff, CountAgreement, MSE, R_2 = evaluate(
            args.option,
            args.val_csv_leaf_number_file,
            generator,
            model,
            save_path=args.save_path,
            calc_det_performance=args.calc_det_performance
        )
    print("CountDiff:", CountDiff, "AbsCountDiff" ,AbsCountDiff, "CountAgreement", CountAgreement, "MSE", MSE)
    # df = pd.read_csv(args.save_res_path)
    # df = df.append(pd.DataFrame({"Exp":[str(args.exp_num)],"Dataset":[args.dataset],"Dic":[str(CountDiff)], "AbsDic":[str(AbsCountDiff)], "Agreement":[str(CountAgreement)], "MSE":[str(MSE)]}),sort=False)
    # df.to_csv(args.save_res_path,index=False)
    if args.calc_det_performance:
        return CountDiff, AbsCountDiff, CountAgreement, MSE, R_2, ap
    return CountDiff, AbsCountDiff, CountAgreement, MSE, R_2

def proccess_image(image, option):
    pass

def _get_prediction_single_image(option, image_path, model, save_path='', calc_det_performance=False):

    visualize_im = False
    all_predicted_counts = []

    image = read_image_bgr(image_path)
    image = preprocess_image(image)
    image, _ = resize_image(image)

    image = proccess_image(image, option)

    for index in range(generator.size()):
        image, output = generator.next()

        image_index = generator.groups[generator.group_index-1][0]
        full_rgbImage_name = generator.rbg_images_names[image_index]
        Image_name = full_rgbImage_name.split("_rgb")[0]

        if visualize_im:
            if not generator.epoch == None:
                if generator.epoch==0 or (generator.epoch+1) % 20 == 0 :
                    visualize_images(output, Image_name, save_path, generator, model, image)
            else:
                visualize_images(output, Image_name, save_path, generator, model, image)


        if option == 'reg_baseline_c5_dubreshko' or option == 'reg_baseline_c5' or option == 'reg_fpn_p3' or \
                option == 'reg_fpn_p3_p7_avg' or option == 'reg_fpn_p3_p7_mle' or option == 'reg_fpn_p3_p7_min_sig' \
                or option == 'reg_fpn_p3_p7_min_sig_L1' or option == 'reg_fpn_p3_p7_mle_L1':
            count = model.predict_on_batch(image)
        if option == 'reg_baseline_c5_dubreshko' or option == 'reg_baseline_c5' or option == 'reg_fpn_p3':
            count = count[0][0]
        if option == 'reg_fpn_p3_p7_mle':
            count = count[0]
        if option == 'reg_fpn_p3_p7_avg':
            mus = [count[i][0][0] for i in range(len(count))]
            count = np.mean(mus)
            print("image:", Image_name, "GT:", output[0][0], ", predictions:", mus, ", count: ", count)

        if option == 'reg_fpn_p3_p7_min_sig' or option == 'reg_fpn_p3_p7_min_sig_L1':
            mus = [count[i][0][0] for i in range(len(count))]
            sigmas = [count[i][0][1] for i in range(len(count))]
            count = mus[np.argmin(sigmas)]
            print("image:", Image_name, "GT:", output[0][0], ", predictions:", mus, ", count: ", count)

        if option == 'reg_fpn_p3_p7_mle_L1':
            mus = np.asarray([count[i][0][0] for i in range(len(count))])
            sigmas = np.asarray([1 / np.exp(count[i][0][1]) for i in range(len(count))])
            sorted_inds = np.argsort(mus)
            mus = mus[sorted_inds]
            sigmas = sigmas[sorted_inds]
            procesed_sigmas = np.cumsum(sigmas) / np.sum(sigmas)
            mle_ind = np.where(procesed_sigmas > 0.5)[0]
            count = mus[mle_ind][0]
        elif option == 2 or option == 3 or option == 10 or option == 1 or option == 20:
            count = model.predict_on_batch(image)[0][0][0]
            #
        print("image:", Image_name, ",", "predicted:", round(count), "(", count, ")",
                  ", abs diff: ", round(abs(output[0][0] - count), 2))

        count = round(count)
        all_predicted_counts.append(count)

    return all_predicted_counts


if __name__ == '__main__':
    args = None

    # parse arguments
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

    args.dataset = 'BL'
    args.random_transform = True
    args.gpu = '0'

    args.pipe = 'reg' #'keyPfinder' #'reg'  #
    args.exp_num = 765
    eval_on_set = 'Test'
    if args.pipe == 'reg':
        '''
        reg options:
        'reg_baseline_c5_dubreshko'
        'reg_baseline_c5'
        'reg_fpn_p3'
        'reg_fpn_p3_p7_avg'
        'reg_fpn_p3_p7_mle'   this is our best regressor
        'reg_fpn_p3_p7_min_sig'
        '''
        args.option = 'reg_fpn_p3_p7_mle'
        args.calc_det_performance = False
    elif args.pipe == 'keyPfinder':
        args.option = 20
        args.calc_det_performance = False
    else:
        print("Choose a relevant pipe - keyPfinder or reg")
        sys.exit

    args.dataset_type = 'csv'
    random.seed(10)

    if args.dataset == 'BL':
        # args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets', 'Banana_leaves', args.dataset)
        args.data_path = 'C:\\Users\\Administrator\\Desktop\\base_dirt\\'

        args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'BL_exp_res', args.pipe, "results",
                                      'exp_' + str(args.exp_num))

        args.val_csv_leaf_number_file = None
        args.val_csv_leaf_location_file = None
    else:
        args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets',
                                      'CVPPP2017_LCC_training', 'training', args.dataset)

        args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results",
                                      'exp_' + str(args.exp_num) + 'testing')

        args.val_csv_leaf_number_file = os.path.join(args.data_path, 'test', args.dataset + '_' + eval_on_set + '.csv')
        args.val_csv_leaf_location_file = os.path.join(args.data_path, 'test', args.dataset + '_' + eval_on_set + '_leaf_location.csv')

    args.snapshot = None
    args.imagenet_weights = True
    args.weights = None

    args.backbone = 'resnet50'
    args.batch_size = 1

    args.multi_gpu = False
    args.multi_gpu_force = False

    args.freeze_backbone = False

    args.evaluation = True
    # TODO - choose min and max image size
    args.image_min_side = 800
    args.image_max_side = 1333

    cv_num = 1
    args.model = os.path.join(GetEnvVar('ModelsPath'), 'Counting_Models_snapshots', args.pipe,
                              'exp_' + str(args.exp_num), 'cv_' + str(cv_num), 'resnet50_csv.h5')

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)
    # print model summary
    print(model.summary())

    # make prediction model
    if args.pipe == 'reg':
        from ..models.gyf_net_reg import gyf_net_LCC
    elif args.pipe == 'keyPfinder':
        from ..models.gyf_net_keyPfinder import gyf_net_LCC


    model = gyf_net_LCC(model=model, option=args.option)

    # start evaluation

    predictions = _get_prediction(
        args.option,
        generator,
        model,
        save_path=args.save_path,
        calc_det_performance=args.calc_det_performance)

    # df = pd.read_csv(args.save_res_path)
    # df = df.append(pd.DataFrame({"Exp":[str(args.exp_num)],"Dataset":[args.dataset],"Dic":[str(CountDiff)], "AbsDic":[str(AbsCountDiff)], "Agreement":[str(CountAgreement)], "MSE":[str(MSE)]}),sort=False)
    # df.to_csv(args.save_res_path,index=False)

    guy  = 1


