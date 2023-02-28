#!/usr/bin/env python
import argparse
import os
import sys

import keras
import pandas as pd
import tensorflow as tf

from counters.MSR_DRN_keras import models
from counters.MSR_DRN_keras.models.DRN import DRN_net_inference
from counters.MSR_DRN_keras.models.MSR import MSR_net_inference
from counters.MSR_DRN_keras.utils.keras_utils import check_keras_version, get_session
from counters.MSR_DRN_keras.preprocessing.csv_DRN_MSR_generator import CSVGenerator_MSR_DRN
from counters.MSR_DRN_keras.utils.evaluation_function import evaluate

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    validation_generator = CSVGenerator_MSR_DRN(
        mode='training',
        model_type=args.model_type,
        classes={f'{args.class_name}': 0},
        csv_object_number_file=args.val_csv_object_number_file,
        csv_object_location_file=args.val_csv_object_location_file,
        batch_size=args.batch_size,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side
    )
    return validation_generator


def parse_args():
    parser = argparse.ArgumentParser(description='Test MSR or DRN network.')
    parser.add_argument('--model_type', type=str, default='MSR_P3_P7_Gauss_MLE', help = 'can be either MSR_P3_L2 / MSR_P3_P7_Gauss_MLE / DRN')
    parser.add_argument('--dataset_name', type=str, default='CherryTomato', help = 'can be either Grapes / WheatSpikelets / BananaLeaves / Banana / A1 / A2 / A3 / A4 / Ac / A1A2A3A4')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--exp_num', type=int, default=0, help = 'if exp_num already exists, num will be increased automaically')
    return parser.parse_args()


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

    # load and test best model
    print('Loading best model, this may take a second...')
    best_model = models.load_model(args.best_model, backbone_name=args.backbone,
                              model_type=args.model_type)

    # make prediction model
    if args.model_type == 'DRN':
        prediction_model = DRN_net_inference(model=best_model)
    elif args.model_type in ['MSR_P3_L2', 'MSR_P3_P7_Gauss_MLE']:
        prediction_model = MSR_net_inference(model_type=args.model_type, model=best_model)

    # start evaluation
    CountDiff, AbsCountDiff, CountAgreement, MSE = evaluate(
        args.model_type,
        generator,
        prediction_model,
        save_path=args.save_path,
        save_results=True
        )
    print("CountDiff:", CountDiff, "AbsCountDiff" ,AbsCountDiff, "CountAgreement", CountAgreement, "MSE", MSE)

    return CountDiff, AbsCountDiff, CountAgreement, MSE



if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    # ------------- DON'T EDIT -------------
    args.image_min_side = 800
    args.image_max_side = 1333
    args.freeze_backbone = False
    args.random_transform = True
    args.evaluation = True
    args.backbone = 'resnet50'
    args.snapshot_path = os.path.join(args.ROOT_DIR, 'Trained_Models', args.dataset_name,
                                      f'{args.model_type}_Models_snapshots', args.model_type,
                                      f'exp_{str(args.exp_num)}')
    args.save_path = os.path.join(args.ROOT_DIR, 'Results', args.dataset_name, args.model_type,
                                  f'exp_{str(args.exp_num)}', 'main_results')

    if args.dataset_name == 'Grapes':
        args.class_name = 'grape'
    elif args.dataset_name == 'WheatSpikelets':
        args.class_name = 'wheat_spikelet'
    elif args.dataset_name == 'BananaLeaves':
        args.class_name = 'banana_leaf'
    elif args.dataset_name == 'Banana':
        args.class_name = 'banana'
    elif args.dataset_name == 'CherryTomato':
        args.class_name = 'cherry_tomato'
    elif args.dataset_name in ['A1', 'A2', 'A3', 'A4', 'Ac', 'A1A2A3A4']:
        args.class_name = 'leafs'
    # ---------------- END ----------------
    args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.dataset_name, 'MSR_DRN')
    if args.dataset_name in ['A1', 'A2', 'A3', 'A4', 'Ac', 'A1A2A3A4']:
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', 'LCC', 'MSR_DRN', args.dataset_name)
    if args.model_type == 'DRN':
        # adding object locations maps
        args.val_csv_object_location_file = os.path.join(args.data_path, 'test',
                                                       args.dataset_name + '_test_location.csv')
    else:
        args.val_csv_object_location_file = ''
    # the number of object is obligatory for all model variants
    args.val_csv_object_number_file = os.path.join(args.data_path, 'test', args.dataset_name+'_test.csv')

    # for cpu - comment args.gpu
    args.gpu = '0'

    args.evaluation = True
    args.best_model = os.path.join(args.ROOT_DIR, 'Trained_Models', args.dataset_name, f'{args.model_type}_Models_snapshots', args.model_type,
                              f'exp_{str(args.exp_num)}', f'resnet50_{args.model_type}_best.h5')
    args.final_model = os.path.join(args.ROOT_DIR, 'Trained_Models', args.dataset_name, f'{args.model_type}_Models_snapshots', args.model_type,
                              f'exp_{str(args.exp_num)}', f'resnet50_final.h5')

    CountDiff, AbsCountDiff, CountAgreement, MSE = main(args)
    pd.DataFrame([{'CountDiff': CountDiff, 'AbsCountDiff': AbsCountDiff, 'CountAgreement': CountAgreement, 'MSE': MSE}]).to_csv(os.path.join(args.save_path, 'main_results.csv'))