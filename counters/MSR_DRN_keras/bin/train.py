#!/usr/bin/env python

import argparse
import os
import keras
import random
import numpy as np

import keras.preprocessing.image
import tensorflow as tf

from counters.MSR_DRN_keras import losses
from counters.MSR_DRN_keras import models
from counters.MSR_DRN_keras.callbacks import RedirectModel
from counters.MSR_DRN_keras.models.DRN import DRN_net_inference
from counters.MSR_DRN_keras.utils.keras_utils import check_keras_version, get_session

from counters.MSR_DRN_keras.callbacks.LLC_eval import Evaluate_LLCtype
from counters.MSR_DRN_keras.preprocessing.csv_LCC_generator import CSVLCCGenerator
from counters.MSR_DRN_keras.utils.anchors import make_shapes_callback, anchor_targets_bbox
from counters.MSR_DRN_keras.utils.general_utils import freeze as freeze_model
from counters.MSR_DRN_keras.utils.transform import random_transform_generator

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
tf.set_random_seed(1234)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train MSR or DRN network.')
    parser.add_argument('--model_type', type=str, default='DRN', help = 'can be either MSR of DRN')
    parser.add_argument('--dataset_type', type=str, default='csv')
    parser.add_argument('--dataset_name', type=str, default='A1')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_num', type=int, default=0, help = 'if exp_num already exists, num will be increased automaically')
    parser.add_argument('--early_stopping_indicator', type=str, default='AbsCountDiff', help = 'if using an early stopping, what should we monitor (AbsCountDiff, MSE)')
    parser.add_argument('--snapshot_index', type=str, default='AbsCountDiff', help = 'save a snapshot if this index is improved (AbsCountDiff, MSE)')
    return parser.parse_args()

def create_models(backbone_counting_net, weights, freeze_backbone=False):
    modifier = freeze_model if freeze_backbone else None

    model = model_with_weights(backbone_counting_net(modifier=modifier), weights=weights, skip_mismatch=True)
    training_model = model

    # make prediction model
    prediction_model = DRN_net_inference(model=model)

    variable_losses = {'detection_subnetwork_final_relu': losses.focal_DRN(),
                      'detection_subnetwork_mid_output_0': losses.focal_DRN(),
                      'detection_subnetwork_mid_output_1': losses.focal_DRN(),
                      'detection_subnetwork_mid_output_2': losses.focal_DRN(),
                      'detection_subnetwork_mid_output_3': losses.focal_DRN(),
                      'counting_reg_output': keras.losses.mae
                       }

    # compile model
    training_model.compile(
        loss=variable_losses,
        optimizer = keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    return model, training_model, prediction_model


def create_callbacks(model, prediction_model, validation_generator, args):
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        evaluation = Evaluate_LLCtype(validation_generator, tensorboard=tensorboard_callback, save_path = args.save_path)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    if args.snapshot_path:
        if args.snapshot_index in ['AbsCountDiff', 'MSE']:
            mode = 'min'
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(args.snapshot_path, exist_ok=True)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(args.snapshot_path, f'{args.backbone}_{args.dataset_type}.h5'),
            verbose=1,
            period= 1,
            save_best_only=True,
            monitor=args.snapshot_index,
            mode=mode
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    if args.early_stopping_indicator:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=args.early_stopping_indicator,
            min_delta=0,
            patience=50,
            verbose=0,
            mode=mode
        )
        early_stopping = RedirectModel(early_stopping, model)
        callbacks.append(early_stopping)

    #TODO - add these parameters to args
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 3,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))

    callbacks.append(keras.callbacks.TerminateOnNaN())
    return callbacks


def create_generators(args):
    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.2,
            max_rotation=0.2,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = None

    train_generator = CSVLCCGenerator(
        args.train_csv_leaf_number_file,
        args.train_csv_leaf_location_file,
        transform_generator=transform_generator,
        batch_size=args.batch_size,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side
    )

    if args.val_csv_leaf_number_file:
        validation_generator = CSVLCCGenerator(
            args.val_csv_leaf_number_file,
            args.val_csv_leaf_location_file,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )
    else:
        validation_generator = None

    return train_generator, validation_generator

def main(args=None):
    # make sure keras is the minimum required version
    # TODO - write specifically what is needed in the requirements
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create output directories
    os.makedirs(args.save_path, exist_ok=True)
    if args.tensorboard_dir:
        os.makedirs(args.tensorboard_dir, exist_ok=True)

    # create object that stores backbone information
    backbone = models.backbone(args.backbone, args.model_type)

    # create the generators
    train_generator, validation_generator = create_generators(args)
    args.steps_per_epoch = args.enlarge_steps_per_epoch*int(train_generator.size() // args.batch_size)

    # create the model
    if args.snapshot is not None:
        print(f'Loading model from checkpoint {args.snapshot_path}, this may take a second...')
        model = models.load_model(os.path.join(args.snapshot_path, 'resnet50_csv_snapshot.h5'), backbone_name=args.backbone)
        training_model = model
        prediction_model = DRN_net_inference(model=model)
        print('loaded model from checkpoint - model will be trained from given checkpoint')
    else:
        weights = backbone.download_imagenet(args.save_pre_trained_weights_path)
        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_counting_net=backbone.create_net,
            weights=weights,
            freeze_backbone=args.freeze_backbone)

    # print model summary
    print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(model, prediction_model, validation_generator, args)

    # start training
    history = training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    # store final model
    model.save(os.path.join(args.snapshot_path, 'resnet50_final.h5'))

    return history

if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    #------------- DON'T EDIT -------------
    args.image_min_side = 800
    args.image_max_side = 1333
    args.enlarge_steps_per_epoch = 5
    args.freeze_backbone = False
    args.random_transform = True
    args.evaluation = True
    args.backbone = 'resnet50'
    args.snapshot_path = os.path.join(args.ROOT_DIR, 'Trained_Models', f'{args.model_type}_Models_snapshots', args.model_type, f'exp_{str(args.exp_num)}')
    args.tensorboard_dir = os.path.join(args.ROOT_DIR, 'Results', args.model_type, args.dataset_name, f'exp_{str(args.exp_num)}', 'log_dir')
    args.save_path = os.path.join(args.ROOT_DIR, 'Results', args.model_type, args.dataset_name, f'exp_{str(args.exp_num)}', 'main_results')
    # ---------------- END ----------------
    # if snapshot is True - model will be uploaded from previous checkpoint (in args.snapshop_path) and continue from there
    args.snapshot = None

    if args.dataset_name in ['A1', 'A2', 'A3', 'A4', 'Ac', 'A1A2A3A4']:
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', 'LCC', 'training', args.dataset_name)

    # model variant
    if args.model_type == 'DRN':
        args.option = 20
    elif args.model_type == 'MSR':
        args.option = ''
    else:
        raise('model type unknown, should be either MSR or DRN')

    args.train_csv_leaf_number_file = os.path.join(args.data_path, 'train', args.dataset_name+'_Train.csv')
    args.train_csv_leaf_location_file = os.path.join(args.data_path, 'train', args.dataset_name+'_Train_leaf_location.csv')

    args.val_csv_leaf_number_file = os.path.join(args.data_path, 'val', args.dataset_name+'_Val.csv')
    args.val_csv_leaf_location_file = os.path.join(args.data_path, 'val', args.dataset_name+'_Val_leaf_location.csv')

    args.imagenet_weights = True
    args.save_pre_trained_weights_path = os.path.join(args.ROOT_DIR, 'Trained_Models', 'pretrained')
    # for cpu - comment args.gpu
    args.gpu = '0'

    main(args)
