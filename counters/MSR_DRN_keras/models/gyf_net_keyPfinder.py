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


import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import keras_retinanet.bin
__package__ = "keras_retinanet.bin"

import keras
from .. import layers

def create_classification_graph(
        inputs,
        num_classes,
        option,
        pyramid_feature_size=256,
        classification_feature_size=256,
        name='classification_submodel',
):
    """ Creates the default regression submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }
    options_for_output = {
        'kernel_size': 1,
        'strides': 1,
        'padding': 'same',
    }
    outputs = inputs
    if option == 2:
        mid_outputs = []
        for i in range(4):
            outputs = keras.layers.Conv2D(
                filters=classification_feature_size,
                activation='relu',
                name='pyramid_classification_{}'.format(i),
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                **options
            )(outputs)
            mid_outputs.append(keras.layers.Conv2D(
                filters=num_classes,
                activation='relu',
                name='pyramid_classification_mid_output{}'.format(i),
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                **options_for_output
            )(outputs)
                               )

        outputs = keras.layers.Conv2D(
            filters=num_classes,
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer=keras.initializers.zeros(),  #initializers.PriorProbability(probability=prior_probability),
            name='pyramid_classification',
            **options_for_output
        )(outputs)

        outputs = keras.layers.Activation('relu', name='pyramid_classification_relu')(outputs)

        return mid_outputs + [outputs]
    elif option == 1:
        for i in range(4):
            outputs = keras.layers.Conv2D(
                filters=classification_feature_size,
                activation='relu',
                name='pyramid_classification_{}'.format(i),
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                **options
            )(outputs)

        outputs = keras.layers.Conv2D(
            filters=num_classes,
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer=keras.initializers.zeros(),  #initializers.PriorProbability(probability=prior_probability),
            name='pyramid_classification',
            **options_for_output
        )(outputs)

        outputs1 = keras.layers.Activation('relu', name='pyramid_classification_relu')(outputs)

        return [outputs] + [outputs1]

    elif option == 10 or option == 20:
        mid_outputs = []
        for i in range(4):
            outputs = keras.layers.Conv2D(
                filters=classification_feature_size,
                activation='relu',
                name='pyramid_classification_{}'.format(i),
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                **options
            )(outputs)
            mid_outputs.append(keras.layers.Conv2D(
                filters=num_classes,
                activation='relu',
                name='pyramid_classification_mid_output{}'.format(i),
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                **options_for_output,

            )(outputs)
            )

        outputs1 = keras.layers.Conv2D(
            filters=num_classes,
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer=keras.initializers.zeros(),  #initializers.PriorProbability(probability=prior_probability),
            name='pyramid_classification',
            **options_for_output
        )(outputs)

        outputs1 = keras.layers.Activation('relu', name='pyramid_classification_relu')(outputs1)

        return mid_outputs + [outputs] + [outputs1]


def create_regression_submodels(input_shape, option, regression_feature_size=256, FC_num_of_nuerons=1024, name='regression_submodel'):
    """ Creates the default regression submodel.

       Args
           num_anchors             : Number of anchors to regress for each feature level.
           pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
           regression_feature_size : The number of filters to use in the layers in the regression submodel.
           name                    : The name of the submodel.

       Returns
           A keras.models.Model that predicts regression values for each anchor.
       """
    if option == 1 or option == 0:
        inputs = keras.layers.Input(shape=(None, None, input_shape))
        # All new conv layers are initialized
        # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            'bias_initializer': 'zeros'
        }
        outputs = inputs
        for i in range(1):
            outputs = keras.layers.Conv2D(
                # TODO -  choose (256) or the same as before (2048+1)?
                filters=regression_feature_size,
                activation='relu',
                name='conv_regression_{}'.format(i),
                **options
            )(outputs)
        outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    elif option == 2 or option == 3:
        inputs = keras.layers.Input(shape=(input_shape,))
        outputs = inputs

    outputs = keras.layers.Dense(FC_num_of_nuerons, name='FC_regression', activation='relu')(outputs)

    outputs = keras.layers.Dense(int(FC_num_of_nuerons/2), name='FC2_regression', activation='relu')(outputs)

    outputs = keras.layers.Dense(1, name='regression', )(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def create_p3_feature(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    return P3

def create_p3_7_feature(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]

# def gyf_net_multi_scale(
#         inputs,
#         backbone_layers,
#         num_classes,
#         option = 1,
#         name='gyf_net'
# ):
#     """ Construct a RetinaNet model on top of a backbone.
#
#     This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).
#
#     Args
#         inputs                  : keras.layers.Input (or list of) for the input to the model.
#         num_classes             : Number of classes to classify.
#         num_anchors             : Number of base anchors.
#         create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
#         submodels               : Submodels to run on each feature map (default is regression and classification submodels).
#         name                    : Name of the model.
#
#     Returns
#         A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.
#
#         The order of the outputs is as defined in submodels:
#         ```
#         [
#             regression, classification, other[0], other[1], ...
#         ]
#         ```
#     """
#     C3, C4, C5 = backbone_layers
#
#     # compute pyramid features as per https://arxiv.org/abs/1708.02002
#     # features = create_pyramid_features(C3, C4, C5)
#     features = create_p3_feature(C3, C4, C5)
#
#     #create the classification model for keypoint finding
#     FC_submodel = submodel()
#     outputs = [FC_submodel(GAF) for GAF in features]
#
#     classification_outputs = create_classification_graph(p3, num_classes, option)
#     classification_output = classification_outputs[-1]
#
#     # cls_output_Step_Function1 = layers.SmoothStepFunction(threshold=0.4, beta = 1)(classification_output)
#     cls_output_MaxPooled = layers.LocalSoftMax(kernel_size = (3,3), strides=(1,1), beta=100, name='LocalSoftMax')(classification_output)
#
#     # cls_output_Step_Function = layers.StepFunction(threshold=0.5)(cls_output_MaxPooled)
#     cls_output_Step_Function2 = layers.SmoothStepFunction1(threshold=0.7, beta = 15)(cls_output_MaxPooled)
#     # cls_output_Step_Function2 = keras.layers.ThresholdedReLU(theta=0.6)(cls_output_MaxPooled)
#     cls_output_downsampled = layers.GlobalSumPooling2D(name='SumPooling_cls_output')(cls_output_Step_Function2)
#
#     cls_output_downsampled = keras.layers.Dense(1, name="sec_reg_output", kernel_initializer=keras.initializers.normal(mean=0.5, stddev=0.1, seed=None), bias_initializer='zeros')(cls_output_downsampled)
#
#     outputs = [cls_output_downsampled] + classification_outputs#[classification_output]
#     # outputs = classification_output
#     return keras.models.Model(inputs=inputs, outputs=outputs, name=name)
#
# def submodel(
#     num_classes = 1,
#     input_shape = (None,None,256),
#     FC_num_of_nuerons = 128,
#     feature_size = 256,
#     name = 'classification_submodel'
# ):
#
#     conv = input_layer
#     for i in range(2):
#         conv = keras.layers.Conv2D(
#             filters=feature_size,
#             activation='relu',
#             name='submodel_conv_{}'.format(i),
#             kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
#             bias_initializer='zeros',
#             **options
#         )(conv)
#     GlobalAvgPool_features = keras.layers.GlobalAveragePooling2D()(conv)
#     FC_regression = keras.layers.Dense(FC_num_of_nuerons, name='FC_regression', activation='relu')(GlobalAvgPool_features)
#     #dropout1 = layers.MC_dropout(level=0.2)(FC_regression)
#     FC2_regression = keras.layers.Dense(FC_num_of_nuerons//2, name='FC2_regression', activation='relu')(FC_regression)
#     #dropout2 = layers.MC_dropout(level=0.2)(FC2_regression)
#
#     regression_output = keras.layers.Dense(2, name='regression_output')(FC2_regression) # activation='relu'
#
#     options = {
#         'kernel_size': 3,
#         'strides': 1,
#         'padding': 'same',
#     }
#     options_for_output = {
#         'kernel_size': 1,
#         'strides': 1,
#         'padding': 'same',
#     }
#     input_layer = keras.layers.Input(shape=input_shape)
#     conv_output = input_layer
#     mid_outputs = []
#     for i in range(4):
#         conv_output = keras.layers.Conv2D(
#             filters=feature_size,
#             activation='relu',
#             name='pyramid_classification_{}'.format(i),
#             kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
#             bias_initializer='zeros',
#             **options
#         )(conv_output)
#         mid_outputs.append(keras.layers.Conv2D(
#             filters=num_classes,
#             activation='relu',
#             name='pyramid_classification_mid_output{}'.format(i),
#             kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
#             bias_initializer='zeros',
#             **options_for_output
#         )(conv_output)
#                            )
#
#     outputs1 = keras.layers.Conv2D(
#         filters=num_classes,
#         kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
#         bias_initializer=keras.initializers.zeros(),
#         # initializers.PriorProbability(probability=prior_probability),
#         name='pyramid_classification',
#         **options_for_output
#     )(conv_output)
#
#     outputs1 = keras.layers.Activation('relu', name='pyramid_classification_relu')(outputs1)
#
#     classification_last_map_for_reg = keras.layers.GlobalAveragePooling2D(name='last_layer_for_reg')(
#         conv_output)
#
#     cls_output_Step_Function1 = layers.SmoothStepFunction(threshold=0.4, beta=1)(outputs1)
#     cls_output_MaxPooled = layers.LocalSoftMax(kernel_size=(3, 3), strides=(1, 1), beta=100, name='LocalSoftMax')(
#         cls_output_Step_Function1)
#
#     cls_output_Step_Function2 = layers.SmoothStepFunction1(threshold=0.8, beta=15, name='smooth_step_function2')(
#         cls_output_MaxPooled)
#     cls_output_downsampled = layers.GlobalSumPooling2D(name='SumPooling_cls_output')(cls_output_Step_Function2)
#
#     reg_output_downsampled = keras.layers.Concatenate(axis=-1)(
#         [classification_last_map_for_reg, cls_output_downsampled])
#
#     reg_output_downsampled = keras.layers.Dense(1, name="sec_reg_output",
#                                                 kernel_initializer=keras.initializers.normal(mean=0.5, stddev=0.1,
#                                                                                              seed=None),
#                                                 bias_initializer='zeros')(reg_output_downsampled)
#
#     outputs = [reg_output_downsampled] + outputs1
# # return mid_outputs + [outputs] + [outputs1]
#
#     return keras.models.Model(inputs=input_layer, outputs=regression_output, name=name)

def gyf_net(
        inputs,
        backbone_layers,
        num_classes,
        option = 1,
        name='gyf_net'
):
    """ Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """
    C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    # features = create_pyramid_features(C3, C4, C5)
    p3 = create_p3_feature(C3, C4, C5)

    #create the classification model for keypoint finding
    classification_outputs = create_classification_graph(p3, num_classes, option)
    classification_output = classification_outputs[-1]

    if option == 2:
        cls_output_Step_Function1 = layers.SmoothStepFunction(threshold=0.4, beta = 1)(classification_output)
        cls_output_MaxPooled = layers.LocalSoftMax(kernel_size=(3, 3), strides=(1, 1), beta=100, name='LocalSoftMax')(
            cls_output_Step_Function1)

        cls_output_Step_Function2 = layers.SmoothStepFunction1(threshold=0.8, beta=15)(cls_output_MaxPooled)
        cls_output_downsampled = layers.GlobalSumPooling2D(name='SumPooling_cls_output')(cls_output_Step_Function2)

        cls_output_downsampled = keras.layers.Dense(1, name="sec_reg_output",
                                                    kernel_initializer=keras.initializers.normal(mean=0.5, stddev=0.1, seed=None),
                                                    bias_initializer='zeros')(cls_output_downsampled)

        outputs = [cls_output_downsampled] + classification_outputs  # [classification_output]
        # outputs = classification_output
        return keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    elif option == 1:
        classification_last_map_for_reg = classification_outputs[-2]
        classification_last_map_for_reg = keras.layers.GlobalAveragePooling2D(name='last_layer_for_reg')(
            classification_last_map_for_reg)

        cls_output_Step_Function1 = layers.SmoothStepFunction(threshold=0.4, beta = 1)(classification_output)
        cls_output_MaxPooled = layers.LocalSoftMax(kernel_size=(3, 3), strides=(1, 1), beta=100, name='LocalSoftMax')(
            cls_output_Step_Function1)

        cls_output_Step_Function2 = layers.SmoothStepFunction1(threshold=0.8, beta=15)(cls_output_MaxPooled)
        cls_output_downsampled = layers.GlobalSumPooling2D(name='SumPooling_cls_output')(cls_output_Step_Function2)

        reg_output_downsampled = keras.layers.Concatenate(axis=-1)(
            [classification_last_map_for_reg, cls_output_downsampled])

        cls_output_downsampled = keras.layers.Dense(1, name="sec_reg_output",
                                                    kernel_initializer=keras.initializers.normal(mean=0.5, stddev=0.1, seed=None),
                                                    bias_initializer='zeros')(reg_output_downsampled)

        outputs = [cls_output_downsampled] + classification_outputs  # [classification_output]
        # outputs = classification_output
        return keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    elif option == 10 or option == 20:
        classification_last_map_for_reg = classification_outputs[-2]
        classification_last_map_for_reg = keras.layers.GlobalAveragePooling2D(name='last_layer_for_reg')(classification_last_map_for_reg)

        cls_output_Step_Function1 = layers.SmoothStepFunction(threshold=0.4, beta = 1)(classification_output)
        cls_output_MaxPooled = layers.LocalSoftMax(kernel_size=(3, 3), strides=(1, 1), beta=100, name='LocalSoftMax')(
            cls_output_Step_Function1)

        cls_output_Step_Function2 = layers.SmoothStepFunction1(threshold=0.8, beta=15, name='smooth_step_function2')(cls_output_MaxPooled)
        # cls_output_Step_Function2 = keras.layers.ThresholdedReLU(theta=0.6)(cls_output_MaxPooled)
        cls_output_downsampled = layers.GlobalSumPooling2D(name='SumPooling_cls_output')(cls_output_Step_Function2)

        reg_output_downsampled = keras.layers.Concatenate(axis=-1)([classification_last_map_for_reg, cls_output_downsampled])

        reg_output_downsampled = keras.layers.Dense(1, name="sec_reg_output",
                                                    kernel_initializer=keras.initializers.normal(mean=0.5, stddev=0.1, seed=None),
                                                    bias_initializer='zeros')(reg_output_downsampled)

        outputs = [reg_output_downsampled] + classification_outputs  # [classification_output]
        # outputs = classification_output
        return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

    elif option == 11:
        classification_last_map_for_reg = classification_outputs[-2]
        classification_last_map_for_reg = keras.layers.GlobalAveragePooling2D(name='last_layer_for_reg')(classification_last_map_for_reg)

        cls_output_Step_Function1 = layers.SmoothStepFunction(threshold=0.4, beta = 1)(classification_output)
        cls_output_MaxPooled = layers.LocalSoftMax(kernel_size=(3, 3), strides=(1, 1), beta=100, name='LocalSoftMax')(
            cls_output_Step_Function1)

        cls_output_Step_Function2 = layers.SmoothStepFunction1(threshold=0.8, beta=15, name='smooth_step_function2')(cls_output_MaxPooled)
        # cls_output_Step_Function2 = keras.layers.ThresholdedReLU(theta=0.6)(cls_output_MaxPooled)
        cls_output_downsampled = layers.GlobalSumPooling2D(name='SumPooling_cls_output')(cls_output_Step_Function2)

        reg_output_downsampled = keras.layers.Concatenate(axis=-1)([classification_last_map_for_reg, cls_output_downsampled])

        reg_output_downsampled = keras.layers.Dense(2, name="sec_reg_output",
                                                    kernel_initializer=keras.initializers.normal(mean=0.5, stddev=0.1, seed=None),
                                                    bias_initializer='zeros')(reg_output_downsampled)

        outputs = [reg_output_downsampled] + classification_outputs  # [classification_output]
        # outputs = classification_output
        return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def gyf_net_LCC(
        model=None,
        option = 1,
        name='gyf_net-LCC',
        **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model             : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        anchor_parameters : Struct containing configuration for anchor generation (sizes, strides, ratios, scales).
        name              : Name of the model.
        *kwargs           : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """
    if model is None:
        model = gyf_net(**kwargs)

    # we expect the anchors, regression and classification values as first output
    regression = model.outputs[0]
    classification = model.outputs[-1]

    # outputs = [classification]
    outputs = [regression, classification]

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=outputs, name=name)