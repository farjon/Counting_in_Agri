import keras
from counters.MSR_DRN_keras import layers


# def create_classification_graf(
#         inputs,
#         num_classes,
#         pyramid_feature_size=256,
#         classification_feature_size=256,
#         name='classification_submodel'
# ):
#     """ Creates the default regression submodel.
#
#     Args
#         num_classes                 : Number of classes to predict a score for at each feature level.
#         pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
#         classification_feature_size : The number of filters to use in the count_layers in the classification submodel.
#         name                        : The name of the submodel.
#
#     Returns
#         A keras.models.Model that predicts classes for each anchor.
#     """
#     options = {
#         'kernel_size': 3,
#         'strides': 1,
#         'padding': 'same',
#     }
#     outputs = inputs
#     for i in range(4):
#         outputs = keras.layers.Conv2D(
#             filters=classification_feature_size,
#             activation='relu',
#             name='pyramid_classification_{}'.format(i),
#             kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
#             bias_initializer='zeros',
#             **options
#         )(outputs)
#
#     outputs = keras.layers.Conv2D(
#         filters=num_classes,
#         #activation='relu',
#         kernel_initializer=keras.initializers.zeros(),
#         bias_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),  #initializers.PriorProbability(probability=prior_probability),
#         name='pyramid_classification',
#         **options
#     )(outputs)
#
#     outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)
#
#     return outputs


# def create_regression_submodels(input_shape, option, regression_feature_size=256, FC_num_of_nuerons=1024, name='regression_submodel'):
#     """ Creates the default regression submodel.
#
#        Args
#            num_anchors             : Number of anchors to regress for each feature level.
#            pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
#            regression_feature_size : The number of filters to use in the count_layers in the regression submodel.
#            name                    : The name of the submodel.
#
#        Returns
#            A keras.models.Model that predicts regression values for each anchor.
#        """
#     if option == 1 or option == 0:
#         inputs = keras.layers.Input(shape=(None, None, input_shape))
#         # All new conv count_layers are initialized
#         # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
#         options = {
#             'kernel_size': 3,
#             'strides': 1,
#             'padding': 'same',
#             'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
#             'bias_initializer': 'zeros'
#         }
#         outputs = inputs
#         for i in range(1):
#             outputs = keras.layers.Conv2D(
#                 # TODO -  choose (256) or the same as before (2048+1)?
#                 filters=regression_feature_size,
#                 activation='relu',
#                 name='conv_regression_{}'.format(i),
#                 **options
#             )(outputs)
#         outputs = keras.layers.GlobalAveragePooling2D()(outputs)
#     elif option == 2 or option == 3:
#         inputs = keras.layers.Input(shape=(input_shape,))
#         outputs = inputs
#
#     outputs = keras.layers.Dense(FC_num_of_nuerons, name='FC_regression', activation='relu')(outputs)
#
#     outputs = keras.layers.Dense(int(FC_num_of_nuerons/2), name='FC2_regression', activation='relu')(outputs)
#
#     outputs = keras.layers.Dense(1, name='regression', )(outputs)
#
#     return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def create_pyramid_features(C3, C4, C5, feature_size=256):
    """ Creates the FPN count_layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

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


def create_p3_feature(C3, C4, C5, feature_size=256):
    """ Creates the FPN count_layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
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

def regression_submodel(input_shape = (None,None,256), FC_num_of_nuerons = 128, feature_size = 256, name = 'FC_submodel'):

    input_layer = keras.layers.Input(shape=input_shape)

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    conv = input_layer
    for i in range(2):
        conv = keras.layers.Conv2D(
            filters=feature_size,
            activation='relu',
            name='submodel_conv_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(conv)

    GAP_features = keras.layers.GlobalAveragePooling2D()(conv)
    FC1_regression = keras.layers.Dense(FC_num_of_nuerons, name='FC1_regression', activation='relu')(GAP_features)
    FC2_regression = keras.layers.Dense(FC_num_of_nuerons//2, name='FC2_regression', activation='relu')(FC1_regression)
    # output - has 2 neurons, representing mu and sigma
    regression_output = keras.layers.Dense(2, name='regression_output')(FC2_regression)

    return keras.models.Model(inputs=input_layer, outputs=regression_output, name=name)


# def submodel_single_out(
#     do_dropout,
#     dropout_param,
#     input_shap = (None,None,256),
#     FC_num_of_nuerons = 128,
#     feature_size = 256,
#     name = 'FC_submodel'
# ):
#     input_layer = keras.layers.Input(shape=input_shap)
#     options = {
#         'kernel_size': 3,
#         'strides': 1,
#         'padding': 'same',
#     }
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
#     #dropout1 = count_layers.MC_dropout(level=0.2)(FC_regression)
#     if do_dropout:
#         FC_regression = keras.layers.Dropout(dropout_param)(FC_regression)
#
#     FC2_regression = keras.layers.Dense(FC_num_of_nuerons//2, name='FC2_regression', activation='relu')(FC_regression)
#     #dropout2 = count_layers.MC_dropout(level=0.2)(FC2_regression)
#     if do_dropout:
#         FC2_regression = keras.layers.Dropout(dropout_param)(FC2_regression)
#
#     regression_output = keras.layers.Dense(1, name='regression_output')(FC2_regression) # activation='relu'
#
#     return keras.models.Model(inputs=input_layer, outputs=regression_output, name=name)


def MSR_net(inputs, backbone_layers, model_variant, name='DRN_net'):


    C3, C4, C5 = backbone_layers

    if model_variant == 'MSR_P3_P7_Gauss_MLE':
        Pyramid_layer_P3_P7 = create_pyramid_features(C3, C4, C5)
        FC_submodel = regression_submodel()
        outputs = [FC_submodel(f) for f in Pyramid_layer_P3_P7]

    elif model_variant == 'MSR_P3_L2':
        Pyramid_layer_P3 = create_p3_feature(C3, C4, C5)
        GAP_features = keras.layers.GlobalAveragePooling2D()(Pyramid_layer_P3)
        FC1_regression = keras.layers.Dense(128, name='FC1_regression', activation='relu')(GAP_features)
        FC2_regression = keras.layers.Dense(64, name='FC2_regression', activation='relu')(FC1_regression)
        outputs = keras.layers.Dense(1, name='regression')(FC2_regression)
    else:
        raise (f'no model_type {model_variant}, please return to train.py and change model_type')

    #
    #
    #
    # if option== 'reg_fpn_p3':
    #     p3 = create_p3_feature(C3, C4, C5)
    #     GlobalAvgPool_features = keras.layers.GlobalAveragePooling2D()(p3)
    #     FC_regression = keras.layers.Dense(128, name='FC_regression', activation='relu')(GlobalAvgPool_features)
    #     FC_regression2 = keras.layers.Dense(64, name='FC_regression2', activation='relu')(FC_regression)
    #     outputs = keras.layers.Dense(1, name='regression')(FC_regression2)
    #
    # if option== 'reg_fpn_p3_p7_avg':
    #     # compute pyramid features as per https://arxiv.org/abs/1708.02002
    #     features = create_pyramid_features(C3, C4, C5)
    #     FC_submodel = submodel_single_out(do_dropout=do_dropout, dropout_param=dropout_param)
    #     outputs = [FC_submodel(GAF) for GAF in features]
    #
    # if option== 'reg_fpn_p3_p7_min_sig' or option== 'reg_fpn_p3_p7_mle' or option== 'reg_fpn_p3_p7_min_sig_L1' or option=='reg_fpn_p3_p7_mle_L1':
    #     # compute pyramid features as per https://arxiv.org/abs/1708.02002
    #     features = create_pyramid_features(C3, C4, C5)
    #     FC_submodel = submodel(do_dropout=do_dropout, dropout_param=dropout_param)
    #     outputs = [FC_submodel(GAF) for GAF in features]
    #
    # if option== 'reg_fpn_p3_p7_min_sig' or option== 'reg_fpn_p3_p7_mle' or option== 'reg_fpn_p3_p7_min_sig_L1' or option=='reg_fpn_p3_p7_mle_L1':
    #     # compute pyramid features as per https://arxiv.org/abs/1708.02002
    #     features = create_pyramid_features(C3, C4, C5)
    #     FC_submodel = submodel(do_dropout=do_dropout, dropout_param=dropout_param)
    #     outputs = [FC_submodel(GAF) for GAF in features]


    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def MSR_net_inference(model_type, model=None, name='MSR_counting_net', **kwargs):

    if model is None:
        model = MSR_net(**kwargs)

    # create the output of the model to attach losses
    if model_type == 'MSR_P3_P7_Gauss_MLE':
        outputs = model.outputs
        concated_outputs = keras.layers.concatenate(outputs,axis=-1)
        output = layers.MLE_layer(name='MLE')(concated_outputs)
    elif model_type == 'MSR_P3_L2':
        output = model.outputs

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=output, name=name)