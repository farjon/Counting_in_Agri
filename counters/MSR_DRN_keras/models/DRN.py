import keras
from counters.MSR_DRN_keras import layers

def create_detection_subnetwork_graph(inputs, classification_feature_size=256):
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

    mid_outputs = []
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='detection_subnetwork_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)
        mid_outputs.append(keras.layers.Conv2D(
            filters=1,
            activation='relu',
            name='detection_subnetwork_mid_output_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options_for_output,
        )(outputs)
        )

    final_conv = keras.layers.Conv2D(
        filters=1,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=keras.initializers.zeros(),
        name='detection_subnetwork_final_conv',
        **options_for_output
    )(outputs)

    final_relu = keras.layers.Activation('relu', name='detection_subnetwork_final_relu')(final_conv)

    return mid_outputs + [outputs] + [final_relu]

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

def DRN_net(inputs, backbone_layers, name='DRN_net'):
    C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    Pyramid_layer_P3 = create_p3_feature(C3, C4, C5)

    #create the classification model for keypoint finding
    subnetwork_outputs = create_detection_subnetwork_graph(Pyramid_layer_P3)

    # create feature map for regression
    classification_last_map_for_reg = subnetwork_outputs[-2]
    classification_last_map_for_reg = keras.layers.GlobalAveragePooling2D(name='last_layer_for_reg')(classification_last_map_for_reg)

    # create counting layers
    classification_output = subnetwork_outputs[-1]
    cls_output_Step_Function = layers.SmoothStepFunction(threshold=0.4, beta = 1)(classification_output)
    cls_output_MaxPooled = layers.LocalSoftMax(kernel_size=(3, 3), strides=(1, 1), beta=100, name='LocalSoftMax')(
        cls_output_Step_Function)

    cls_output_Step_Function1 = layers.SmoothStepFunction1(threshold=0.8, beta=15, name='smooth_step_function1')(cls_output_MaxPooled)

    cls_output_downsampled = layers.GlobalSumPooling2D(name='SumPooling_cls_output')(cls_output_Step_Function1)

    reg_output_downsampled = keras.layers.Concatenate(axis=-1)([classification_last_map_for_reg, cls_output_downsampled])

    reg_output_downsampled = keras.layers.Dense(1, name="counting_reg_output",
                                                kernel_initializer=keras.initializers.normal(mean=0.5, stddev=0.1, seed=None),
                                                bias_initializer='zeros'
                                                )(reg_output_downsampled)

    outputs = [reg_output_downsampled] + subnetwork_outputs

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def DRN_net_inference(model=None, name='DRN_counting_net', **kwargs):

    if model is None:
        model = DRN_net(**kwargs)

    # create the output of the model to attach losses
    regression = model.outputs[0]
    classification = model.outputs[-1]

    outputs = [regression, classification]

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=outputs, name=name)