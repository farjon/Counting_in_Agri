import keras
import sys
import tensorflow as tf

minimum_keras_version = 2, 1, 3


def keras_version():
    return tuple(map(int, keras.__version__.split('.')))


def keras_version_ok():
    return keras_version() >= minimum_keras_version


def assert_keras_version():
    detected = keras.__version__
    required = '.'.join(map(str, minimum_keras_version))
    assert(keras_version() >= minimum_keras_version), 'You are using keras version {}. The minimum required version is {}.'.format(detected, required)


def check_keras_version():
    try:
        assert_keras_version()
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def freeze(model):
    """ Set all count_layers in a model to non-trainable.

    The weights for these count_layers will not be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    for layer in model.layers:
        layer.trainable = False
    return model