class Backbone(object):
    """ This class stores additional information on backbones.
    """
    def __init__(self, backbone):
        # a dictionary mapping custom layer names to the correct classes
        from .. import layers
        from counters.MSR_DRN_keras.layers import losses
        from counters.MSR_DRN_keras.utils import initializers
        self.custom_objects = {
            'UpsampleLike'      : layers.UpsampleLike,
            'PriorProbability'  : initializers.PriorProbability,
            'RegressBoxes'      : layers.RegressBoxes,
            'Anchors'           : layers.Anchors,
            'ClipBoxes'         : layers.ClipBoxes,
            'GlobalSumPooling2D': layers.GlobalSumPooling2D,
            'LocalSoftMax'      : layers.LocalSoftMax,
            'StepFunction'      : layers.StepFunction,
            'SmoothStepFunction': layers.SmoothStepFunction,
            'SmoothStepFunction1': layers.SmoothStepFunction1,
            '_smooth_l1'        : losses.smooth_l1(),
            '_focal'            : losses.focal(),
            '_focal_DRN'        : losses.focal_DRN(),
            '_mu_sigma_MSR'    : losses.mu_sigma_MSR(),
            '_mu_sig_gyf_L1'    : losses.mu_sig_gyf_L1(),
            'MLE_layer'         : layers.MLE_layer
        }

        self.backbone = backbone
        self.validate()

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')


def backbone(backbone_name, model_type):
    """ Returns a backbone object for the given backbone.
    """
    if 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name, model_type)


def load_model(filepath, model_type, backbone_name='resnet50', convert=False, nms=True):
    """ Loads a retinanet model using the correct custom objects.

    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name: Backbone with which the model was trained.
        convert: Boolean, whether to convert the model to an inference model.
        nms: Boolean, whether to add NMS filtering to the converted model. Only valid if convert=True.

    # Returns
        A keras.models.Model object.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import keras.models

    model = keras.models.load_model(filepath, custom_objects=backbone(backbone_name, model_type=model_type).custom_objects)

    return model
