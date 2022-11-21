import keras
from counters.MSR_DRN_keras.utils.evaluation_function import evaluate

class Evaluate_Counting_Callback(keras.callbacks.Callback):
    def __init__(self, model_type, generator, save_path=None, tensorboard=None, verbose=1):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            save_path       : The path to save images with visualized detections to.
            tensorboard     : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            verbose         : Set the verbosity level, by default this is set to 1.
        """
        self.model_type     = model_type
        self.generator      = generator
        self.save_path      = save_path
        self.tensorboard    = tensorboard
        self.verbose        = verbose

        super(Evaluate_Counting_Callback, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.generator.set_epoch(epoch)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # start evaluation
        self.DiC, self.abs_DiC, self.agreement, self.mse = evaluate(
            self.model_type,
            self.generator,
            self.model,
            save_path=self.save_path
        )


        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()

            summary_value = summary.value.add()
            summary_value.simple_value = self.DiC
            summary_value.tag = 'DiC'

            summary_value = summary.value.add()
            summary_value.simple_value = self.abs_DiC
            summary_value.tag = 'abs_DiC'

            summary_value = summary.value.add()
            summary_value.simple_value = self.agreement
            summary_value.tag = '%-Agreement'

            summary_value = summary.value.add()
            summary_value.simple_value = self.mse
            summary_value.tag = 'MSE'

            self.tensorboard.writer.add_summary(summary, epoch)

        logs['DiC'] = self.DiC
        logs['abs_DiC'] = self.abs_DiC
        logs['agreement'] = self.agreement
        logs['MAE'] = self.mse

        if self.verbose == 1:
            print('DiC - ', self.DiC, '; abs_DiC - ', self.abs_DiC, '; %-Agreement - ', self.agreement*100, '; MSE - ', self.mse)
