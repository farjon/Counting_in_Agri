import argparse
import os
import keras
import random

from counters.MSR_DRN_keras import models
from counters.MSR_DRN_keras.utils.evaluation_function import collect_predictions
from counters.MSR_DRN_keras.utils.keras_utils import check_keras_version, get_session
from counters.MSR_DRN_keras.preprocessing.csv_DRN_MSR_generator import CSVGenerator_MSR_DRN


def parse_args():
    parser = argparse.ArgumentParser(description='Count using MSR or DRN networks.')
    parser.add_argument('--model_type', type=str, default='DRN',
                        help='can be either MSR_P3_L2 / MSR_P3_P7_Gauss_MLE / DRN')
    parser.add_argument('--dataset_name', type=str, default='A1')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--exp_num', type=int, default=0,
                        help='if exp_num already exists, num will be increased automaically')
    parser.add_argument('--eval_on_set', type=str, default='Test')
    return parser.parse_args()

def report_and_save_results(predictions):
    pass


def create_generator(args):
    test_generator = CSVGenerator_MSR_DRN(
        mode='inference',
        model_type=args.model_type,
        base_dir=args.data_path,
        batch_size=args.batch_size,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side
    )

    return test_generator


def main(args=None):
    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model_path, backbone_name=args.backbone, model_type=args.model_type)
    # print model summary
    print(model.summary())

    # make prediction model
    if args.model_type in ['MSR_P3_P7_Gauss_MLE', 'MSR_P3_L2']:
        from counters.MSR_DRN_keras.models.MSR import MSR_net_inference
        model = MSR_net_inference(option=args.model_type, model=model)
    elif args.model_type == 'DRN':
        from counters.MSR_DRN_keras.models.DRN import DRN_net_inference
        model = DRN_net_inference(model=model)

    # start counting
    predictions = collect_predictions(
        args.model_type,
        generator,
        model)
    # TODO - report results
    if args.save_path is not None:
        report_and_save_results(predictions)

if __name__ == '__main__':
    random.seed(10)
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    # ------------- DON'T EDIT -------------
    args.image_min_side = 800
    args.image_max_side = 1333
    args.backbone = 'resnet50'
    # ---------------- END ----------------

    args.model_path = os.path.join(args.ROOT_DIR, 'Trained_Models', f'{args.model_type}_Models_snapshots',
                                      args.model_type, args.dataset_name, f'exp_{str(args.exp_num)}',
                                   f'{args.backbone}_{args.model_type}_best.h5')
    args.save_path = os.path.join(args.ROOT_DIR, 'Results', args.model_type, args.dataset_name,
                                  f'exp_{str(args.exp_num)}', f'{args.eval_on_set}_results')

    if args.dataset_name in ['A1', 'A2', 'A3', 'A4', 'A5']:
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', 'LCC', 'CVPPP2017_testing', 'testing', args.dataset_name)

    if args.model_type in ['MSR_P3_P7_Gauss_MLE', 'MSR_P3_L2']:
        args.calc_det_performance = False
    elif args.model_type == 'DRN':
        args.calc_det_performance = False
    else:
        raise ("Choose a relevant model type - MSR_P3_P7_Gauss_MLE / MSR_P3_L2 / DRN")

    args.test_csv_number_file = None
    args.test_csv_location_file = None


    args.snapshot = None
    args.imagenet_weights = True
    args.weights = None



    # for cpu - comment args.gpu
    args.gpu = '0'
    main(args)



