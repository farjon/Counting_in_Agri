import os
import torch
import numpy as np
import pandas as pd
import argparse
from counters.results_graphs import counting_results


def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='Melons_split', help='choose a dataset')
    parser.add_argument('-si', '--split_images', type=bool, default=False, help='should we split the images into tiles')
    parser.add_argument('-nt', '--num_of_tiles', type=int, default=7, help='number of tiles')
    parser.add_argument('-p', '--padding', type=int, default=100, help='padding size in case of splitting')
    # --------------------------- Training Arguments -----------------------
    parser.add_argument('-det', '--detector', type=str, default='efficientDet_2', help='choose a detector efficientDet_i / yolov5_i / fasterRCNN / RetinaNet'
                               'in case you choose efficientDet, please add "_i" where i is the compound coefficient'
                                'in case you choose yolov5, please add "_i" where i is the model size')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('-exp', '--exp_number', type=int, default=0, help='number of current experiment')
    parser.add_argument('-c', '--criteria', type=str, default='mse', help='criteria can be mse / mae')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='set learning rate')
    parser.add_argument('-o', '--optim', type=str, default='sgd', help='choose optimizer adam / adamw / sgd')
    parser.add_argument('-ve', '--val_interval', type=int, default=5, help='run model validation every X epochs')
    parser.add_argument('-se', '--save_interval', type=int, default=100, help='save checkpoint every X steps')
    parser.add_argument('-dt', '--det_test_thresh', type=float, default=0.4, help='detection threshold (for test), defualt is 0.2')
    parser.add_argument('-iou', '--iou_threshold', type=float, default=0.2, help='iou threshold (for test), defualt is 0.2')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--save_predictions', action='store_true', help='either to save or not, the predicted results of the detection')
    args = parser.parse_args()
    return args


def main(args):
    # --------------------------- Don't edit --------------------------- #
    # define device (use cuda if available)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # torch and numpy reproducibility setup
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(10)
    np.random.seed(10)

    if args.detector == 'fasterRCNN' or args.detector == 'RetinaNet' or args.detector.split('_')[0] == 'efficientDet':
        labels_format = 'coco'
    elif args.detector.split('_')[0] == 'yolov5':
        labels_format = 'yolo'
    else:
        raise ValueError('Detector not supported')

    if args.split_images:
        from utils.tile_images.split_raw_images_anno_coco import split_to_tiles
        print('Notice - to split the images, bbox annotations are needed')
        split_to_tiles(args, args.num_of_tiles, args.padding)
        #TODO - make sure that the split function works on other data formats
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data + '_split', 'Detection', labels_format)
    else:
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data, 'Detection', labels_format)

    # what is the test set?
    # in case no test set exists, use validation set
    args.test_set = 'test' if os.path.exists(os.path.join(args.data_path, 'test')) else 'val'

    if args.detector == 'fasterRCNN' or args.detector == 'RetinaNet':
        if args.detector == 'fasterRCNN':
            args.det_model = 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'
        else:
            args.det_model = 'retinanet_R_50_FPN_3x.yaml'
        import detectron2_windows.detectron2
        from detectron2_windows.detectron2.utils.logger import setup_logger
        setup_logger()
        # import some common detectron2 utilities
        from detectron2_windows.detectron2.data.datasets import register_coco_instances

        # register_coco_instances : name, metadata, json_file, image_root
        register_coco_instances("train", {}, f'{args.data_path}/train/instances_train.json', f'{args.data_path}/train/')
        register_coco_instances("val", {}, f'{args.data_path}/val/instances_val.json', f'{args.data_path}/val/')
        if args.test_set == 'test':
            register_coco_instances("test", {}, f'{args.data_path}/test/instances_test.json', f'{args.data_path}/test/')

    # --------------------------- Start edit ---------------------------
    # setting up path to save trained models
    args.save_trained_models = os.path.join(args.ROOT_DIR, 'Trained_Models', args.data)
    torch.hub.set_dir(args.save_trained_models)
    # setting up log directory
    args.log_path = os.path.join(args.ROOT_DIR, 'Logs', 'EfficientDet')
    os.makedirs(args.log_path, exist_ok=True)

    args.save_counting_results = os.path.join(args.ROOT_DIR, 'Results', args.data, 'Detection')

    # args.save_checkpoint_path = os.path.join(args.ROOT_DIR, 'Trained_Models', args.data, 'Detection', 'model' str(args.exp_number))
    # os.makedirs(args.save_checkpoint_path, exist_ok=True)

    # --------------------------- Stage 1 - Detection ---------------------------
    if args.detector == 'fasterRCNN' or args.detector == 'RetinaNet':
        from counters.Detection_based.bin.train_detectors import train_detectron2
        from counters.Detection_based.bin.test_detectors import test_detectron2
        train_detectron2(args)
        images_counting_results, images_detection_results = test_detectron2(args)
        counting_save_results = os.path.join(args.save_counting_results, f'{args.detector}_{str(args.exp_number)}')
        os.makedirs(counting_save_results, exist_ok=True)

    elif args.detector.split('_')[0] == 'efficientDet':
        from counters.Detection_based.bin.train_detectors import train_efficientDet
        from counters.Detection_based.bin.test_detectors import test_efficientDet
        best_epoch, eff_det_args = train_efficientDet(args)
        images_counting_results, images_detection_results = test_efficientDet(args, eff_det_args, best_epoch)

    elif args.detector.split('_')[0] == 'yolov5':
        from counters.Detection_based.bin.train_detectors import train_yolov5
        from counters.Detection_based.bin.test_detectors import test_yolov5
        # train_yolov5(args)
        images_counting_results, images_detection_results, yolo_infer_args = test_yolov5(args)
        counting_save_results = os.path.join(yolo_infer_args.project, yolo_infer_args.name)

    # Report results
    mse_dict, mse = counting_results.report_mse(images_counting_results['image_name'], images_counting_results['gt_count'], images_counting_results['pred_count'])
    mae_dict, mae = counting_results.report_mae(images_counting_results['image_name'], images_counting_results['gt_count'], images_counting_results['pred_count'])
    agreement_dict, agreement = counting_results.report_agreement(images_counting_results['image_name'], images_counting_results['gt_count'], images_counting_results['pred_count'])
    mrd_dict, mrd = counting_results.report_mrd(images_counting_results['image_name'], images_counting_results['gt_count'], images_counting_results['pred_count'])
    r_squared = counting_results.report_r_squared(images_counting_results['image_name'], images_counting_results['gt_count'], images_counting_results['pred_count'])

    # save results to csv
    results_df = pd.DataFrame({'image_name': images_counting_results['image_name'], 'gt_count': images_counting_results['gt_count'], 'pred_count': images_counting_results['pred_count']})
    results_df.to_csv(os.path.join(counting_save_results, 'complete_results.csv'), index=False)

    metric_results_df = pd.DataFrame({'mse': mse, 'mae': mae, 'agreement': agreement, 'mrd': mrd, 'r_squared': r_squared}, index=[0])
    metric_results_df.to_csv(os.path.join(counting_save_results, 'metric_results.csv'), index=False)

if __name__ =='__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)
