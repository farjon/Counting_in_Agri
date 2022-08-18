import os
import torch
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='Banana', help='choose a dataset')
    parser.add_argument('-si', '--split_images', type=bool, default=False, help='should we split the images into tiles')
    parser.add_argument('-nt', '--num_of_tiles', type=int, default=7, help='number of tiles')
    parser.add_argument('-p', '--padding', type=int, default=100, help='padding size in case of splitting')
    # --------------------------- Training Arguments -----------------------
    parser.add_argument('-det', '--detector', type=str, default='RetinaNet', help='choose a detector efficientDet_i / yolov5_i / fasterRCNN / RetinaNet'
                               'in case you choose efficientDet, please add "_i" where i is the compound coefficient')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('-exp', '--exp_number', type=int, default=0, help='number of current experiment')
    parser.add_argument('-c', '--criteria', type=str, default='mse', help='criteria can be mse / mae')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='set learning rate')
    parser.add_argument('-o', '--optim', type=str, default='sgd', help='choose optimizer adam / adamw / sgd')
    parser.add_argument('-ve', '--val_interval', type=int, default=2, help='run model validation every X epochs')
    parser.add_argument('-se', '--save_interval', type=int, default=100, help='save checkpoint every X steps')
    parser.add_argument('-dt', '--det_test_thresh', type=float, default=0.2, help='detection threshold (for test), defualt is 0.2')
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

    if args.split_images:
        from utils.split_raw_images_anno_coco import split_to_tiles
        print('Notice - to split the images, bbox annotations are needed')
        split_to_tiles(args, args.num_of_tiles, args.padding)
        #TODO - make sure that the split function works on other data formats
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data + '_split', labels_format)
    else:
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data, labels_format)


    test_folder = os.path.join(args.data_path, 'test')

    if args.detector == 'fasterRCNN' or args.detector == 'RetinaNet':
        args.det_model = 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'
        import detectron2_windows.detectron2
        from detectron2_windows.detectron2.utils.logger import setup_logger
        setup_logger()
        # import some common detectron2 utilities
        from detectron2_windows.detectron2.data.datasets import register_coco_instances

        # register_coco_instances : name, metadata, json_file, image_root
        register_coco_instances("train", {}, f'{args.data_path}/train/instances_train.json', f'{args.data_path}/train/')
        register_coco_instances("val", {}, f'{args.data_path}/val/instances_val.json', f'{args.data_path}/val/')
        register_coco_instances("test", {}, f'{args.data_path}/test/instances_test.json', f'{args.data_path}/test/')

    # --------------------------- Start edit ---------------------------
    # setting up path to save trained models
    args.save_trained_models = os.path.join(args.ROOT_DIR, 'Trained_Models', args.data)
    torch.hub.set_dir(args.save_trained_models)
    # setting up log directory
    args.log_path = os.path.join(args.ROOT_DIR, 'Logs', 'EfficientDet')
    os.makedirs(args.log_path, exist_ok=True)

    # args.save_checkpoint_path = os.path.join(args.ROOT_DIR, 'Trained_Models', args.data, str(args.exp_number))
    # os.makedirs(args.save_checkpoint_path, exist_ok=True)

    # --------------------------- Stage 1 - Detection ---------------------------
    if args.detector == 'fasterRCNN' or args.detector == 'RetinaNet':
        from counters.Detection_Regression.bin.train_detectors import train_detectron2
        from counters.Detection_Regression.bin.test_detectors import test_detectron2
        train_detectron2(args)
        images_counting_results, images_detection_results = test_detectron2(args)

    elif args.detector.split('_')[0] == 'efficientDet':
        from counters.Detection_Regression.bin.train_detectors import train_efficientDet
        from counters.Detection_Regression.bin.test_detectors import test_efficientDet
        best_epoch, eff_det_args = train_efficientDet(args)
        images_counting_results, images_detection_results = test_efficientDet(args, eff_det_args, best_epoch)

    elif args.detector.split('_')[0] == 'yolov5':
        from counters.Detection_Regression.bin.train_detectors import train_yolov5

    # Report results


if __name__ =='__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)
