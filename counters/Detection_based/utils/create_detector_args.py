import os
import argparse


def create_yolov5_train_args(args, cfg_path, data_yaml_path, path_to_pretrained_model):
    # used parameters
    from yolov5.train import parse_opt as yolo_parse_args
    yolo_det_args = yolo_parse_args()
    yolo_det_args.cfg = cfg_path
    yolo_det_args.imgsz = 640
    yolo_det_args.batch_size = args.batch_size
    yolo_det_args.epochs = args.epochs
    yolo_det_args.data = data_yaml_path
    yolo_det_args.cfg = cfg_path
    yolo_det_args.weights = path_to_pretrained_model
    yolo_det_args.name = args.detector + '_' + args.data + '_results_exp_' + str(args.exp_number)
    yolo_det_args.project = args.save_trained_models
    yolo_det_args.device = 0

    # for visualization needs
    import wandb
    wandb.login()
    yolo_det_args.entity = None
    yolo_det_args.upload_dataset = False
    yolo_det_args.bbox_interval = -1
    yolo_det_args.artifact_alias = 'latest'

    return yolo_det_args

def create_yolov5_infer_args(args):
    from yolov5.detect import parse_opt as yolo_parse_args
    yolo_infer_args = yolo_parse_args()
    yolo_infer_args.project = args.save_counting_results
    yolo_infer_args.name = args.detector + '_' + args.data + '_results_exp_' + str(args.exp_number)
    yolo_infer_args.weights = os.path.join(args.save_trained_models, yolo_infer_args.name, 'weights', 'best.pt')
    yolo_infer_args.source = os.path.join(args.data_path, 'images', args.test_set)
    yolo_infer_args.conf_thres = args.det_test_thresh
    yolo_infer_args.iou_thres = args.iou_threshold  # NMS iou threshold
    yolo_infer_args.nosave = True
    yolo_infer_args.save_txt = True
    return yolo_infer_args