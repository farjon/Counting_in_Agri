import os
import argparse

def create_efficientDet_args(args, cfg_path):
    eff_det_args = argparse.ArgumentParser()
    eff_det_args.project = cfg_path
    eff_det_args.compound_coef = int(args.detector.split('_')[1])
    eff_det_args.batch_size = args.batch_size
    eff_det_args.lr = args.lr
    eff_det_args.optim = args.optim
    eff_det_args.num_epochs = args.epochs
    eff_det_args.val_interval = args.val_interval
    eff_det_args.save_interval = args.save_interval
    eff_det_args.es_patience = args.es_patience
    eff_det_args.data_path = args.data_path
    eff_det_args.log_path = args.log_path
    eff_det_args.save_path = args.save_trained_models
    eff_det_args.head_only = False
    eff_det_args.load_weights = None
    eff_det_args.num_workers = 0
    eff_det_args.debug = False
    eff_det_args.es_min_delta = 0.0
    return eff_det_args

def create_yolov5_train_args(args, cfg_path, data_yaml_path, path_to_pretrained_model):
    # used parameters
    yolo_det_args = argparse.ArgumentParser()
    yolo_det_args = yolo_det_args.parse_args()
    yolo_det_args.cfg = cfg_path
    yolo_det_args.imgsz = 640
    yolo_det_args.batch_size = args.batch_size
    yolo_det_args.epochs = args.epochs
    yolo_det_args.data = data_yaml_path
    yolo_det_args.cfg = cfg_path
    yolo_det_args.weights = path_to_pretrained_model
    yolo_det_args.name = args.detector + '_' + args.data + '_results'
    yolo_det_args.project = args.save_trained_models
    yolo_det_args.device = 0

    # unused, but needed parameters
    yolo_det_args.resume = False
    yolo_det_args.hyp = os.path.join(args.ROOT_DIR, 'yolov5', 'data', 'hyps', 'hyp.scratch.yaml')
    yolo_det_args.evolve = False
    yolo_det_args.exist_ok = True
    yolo_det_args.single_cls = True
    yolo_det_args.noval = True
    yolo_det_args.nosave = False
    yolo_det_args.workers = 0
    yolo_det_args.freeze = 0
    yolo_det_args.adam = True
    yolo_det_args.linear_lr = True
    yolo_det_args.sync_bn = True
    yolo_det_args.cache = True
    yolo_det_args.rect = True
    yolo_det_args.image_weights = True
    yolo_det_args.quad = False
    yolo_det_args.noautoanchor = True
    yolo_det_args.label_smoothing = 0.0
    yolo_det_args.patience = 10
    yolo_det_args.multi_scale = True
    yolo_det_args.save_period = -1  # -1 to disable

    # for visualization needs
    import wandb
    wandb.login()
    yolo_det_args.entity = None
    yolo_det_args.upload_dataset = False
    yolo_det_args.bbox_interval = -1
    yolo_det_args.artifact_alias = 'latest'

    return yolo_det_args

def create_yolov5_infer_args(args, yolo_det_args):
    yolo_infer_args = argparse.ArgumentParser()
    yolo_infer_args = yolo_infer_args.parse_args()
    yolo_infer_args.project = os.path.join(args.ROOT_DIR, 'Results', 'detect', 'yolo')
    yolo_infer_args.name = 'exp'
    yolo_infer_args.weights = os.path.join(args.save_trained_models, yolo_det_args.name, 'weights', 'best.pt')
    yolo_infer_args.source = os.path.join(args.data_path, 'test')
    yolo_infer_args.conf_thres = 0.25
    yolo_infer_args.iou_thres = 0.45  # NMS iou threshold
    yolo_infer_args.imgsz = [640]
    yolo_infer_args.max_det = 1000
    yolo_infer_args.view_img = False
    yolo_infer_args.save_txt = True
    yolo_infer_args.save_conf = True
    yolo_infer_args.save_crop = True
    yolo_infer_args.nosave = True
    yolo_infer_args.exist_ok = True
    yolo_infer_args.classes = None
    yolo_infer_args.agnostic_nms = True
    yolo_infer_args.augment = True
    yolo_infer_args.visualize = True
    yolo_infer_args.update = True
    yolo_infer_args.line_thickness = 3
    yolo_infer_args.hide_conf = False
    yolo_infer_args.hide_labels = False
    yolo_infer_args.half = False
    yolo_infer_args.imgsz *= 2 if len(yolo_infer_args.imgsz) == 1 else 1  # expand
    return yolo_infer_args