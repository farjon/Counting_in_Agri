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

def create_yolov5_args(args, yolo_det_args):
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