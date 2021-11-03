import os
from detectron2_windows.detectron2.config import get_cfg
from detectron2_windows.detectron2 import model_zoo

def create_cfg(args):
    model_name = args.det_model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model_name}"))
    cfg.DATASETS.TRAIN = ('train',)
    cfg.DATASETS.TEST = ('val',)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{model_name}")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.TEST.EVAL_PERIOD = 500
    return cfg