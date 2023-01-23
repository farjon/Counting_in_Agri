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
    cfg.SOLVER.WARMUP_ITERS = 50
    cfg.SOLVER.MAX_ITER = 150 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (100, 150)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.TEST.EVAL_PERIOD = 50
    cfg.SOLVER.CHECKPOINT_PERIOD = 50
    return cfg