import os
import json

def train_detectron2(args):
    from detectron2_windows.detectron2 import model_zoo
    from detectron2_windows.detectron2.engine import DefaultTrainer
    from detectron2_windows.detectron2.evaluation import COCOEvaluator
    from detectron2_windows.detectron2.checkpoint import DetectionCheckpointer
    from counters.Detection_Regression.config.adjust_detectron_cfg import create_cfg

    cfg = create_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    class CocoTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                os.makedirs("coco_eval", exist_ok=True)
                output_folder = "coco_eval"
            return COCOEvaluator(dataset_name, cfg, False, output_folder)

    cfg.OUTPUT_DIR = os.path.join(args.save_trained_models, args.detector + '_' + str(args.exp_number))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print('Training started, to see the progess, open tensorboard with "tensorboard --logdir /path/to/dir"')
    trainer.train()


def train_efficientDet(args):
    cfg_path = os.path.join(args.ROOT_DIR, 'counters', 'Detection_Regression', 'config',
                            'efficient_det_' + args.data + '_cfg')
    from EfficientDet_Pytorch.train import train as eff_train
    from counters.Detection_Regression.utils.create_detector_args import create_efficientDet_args

    eff_det_args = create_efficientDet_args(args, cfg_path)

    # model will be stored at eff_det_args.save_path
    # under the name 'args.data/'efficientdet-d{opt.compound_coef}_{best_epoch}.pth''
    best_epoch = eff_train(eff_det_args)
    return best_epoch, eff_det_args



def train_yolov5(args):
    # configuration files
    data_yaml_path = os.path.join(args.ROOT_DIR, 'counters', 'Detection_Regression', 'yolo_data_yaml',
                                  args.data + '.yaml')
    cfg_path = os.path.join(args.ROOT_DIR, 'counters', 'Detection_Regression', 'config',
                            args.detector + '_' + args.data + '_cfg.yaml')
    detector_version = args.detector.split('_')[0] + args.detector.split('_')[1]
    path_to_pretrained_model = os.path.join(args.ROOT_DIR, 'Trained_Models', 'pretrained', 'yolo',
                                            detector_version + '.pt')

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

    from yolov5.train import main as yolo_train
    yolo_train(yolo_det_args)
