import os
import json

def train_detectron2(args):
    from detectron2_windows.detectron2 import model_zoo
    from detectron2_windows.detectron2.engine import DefaultTrainer
    from detectron2_windows.detectron2.evaluation import COCOEvaluator
    from detectron2_windows.detectron2.checkpoint import DetectionCheckpointer
    from counters.Detection_based.config.adjust_detectron_cfg import create_cfg

    cfg = create_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    class CocoTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                os.makedirs("coco_eval", exist_ok=True)
                output_folder = "coco_eval"
            return COCOEvaluator(dataset_name, cfg, False, output_folder, use_fast_impl=False)

    cfg.OUTPUT_DIR = os.path.join(args.save_trained_models, args.detector + '_' + str(args.exp_number))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print('Training started, to see the progess, open tensorboard with "tensorboard --logdir /path/to/dir"')
    trainer.train()


def train_efficientDet(args):
    cfg_path = os.path.join(args.ROOT_DIR, 'counters', 'Detection_based', 'config',
                            'efficient_det_' + args.data + '_cfg')
    from EfficientDet_Pytorch.train import train as eff_train
    from counters.Detection_based.utils.create_detector_args import create_efficientDet_args

    eff_det_args = create_efficientDet_args(args, cfg_path)

    # model will be stored at eff_det_args.save_path
    # under the name 'args.data/'efficientdet-d{opt.compound_coef}_{best_epoch}.pth''
    best_epoch = eff_train(eff_det_args)
    return best_epoch, eff_det_args



def train_yolov5(args):
    from counters.Detection_based.utils.create_detector_args import create_yolov5_train_args
    # configuration files
    data_yaml_path = os.path.join(args.ROOT_DIR, 'counters', 'Detection_based', 'yolo_data_yaml',
                                  args.data + '.yaml')
    cfg_path = os.path.join(args.ROOT_DIR, 'counters', 'Detection_based', 'config',
                            args.detector + '_' + args.data + '_cfg.yaml')
    detector_version = args.detector.split('_')[0] + args.detector.split('_')[1]
    path_to_pretrained_model = os.path.join(args.ROOT_DIR, 'Trained_Models', 'pretrained', 'yolo',
                                            detector_version + '.pt')

    yolo_det_args = create_yolov5_train_args(args, cfg_path, data_yaml_path, path_to_pretrained_model)

    from yolov5.train import main as yolo_train
    yolo_train(yolo_det_args)
