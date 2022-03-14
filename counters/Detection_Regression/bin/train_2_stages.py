import os
import torch
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='CherryTomato', help='choose a dataset')
    parser.add_argument('-si', '--split_images', type=bool, default=False, help='should we split the images into tiles')
    parser.add_argument('-nt', '--num_of_tiles', type=int, default=7, help='number of tiles')
    parser.add_argument('-p', '--padding', type=int, default=100, help='padding size in case of splitting')
    # --------------------------- Training Arguments -----------------------
    parser.add_argument('-det', '--detector', type=str, default='efficientDet_0', help='choose a detector efficientDet_i / yolov5_i / fasterRCNN / RetinaNet'
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
        register_coco_instances("train", {}, f'{args.data_path}/train/train.json', os.path.join(args.data_path, 'train'))
        register_coco_instances("val", {}, f'{args.data_path}/val/val.json', os.path.join(args.data_path, 'val'))
        register_coco_instances("test", {}, f'{args.data_path}/test/test.json', os.path.join(args.data_path, 'test'))

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
        from detectron2_windows.detectron2.engine import DefaultTrainer
        from detectron2_windows.detectron2.evaluation import COCOEvaluator
        from counters.Detection_Regression.config.adjust_detectron_cfg import create_cfg
        cfg = create_cfg(args)
        class CocoTrainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                if output_folder is None:
                    os.makedirs("coco_eval", exist_ok=True)
                    output_folder = "coco_eval"
                return COCOEvaluator(dataset_name, cfg, False, output_folder)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # Run inference over the test set towards the regression phase
        from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.engine import DefaultPredictor
        import cv2
        from glob import glob

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.DATASETS.TEST = ("test",)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.det_test_thresh  # set the testing threshold for this model
        predictor = DefaultPredictor(cfg)
        #TODO - add an iou threshold
        #TODO - run on folder and collect results
        outputs = []
        for imgae_path in glob(os.path.join(args.data_path, '*jpg')):
            im = cv2.imread(imgae_path)
            out = predictor(im)
            outputs.append({
                'image_name': imgae_path,
                'predictions': out
            })


    elif args.detector.split('_')[0] == 'efficientDet':
        cfg_path = os.path.join(args.ROOT_DIR, 'counters', 'Detection_Regression', 'config',
                                'efficient_det_'+args.data+'_cfg')
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
        eff_det_args.saved_path = args.save_trained_models
        eff_det_args.head_only = False
        eff_det_args.load_weights = None
        eff_det_args.num_workers = 0
        eff_det_args.debug = False
        eff_det_args.es_min_delta = 0.0

        from EfficientDet_Pytorch.train import train as eff_train
        from counters.Detection_Regression.bin.test_eff import test_eff_on_folder
        best_epoch = eff_train(eff_det_args)
        # model will be stored at eff_det_args.saved_path
        # under the name 'args.data/'efficientdet-d{opt.compound_coef}_{best_epoch}.pth''
        model_path = os.path.join(args.data, f'efficientdet-d{eff_det_args.compound_coef}_{best_epoch}.pth')
        # outputs = test_eff_on_folder(eff_det_args, model_path, test_folder)


    elif args.detector.split('_')[0] == 'yolov5':
        # configuration files
        data_yaml_path = os.path.join(args.ROOT_DIR, 'counters', 'Detection_Regression', 'yolo_data_yaml',
                                args.data+'.yaml')
        cfg_path = os.path.join(args.ROOT_DIR, 'counters', 'Detection_Regression', 'config',
                                args.detector+'_'+args.data+'_cfg.yaml')
        detector_version = args.detector.split('_')[0]+args.detector.split('_')[1]
        path_to_pretrained_model = os.path.join(args.ROOT_DIR, 'Trained_Models', 'pretrained', 'yolo', detector_version + '.pt')

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
        yolo_det_args.save_period = -1 # -1 to disable

        # for visualization needs
        import wandb
        wandb.login()
        yolo_det_args.entity = None
        yolo_det_args.upload_dataset = False
        yolo_det_args.bbox_interval = -1
        yolo_det_args.artifact_alias = 'latest'

        from yolov5.train import main as yolo_train
        yolo_train(yolo_det_args)
        from yolov5 import detect
        yolo_infer_args = argparse.ArgumentParser()
        yolo_infer_args = yolo_infer_args.parse_args()
        yolo_infer_args.project = os.path.join(args.ROOT_DIR, 'Results', 'detect', 'yolo')
        os.makedirs(yolo_infer_args.project, exist_ok=True)
        yolo_infer_args.name = 'exp'
        yolo_infer_args.weights = os.path.join(args.save_trained_models, yolo_det_args.name, 'weights', 'best.pt')
        yolo_infer_args.source = os.path.join(args.data_path, 'test')
        yolo_infer_args.conf_thres = 0.25
        yolo_infer_args.iou_thres = 0.45 # NMS iou threshold
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
        # TODO - detect.run does not return anything, create a wrapper
        detect.run(**vars(yolo_infer_args))
    # Stage 1 is now complete, dump results into a json file and move on


    # Report results


if __name__ =='__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)
