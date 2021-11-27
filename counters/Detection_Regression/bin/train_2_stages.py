import os
import torch
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='Melons', help='choose a dataset')
    parser.add_argument('-si', '--split_images', type=bool, default=False, help='should we split the images into tiles')
    parser.add_argument('-nt', '--num_of_tiles', type=int, default=10, help='number of tiles')
    parser.add_argument('-p', '--padding', type=int, default=10, help='padding size in case of splitting')
    # --------------------------- Training Arguments -----------------------
    parser.add_argument('-det', '--detector', type=str, default='efficientDet_0', help='choose a detector efficientDet_i / yolov5 / fasterRCNN / RetinaNet'
                               'in case you choose efficientDet, please add "_i" where i is the compound coefficient')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='number of epochs for training')
    parser.add_argument('-exp', '--exp_number', type=int, default=0, help='number of current experiment')
    parser.add_argument('-c', '--criteria', type=str, default='mse', help='criteria can be mse / mae')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='set learning rate')
    parser.add_argument('-o', '--optim', type=str, default='sgd', help='choose optimizer adam / adamw / sgd')
    parser.add_argument('-ve', '--val_interval', type=int, default=2, help='run model validation every X epochs')
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

    if args.split_images:
        from utils.split_raw_images_anno_csv import split_to_tiles
        print('Notice - to split the images, bbox annotations are needed')
        split_to_tiles(args, args.num_of_tiles, args.padding)
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data + '_split', 'coco')
    else:
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data, 'coco')


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
    # setting up path to save pretrained models
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
        eff_det_args.save_interval = args.epochs
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
        outputs = test_eff_on_folder(args, model_path, test_folder)


    elif args.detector == 'yolov5':
        pass


    # parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    # parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    # parser.add_argument('--epochs', type=int, default=300)
    # parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    # parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # parser.add_argument('--rect', action='store_true', help='rectangular training')
    # parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    # parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    # parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    # parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    # parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    # parser.add_argument('--name', default='exp', help='save to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    # parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    # parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    # parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    #
    # # Weights & Biases arguments
    # parser.add_argument('--entity', default=None, help='W&B: Entity')
    # parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    # parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    # parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')


    # Stage 1 is now complete, dump results into a json file and move on


    # Report results


if __name__ =='__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)
