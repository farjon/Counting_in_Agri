import os
import torch
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='Hens', help='choose a dataset')
    parser.add_argument('-si', '--split_images', type=bool, default=False, help='should we split the images into tiles')
    parser.add_argument('-nt', '--num_of_tiles', type=int, default=10, help='number of tiles')
    parser.add_argument('-p', '--padding', type=int, default=10, help='padding size in case of splitting')
    # --------------------------- Training Arguments -----------------------
    parser.add_argument('-det', '--detector', type=str, default='fasterRCNN', help='choose a detector efficientDet / yolov5 / fasterRCNN / RetinaNet')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='number of epochs for training')
    parser.add_argument('-exp', '--exp_number', type=int, default=0, help='number of current experiment')
    parser.add_argument('-c', '--criteria', type=str, default='mse', help='criteria can be mse / mae')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='set learning rate')
    parser.add_argument('-o', '--optim', type=str, default='sgd', help='choose optimizer adam / adamw / sgd')
    parser.add_argument('-ve', '--val_interval', type=int, default=2, help='run model validation every X epochs')
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
        from utils.split_raw_images import split_to_tiles
        print('Notice - to split the images, bbox annotations are needed')
        split_to_tiles(args, args.num_of_tiles, args.padding)
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data + '_split', 'coco')
    else:
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data, 'coco')

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
        register_coco_instances("test", {}, f'{args.data_path}/val/val.json', os.path.join(args.data_path, 'val'))

    # --------------------------- Start edit ---------------------------
    # setting up path to save pretrained models
    torch.hub.set_dir(os.path.join(args.ROOT_DIR, 'Trained_Models', 'pretrained'))
    # args.save_checkpoint_path = os.path.join(args.ROOT_DIR, 'Trained_Models', args.data, str(args.exp_number))
    # os.makedirs(args.save_checkpoint_path, exist_ok=True)

    # --------------------------- Stage 1 - Detection ---------------------------
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
    # Create model
    # if args.model_type == 'resnet50':
    #     from counters.Basic_Regression.models.ResNet50_reg import ResNet_50_regressor as Reg_Model
    # model = Reg_Model()
    #
    # # define loss function
    # if args.criteria == 'mse':
    #     loss_func = torch.nn.MSELoss(reduction='mean')
    # elif args.criteria == 'mae':
    #     loss_func = torch.nn.L1Loss(reduction='mean')
    #
    # if args.optim == 'adamw':
    #     optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    # elif args.optim == 'adam':
    #     optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # elif args.optim == 'sgd':
    #     optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True)
    #
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    #
    # # Train model
    # train_and_eval(args, train_dataset, val_dataset, model, loss_func, optimizer, scheduler)
    #
    # best_model = Reg_Model().load_state_dict(torch.load(os.path.join(
    #     args.save_checkpoint_path, f'best_{args.model_type}_model.pth'
    # )))
    # final_model = Reg_Model().load_state_dict(torch.load(os.path.join(
    #     args.save_checkpoint_path, f'final_{args.model_type}_model.pth'
    # )))
    #
    # # Test Models
    # final_scores, best_scores = test_models(args, test_dataset, models = [final_model, best_model])

    # Report results


if __name__ =='__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)
