import os
import torch
from glob import glob
import argparse

from EfficientDet_Pytorch.train import Params

from EfficientDet_Pytorch.backbone import EfficientDetBackbone
from EfficientDet_Pytorch.efficientdet.utils import BBoxTransform, ClipBoxes
from EfficientDet_Pytorch.utils.utils import preprocess, invert_affine, postprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--dataset', type=str, default='CherryTomato', help='choose a dataset')
    # --------------------------- Testing Arguments -----------------------
    parser.add_argument('-m', '--model', type=str, default='fasterRCNN', help='choose a detector fasterRCNN / RetinaNet')
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

def test_eff_on_folder(args, model_path, path_to_folder):
    torch.backends.cudnn.fastest = True
    torch.backends.cudnn.benchmark = True

    params = Params(f'{args.project}.yml')

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[args.compound_coef]

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    predictor = EfficientDetBackbone(compound_coef=args.compound_coef, num_classes=len(params.obj_list),
                                     ratios=params.anchor_ratios, scales=params.anchor_scales)
    predictor.load_state_dict(torch.load(model_path, map_location='cpu'))
    predictor.requires_grad_(False)
    predictor.eval()
    predictor.to(args.device)

    outputs = []
    with torch.no_grad():
        for imgae_path in glob(os.path.join(path_to_folder, '*.jpj')):

            # we are only using batch_size = 1 in our work
            ori_imgs, framed_imgs, framed_metas = preprocess(imgae_path, max_size=input_size)
            pred_on_imgs = torch.stack([torch.from_numpy(fi).to(args.device) for fi in framed_imgs], 0)
            pred_on_imgs = pred_on_imgs.to(torch.float32).permute(0, 3, 1, 2)

            features, regression, classification, anchors = predictor(pred_on_imgs)
            out = postprocess(pred_on_imgs,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        args.det_test_thresh, args.iou_threshold)
            out = invert_affine(framed_metas, out)
            # collect the results
            outputs.append({
                'image_name': imgae_path,
                'predictions': out
            })

        return outputs

def test_eff_on_image(args, model, images):

    with torch.no_grad():
        features, regression, classification, anchors = model(images)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(images,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          args.det_test_thresh, args.iou_threshold)

        return out






