import os
import torch
from glob import glob

from EfficientDet_Pytorch.train import Params

from EfficientDet_Pytorch.backbone import EfficientDetBackbone
from EfficientDet_Pytorch.efficientdet.utils import BBoxTransform, ClipBoxes
from EfficientDet_Pytorch.utils.utils import preprocess, invert_affine, postprocess


def test_eff_on_folder(args, model_path, path_to_folder):
    torch.backends.cudnn.fastest = True
    torch.backends.cudnn.benchmark = True

    params = Params(f'{args.project}.yml')

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[args.compound_coef]

    predictor = EfficientDetBackbone(compound_coef=args.compound_coef, num_classes=len(params.obj_list),
                                     ratios=params.anchor_ratios, scales=params.anchor_scales)
    predictor.load_state_dict(torch.load(model_path, map_location='cpu'))
    predictor.requires_grad_(False)
    predictor.eval()
    predictor.to(args.device)

    outputs = []
    for imgae_path in glob(os.path.join(path_to_folder, '*.jpj')):

        # we are only using batch_size = 1 in our work
        ori_imgs, framed_imgs, framed_metas = preprocess(imgae_path, max_size=input_size)
        pred_on_imgs = torch.stack([torch.from_numpy(fi).to(args.device) for fi in framed_imgs], 0)
        pred_on_imgs = pred_on_imgs.to(torch.float32).permute(0, 3, 1, 2)
        out = test_eff_on_image(args, predictor, pred_on_imgs)
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






