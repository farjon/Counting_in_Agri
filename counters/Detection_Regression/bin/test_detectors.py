import os
import json
import glob
import torch

def test_detectron2(args):
    # Run inference over the test set towards the regression phase
    from counters.Detection_Regression.config.adjust_detectron_cfg import create_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.engine import DefaultPredictor
    import cv2
    from glob import glob

    cfg = create_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.save_predictions:
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'prediction_results'), exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = ("test",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.det_test_thresh  # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    # TODO - add an iou threshold
    # TODO - run on folder and collect results
    with open(f'{args.data_path}/test/instances_test.json', 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)

    images_counting_results = {
        'image_name': [],
        'gt_count': [],
        'pred_count': []
    }

    images_detection_results = {
        'image_name': [],
        'gt_bboxes': [],
        'pred_bboxes': []
    }

    for image in coco['images']:
        image_id = image['id']
        image_name = image['file_name']
        images_counting_results['image_name'].append(image_name)
        images_detection_results['image_name'].append(image_name)

        # store ground truth information - bboxes and number of objects
        gt_bboxes = [a for a in coco['annotations'] if a['image_id'] == image_id]
        images_counting_results['gt_count'].append(len(gt_bboxes))
        images_detection_results['gt_bboxes'].append(gt_bboxes)

        im = cv2.imread(os.path.join(args.data_path, 'test', image_name))
        predictions = predictor(im)

        # collect the results
        images_counting_results['pred_count'].append(len(predictions['instances']))
        images_detection_results['pred_bboxes'].append(predictions['instances'])

        # Save the images with the infered bounding boxes
        if args.save_predictions:
            cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, 'prediction_results', image_name), predictions.get_image()[:, :, ::-1])

    return images_counting_results, images_detection_results

def test_efficientDet(args, eff_det_args, best_epoch):
    from EfficientDet_Pytorch.train import Params

    from EfficientDet_Pytorch.backbone import EfficientDetBackbone
    from EfficientDet_Pytorch.efficientdet.utils import BBoxTransform, ClipBoxes
    from EfficientDet_Pytorch.utils.utils import preprocess, invert_affine, postprocess

    model_path = os.path.join(args.data, f'efficientdet-d{eff_det_args.compound_coef}_{best_epoch}.pth')

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

    with open(f'{args.data_path}/test/instances_test.json', 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)

    images_counting_results = {
        'image_name': [],
        'gt_count': [],
        'pred_count': []
    }

    images_detection_results = {
        'image_name': [],
        'gt_bboxes': [],
        'pred_bboxes': []
    }

    outputs = []
    with torch.no_grad():
        for image in coco['images']:
            image_id = image['id']
            image_name = image['file_name']
            images_counting_results['image_name'].append(image_name)
            images_detection_results['image_name'].append(image_name)

            # store ground truth information - bboxes and number of objects
            gt_bboxes = [a for a in coco['annotations'] if a['image_id'] == image_id]
            images_counting_results['gt_count'].append(len(gt_bboxes))
            images_detection_results['gt_bboxes'].append(gt_bboxes)

            image_path = os.path.join(args.data_path, image_name)
            # we are only using batch_size = 1 in our work
            ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_size)
            pred_on_imgs = torch.stack([torch.from_numpy(fi).to(args.device) for fi in framed_imgs], 0)
            pred_on_imgs = pred_on_imgs.to(torch.float32).permute(0, 3, 1, 2)

            features, regression, classification, anchors = predictor(pred_on_imgs)
            out = postprocess(pred_on_imgs,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              args.det_test_thresh, args.iou_threshold)
            predictions = invert_affine(framed_metas, out)

            # collect the results
            images_counting_results['pred_count'].append(len(predictions['rois']))
            images_detection_results['pred_bboxes'].append(predictions['rois'])

            # TODO - add the option to visualize results
        return images_counting_results, images_detection_results

def test_yolov5(args, yolo_det_args):
    from yolov5.detect import run as yolov5_detect
    from counters.Detection_Regression.utils.create_detector_args import create_yolov5_args
    yolo_infer_args = create_yolov5_args(args, yolo_det_args)
    os.makedirs(yolo_infer_args.project, exist_ok=True)
    # TODO - detect.run does not return anything, create a wrapper
    yolov5_detect(**vars(yolo_infer_args))
