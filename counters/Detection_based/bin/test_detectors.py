import os
import json
import cv2
import torch
import shutil
import numpy as np
from utils.visualization import drawing_detections
def test_detectron2(args):
    # Run inference over the test set towards the regression phase
    from counters.Detection_based.config.adjust_detectron_cfg import create_cfg
    from detectron2.engine import DefaultPredictor
    import cv2

    cfg = create_cfg(args)
    cfg.OUTPUT_DIR = os.path.join(args.save_trained_models, args.detector + '_' + str(args.exp_number))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    if args.visualize:
        path_to_save_visualization = os.path.join(args.save_counting_results, f'{args.detector}_{str(args.exp_number)}', 'visualize')
        os.makedirs(path_to_save_visualization, exist_ok=True)

    if args.save_predictions:
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'prediction_results'), exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = (args.test_set,)
    predictor = DefaultPredictor(cfg)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.det_test_thresh
    with open(f'{args.data_path}/{args.test_set}/instances_{args.test_set}.json', 'rt', encoding='UTF-8') as annotations:
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
    if 'split' in args.data:
        images_detection_results['pred_scores'] = []

    for image in coco['images']:
        image_id = image['id']
        image_name = image['file_name']
        images_counting_results['image_name'].append(image_name)
        images_detection_results['image_name'].append(image_name)

        # store ground truth information - bboxes and number of objects
        gt_bboxes = [a for a in coco['annotations'] if a['image_id'] == image_id]
        images_counting_results['gt_count'].append(len(gt_bboxes))
        images_detection_results['gt_bboxes'].append(gt_bboxes)

        im = cv2.imread(os.path.join(args.data_path, args.test_set, image_name))
        predictions = predictor(im)
        
        predictions = predictions['instances'].to('cpu')
        predictions = predictions[predictions.scores > args.det_test_thresh]

        # collect the results
        images_counting_results['pred_count'].append(len(predictions.get('scores')))
        images_detection_results['pred_bboxes'].append(predictions.get('pred_boxes'))
        # towards combining the detections to the full scale image
        if "split" in args.data:
            images_detection_results['pred_scores'].append(predictions.get('scores'))
        if args.visualize:
            detections = predictions.get('pred_boxes').tensor.numpy()
            detections = detections.astype(int)
            # convert xyxy to xywh
            detections[:, 2] = detections[:, 2] - detections[:, 0]
            detections[:, 3] = detections[:, 3] - detections[:, 1]
            img_to_draw = cv2.imread(os.path.join(args.data_path, 'test', image_name))
            img = drawing_detections.draw_detections_and_annotations(img_to_draw, gt_bboxes,
                                                                     detections,
                                                                     args.data)
            cv2.imwrite(os.path.join(path_to_save_visualization, image_name), img)

    return images_counting_results, images_detection_results

def test_yolov5(args):
    from yolov5.detect import run as yolov5_detect
    from counters.Detection_based.utils.create_detector_args import create_yolov5_infer_args
    yolo_infer_args = create_yolov5_infer_args(args)

    if os.path.exists(os.path.join(yolo_infer_args.project, yolo_infer_args.name)):
        shutil.rmtree(os.path.join(yolo_infer_args.project, yolo_infer_args.name))
    yolov5_detect(**vars(yolo_infer_args))

    if args.visualize:
        path_to_save_visualization = os.path.join(yolo_infer_args.project, yolo_infer_args.name, 'visualize')
        os.makedirs(path_to_save_visualization, exist_ok=True)

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
    labels_dir = os.path.join(args.data_path, 'labels', args.test_set)
    predictions_dir = os.path.join(yolo_infer_args.project, yolo_infer_args.name, 'labels')

    # find image format
    for image_name in os.listdir(os.path.join(args.data_path, 'images', 'test')):
        if image_name.endswith('.jpg'):
            image_format = '.jpg'
            break
        elif image_name.endswith('.png'):
            image_format = '.png'
            break
        elif image_name.endswith('.jpeg'):
            image_format = '.jpeg'
            break
        else:
            raise ValueError('Image format is not supported')

    for image_res_name in os.listdir(labels_dir):
        image_name = image_res_name.replace('.txt', image_format)
        # collect image names for both counting and detection
        images_counting_results['image_name'].append(image_res_name)
        images_detection_results['image_name'].append(image_res_name)

        # read annotation text file
        with open(os.path.join(labels_dir, image_res_name), 'rt') as f:
            annotations = f.readlines()
        collected_annotations = []
        for ann in annotations:
            ann = ann.split(' ')
            ann = [float(a) for a in ann]
            ann[0] = int(ann[0])
            to_collect = {'bbox': ann[1:5], 'category_id': ann[0]}
            collected_annotations.append(to_collect)
        images_detection_results['gt_bboxes'].append(collected_annotations)
        images_counting_results['gt_count'].append(len(annotations))

        # read prediction text file
        with open(os.path.join(predictions_dir, image_res_name), 'rt') as f:
            predictions = f.readlines()
        collected_predictions = []
        for pred in predictions:
            pred = pred.split(' ')
            pred = [float(p) for p in pred]
            pred[0] = int(pred[0])
            collected_predictions.append(pred)
        images_detection_results['pred_bboxes'].append(collected_predictions)
        images_counting_results['pred_count'].append(len(predictions))

        # visualize results
        if args.visualize:
            from utils.parsers.parse_yolo2coco import yolo_label_to_coco_style
            img_to_draw = cv2.imread(os.path.join(args.data_path, 'images', 'test', image_name))
            annotations_to_draw = []
            for a in collected_annotations:
                annotations_to_draw.append({'bbox': yolo_label_to_coco_style(a['bbox'], img_to_draw.shape[1], img_to_draw.shape[0])})
            predictions_to_draw = [yolo_label_to_coco_style(p[1:], img_to_draw.shape[1], img_to_draw.shape[0]) for p in collected_predictions]
            img = drawing_detections.draw_detections_and_annotations(img_to_draw, annotations_to_draw, np.array(predictions_to_draw), args.data)
            cv2.imwrite(os.path.join(path_to_save_visualization, image_name), img)
    return images_counting_results, images_detection_results, yolo_infer_args


def test_combined_tiles(args, images_detection_results):
    from utils.tile_images.combind_split_to_raw import combine_predictions_to_full_scale
    #TODO - this function is currently not working for yolo format
    # from utils.parsers.parse_yolo2coco import yolo_label_to_coco_style
    original_data_name = args.data.split('_')[0]
    original_images_dir = os.path.join(args.ROOT_DIR, 'Data', original_data_name, 'Detection', args.labels_format, args.test_set)

    if args.visualize:
        from utils.visualization.drawing_detections import draw_detections
        path_to_save_visualization = os.path.join(args.save_counting_results, f'{args.detector}_{str(args.exp_number)}', 'visualize_full_scale')
        os.makedirs(path_to_save_visualization, exist_ok=True)

    images_files = os.listdir(original_images_dir)
    # read annotations
    with open(f'{original_images_dir}/instances_{args.test_set}.json', 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)

    images_counting_results = {
        'image_name': [],
        'gt_count': [],
        'pred_count': []
    }

    for img_name in images_files:
        if not img_name.split('.')[1] in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:
            continue
        # collect image name
        images_counting_results['image_name'].append(img_name)
        # collect gt count
        image_id = [a['id'] for a in coco['images'] if a['file_name'] == img_name][0]
        gt_bboxes = [a for a in coco['annotations'] if a['image_id'] == image_id]
        images_counting_results['gt_count'].append(len(gt_bboxes))

        image_full_path = os.path.join(original_images_dir, img_name)
        img = cv2.imread(image_full_path)
        img_h, img_w, _ = img.shape
        image_predictions = {}
        for n, p, s in zip(images_detection_results['image_name'], images_detection_results['pred_bboxes'], images_detection_results['pred_scores']):
            if n.split('_')[0] == img_name.split('.')[0]:
                # add tile detections
                detections = p.tensor.numpy().astype(int)
                # convert xyxy to xywh
                detections[:, 2] = detections[:, 2] - detections[:, 0]
                detections[:, 3] = detections[:, 3] - detections[:, 1]
                image_predictions[n] = {'det': detections, 'scores': s.numpy().astype(float)}
        combined_preds = combine_predictions_to_full_scale(img_name, image_predictions, (img_w, img_h), args.num_of_tiles)
        from torchvision.ops import nms
        combined_preds_nms = nms(torch.tensor(np.array(combined_preds)[:, 1:]), torch.tensor(np.array(combined_preds)[:, 0]), 0.5)
        images_counting_results['pred_count'].append(combined_preds_nms.shape[0])
        if args.visualize:
            img = draw_detections(img, combined_preds, args.data)
            cv2.imwrite(os.path.join(path_to_save_visualization, img_name), img)
    return images_counting_results



