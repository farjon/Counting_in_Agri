import os
import cv2
import copy
import json
import numpy as np
from utils.visualization.image import draw_rect_on_image, draw_dot_on_image, draw_text_on_image


def draw_annotations(args):
    args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data, 'Detection', 'coco')
    args.output_path = os.path.join(args.ROOT_DIR, 'Data', args.data, 'Detection', 'images_with_annotations')
    os.makedirs(args.output_path, exist_ok=True)
    sets_to_vis = ['train', 'val', 'test']

    for current_set in sets_to_vis:
        args.set_output_path = os.path.join(args.output_path, current_set)
        os.makedirs(args.set_output_path, exist_ok=True)
        set_annotations_path = os.path.join(args.data_path, 'annotations', 'instances_' + current_set + '.json')
        with open(set_annotations_path) as f:
            json_decoded = json.load(f)

        for sample in json_decoded['images']:
            im_original = cv2.imread(os.path.join(args.data_path, current_set, sample['file_name']))
            im_drawing = copy.deepcopy(im_original)
            annotations_for_image = [x for x in json_decoded['annotations'] if x['image_id'] == sample['id']]
            for anno in annotations_for_image:
                im_drawing = draw_rect_on_image(im_drawing, anno['bbox'], color="red", thickness=5)

            new_file_name = os.path.join(args.set_output_path, sample['file_name'])
            im_drawing.save(new_file_name)


def draw_detections_and_annotations(img, annotations, detections, class_name):
    """
    Draw annotations and detections on image
    :param img: image to draw on
    :param annotations: list of annotations
    :param detections:  list of detections
    :param class_name: name of classs
    :return: image with annotations and detections - blue for annotations, red for detections
    """
    img = copy.deepcopy(img)
    for anno in annotations:
        img = draw_rect_on_image(img, anno['bbox'], color=(0,0,255), thickness=2)
        img = draw_text_on_image(img, class_name, (50,50), color=(0,0,255), thickness=2)
    for det in detections:
        img = draw_rect_on_image(img, det.astype(np.int32), color=(255,0,0), thickness=2)
        img = draw_text_on_image(img, class_name, (50,50), color=(255,0,0), thickness=2)
    return img