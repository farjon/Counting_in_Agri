import os
import json
import cv2
import copy
import numpy as np
import argparse
from image import draw_rect_on_image, draw_dot_on_image, draw_text_on_image
def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='AppleFlowers_split', help='choose a dataset')
    parser.add_argument('-o', '--output_path', type=str, default='', help='path to save images, defualt is NONE')
    args = parser.parse_args()
    return args

def main(args):
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
                im_drawing = draw_rect_on_image(im_drawing, np.array(anno['bbox']).astype(np.int32), color=(0,255,0), thickness=5)

            new_file_name = os.path.join(args.set_output_path, sample['file_name'])
            cv2.imwrite(new_file_name, im_drawing)

if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)