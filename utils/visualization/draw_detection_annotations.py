import os
import json
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import copy
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def drawrect(drawcontext, box, outline=None, width=0):
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='Grapes', help='choose a dataset')
    parser.add_argument('-o', '--output_path', type=str, default='', help='path to save images, defualt is NONE')
    args = parser.parse_args()
    return args

def main(args):
    args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data, 'coco')
    args.output_path = os.path.join(args.ROOT_DIR, 'Data', args.data, 'images_with_annotations')
    os.makedirs(args.output_path, exist_ok=True)
    sets_to_vis = ['train', 'val', 'test']

    for current_set in sets_to_vis:
        args.set_output_path = os.path.join(args.output_path, current_set)
        os.makedirs(args.set_output_path, exist_ok=True)
        set_annotations_path = os.path.join(args.data_path, 'annotations', 'instances_' + current_set + '.json')
        with open(set_annotations_path) as f:
            json_decoded = json.load(f)

        for sample in json_decoded['images']:
            im_original = Image.open(os.path.join(args.data_path, current_set, sample['file_name']))
            im_drawing = copy.deepcopy(im_original)
            draw_mask = ImageDraw.Draw(im_drawing)
            annotations_for_image = [x for x in json_decoded['annotations'] if x['image_id'] == sample['id']]
            for anno in annotations_for_image:
                drawrect(draw_mask, anno['bbox'], outline="red", width=5)

            new_im = Image.new('RGB', im_original.size)
            new_im.paste(im_drawing, (0, 0))

            new_file_name = os.path.join(args.set_output_path, sample['file_name'])
            new_im.save(new_file_name)

if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)