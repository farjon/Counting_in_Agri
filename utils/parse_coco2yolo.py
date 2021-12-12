import os
import cv2
import json
import shutil
import argparse

# this script assumes that the annotatoins are arranged in a coco style
# meaning - there are 4 folders:
#   1. train - contains the images for training
#   2. val - contains the images for evaluation
#   3. test - contains the images for testing
#   4. annotations - contains 3 '.json' files for each set

def coco_label_to_yolo_style(coco_style, im_width, im_height):
    '''
    convert coco labels style to yolo style: bbox[x_min, y_min, w, h] -> x_center, y_center, w, h
    :param coco_style: a dictionary with x, y, w, h, normalized
    :return: yolo style labels in a list
    '''
    x_center_norm = (coco_style['bbox'][0] + int(coco_style['bbox'][2]/2))/im_width
    y_center_norm = (coco_style['bbox'][1] + int(coco_style['bbox'][3]/2))/im_height
    w_norm = coco_style['bbox'][2] / im_width
    h_norm = coco_style['bbox'][3] /im_height
    yolo_style = [x_center_norm, y_center_norm, w_norm, h_norm]
    return yolo_style


def parse_args():
    parser = argparse.ArgumentParser(description='Annotations parser from VOC to COCO')
    # --------------------------- Data Arguments ---------------------------
    # parser.add_argument('-r', '--ROOT_DIR', type=str, help='path to data root folder')
    parser.add_argument('-d', '--data', type=str, default='Banana', help='choose a dataset')

    args = parser.parse_args()
    return args


def main(args):
    # set up paths
    path_to_coco = os.path.join(args.ROOT_DIR, args.data, 'coco')
    path_to_yolo = os.path.join(args.ROOT_DIR, args.data, 'yolo')
    path_to_yolo_images = os.path.join(path_to_yolo, 'images')
    path_to_yolo_labels = os.path.join(path_to_yolo, 'labels')
    # os.makedirs(path_to_yolo_images, exist_ok=True)
    # os.makedirs(path_to_yolo_labels, exist_ok=True)

    sets = ['train', 'val', 'test']
    for current_set in sets:
        path_to_set_yolo = os.path.join(path_to_yolo, current_set)
        os.makedirs(path_to_set_yolo, exist_ok=True)
        set_label_path = os.path.join(path_to_yolo, current_set + '.txt')
        set_label_txt = open(set_label_path, 'w+')
        coco_json_file = os.path.join(path_to_coco, 'annotations', 'instances_'+current_set+'.json')
        with open(coco_json_file, 'r') as annotations:
            coco_json = json.load(annotations)

        # run over images, collect annotations
        for img_coco in coco_json['images']:
            img_name = img_coco['file_name']
            img_width = img_coco['width']
            img_height = img_coco['height']
            open(set_label_path, 'a').write(current_set + '\\' + img_name +'\n')
            path_to_im_src = os.path.join(path_to_coco, current_set, img_name)
            path_to_im_dest = os.path.join(path_to_set_yolo, img_name)
            shutil.copy(path_to_im_src, path_to_im_dest)
            img_label_path = os.path.join(path_to_set_yolo, img_name.split('.')[0] + '.txt')
            img_label_txt = open(img_label_path, 'w+')
            annotations_for_im_coco = [x for x in coco_json['annotations'] if x['image_id'] == img_coco['id']]
            for annot in annotations_for_im_coco:
                yolo_style_annot = coco_label_to_yolo_style(annot, img_width, img_height)
                yolo_style_annot.insert(0, annot['category_id']-1)
                open(img_label_path, 'a').write(' '.join([str(i) for i in yolo_style_annot]) + '\n')
            img_label_txt.close()
        set_label_txt.close()

if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data'
    main(args)


