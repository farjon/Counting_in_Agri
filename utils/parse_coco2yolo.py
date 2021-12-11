import os
import json
import argparse

# this script assumes that the annotatoins are arranged in a coco style
# meaning - there are 4 folders:
#   1. train - contains the images for training
#   2. val - contains the images for evaluation
#   3. test - contains the images for testing
#   4. annotations - contains 3 '.json' files for each set

def parse_args():
    parser = argparse.ArgumentParser(description='Annotations parser from VOC to COCO')
    # --------------------------- Data Arguments ---------------------------
    # parser.add_argument('-r', '--ROOT_DIR', type=str, help='path to data root folder')
    parser.add_argument('-d', '--data', type=str, default='Banana', help='choose a dataset')

    args = parser.parse_args()
    return args


def main(args):
    path_to_coco = os.path.join(args.ROOT_DIR, args.data, 'coco')
    path_to_yolo = os.path.join(args.ROOT_DIR, args.data, 'yolo')
    path_to_yolo_images = os.path.join(path_to_yolo, 'images')
    path_to_yolo_labels = os.path.join(path_to_yolo, 'labels')
    os.makedirs(path_to_yolo_images, exist_ok=True)
    os.makedirs(path_to_yolo_labels, exist_ok=True)

    sets = ['train', 'val', 'test']
    for current_set in sets:
        coco_json_file = os.path.join(path_to_coco, 'annotations', 'instances_'+current_set+'.json')
        with open(coco_json_file, 'r') as annotations:
            coco_json = json.load(annotations)


        # notice - yolo expects to get the annotation as x_center, y_center, width, height


if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)


