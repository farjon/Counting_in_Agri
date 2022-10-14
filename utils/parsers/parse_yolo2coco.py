import os
import sys
import json
import shutil
import argparse

# this script assumes that the annotations are arranged in a coco style
# meaning - there are 4 folders:
#   1. train - contains the images for training
#   2. val - contains the images for evaluation
#   3. test - contains the images for testing
#   4. annotations - contains 3 '.json' files for each set


def yolo_label_to_coco_style(yolo_style, im_width, im_height):
    '''
    convert yolo labels style to coco style: bbox[x_center_norm, x_center_norm, w_norm, h_norm] -> x, y, w, h
    :param yolo_style: a dictionary with x, y, w, h, normalized
    :return: coco style labels in a list
    '''
    w = int(yolo_style[2] * im_width)
    h = int(yolo_style[3] * im_height)
    x = int(yolo_style[0] * im_width - w/2)
    y =  int(yolo_style[1] * im_height - h/2)
    coco_style = [x, y, w, h]
    return coco_style

def parse_args():
    parser = argparse.ArgumentParser(description='Annotations parser from VOC to COCO')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-r', '--ROOT_DIR', type=str, default="C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data", help='path to data root folder')
    parser.add_argument('-d', '--data', type=str, default='Hens', help='choose a dataset')
    args = parser.parse_args()
    return args


def main(args):
    # set up paths
    path_to_coco = os.path.join(args.ROOT_DIR, args.data, 'coco')
    if not os.path.isdir(path_to_coco):
        print(f'no coco annotation found for dataset {args.data}')
        sys.exit(0)
    path_to_yolo = os.path.join(args.ROOT_DIR, args.data, 'yolo')
    path_to_yolo_images = os.path.join(path_to_yolo, 'images')
    path_to_yolo_labels = os.path.join(path_to_yolo, 'labels')
    os.makedirs(path_to_yolo_images, exist_ok=True)
    os.makedirs(path_to_yolo_labels, exist_ok=True)
    #TODO - read yolo and write to coco!
    sets = ['train', 'val', 'test']
    for current_set in sets:
        if not os.path.isdir(os.path.join(path_to_coco, current_set)):
            print(f'there is no {current_set} set in the coco style folder')
            continue
        print(f'creating {current_set} set in yolo style')
        set_images_path = os.path.join(path_to_yolo_images, current_set)
        os.makedirs(set_images_path, exist_ok=True)
        set_labels_path = os.path.join(path_to_yolo_labels, current_set)
        os.makedirs(set_labels_path, exist_ok=True)
        coco_json_file = os.path.join(path_to_coco, 'annotations', 'instances_'+current_set+'.json')
        with open(coco_json_file, 'r') as annotations:
            coco_json = json.load(annotations)

        # run over images, collect annotations
        for img_coco in coco_json['images']:
            img_name = img_coco['file_name']
            img_width = img_coco['width']
            img_height = img_coco['height']
            img_label_path = os.path.join(set_labels_path, img_name.split('.')[0] + '.txt')
            img_label_txt = open(img_label_path,  'w+')
            # moving the image into yolo
            path_to_im_src = os.path.join(path_to_coco, current_set, img_name)
            path_to_im_dest = os.path.join(set_images_path, img_name)
            shutil.copy(path_to_im_src, path_to_im_dest)
            # creating label file
            annotations_for_im_coco = [x for x in coco_json['annotations'] if x['image_id'] == img_coco['id']]
            for annot in annotations_for_im_coco:
                yolo_style_annot = coco_label_to_yolo_style(annot, img_width, img_height)
                yolo_style_annot.insert(0, annot['category_id']-1)
                open(img_label_path, 'a').write(' '.join([str(i) for i in yolo_style_annot]) + '\n')
            img_label_txt.close()
    print('finished creating yolo style data')

if __name__ == '__main__':
    args = parse_args()
    main(args)


