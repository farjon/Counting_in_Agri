import os
import xml.etree.ElementTree as ET
import json
import argparse
from PIL import Image
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Annotations parser from VOC to COCO')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-r', '--ROOT_DIR', type=str, default="C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data", help='path to data root folder')
    parser.add_argument('-d', '--data', type=str, default='Pears', help='choose a dataset')
    args = parser.parse_args()
    return args

def main(args):
    data_dir = os.path.join(args.ROOT_DIR, args.data, 'voc')
    output_dir = os.path.join(args.ROOT_DIR, args.data, 'coco')
    annotations_output_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(annotations_output_dir, exist_ok=True)

    sets = ['train', 'val', 'test']
    for current_set in sets:
        images_set_txt = os.path.join(data_dir, 'ImageSets', 'Main', current_set + '.txt')
        set_images_to_take = open(images_set_txt, 'r').read().splitlines()

        set_images_output_dir = os.path.join(output_dir, current_set)
        os.makedirs(set_images_output_dir, exist_ok=True)
        coco_output_json_file = os.path.join(annotations_output_dir, 'instances_' + current_set + '.json')

        categories = [{'supercategory': None, 'id': 1, 'name': args.data}]
        images = []
        annotations = []
        image_id = 0
        annotations_id = 0

        for img_to_take in set_images_to_take:
            xml_path = os.path.join(data_dir, 'Annotations', img_to_take + '.xml')
            image_path = os.path.join(data_dir, 'JPEGImages', img_to_take + '.jpg')
            image_new_path = os.path.join(set_images_output_dir, img_to_take + '.jpg')
            shutil.copyfile(image_path, image_new_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            images.append({
                'file_name': root.find('filename').text, 'height': height, 'width': width,
                'id': image_id
            })

            for object in root.findall('object'):
                bndbox = object.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                o_width = xmax - xmin
                o_height = ymax - ymin
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                    image_id, 'bbox': [xmin, ymin, o_width, o_height],
                    'category_id': 1, 'id': annotations_id, 'ignore': 0}
                annotations.append(ann)
                annotations_id += 1
            image_id += 1
        json_dict = {
            'categories': categories,
            'images': images,
            'annotations': annotations
        }
        json_fp = open(coco_output_json_file, 'w')
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)

if __name__ == '__main__':
    args = parse_args()
    main(args)
