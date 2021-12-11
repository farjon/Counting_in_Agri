import os
import xml.etree.ElementTree as ET
import json
from glob import glob
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Annotations parser from VOC to COCO')
    # --------------------------- Data Arguments ---------------------------
    # parser.add_argument('-r', '--ROOT_DIR', type=str, help='path to data root folder')
    parser.add_argument('-d', '--data', type=str, default='Hens', help='choose a dataset')

    args = parser.parse_args()
    return args

def main(args):
    sets = ['train', 'val']
    for current_set in sets:
        data_dir = os.path.join(args.ROOT_DIR, args.data, 'voc', current_set)
        output_dir = os.path.join(args.ROOT_DIR, args.data, 'coco', current_set)
        os.makedirs(output_dir, exist_ok=True)

        json_file = os.path.join(output_dir, current_set +'.json')

        categories = [{'supercategory': None, 'id': 1, 'name': args.data}]
        images = []
        annotations = []
        image_id = 0
        annotations_id = 0
        for xml_file in glob(os.path.join(data_dir, '*.xml')):
            image_path = xml_file[:-4] + '.JPG'
            image_name = image_path.split('\\')[-1]
            image_new_path = os.path.join(output_dir, image_name)
            im = Image.open(image_path)
            im.save(image_new_path)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            images.append({
                'file_name': image_name, 'height': height, 'width': width,
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
        json_fp = open(json_file, 'w')
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)

if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data\\'
    main(args)
