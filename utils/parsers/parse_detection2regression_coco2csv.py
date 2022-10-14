import os
import json
import shutil
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-r', '--ROOT_DIR', type=str,
                        default="C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data",
                        help='path to data root folder')
    parser.add_argument('-d', '--data', type=str, default='Hens', help='choose a dataset')
    args = parser.parse_args()
    return args

def main(args):
    output_dir = os.path.join(args.ROOT_DIR, args.data, 'regression_csv')
    os.makedirs(output_dir, exist_ok=True)
    sets = ['train', 'val', 'test']
    for current_set in sets:
        data_dir = os.path.join(args.ROOT_DIR, args.data, 'coco', current_set)
        images_output_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_output_dir, exist_ok=True)
        input_file = os.path.join(args.ROOT_DIR, args.data, 'coco', 'annotations', 'instances_' + current_set + '.json')
        output_file = os.path.join(output_dir, current_set + '.csv')
        current_anno_file = json.load(open(input_file))
        csv_annotations = {
            'image_name': [],
            'GT_number': []
        }
        for image_desc in current_anno_file['images']:
            image_name = image_desc['file_name']
            image_input_path = os.path.join(data_dir, image_name)
            images_output_path = os.path.join(images_output_dir, image_name)
            shutil.copy(image_input_path, images_output_path)
            annotations_for_image = [x for x in current_anno_file['annotations'] if x['image_id'] == image_desc['id']]
            csv_annotations['image_name'].append(image_name)
            csv_annotations['GT_number'].append(len(annotations_for_image))

        pd.DataFrame(csv_annotations).to_csv(output_file, index=False)
if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data\\'
    main(args)
