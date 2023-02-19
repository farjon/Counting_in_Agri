import os
import cv2
import numpy as np
import pandas as pd
import json

def check_point_inside_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    if x1 < x < x2 and y1 < y < y2:
        return True
    return False

def main():
    ROOT = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data\\'
    data_name = 'CherryTomato'
    data_path = os.path.join(ROOT, data_name, 'Raw', 'CherryTomatoCounting')
    output_path = os.path.join(data_path, 'output')

    set_name = 'test'
    annotations_dir = os.path.join(data_path, 'annotations')
    images_dir = os.path.join(data_path, 'images')

    os.makedirs(os.path.join(output_path, set_name), exist_ok=True)
    collect_count_annotations = {
        'image_name': [],
        'count': []
    }
    image_number = 1
    for image_name in os.listdir(images_dir):
        img = cv2.imread(os.path.join(images_dir, image_name))
        with open(os.path.join(annotations_dir, f'{image_name}.txt')) as f:
            lines = f.readlines()
        for line in lines:
            img_count = int(line.split(' ')[0])
        cv2.imwrite(os.path.join(output_path, set_name, f'{image_number}.jpg'), img)
        collect_count_annotations['image_name'].append(f'{image_number}.jpg')
        collect_count_annotations['count'].append(img_count)
        image_number += 1
    pd.DataFrame(collect_count_annotations).to_csv(os.path.join(output_path, set_name, f'{data_name}_{set_name}.csv'), index=False)
if __name__ == '__main__':
    main()