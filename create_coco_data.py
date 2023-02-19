import os
import cv2
import numpy as np
import pandas as pd
import json

def main():
    ROOT = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data\\'
    data_name = 'wheat_spikes_and_spikelets'
    data_path = os.path.join(ROOT, data_name)
    output_path = os.path.join(data_path, 'output')
    set_name = 'test'
    raw_data_folder = os.path.join(data_path, 'from Faina', set_name)
    os.makedirs(os.path.join(output_path, set_name), exist_ok=True)

    df = pd.read_csv(os.path.join(raw_data_folder, f'{set_name}_MS5_detonly.kcsv'), header=None)

    # coco format
    categories = [{'supercategory': None, 'id': 1, 'name': 'wheat_spikes'}]
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    # use only bboxes first to create the cluster images
    for image_name in pd.unique(df[0]):
        image_info = df[df[0] == image_name]
        img = cv2.imread(os.path.join(raw_data_folder, image_name))
        images.append({
            'file_name': image_name, 'height': img.shape[0], 'width': img.shape[1],
            'id': image_id
        })

        for i, row in image_info.iterrows():
            if np.isnan(row[4]):
                # this is a center point
                continue
            # bbox is [x1, y1, x2, y2]
            x1 = int(row[2])
            y1 = int(row[3])
            x2 = int(row[4])
            y2 = int(row[5])

            annotations.append({'bbox': [x1, y1, x2 - x1, y2 - y1],
                                'category_id': 1,
                                'area': (x2 - x1) * (y2 - y1),
                                'id': annotation_id,
                                'image_id': image_id,
                                'iscrowd': 0,
                                'ignore': 0})
            annotation_id += 1
        image_id += 1

    # dump coco format annotations
    json_dict = {
        'categories': categories,
        'images': images,
        'annotations': annotations
    }
    with open(os.path.join(output_path, f'instances_{set_name}.json'), "w") as outfile:
        json.dump(json_dict, outfile)

if __name__ == '__main__':
    main()



