import os
import cv2
import numpy as np
import pandas as pd

def check_point_inside_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    if x1 < x < x2 and y1 < y < y2:
        return True
    return False

def main():
    ROOT = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data\\'
    data_name = 'wheat_spikes_and_spikelets'
    data_path = os.path.join(ROOT, data_name)
    output_path = os.path.join(data_path, 'output')
    set_name = 'test'
    raw_data_folder = os.path.join(data_path, 'from Faina', set_name)
    os.makedirs(os.path.join(output_path, set_name), exist_ok=True)

    df = pd.read_csv(os.path.join(raw_data_folder, f'{set_name}_MS5.kcsv'), header=None)
    collect_point_annotations = {
        'image_name': [],
        'x': [],
        'y': []
    }
    collect_count_annotations = {
        'image_name': [],
        'count': []
    }
    # use only bboxes first to create the cluster images
    for image_name in pd.unique(df[0]):
        image_info = df[df[0] == image_name]
        collect_bboxes_for_image = {}
        img = cv2.imread(os.path.join(raw_data_folder, image_name))
        for i, row in image_info.iterrows():
            image_base_name = image_name.split('.')[0]
            if np.isnan(row[4]):
                # this is a center point
                continue
            # bbox is [x1, y1, x2, y2]
            x1 = int(row[2])
            y1 = int(row[3])
            x2 = int(row[4])
            y2 = int(row[5])

            collect_bboxes_for_image[f'{image_base_name}_{i}'] = [x1, y1, x2, y2]
            img_crop = img[y1:y2, x1:x2]
            print(f'{image_base_name}_{i}.jpg')
            cv2.imwrite(os.path.join(output_path, set_name, f'{image_base_name}_{i}.jpg'), img_crop)
        object_count = {}
        for i, row in image_info.iterrows():
            if not np.isnan(row[4]):
                # this is a bbox
                continue
            # center point is [x, y]
            x = int(row[2])
            y = int(row[3])
            for sub_image_name in collect_bboxes_for_image:
                bbox = collect_bboxes_for_image[sub_image_name]
                if check_point_inside_bbox([x, y], bbox):
                    relative_x, relative_y = x - bbox[0], y - bbox[1]
                    collect_point_annotations['image_name'].append(sub_image_name)
                    collect_point_annotations['x'].append(relative_x)
                    collect_point_annotations['y'].append(relative_y)

                    if sub_image_name not in object_count:
                        object_count[sub_image_name] = 1
                    else:
                        object_count[sub_image_name] += 1
                    break
        for sub_image_name in object_count:
            collect_count_annotations['image_name'].append(sub_image_name)
            collect_count_annotations['count'].append(object_count[sub_image_name])

    pd.DataFrame(collect_point_annotations).to_csv(os.path.join(output_path, set_name, f'{data_name}_{set_name}_location.csv'), index=False)
    pd.DataFrame(collect_count_annotations).to_csv(os.path.join(output_path, set_name, f'{data_name}_{set_name}.csv'), index=False)
if __name__ == '__main__':
    main()