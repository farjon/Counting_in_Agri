import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='A4', help='choose a dataset')
    parser.add_argument('-lf', '--labels_format', type=str, default='csv', help='labels format can be csv / coco (.json)')
    parser.add_argument('-s', '--set_split', type=bool, default=True, help='state if the dataset is already split to train-val-test sets')
    args = parser.parse_args()
    return args

def load_dataset_labels_csv(args):
    print('loading the data')
    sets = ['train', 'val', 'test']
    sets_information = {}
    for current_set in sets:
        path_to_csv = os.path.join(args.ROOT_DIR, current_set + '.csv')
        if not os.path.exists(path_to_csv):
            continue
        csv_data = pd.read_csv(path_to_csv)
        sets_information[current_set] = []
        sets_information[current_set].append(csv_data['image_name'].to_list())
        sets_information[current_set].append(csv_data['count'].to_list())
        print(f'loaded {current_set} dataset')
    return sets_information

def load_dataset_labels_coco(args):
    print('loading the data')
    sets = ['train', 'val', 'test']
    sets_information = {}
    for current_set in sets:
        path_to_json = os.path.join(args.ROOT_DIR, 'instances_' + current_set + '.json')
        if not os.path.exists(path_to_json):
            continue
        sets_information[current_set] = []
        with open(path_to_json, 'r') as f:
            json_data = json.loads(f.read())

        labels = json_data['annotations']
        images = json_data['images']
        sets_information[current_set].append(labels)
        sets_information[current_set].append(images)
        print(f'loaded {current_set} dataset')
    return sets_information


def objects_numbers_stats(sets_information, dataset_stats):
    print('calculating some simple number statistics, all will appear in the final report')
    # number of objects
    dataset_stats['total_objects'] = 0
    dataset_stats['total_images'] = 0
    for current_set in sets_information:
        dataset_stats[f'{current_set}_objects'] = len(sets_information[current_set][0])
        dataset_stats['total_objects'] += dataset_stats[f'{current_set}_objects']
        # number of images
        dataset_stats[f'{current_set}_images'] = len(sets_information[current_set][1])
        dataset_stats['total_images'] += dataset_stats[f'{current_set}_images']

    return dataset_stats

def object_per_image_stats_coco_style(sets_information, dataset_stats):
    print('calculating number of objects in images stats')
    image_objects_pairs = []
    max_OPI = 0
    min_OPI = 100000
    for current_set in sets_information:
        print(f'analyzing {current_set} set')
        for row in sets_information[current_set][1]:
            image_id = row['id']
            labels_for_image = [x for x in sets_information[current_set][0] if x['image_id'] == image_id]
            number_of_objects_for_image = len(labels_for_image)
            image_objects_pairs.append(number_of_objects_for_image)
            if number_of_objects_for_image > max_OPI:
                max_OPI = number_of_objects_for_image
            if number_of_objects_for_image < min_OPI:
                min_OPI = number_of_objects_for_image
    dataset_stats['min_object_per_image'] = min_OPI
    dataset_stats['max_object_per_image'] = max_OPI
    dataset_stats['mean_object_per_image'] = sum(image_objects_pairs) / len(image_objects_pairs)

    return dataset_stats


def object_per_image_stats_csv_style(sets_information, dataset_stats):
    print('calculating number of objects in images stats')
    image_objects_pairs = []
    max_OPI = 0
    min_OPI = 100000
    for current_set in sets_information:
        print(f'analyzing {current_set} set')
        max_OPI = max(max_OPI, max(sets_information[current_set][1]))
        min_OPI = min(min_OPI, min(sets_information[current_set][1]))
        image_objects_pairs.extend(sets_information[current_set][1])

    dataset_stats['min_object_per_image'] = min_OPI
    dataset_stats['max_object_per_image'] = max_OPI
    dataset_stats['mean_object_per_image'] = sum(image_objects_pairs) / len(image_objects_pairs)

    return dataset_stats

def objects_images_sizes_stats(sets_information, dataset_stats):
    images_sizes = {
        'area': [],
        'width': [],
        'height': []
    }
    objects_sizes_per_image = []
    objects_sizes = {
        'area': [],
        'width': [],
        'height': []
    }
    for current_set in sets_information:
        print(f'analyzing {current_set} set')
        for row in sets_information[current_set][1]:
            images_sizes['area'].append(row['width'] * row['height'])
            images_sizes['width'].append(row['width'])
            images_sizes['height'].append(row['height'])
            labels_for_image = [x for x in sets_information[current_set][0] if x['image_id'] == row['id']]
            objects_per_image_areas = [x['area'] for x in labels_for_image]
            objects_sizes['area'].extend(objects_per_image_areas)
            objects_sizes['width'].extend([x['bbox'][2] for x in labels_for_image])
            objects_sizes['height'].extend([x['bbox'][3] for x in labels_for_image])
            objects_sizes_per_image.append(objects_per_image_areas)
    # objects average area
    dataset_stats['objects_mean_area'] = sum(objects_sizes['area'])/len(objects_sizes['area'])
    # images average area
    dataset_stats['images_mean_area'] = sum(images_sizes['area'])/len(images_sizes['area'])
    # object area / image area
    image_object_ratio = []
    for i in range(len(images_sizes['area'])):
        for j in range(len(objects_sizes_per_image[i])):
            image_object_ratio.append(objects_sizes_per_image[i][j] / images_sizes['area'][i])
    # create and save an histogram of the ratios
    plt.hist(image_object_ratio, bins=100)
    plt.title('object area / image area')
    plt.savefig(os.path.join(args.ROOT_DIR, 'object_area_image_area.png'))
    plt.close()
    return dataset_stats

def main(args):
    # either coco or csv formats are load into a pandas data frame
    if args.labels_format == 'coco':
        args.ROOT_DIR = os.path.join(args.ROOT_DIR, 'Detection', 'coco', 'annotations')
        sets_information = load_dataset_labels_coco(args)
    elif args.labels_format == 'csv':
        if args.data in ['A1', 'A2', 'A3', 'A4']:
            args.ROOT_DIR = os.path.join(args.ROOT_DIR, 'Direct_Regression', args.data, 'annotations')
        else:
            args.ROOT_DIR = os.path.join(args.ROOT_DIR, 'Direct_Regression', 'annotations')
        sets_information = load_dataset_labels_csv(args)
    else:
        raise ("only coco format is available, please use the parsers in the 'utils' directory")
    dataset_stats = {}
    # how many objects (in general, and in each set)
    # how many images (in general, and in each set)
    dataset_stats = objects_numbers_stats(sets_information, dataset_stats)

    # max objects per images
    # min objects per images
    # mean objects per image
    if args.labels_format == 'coco':
        dataset_stats = object_per_image_stats_coco_style(sets_information, dataset_stats)
        # images sizes
        # average object size
        # object size to image size
        dataset_stats = objects_images_sizes_stats(sets_information, dataset_stats)
    elif args.labels_format == 'csv':
        dataset_stats = object_per_image_stats_csv_style(sets_information, dataset_stats)

    # TODO - calculate overlap between objects

    #print dataset stats
    print(dataset_stats)



if __name__ == '__main__':
    args = parse_args()
    if args.data in ['A1', 'A2', 'A3', 'A4']:
        args.ROOT_DIR = os.path.join('C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data', 'LCC')
    else:
        args.ROOT_DIR = os.path.join('C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data', args.data)
    main(args)