import os
import json
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='Grapes', help='choose a dataset')
    parser.add_argument('-lf', '--labels_format', type=str, default='coco', help='labels format can be csv / coco (.json)')
    parser.add_argument('-s', '--set_split', type=bool, default=True, help='state if the dataset is already split to train-val-test sets')
    args = parser.parse_args()
    return args

def load_dataset_labels_coco(args):
    print('loading the data')
    sets = ['train', 'val', 'test']
    sets_information = {
        'train': [],
        'val': [],
        'test': []
    }
    for current_set in sets:
        path_to_json = os.path.join(args.ROOT_DIR, 'instances_' + current_set + '.json')
        with open(path_to_json, 'r') as f:
            json_data = json.loads(f.read())
        labels_df = pd.json_normalize(json_data, record_path=['annotations'])
        images_df = pd.json_normalize(json_data, record_path=['images'])
        sets_information[current_set].append(labels_df)
        sets_information[current_set].append(images_df)
        print(f'loaded {current_set} dataset')
    return sets_information


def objects_numbers_stats(sets_information, dataset_stats):
    print('calculating some simple number statistics, all will appear in the final report')
    # number of objects
    dataset_stats['train_objects'] = sets_information['train'][0].shape[0]
    dataset_stats['val_objects'] = sets_information['val'][0].shape[0]
    dataset_stats['test_objects'] = sets_information['test'][0].shape[0]
    dataset_stats['total_objects'] = dataset_stats['train_objects'] + dataset_stats['val_objects'] + dataset_stats['test_objects']
    # number of images
    dataset_stats['train_images'] = sets_information['train'][1].shape[0]
    dataset_stats['val_images'] = sets_information['val'][1].shape[0]
    dataset_stats['test_images'] = sets_information['test'][1].shape[0]
    dataset_stats['total_images'] = dataset_stats['train_images'] + dataset_stats['val_images'] + dataset_stats['test_images']

    return dataset_stats

def object_per_image_stats(sets_information, dataset_stats):
    print('calculating number of objects in images stats')
    image_objects_pairs = []
    max_OPI = 0
    min_OPI = 100000
    for current_set in sets_information:
        print(f'analyzing {current_set} set')
        for i, row in sets_information[current_set][1].iterrows():
            image_id = row['id']
            labels_for_image = [x for j, x in sets_information[current_set][0].iterrows() if x['image_id'] == image_id]
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

def objects_images_sizes_stats(sets_information, dataset_stats):
    images_sizes = {
        'width': [],
        'height': []
    }
    objects_sizes = {
        'width': [],
        'height': []
    }
    for current_set in sets_information:
        set_images_sizes = {
            'width': [],
            'height': []
        }
        set_objects_sizes = {
            'width': [],
            'height': []
        }
        print(f'analyzing {current_set} set')

def main(args):
    # either coco or csv formats are load into a pandas data frame
    if args.labels_format == 'coco':
        args.ROOT_DIR = os.path.join(args.ROOT_DIR, 'coco', 'annotations')
        sets_information = load_dataset_labels_coco(args)
    dataset_stats = {}
    # how many objects (in general, and in each set)
    # how many images (in general, and in each set)
    dataset_stats = objects_numbers_stats(sets_information, dataset_stats)

    # max objects per images
    # min objects per images
    # mean objects per image
    dataset_stats = object_per_image_stats(sets_information, dataset_stats)

    # images sizes
    # average object size
    # object size to image size
    dataset_stats = objects_images_sizes_stats(sets_information, dataset_stats)

    # overlap between objects



if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = os.path.join('C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data', args.data)
    main(args)