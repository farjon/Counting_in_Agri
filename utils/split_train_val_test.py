import os
import cv2
import shutil
import numpy as np
import json
import pandas as pd
def split_train_val_test_coco_style(images_names, annotations, train_val_test_split=[0.7, 0.2, 0.1], output_dir='output', categories=None):
    """
    Split the images, annotations to train, validation and test sets
    :param images_names: list of images information (coco style)
    :param annotations: list of annotations
    :param train_val_test_split: list of the split ratio of train, validation and test sets
    :param output_dir: output directory
    :param categories: categories
    :return: None
    """
    # make sure the split ratio is valid
    assert np.sum(train_val_test_split) == 1, 'The split ratio must sum to 1'
    # make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # make sure the output directory is empty
    if len(os.listdir(output_dir)) > 0:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    # create the directories for the train, validation and test sets
    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir)
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(val_dir)
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_dir)
    # create the directories for the images, annotations and predictions
    train_images_dir = os.path.join(train_dir, 'images')
    os.makedirs(train_images_dir)
    train_annotations_dir = os.path.join(train_dir, 'annotations')
    os.makedirs(train_annotations_dir)
    val_images_dir = os.path.join(val_dir, 'images')
    os.makedirs(val_images_dir)
    val_annotations_dir = os.path.join(val_dir, 'annotations')
    os.makedirs(val_annotations_dir)
    test_images_dir = os.path.join(test_dir, 'images')
    os.makedirs(test_images_dir)
    test_annotations_dir = os.path.join(test_dir, 'annotations')
    os.makedirs(test_annotations_dir)
    # split the images, annotations to train, validation and test sets
    train_images = images_names[:int(len(images_names)*train_val_test_split[0])]
    val_images = images_names[int(len(images_names)*train_val_test_split[0]):int(len(images_names)*(train_val_test_split[0]+train_val_test_split[1]))]
    test_images = images_names[int(len(images_names)*(train_val_test_split[0]+train_val_test_split[1])):]
    train_annotations = annotations[:int(len(annotations)*train_val_test_split[0])]
    val_annotations = annotations[int(len(annotations)*train_val_test_split[0]):int(len(annotations)*(train_val_test_split[0]+train_val_test_split[1]))]
    test_annotations = annotations[int(len(annotations)*(train_val_test_split[0]+train_val_test_split[1])):]
    # copy the images to the train, validation and test sets
    for image_name in train_images:
        shutil.copy(image_name, train_images_dir)
    for image_name in val_images:
        shutil.copy(image_name, val_images_dir)
    for image_name in test_images:
        shutil.copy(image_name, test_images_dir)

    # write annotations to disk
    train_annotations_file = os.path.join(train_annotations_dir, f'instances_train.json')
    train_json_dict = {'categories': categories, 'images': train_images, 'annotations': train_annotations}
    with open(train_annotations_file, "w") as outfile:
        json.dump(train_json_dict, outfile)

    val_annotations_file = os.path.join(val_annotations_dir, f'instances_val.json')  
    val_json_dict = {'categories': categories, 'images': val_images, 'annotations': val_annotations}
    with open(val_annotations_file, "w") as outfile:
        json.dump(val_json_dict, outfile)

    test_annotations_file = os.path.join(test_annotations_dir, f'instances_test.json')
    test_json_dict = {'categories': categories, 'images': test_images, 'annotations': test_annotations}
    with open(test_annotations_file, "w") as outfile:
        json.dump(test_json_dict, outfile)

def split_train_val_test_csv_style(root_dir, images_names, annotations_df, train_val_test_split=[0.7, 0.2, 0.1], output_dir='output'):
    """
    Split the images, annotations to train, validation and test sets
    :param images_names: list of images information (csv style)
    :param annotations: list of annotations
    :param train_val_test_split: list of the split ratio of train, validation and test sets
    :param output_dir: output directory
    :return: None
    """
    # make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # create the directories for the train, validation and test sets
    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    annotations_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    # randomize indices for split
    indices = np.arange(len(images_names))
    np.random.shuffle(indices)
    train_indices = indices[:int(len(images_names)*train_val_test_split[0])]
    val_indices = indices[int(len(images_names)*train_val_test_split[0]):int(len(images_names)*(train_val_test_split[0]+train_val_test_split[1]))]
    test_indices = indices[int(len(images_names)*(train_val_test_split[0]+train_val_test_split[1])):]

    train_images = list(np.array(images_names)[train_indices])
    val_images = list(np.array(images_names)[val_indices])
    test_images = list(np.array(images_names)[test_indices])

    train_annotations = annotations_df[annotations_df['image name'].isin(train_images)]
    val_annotations = annotations_df[annotations_df['image name'].isin(val_images)]
    test_annotations = annotations_df[annotations_df['image name'].isin(test_images)]
    # copy the images to the train, validation and test sets
    for image_name in train_images:
        shutil.copy(os.path.join(root_dir, image_name), os.path.join(train_dir, image_name))
    for image_name in val_images:
        shutil.copy(os.path.join(root_dir, image_name), os.path.join(val_dir, image_name))
    for image_name in test_images:
        shutil.copy(os.path.join(root_dir, image_name), os.path.join(test_dir, image_name))

    # write annotations to disk as csv format
    train_annotations.to_csv(os.path.join(annotations_dir, f'train.csv'), index=False)
    val_annotations.to_csv(os.path.join(annotations_dir, f'val.csv'), index=False)
    test_annotations.to_csv(os.path.join(annotations_dir, f'test.csv'), index=False)


def draw_annotations_on_images(images_names, annotations, output_dir='output'):
    """
    Draw annotations on images
    :param images_names: list of images names
    :param annotations: list of annotations
    :param output_dir: output directory
    :return: None
    """
    # make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # make sure the output directory is empty
    if len(os.listdir(output_dir)) > 0:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    # draw annotations on images
    for image_name, annotation in zip(images_names, annotations):
        image = cv2.imread(image_name)
        for box in annotation:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_name)), image)


if __name__ == '__main__':
    ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    split_train_val_test_coco_style()