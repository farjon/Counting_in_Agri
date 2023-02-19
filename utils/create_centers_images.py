import os
import cv2
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Create center images for the dataset')
    parser.add_argument('--path_to_data', default='Grapes',  help='Path to the dataset', required=True)
    parser.add_argument('--dataset_name', default='',   help='Name of the dataset', required=True)
    parser.add_argument('--image_format', default='jpg', help='Image format', required=True)
    args = parser.parse_args()
    return args


def create_center_images(args):

    sets = ['train', 'val', 'test']
    for current_set in sets:
        output_path = os.path.join(args.path_to_data, current_set, 'centers')
        # Create a folder for the center images
        os.makedirs(output_path, exist_ok=True)

        # Read the csv file
        df = pd.read_csv(os.path.join(args.path_to_data, current_set, f'{args.dataset_name}_{current_set}_location.csv'))

        for image_name in pd.unique(df.iloc[:,0]):
            print(f'Processing image:{image_name}')
            image_df = df[df.iloc[:,0] == image_name]
            image = cv2.imread(os.path.join(args.path_to_data, current_set, 'RGB', f'{image_name}.{args.image_format}'))
            empty_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            for index, row in image_df.iterrows():
                x = row[1]
                y = row[2]
                empty_image[y, x] = 255
            cv2.imwrite(os.path.join(output_path, f'{image_name}_centers.{args.image_format}'), empty_image)


if __name__ == '__main__':
    ROOT_DIR = os.path.join(os.path.abspath("../"), 'Data')
    args = parse_args()
    args.path_to_data = os.path.join(ROOT_DIR, args.dataset_name, 'MSR_DRN')
    create_center_images(args)