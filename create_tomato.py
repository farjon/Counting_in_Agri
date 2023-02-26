import os
import shutil
import pandas as pd
ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'


image_path = os.path.join(ROOT_DIR, 'Data', 'CherryTomato', 'Direct_Regression', 'all_images')
annotations_path = os.path.join(ROOT_DIR, 'Data', 'CherryTomato', 'Direct_Regression', 'annotations', 'all_annotations.csv')
images_dir = os.listdir(image_path)
annotations = pd.read_csv(annotations_path)

output_dir = os.path.join(ROOT_DIR, 'Data', 'CherryTomato', 'Direct_Regression')
from utils.split_train_val_test import split_train_val_test_csv_style
split_train_val_test_csv_style(image_path, images_dir, annotations, output_dir=output_dir)


