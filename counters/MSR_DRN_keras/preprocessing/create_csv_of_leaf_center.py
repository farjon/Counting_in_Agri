import cv2
import os
import numpy as np
import pandas as pd
'''
This script generates a csv file that contains the coordinates of the leaves center points, as provided by 
the XXX_centers.gpg images within a given dataset directory.

Thw main function gets as input:
Ai_data_path - the path to the relevant dataset directory
dataset_name -  the specific dataset name

'''


def create_data_to_write(data_path, csv_file_path, dataset_name):
    csv_file_pd = pd.read_csv(csv_file_path)
    new_csv_file_data = {
        'names': [],
        'Xs': [],
        'Ys': []
    }
    for _, centers_image in csv_file_pd.iterrows():
        center_image_name = f'{centers_image[0].split("_")[0]}_centers.png'

        centers_path = os.path.join(data_path, 'all images', dataset_name + '_' + center_image_name)
        # read the mask in gray scale
        centers_image = cv2.imread(centers_path, 0)

        ######################################################################
        # fixing A4 data bug
        if dataset_name!="A4":
            Ys, Xs = np.nonzero(centers_image)
        else:
            Xs, Ys  = np.nonzero(centers_image)
        ######################################################################

        for i in range(len(Xs)):
            new_csv_file_data['names'].append(center_image_name)
            new_csv_file_data['Xs'].append(Xs[i])
            new_csv_file_data['Ys'].append(Ys[i])

    return pd.DataFrame.from_dict(new_csv_file_data)

def main(data_path, dataset_name):

    print(f'creating leaf center csv on dataset {dataset_name}')

    csv_file_path = os.path.join(data_path, dataset_name + '.csv')
    centers_file_path = os.path.join(data_path, dataset_name + '_leaf_location1.csv')

    new_csv_file_data = create_data_to_write(data_path, csv_file_path, dataset_name)

    # write the data
    new_csv_file_data.to_csv(centers_file_path, index=0)

if __name__ == "__main__":

    dataset_name = 'A4'
    ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    data_path = os.path.join(ROOT_DIR, 'Data', 'LCC', 'training', dataset_name)

    main(data_path, dataset_name)