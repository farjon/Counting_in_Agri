import os
import csv
import shutil
import argparse
import numpy as np
import pandas as pd

from counters.MSR_DRN_keras.preprocessing import create_csv_of_leaf_center

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_rate',type=float, default=0.5, help='choose portion of training set, the rest will be equally divided in to test and validation')
    return parser.parse_args()

def create_splited_dirs(data_dir, sub_datasets, sets_files, sets_names):
    new_data_folder = ''.join(sub_datasets)
    for current_set_files, current_set_name in zip(sets_files, sets_names):
        current_set_data_path = os.path.join(data_dir, new_data_folder, f'{new_data_folder}_{current_set_name}')
        os.makedirs(current_set_data_path, exist_ok=True)
        set_output_count_csv = {}
        set_output_location_csv = {}
        for i in range(len(current_set_files['rgb_images'])):
            subset_origin = current_set_files['rgb_images'][i].split('\\')[-2]
            plant_name = os.path.basename(current_set_files['rgb_images'][i]).split('_')[0]

            # copy rgb images
            rgb_image_path = current_set_files['rgb_images'][i]
            dst_file = os.path.join(current_set_data_path, subset_origin + '_' + plant_name + '_rgb.png')
            shutil.copyfile(rgb_image_path, dst_file)

            # copy centers images
            centers_image_path = current_set_files['centers_images'][i]
            dst_file = os.path.join(current_set_data_path, subset_origin + '_' + plant_name + '_centers.png')
            shutil.copyfile(centers_image_path, dst_file)

            # copy fg images
            mask_image_path = current_set_files['mask_images'][i]
            dst_file = os.path.join(current_set_data_path, subset_origin + '_' + plant_name + '_fg.png')
            shutil.copyfile(mask_image_path, dst_file)

        # Create a csv file of leaf counts for the relevant set
        new_counts_file_path = os.path.join(current_set_data_path, new_data_folder + '_' +current_set_name + '.csv')

        with open(new_counts_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for count, image_name in zip(current_set_files['leaf_counts'], current_set_files['rgb_images']):
                subset_origin = current_set_files['rgb_images'][i].split('\\')[-2]
                name = subset_origin + '_' + image_name.split('\\')[-1].split('_')[0] + "_rgb.png"
                writer.writerow([name, count])

        # Create a csv file of center points for the relevant set
        new_centers_file_path = os.path.join(current_set_data_path, new_data_folder + '_' + current_set_name+ '_leaf_location.csv')
        with open(new_centers_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(current_set_files['leaf_location_coord'])):
                line = current_set_files['leaf_location_coord'][i]
                name = line[0] + "_centers.png"
                points = line[1]
                for j in range(len(points)):
                    x = points[j][0]
                    y = points[j][1]
                    writer.writerow([name, x, y])

def create_set_files(leaf_counts, rgb_images_paths, masks_images_paths, centers_images_paths, leaf_location_coord, set_indices):
    set_files = {}
    set_files['leaf_counts'] = [leaf_counts[i] for i in set_indices]
    set_files['rgb_images'] = [rgb_images_paths[i] for i in set_indices]
    set_files['mask_images'] = [masks_images_paths[i] for i in set_indices]
    set_files['centers_images'] = [centers_images_paths[i] for i in set_indices]
    set_files['leaf_location_coord'] = [leaf_location_coord[i] for i in set_indices]
    return set_files

def main():
    '''
    For a given dataset, and chosen train split rate (default split_rate = 0.5, the rest is equally divided between validation and test),
    this script creats experiment folders for the train, val, and test sets.
    Each of this sub folder will have
    1. all relevant images
    2. {subset_data}_{set_name}_leaf_location.csv file containing the leaves locations in each image
    3. {subset_data}_{set_name}.csv containing the number of leaves in each image
    where subset_data is ['A1', 'A2', 'A3', 'A4']

    This script can also do the same for a new concatenation of datasets
    For example, if sub_datasets=["A1", "A2"], it will generate a new directory called "A1A2", that will have all the
    data from "A1" and "A2" , splitted to train, val and test sub directories.
    '''
    np.random.seed(10)
    # Enter the names of the datasets you want to use:
    # choose any of A1-A4, or any combination of those
    sub_datasets = ['A2', 'A3']

    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    args.data_path = os.path.join(args.ROOT_DIR, 'Data', 'LCC', 'training')

    train_split_rate = args.split_rate
    val_split_rate = (1-args.split_rate)/2

    # collect all data from provided sub_datasets
    # these files are provided by the Leaf Counting Challenge
    leaf_counts = []
    rgb_images_paths = []
    masks_images_paths = []
    centers_images_paths = []

    # the center coordinates of the leaves of each image at each dataset
    coord_dict = {}
    for data_name in sub_datasets:
        print(f'Working on dataset: {data_name}')
        current_data_path = os.path.join(args.data_path, data_name)

        # read the leaf counts csv file
        leaf_count_annotations = pd.read_csv(os.path.join(current_data_path, data_name + '.csv'))

        count = 0
        for _, row in leaf_count_annotations.iterrows():
            print(row)
            rgb_image_name = row[0]
            plant_name = rgb_image_name.split("_")[0]

            # collect number of leaves for image
            leaf_counts.append(int(row[1]))
            # collect the image itself - path only
            rgb_images_paths.append(os.path.join(current_data_path, rgb_image_name))
            # collect the mask image - path only (only foreground vs. background annotation)
            masks_images_paths.append(os.path.join(current_data_path, plant_name + '_fg.png'))
            # collect centers image - path only
            centers_images_paths.append(os.path.join(current_data_path, plant_name + '_centers.png'))

            count+=1

        # create the centers coordinates csv, if doesn't exist yet
        leaf_location_csv_path = os.path.join(current_data_path, data_name + '_leaf_location.csv')
        if not os.path.isfile(leaf_location_csv_path):
            create_csv_of_leaf_center.main(current_data_path, data_name)

        leaf_location_annotaitons = pd.read_csv(leaf_location_csv_path)
        print(f'Reading leaf coordinates: {data_name}')
        # create a dictionary for the center coordinates of each plant in each dataset
        for _, row in leaf_location_annotaitons.iterrows():
            plant_name = row[0].split("_")[0]
            x = int(row[1])
            y = int(row[2])
            key = f'{data_name}_{plant_name}'
            if not key in coord_dict.keys():
                coord_dict[key] = []
            coord_dict[key].append([x, y])

        print(f'Done with {data_name} set - found {count} images')

    # create a list where each item is a pair: plant key, list of center coordinates
    leaf_location_coord = []
    for key,value in coord_dict.items():
        leaf_location_coord.append([key,value])

    print("Done reading the datasets, start random split of the data... \n")

    # Create a random datasets split
    num_of_images = len(rgb_images_paths)
    N_train = round(train_split_rate * num_of_images)
    N_val = round(val_split_rate*num_of_images)

    Perm = np.random.permutation(num_of_images)   # Randomly permute a sequence
    train_indices = Perm[0:N_train]                       # indices for training
    val_indices = Perm[N_train:N_train+N_val]          # indices for validation
    test_indices = Perm[N_train+N_val:]                  # indices for testing

    Train_files = create_set_files(leaf_counts, rgb_images_paths, masks_images_paths, centers_images_paths, leaf_location_coord, train_indices)
    Val_files = create_set_files(leaf_counts, rgb_images_paths, masks_images_paths, centers_images_paths, leaf_location_coord, val_indices)
    Test_files = create_set_files(leaf_counts, rgb_images_paths, masks_images_paths, centers_images_paths, leaf_location_coord, test_indices)

    print("Done splitting the data..")
    print ("Total num of images: ", num_of_images)
    print ("Num of Train images: ", len(train_indices))
    print ("Num of Val images: ", len(val_indices))
    print ("Num of Test images: ", len(test_indices))


    create_splited_dirs(args.data_path, sub_datasets, [Train_files, Val_files, Test_files], ['Train', 'Val', 'Test'])

if __name__ == "__main__":
    main()