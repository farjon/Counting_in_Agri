import csv
import random
import sys
import os
import argparse
import numpy as np
import pandas as pd

import counters.MSR_DRN_keras.bin.train
import evaluate_LCC


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    return parser.parse_args(args)


def get_centers_data(leaf_location_csvPath):

    coord_dict = {}
    # read the leaf_location csv file
    with open(leaf_location_csvPath) as csvfile_2:
        readCSV_2 = csv.reader(csvfile_2)
        # create a dictionary for the center coordinates of each plant in each dataset
        for row in readCSV_2:
            plant_name = row[0].split("_")[0]
            x = int(row[1])
            y = int(row[2])
            #key = data + "_" + plant_name
            key = plant_name
            if len(coord_dict) == 0:
                coord_dict[key] = []
                coord_dict[key].append([x, y])
            else:
                if key in coord_dict.keys():
                    coord_dict[key].append([x, y])
                else:
                    coord_dict[key] = []
                    coord_dict[key].append([x, y])

    return coord_dict

def read_leaf_count(dataset_path, dataset):
    leaf_count = []

    # read the leaf counts csv file
    csvPath = os.path.join(dataset_path, dataset + ".csv")
    with open(csvPath) as csvfile:
        readCSV = csv.reader(csvfile)
        print("Working on spliting dataset: ", dataset, "\n")
        count = 0
        for row in readCSV:
            print(row)
            rgbImage_name = row[0]
            if dataset == 'BL':
                plant_name = rgbImage_name.split(".")[0]
            else:
                plant_name = rgbImage_name.split("_")[0]

            current_leaf_count = {}
            current_leaf_count[plant_name] = int(row[1])
            current_leaf_count = [current_leaf_count]
            leaf_count.append(current_leaf_count)
            count += 1
        print()

    print("Done, ", dataset, "set - has", count, "images \n")
    return leaf_count

def read_lead_location(dataset_path, dataset_name):

    leaf_location_csvPath = os.path.join(dataset_path, dataset_name + "_leaf_location.csv")
    # create a list where each item is a pair: plant key, list of center coordinates
    # the center coordinates of the leaves of each image at each dataset
    coord_dict = get_centers_data(leaf_location_csvPath)

    leaf_location_coord = []
    for key, value in coord_dict.items():
        leaf_location_coord.append([key, value])

    return leaf_location_coord


def data_split(dataset_path, dataset_name, test_set_size):
    # read annotations files
    leaf_count = read_leaf_count(dataset_path, dataset_name)
    leaf_location_coord = read_lead_location(dataset_path, dataset_name)

    print("Done reading the datasets, start random split of the data... \n")

    randomized_plants = np.random.permutation(len(leaf_count))
    test_indices = randomized_plants[:test_set_size]
    train_indices = randomized_plants[test_set_size:]

    train_set_leaf_count = [leaf_count[i] for i in train_indices]
    test_set_leaf_count = [leaf_count[i] for i in test_indices]

    train_set_leaf_centers = [leaf_location_coord[i] for i in train_indices]
    test_set_leaf_centers = [leaf_location_coord[i] for i in test_indices]

    return [train_set_leaf_count, train_set_leaf_centers], [test_set_leaf_count, test_set_leaf_centers]


def create_csv_file(csv_leaf_number_file, csv_leaf_location_file, leaf_counts, leaf_location_coord, ds):
    '''
    input: indices of images
    output: generates csv file of a subset from the given file, based on the required indices
    '''
    # Create a csv file of leaf counts for the relevant set

    new_counts_file_path = csv_leaf_number_file
    with open(new_counts_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(leaf_counts)):
            line = leaf_counts[i][0]
            keys = line.keys()
            for key in keys:
                count = line[key]
                if ds == 'BL':
                    name = key + '.jpg'
                else:
                    name = key + "_rgb.png"
                writer.writerow([name, count])

    # Create a csv file of center points for the relevant set
    new_centers_file_path = csv_leaf_location_file
    with open(new_centers_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(leaf_location_coord)):
            line = leaf_location_coord[i]
            if ds == 'BL':
                name = line[0]
            else:
                name = line[0] + "_centers.png"
            points = line[1]
            for j in range(len(points)):
                x = points[j][0]
                y = points[j][1]
                writer.writerow([name, x, y])




def split_to_train_val_sets(train_set, sets_sizes, val_rate):

    train_sets = []
    val_sets = []

    for size in sets_sizes:
        train_data_leaf_count = train_set[0][:size]
        train_data_leaf_centers =  train_set[1][:size]

        train_set_size = int(len(train_data_leaf_count)*(1-val_rate))

        random_indices = np.random.permutation(len(train_data_leaf_count))
        train_indices = random_indices[:train_set_size]
        val_indices = random_indices[train_set_size:]

        train_set_leaf_count = [train_data_leaf_count[i] for i in train_indices]
        train_set_leaf_centers = [train_data_leaf_centers[i] for i in train_indices]
        train_sets.append([train_set_leaf_count, train_set_leaf_centers])

        val_set_leaf_count = [train_data_leaf_count[i] for i in val_indices]
        val_set_leaf_centers = [train_data_leaf_centers[i] for i in val_indices]
        val_sets.append([val_set_leaf_count, val_set_leaf_centers])

    return train_sets, val_sets

def create_csv_file_for_experiment(data_path, ds, exp_name, dataset, dataset_name):
    '''

    :param data_path:
    :param ds:
    :param exp_name:
    :param dataset:
    :return:
    '''
    csv_file_names = {}
    csv_file_start = ds + '_' + exp_name

    csv_file_names[dataset_name + '_count_file'] = os.path.join(data_path, csv_file_start + '_' + dataset_name + '.csv')
    csv_file_names[dataset_name + '_centers_files'] = os.path.join(data_path, csv_file_start + '_' + dataset_name + '_leaf_location.csv')

    # Remove files from prev runs if mistakenly exist
    for file in csv_file_names:
        if os.path.isfile(csv_file_names[file]):
            os.remove(csv_file_names[file])

    create_csv_file(csv_file_names[dataset_name + '_count_file'],
                        csv_file_names[dataset_name + '_centers_files'],
                        dataset[0],
                        dataset[1],
                        ds)

    return csv_file_names

def main(args=None):

    random.seed(130)
    np.random.seed(40)
    args.pipe = 'keyPfinder'

    args.random_transform = True

    # keyPfinder options:
        # detection_option_20

    args.exp_name = 'detection_option_20_dataset_size'

    args.lr = 1e-5
    args.reduce_lr = True
    args.reduceLR_patience = 5
    args.reduceLR_factor = 0.05

    args.early_stopping_indicator = "AbsCountDiff"
    args.early_stopping_patience = 50

    args.step_multi = 5

    args.multi_gpu = False
    args.multi_gpu_force = False

    # key point detection options:
    # 10 - best option, as in the paper
    # 20 - reducing size GT Gaussian maps for the sub-model
    args.option = 20
    args.calc_det_performance = False


    args.save_res_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results", 'results_' + args.pipe + '_exp_'+ args.exp_name+ '_'+str(args.exp_num) + ".csv")

    images_num = {}
    images_num['A4'] = 624
    images_num['BL'] = 1016

    chosen_datasets = ['BL']
    test_set_size = 150
    dataset_sizes = [50, 100, 250, 450, 700, images_num['BL'] - test_set_size]
    validation_rate = 0.2

    total_num_of_images = 0

    # agreement_per_ds = {}
    # total_mean_agreement = 0

    for ds in chosen_datasets:

        total_num_of_images += images_num[ds]

        if ds == 'A1' or ds == 'A2' or ds == 'A3' or ds == 'A4':
            args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets',
                                      'CVPPP2017_LCC_training', 'training', ds)
        elif ds == 'BL':
            args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets',
                                          'Banana_leaves', ds)
        print('Working on dataset:', ds)

        if not os.path.isfile(args.save_res_path) :
            with open(args.save_res_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Exp", "Augmantation", "dataset", "dic", "absDic", "agree", "MSE", "dataset size"])

        [train_set, test_set] = data_split(args.data_path, ds, test_set_size)
        [train_sets, val_sets] = split_to_train_val_sets(train_set, dataset_sizes, validation_rate)

        test_csv_files_names = create_csv_file_for_experiment(args.data_path, ds, args.exp_name, test_set, 'Test')

        for i in range(len(dataset_sizes)):

            saving_path_name = os.path.join('exp_' + str(args.exp_num), 'dataset_size_' + str(dataset_sizes[i]))

            args.snapshot_path = os.path.join(GetEnvVar('ModelsPath'), 'Counting_Models_snapshots', args.pipe,
                                              saving_path_name)

            args.model = os.path.join(args.snapshot_path, 'resnet50_csv.h5')

            args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results",
                                          saving_path_name)
            args.tensorboard_dir = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, 'log_dir',
                                                saving_path_name)

            train_csv_files_names = create_csv_file_for_experiment(args.data_path, ds, args.exp_name, train_sets[i], 'Train')
            val_csv_files_names = create_csv_file_for_experiment(args.data_path, ds, args.exp_name, val_sets[i], 'Val')


            args.train_csv_leaf_number_file = train_csv_files_names['Train_count_file']
            args.train_csv_leaf_location_file = train_csv_files_names['Train_centers_files']

            args.val_csv_leaf_number_file = val_csv_files_names['Val_count_file']
            args.val_csv_leaf_location_file = val_csv_files_names['Val_centers_files']

            # Train the model based on current split
            if args.pipe == 'keyPfinder':
                history = train.main(args)

            # Test the model

            # update args for evaluation
            args.val_csv_leaf_number_file = test_csv_files_names['Test_count_file']
            args.val_csv_leaf_location_file = test_csv_files_names['Test_centers_files']

            if args.calc_det_performance:
                CountDiff, AbsCountDiff, CountAgreement, MSE, R_2, ap = evaluate_LCC.main(args)
                ap = str(round(ap, 3))

            else:
                CountDiff, AbsCountDiff, CountAgreement, MSE, R_2 = evaluate_LCC.main(args)
                ap = 'not available'

            with open(args.save_res_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([str(args.exp_num),
                                 str(args.random_transform),
                                 ds,
                                 str(round(CountDiff, 3)),
                                 str(round(AbsCountDiff, 3)),
                                 str(round(CountAgreement, 3)),
                                 str(round(MSE, 3)),
                                 str(dataset_sizes[i])])

            # Delete current temp csv files

            for file in train_csv_files_names:
                if os.path.isfile(train_csv_files_names[file]):
                    os.remove(train_csv_files_names[file])
            for file in val_csv_files_names:
                if os.path.isfile(val_csv_files_names[file]):
                    os.remove(val_csv_files_names[file])


        args.exp_num += 1

    print("Done")
    for file in test_csv_files_names:
        if os.path.isfile(test_csv_files_names[file]):
            os.remove(test_csv_files_names[file])

    return history


if __name__ == '__main__':
    args = None
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

    args.snapshot = None
    args.imagenet_weights = True
    args.weights = None

    args.backbone = 'resnet50'
    args.batch_size = 1

    args.freeze_backbone = False

    args.random_transform = True
    args.evaluation = True
    # TODO - choose min and max image size
    args.image_min_side = 800
    args.image_max_side = 1333

    args.dataset_type = 'csv'

    args.exp_num = 2070
    args.gpu = '0'

    args.epochs = 100
    main(args)









