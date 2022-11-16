import csv
import random
import sys
import os
import argparse
import numpy as np
from GetEnvVar import GetEnvVar
import pandas as pd


#Allow relative imports when being executed as script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#import keras_retinanet.bin
__package__ = "keras_retinanet.bin"

from run_CV import get_aggregated_results
from run_CV import get_current_data
from .import train
from .import train_reg
from . import evaluate_LCC
from ..preprocessing import create_csv_of_leaf_center


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    return parser.parse_args(args)


def get_paths_dict(Ac_files_path, exp_id, cv_fold):

    train_count_file = os.path.join(Ac_files_path, 'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_Train.csv')
    train_centers_file = os.path.join(Ac_files_path,
                                      'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_Train_leaf_location.csv')

    val_count_file = os.path.join(Ac_files_path, 'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_Val.csv')
    val_centers_file = os.path.join(Ac_files_path,
                                    'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_Val_leaf_location.csv')

    # test_count_file = os.path.join(Ac_files_path,'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_Test.csv')
    # test_centers_file = os.path.join(Ac_files_path,'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_Test_leaf_location.csv')

    test_count_file_A1 = os.path.join(Ac_files_path, 'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_A1_Test.csv')
    test_centers_file_A1 = os.path.join(Ac_files_path,
                                         'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_A1_Test_leaf_location.csv')

    test_count_file_A2 = os.path.join(Ac_files_path, 'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_A2_Test.csv')
    test_centers_file_A2 = os.path.join(Ac_files_path,
                                         'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_A2_Test_leaf_location.csv')

    test_count_file_A3 = os.path.join(Ac_files_path, 'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_A3_Test.csv')
    test_centers_file_A3 = os.path.join(Ac_files_path,
                                         'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_A3_Test_leaf_location.csv')

    test_count_file_A4 = os.path.join(Ac_files_path, 'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_A4_Test.csv')
    test_centers_file_A4 = os.path.join(Ac_files_path,
                                         'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_A4_Test_leaf_location.csv')

    test_count_file_Ac = os.path.join(Ac_files_path, 'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_Ac_Test.csv')
    test_centers_file_Ac = os.path.join(Ac_files_path,
                                        'Ac_cv' + str(cv_fold) + '_exp_' + str(exp_id) + '_Ac_Test_leaf_location.csv')


    files_paths = {}
    files_paths['train_count_file'] = train_count_file
    files_paths['train_centers_file'] = train_centers_file

    files_paths['val_count_file'] = val_count_file
    files_paths['val_centers_file'] = val_centers_file

    files_paths['test_count_file_A1'] = test_count_file_A1
    files_paths['test_centers_file_A1'] = test_centers_file_A1

    files_paths['test_count_file_A2'] = test_count_file_A2
    files_paths['test_centers_file_A2'] = test_centers_file_A2

    files_paths['test_count_file_A3'] = test_count_file_A3
    files_paths['test_centers_file_A3'] = test_centers_file_A3

    files_paths['test_count_file_A4'] = test_count_file_A4
    files_paths['test_centers_file_A4'] = test_centers_file_A4

    files_paths['test_count_file_Ac'] = test_count_file_Ac
    files_paths['test_centers_file_Ac'] = test_centers_file_Ac

    return files_paths


def data_split_for_Ac(DATASET_DIR, dataset):
    leaf_counts = []

    # read the leaf counts csv file
    csvPath = os.path.join(DATASET_DIR, dataset + ".csv")
    with open(csvPath) as csvfile:
        readCSV = csv.reader(csvfile)
        print("Working on spliting dataset: ", dataset, "\n")
        count = 0
        for row in readCSV:
            print(row)
            rgbImage_name = row[0]
            plant_name = dataset + '_' + rgbImage_name

            current_leaf_count = {}
            current_leaf_count[plant_name] = int(row[1])
            current_leaf_count = [current_leaf_count]
            leaf_counts.append(current_leaf_count)
            count += 1
        print()

    print("Done, ", dataset, "set - has", count, "images \n")


    leaf_location_csvPath = os.path.join(DATASET_DIR, dataset + "_leaf_location.csv")

    # create the centers coordinates csv, if doesn't exist yet - of the whole dataset
    if os.path.isfile(leaf_location_csvPath) == False:
        create_csv_of_leaf_center.main(DATASET_DIR, dataset)

    # create a list where each item is a pair: plant key, list of center coordinates
    # the center coordinates of the leaves of each image at each dataset
    coord_dict = get_centers_data_for_Ac(dataset, leaf_location_csvPath)

    leaf_location_coord = []
    for key, value in coord_dict.items():
        leaf_location_coord.append([key, value])


    print("Done reading the datasets, start random split of the data... \n")

    # Create a random datasets split
    num_of_images = len(leaf_location_coord)
    N_f1 = round(0.25* num_of_images)
    N_f2 = round(0.25* num_of_images)
    N_f3 = round(0.25* num_of_images)
    N_f4 = num_of_images-N_f1-N_f2-N_f3

    np.random.seed(0)

    Perm = np.random.permutation(num_of_images)  # Randomly permute a sequence
    f1_inx = Perm[0:N_f1]                       # indices for fold 1
    f2_inx = Perm[N_f1: N_f1+N_f2]             # indices for fold 2
    f3_inx = Perm[N_f1+N_f2: N_f1+N_f2+N_f3]    # indices for fold 3
    f4_inx = Perm[N_f1+N_f2+N_f3:]              # indices for fold 4

    # sorting the lists so they will be correlated
    leaf_counts[0].sort()
    leaf_location_coord.sort()

    # Create F1 data
    f1_leaf_counts = [leaf_counts[i] for i in f1_inx]
    f1_leaf_location_coord = [leaf_location_coord[i] for i in f1_inx]

    # Create F2 data
    f2_leaf_counts = [leaf_counts[i] for i in f2_inx]
    f2_leaf_location_coord = [leaf_location_coord[i] for i in f2_inx]

    # Create F3 data
    f3_leaf_counts = [leaf_counts[i] for i in f3_inx]
    f3_leaf_location_coord = [leaf_location_coord[i] for i in f3_inx]

    # Create F4 data
    f4_leaf_counts = [leaf_counts[i] for i in f4_inx]
    f4_leaf_location_coord = [leaf_location_coord[i] for i in f4_inx]


    print("Done splitting the data..")
    print("Total num of images: ", num_of_images)
    print("Num of F1 images: ", len(f1_inx))
    print("Num of F2 images: ", len(f2_inx))
    print("Num of F3 images: ", len(f3_inx))
    print("Num of F4 images: ", len(f4_inx))
    print()

    All_Splitted_Data = {}

    All_Splitted_Data["f1_leaf_counts"] = f1_leaf_counts
    All_Splitted_Data["f1_leaf_location_coord"] = f1_leaf_location_coord

    All_Splitted_Data["f2_leaf_counts"] = f2_leaf_counts
    All_Splitted_Data["f2_leaf_location_coord"] = f2_leaf_location_coord

    All_Splitted_Data["f3_leaf_counts"] = f3_leaf_counts
    All_Splitted_Data["f3_leaf_location_coord"] = f3_leaf_location_coord

    All_Splitted_Data["f4_leaf_counts"] = f4_leaf_counts
    All_Splitted_Data["f4_leaf_location_coord"] = f4_leaf_location_coord

    return All_Splitted_Data


def get_data_for_current_fold(files_paths,cv_fold, all_data_path, Ac_files_path, exp_id, num_of_CV):

    # Ac files for current fold

    train_count_data = []
    train_centers_data = []

    val_count_data = []
    val_centers_data = []

    # test_count_data = []
    # test_centers_data = []

    test_count_data_A1 = []
    test_centers_data_A1 = []
    test_count_data_A2 = []
    test_centers_data_A2 = []
    test_count_data_A3 = []
    test_centers_data_A3 = []
    test_count_data_A4 = []
    test_centers_data_A4 = []
    test_count_data_Ac = []
    test_centers_data_Ac = []

    all_Ac_files = [files_paths['train_count_file'], files_paths['train_centers_file'], files_paths['val_count_file'], files_paths['val_centers_file']]

    all_sub_Test_files = [files_paths['test_count_file_A1'], files_paths['test_centers_file_A1'], files_paths['test_count_file_A2'], files_paths['test_centers_file_A2'],
                          files_paths['test_count_file_A3'], files_paths['test_centers_file_A3'], files_paths['test_count_file_A4'], files_paths['test_centers_file_A4'],
                          files_paths['test_count_file_Ac'], files_paths['test_centers_file_Ac']]

    # delete previous files if exist
    for file_path in all_Ac_files:
        if os.path.isfile(file_path):
            os.remove(file_path)

    for file_path in all_sub_Test_files:
        if os.path.isfile(file_path):
            os.remove(file_path)

    # from each dataset, get train , val, test sets
    Test_fold_num = (cv_fold) % num_of_CV + 1
    Val_fold_num = (cv_fold + 1) % num_of_CV + 1
    Train_fold_num = [(cv_fold + 2) % num_of_CV + 1, (cv_fold + 3) % num_of_CV + 1]

    data_sets = ['A1', 'A2', 'A3', 'A4']

    for ds in data_sets:

        args.data_path = os.path.join(all_data_path, ds)

        All_Splitted_Data = data_split_for_Ac(args.data_path, ds)

        current_data_dict = get_current_data(All_Splitted_Data, Test_fold_num, Val_fold_num, Train_fold_num)

        for value in current_data_dict['Train_leaf_counts']:
            train_count_data.append(value)
        for value in current_data_dict['Train_leaf_location_coord']:
            train_centers_data.append(value)
        for value in current_data_dict['Val_leaf_counts']:
            val_count_data.append(value)
        for value in current_data_dict['Val_leaf_location_coord']:
            val_centers_data.append(value)

        v_name = 'test_count_data_' + ds
        for value in current_data_dict['Test_leaf_counts']:
            vars()[v_name].append(value)

        v_name = 'test_centers_data_' + ds
        for value in current_data_dict['Test_leaf_location_coord']:
            vars()[v_name].append(value)


    # create test set for Ac
    for ds in data_sets:

        v_name = 'test_count_data_' + ds
        for value in vars()[v_name]:
            test_count_data_Ac.append(value)

        v_name = 'test_centers_data_' + ds
        for value in vars()[v_name]:
            test_centers_data_Ac.append(value)



    # check data
    train_counts_names=[]
    for i in range(len(train_count_data)):
        for key in train_count_data[i][0].keys():
            train_counts_names.append([key])
    train_centers_names = []
    for i in range(len(train_centers_data)):
        for key in train_centers_data[i][0]:
            train_centers_names.append([key])

    if not train_counts_names.sort()==train_centers_names.sort():
        raise()

    val_counts_names = []
    for i in range(len(val_count_data)):
        for key in val_count_data[i][0].keys():
            val_counts_names.append([key])
    val_centers_names = []
    for i in range(len(val_centers_data)):
        for key in val_centers_data[i][0]:
            val_centers_names.append([key])

    if not val_counts_names.sort() == val_centers_names.sort():
        raise ()


    test_counts_names_A1 = []
    for i in range(len(test_count_data_A1)):
        for key in test_count_data_A1[i][0].keys():
            test_counts_names_A1.append([key])
    test_centers_names_A1 = []
    for i in range(len(test_centers_data_A1)):
        for key in test_centers_data_A1[i][0]:
            test_centers_names_A1.append([key])

    if not test_counts_names_A1.sort() == test_centers_names_A1.sort():
        raise ()

    test_counts_names_A2 = []
    for i in range(len(test_count_data_A2)):
        for key in test_count_data_A2[i][0].keys():
            test_counts_names_A2.append([key])
    test_centers_names_A2 = []
    for i in range(len(test_centers_data_A2)):
        for key in test_centers_data_A2[i][0]:
            test_centers_names_A2.append([key])

    if not test_counts_names_A2.sort() == test_centers_names_A2.sort():
        raise ()

    test_counts_names_A3 = []
    for i in range(len(test_count_data_A3)):
        for key in test_count_data_A3[i][0].keys():
            test_counts_names_A3.append([key])
    test_centers_names_A3 = []
    for i in range(len(test_centers_data_A3)):
        for key in test_centers_data_A3[i][0]:
            test_centers_names_A3.append([key])

    if not test_counts_names_A3.sort() == test_centers_names_A3.sort():
        raise ()

    test_counts_names_A4 = []
    for i in range(len(test_count_data_A4)):
        for key in test_count_data_A4[i][0].keys():
            test_counts_names_A4.append([key])
    test_centers_names_A4 = []
    for i in range(len(test_centers_data_A4)):
        for key in test_centers_data_A4[i][0]:
            test_centers_names_A4.append([key])

    if not test_counts_names_A4.sort() == test_centers_names_A4.sort():
        raise ()

    create_sub_csv_file_for_Ac(files_paths['train_count_file'],   files_paths['train_centers_file'], train_count_data, train_centers_data)
    create_sub_csv_file_for_Ac(files_paths['val_count_file'],     files_paths['val_centers_file'], val_count_data, val_centers_data)
    create_sub_csv_file_for_Ac(files_paths['test_count_file_A1'], files_paths['test_centers_file_A1'], test_count_data_A1, test_centers_data_A1)
    create_sub_csv_file_for_Ac(files_paths['test_count_file_A2'], files_paths['test_centers_file_A2'], test_count_data_A2, test_centers_data_A2)
    create_sub_csv_file_for_Ac(files_paths['test_count_file_A3'], files_paths['test_centers_file_A3'], test_count_data_A3, test_centers_data_A3)
    create_sub_csv_file_for_Ac(files_paths['test_count_file_A4'], files_paths['test_centers_file_A4'], test_count_data_A4, test_centers_data_A4)
    create_sub_csv_file_for_Ac(files_paths['test_count_file_Ac'], files_paths['test_centers_file_Ac'], test_count_data_Ac, test_centers_data_Ac)


def get_centers_data_for_Ac(data, leaf_location_csvPath):

    coord_dict = {}

    # read the leaf_location csv file
    with open(leaf_location_csvPath) as csvfile_2:
        readCSV_2 = csv.reader(csvfile_2)
        print("Reading leaf coordinates: ", "\n")
        # create a dictionary for the center coordinates of each plant in each dataset
        for row in readCSV_2:
            plant_name = row[0].split("_")[0]
            x = int(row[1])
            y = int(row[2])
            #key = data + "_" + plant_name
            key = data+'_'+plant_name
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



def create_sub_csv_file_for_Ac(csv_leaf_number_file, csv_leaf_location_file, leaf_counts, leaf_location_coord ):
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
                writer.writerow([key, count])

    # Create a csv file of center points for the relevant set
    new_centers_file_path = csv_leaf_location_file
    with open(new_centers_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(leaf_location_coord)):
            line = leaf_location_coord[i]
            name = line[0] + "_centers.png"
            points = line[1]
            for j in range(len(points)):
                x = points[j][0]
                y = points[j][1]
                writer.writerow([name, x, y])




def main(args=None):
    random.seed(50)

    # initial values of hyper-params
    args.lr =  1e-5 #0.0000182839516698763  #1e-5

    args.reduce_lr = True
    args.reduceLR_patience = 5
    args.reduceLR_factor = 0.5

    args.early_stopping_patience = 75

    args.step_multi = 5


    args.pipe = 'keyPfinder' #'reg'
    args.exp_num = 112211

    args.do_dropout = False

    args.random_transform = True
    args.early_stopping_indicator = "AbsCountDiff"
    args.epochs = 200

    args.gpu = '3'
    args.multi_gpu = False
    args.multi_gpu_force = False


    args.exp_name = 'detection_20'


    if args.pipe == 'reg':
        '''
        reg options:
        'reg_baseline_c5_dubreshko'
        'reg_baseline_c5'
        'reg_fpn_p3'
        'reg_fpn_p3_p7_avg'
        'reg_fpn_p3_p7_mle'
        'reg_fpn_p3_p7_min_sig'
        'reg_fpn_p3_p7_mle_L1'
        'reg_fpn_p3_p7_min_sig_L1'
        '''
        args.option = args.exp_name

        args.calc_det_performance = False
    elif args.pipe == 'keyPfinder':
        args.option = 20
        args.calc_det_performance = True
    else:
        print("Choose a relevant pipe - keyPfinder or reg")
        return



    train_Ac_Test_others = True
    if train_Ac_Test_others:
        data_sets = ['A1', 'A2', 'A3', 'A4', 'Ac']
    else:
        print("Please use run_cv.py script")

    args.save_res_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results", 'results_' + args.pipe + '_exp_'+ args.exp_name+ '_'+str(args.exp_num) + ".csv")

    all_data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets',
                                  'CVPPP2017_LCC_training', 'training')

    Ac_files_path = os.path.join(all_data_path, 'Ac')


    stats = {}

    for data in data_sets:
        args.train_on = 'Ac'
        args.test_on = data
        stats['train_on_' + args.train_on + '_test_on_' + args.test_on] = {}
        stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountDiff'] = []
        stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['AbsCountDiff'] = []
        stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountAgreement'] = []
        stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['MSE'] = []
        stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['R_2'] = []
        stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['ap'] = []


    num_of_CV = 4
    for cv_fold in range(1, num_of_CV + 1):

        saving_path_name = os.path.join('exp_' + str(args.exp_num), 'cv_' + str(cv_fold))

        args.snapshot_path = os.path.join(GetEnvVar('ModelsPath'), 'Counting_Models_snapshots', args.pipe,
                                          saving_path_name)

        args.model = os.path.join(args.snapshot_path, 'resnet50_csv.h5')

        args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results",
                                      saving_path_name)
        args.tensorboard_dir = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, 'log_dir',
                                            saving_path_name)

        files_paths = get_paths_dict(Ac_files_path, args.exp_num, cv_fold)

        get_data_for_current_fold(files_paths, cv_fold, all_data_path, Ac_files_path, args.exp_num, num_of_CV)


        # having train and val data from all datasets together, and the test data splitted by sets, for the curent fold
        # perfom training and evaluation  per each test set

        train_count_file = files_paths['train_count_file']
        train_centers_file = files_paths['train_centers_file']

        val_count_file = files_paths['val_count_file']
        val_centers_file = files_paths['val_centers_file']

        args.train_csv_leaf_number_file = train_count_file
        args.train_csv_leaf_location_file = train_centers_file

        args.val_csv_leaf_number_file = val_count_file
        args.val_csv_leaf_location_file = val_centers_file

        # Train the model based on current split
        print('Start training on Ac')
        if args.pipe == 'keyPfinder':
            train.main(args)

        elif args.pipe == 'reg':
            train_reg.main(args)


        for ds in data_sets:

            args.train_on = 'Ac'
            args.test_on = ds

            print('Testing on dataset:', ds)

            # current test files
            test_count_file = files_paths['test_count_file_' + ds]
            test_centers_file = files_paths['test_centers_file_'+ds]


            # Test the model
            #update args for evaluation

            args.val_csv_leaf_number_file = test_count_file
            args.val_csv_leaf_location_file = test_centers_file

            if args.calc_det_performance:
                CountDiff, AbsCountDiff, CountAgreement, MSE, R_2, ap = evaluate_LCC.main(args)
            else:
                CountDiff, AbsCountDiff, CountAgreement, MSE, R_2 = evaluate_LCC.main(args)
                ap = None


            print('Result of cv_',str(cv_fold),': Testing on ', ds)
            print('CountDiff:',CountDiff, 'AbsCountDiff:', AbsCountDiff, 'CountAgreement', CountAgreement, 'MSE:', MSE)

            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountDiff'].append(CountDiff)
            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['AbsCountDiff'].append(AbsCountDiff)
            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountAgreement'].append(CountAgreement)
            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['MSE'].append(MSE)
            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['R_2'].append(R_2)
            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['ap'].append(ap)

    # get mean and std errors, and save to results file

    if not os.path.isfile(args.save_res_path) :
        with open(args.save_res_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Exp", "Augmantation", "Train_set", "Test_set", "mean_Dic", "std_Dic",
                             "mean_AbsDic", "std_AbsDic", "mean_Agreement", "std_Agreement",
                             "mean_MSE", "std_MSE", 'mean_R_2', "std_R_2", "mean_ap", "std_ap",
                             "all_dic", "all_AbsDic", "all_CountAgreement", "all_MSE",
                             "all_R_2", "all_ap"])

    for data in data_sets:
        args.train_on = 'Ac'
        args.test_on = data
        get_aggregated_results(args, stats)


    print("Done")



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

    args.evaluation = True
    # TODO - choose min and max image size
    args.image_min_side = 800
    args.image_max_side = 1333

    args.dataset_type = 'csv'

    args.nd=[0, 0 ,0, 0, 0]
    args.wd = [0, 0, 0, 0, 0]

    main(args)
