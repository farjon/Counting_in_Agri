import csv
import random
import sys
import os
import argparse
import numpy as np
import pandas as pd


from counters.MSR_DRN_keras.bin import train

from counters.MSR_DRN_keras.bin import evaluate_LCC
from counters.MSR_DRN_keras.preprocessing import create_csv_of_leaf_center

#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import seaborn as sns



def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    return parser.parse_args(args)


def get_centers_data(data, leaf_location_csvPath):

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


def data_split(DATASET_DIR, dataset):
    leaf_counts = []
    rgb_images = []

    # read the leaf counts csv file
    csvPath = os.path.join(DATASET_DIR, dataset + ".csv")
    with open(csvPath) as csvfile:
        readCSV = csv.reader(csvfile)
        print("Working on spliting dataset: ", dataset, "\n")
        count = 0
        for row in readCSV:
            print(row)
            rgbImage_name = row[0]
            plant_name = rgbImage_name.split("_")[0]

            current_leaf_count = {}
            #current_leaf_count[dataset + "_" + plant_name] = int(row[1])
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
    coord_dict = get_centers_data(dataset, leaf_location_csvPath)

    ############################################################################

    # Not the same as coord_dict??????????????????????????
    leaf_location_coord = []
    for key, value in coord_dict.items():
        leaf_location_coord.append([key, value])
    ########################################################################

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

    #sorting the lists so they will be correlated
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


def create_sub_csv_file(csv_leaf_number_file, csv_leaf_location_file, leaf_counts, leaf_location_coord ):
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
                name = key + "_rgb.png"
                writer.writerow([name, count])

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


def get_current_data(All_Splitted_Data, Test_fold_num,Val_fold_num, Train_fold_num):
    Train_leaf_counts = []
    Train_leaf_location_coord = []

    for i in range(len(All_Splitted_Data['f' + str(Train_fold_num[0]) + '_leaf_counts'])):
        Train_leaf_counts.append(All_Splitted_Data['f' + str(Train_fold_num[0]) + '_leaf_counts'][i])

    for i in range(len(All_Splitted_Data['f' + str(Train_fold_num[1]) + '_leaf_counts'])):
        Train_leaf_counts.append(All_Splitted_Data['f' + str(Train_fold_num[1]) + '_leaf_counts'][i])

    for i in range(len(All_Splitted_Data['f' + str(Train_fold_num[0]) + '_leaf_location_coord'])):
        Train_leaf_location_coord.append(All_Splitted_Data['f' + str(Train_fold_num[0]) + '_leaf_location_coord'][i])
    for i in range(len(All_Splitted_Data['f' + str(Train_fold_num[1]) + '_leaf_location_coord'])):
        Train_leaf_location_coord.append(All_Splitted_Data['f' + str(Train_fold_num[1]) + '_leaf_location_coord'][i])

    Val_leaf_counts = All_Splitted_Data['f' + str(Val_fold_num) + '_leaf_counts']
    Val_leaf_location_coord = All_Splitted_Data['f' + str(Val_fold_num) + '_leaf_location_coord']

    Test_leaf_counts = All_Splitted_Data['f' + str(Test_fold_num) + '_leaf_counts']
    Test_leaf_location_coord = All_Splitted_Data['f' + str(Test_fold_num) + '_leaf_location_coord']

    current_data_dict = {}
    current_data_dict['Train_leaf_counts'] = Train_leaf_counts
    current_data_dict['Train_leaf_location_coord'] = Train_leaf_location_coord
    current_data_dict['Val_leaf_counts'] = Val_leaf_counts
    current_data_dict['Val_leaf_location_coord'] = Val_leaf_location_coord
    current_data_dict['Test_leaf_counts'] = Test_leaf_counts
    current_data_dict['Test_leaf_location_coord'] = Test_leaf_location_coord

    return current_data_dict


def get_aggregated_results(args, stats):
    all_CountDiff = stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountDiff']
    all_AbsCountDiff = stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['AbsCountDiff']
    all_CountAgreement =stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountAgreement']
    all_MSE = stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['MSE']
    all_R_2 = stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['R_2']
    all_ap = stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['ap']

    mean_CountDiff = np.mean(all_CountDiff)
    mean_AbsCountDiff = np.mean(all_AbsCountDiff)
    mean_CountAgreement = np.mean(all_CountAgreement)
    mean_MSE = np.mean(all_MSE)
    mean_R_2 = np.mean(all_R_2)

    std_CountDiff = np.std(all_CountDiff)
    std_AbsCountDiff = np.std(all_AbsCountDiff)
    std_CountAgreement = np.std(all_CountAgreement)
    std_MSE = np.std(all_MSE)
    std_R_2 = np.std(all_R_2)

    #ap values
    if args.calc_det_performance:
        mean_ap = round(np.mean(all_ap), 3)
        std_ap = round(np.std(all_ap), 3)
    else:
        mean_ap = None
        std_ap = None

    print()
    print('All CV results:', 'train on:', args.train_on, ',', 'test on:', args.test_on )
    print('The CountDiff values:', [ '%.3f' % elem for elem in all_CountDiff ])
    print('The AbsCountDiff values:', [ '%.3f' % elem for elem in all_AbsCountDiff])
    print('The CountAgreement values:', [ '%.3f' % elem for elem in all_CountAgreement])
    print('The MSE values:', [ '%.3f' % elem for elem in all_MSE])
    print('The R_2 values:', ['%.3f' % elem for elem in all_R_2])

    if args.calc_det_performance:
        print('The ap values:', ['%.3f' % elem for elem in all_ap])
    else:
        print('The ap values:', 'None')

    print()

    print('Summarized results for: train on:', args.train_on, ',', 'test on:', args.test_on)
    print('mean_CountDiff:', round(mean_CountDiff, 3), 'std_CountDiff:', round(std_CountDiff, 3))
    print('mean_AbsCountDiff:', round(mean_AbsCountDiff,3), 'std_AbsCountDiff:', round(std_AbsCountDiff, 3))
    print('mean_CountAgreement:', round(mean_CountAgreement, 3), 'std_CountAgreement:', round(std_CountAgreement, 3))
    print('mean_MSE:', round(mean_MSE, 3), 'std_MSE:', round(std_MSE, 3))
    print('mean_R_2:', round(mean_R_2, 3), 'std_R_2:', round(std_R_2, 3))

    if args.calc_det_performance:
        print('mean_ap:', round(mean_ap, 3), 'std_ap:', round(std_ap, 3))
        ap_data =[['%.3f' % elem for elem in all_ap]]
    else:
        print('mean_ap:', 'None', 'std_ap:', 'None')
        ap_data = ['None']


    df = pd.read_csv(args.save_res_path)
    new_data = pd.DataFrame({"Exp":[str(args.exp_num)], "Augmantation": str(args.random_transform), "Train_set":[args.train_on], "Test_set":[args.test_on],
                                 "mean_Dic":[str(round(mean_CountDiff, 3))], "std_Dic":[str(round(std_CountDiff, 3))],
                                 "mean_AbsDic":[str(round(mean_AbsCountDiff, 3))],  "std_AbsDic":[str(round(std_AbsCountDiff, 3))],
                                 "mean_Agreement":[str(round(mean_CountAgreement, 3))],"std_Agreement":[str(round(std_CountAgreement, 3))],
                                 "mean_MSE":[str(round(mean_MSE,3))], "std_MSE":[str(round(std_MSE, 3))],
                                 "mean_R_2":[str(round(mean_R_2,3))], "std_R_2":[str(round(std_R_2, 3))],
                                 "mean_ap": [str(mean_ap)], "std_ap": [str(std_ap)],
                                 "all_dic": [['%.3f' % elem for elem in all_CountDiff]], "all_AbsDic": [['%.3f' % elem for elem in all_AbsCountDiff]],
                                 "all_CountAgreement": [[ '%.3f' % elem for elem in all_CountAgreement]], "all_MSE": [[ '%.3f' % elem for elem in all_MSE]],
                                 "all_R_2": [['%.3f' % elem for elem in all_R_2]], "all_ap": ap_data})
    df = df.append(new_data)
    df.to_csv(args.save_res_path,index=False)


def get_aggregated_results_withHyper(args, stats):
    all_CountDiff = stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountDiff']
    all_AbsCountDiff = stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['AbsCountDiff']
    all_CountAgreement =stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountAgreement']
    all_MSE = stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['MSE']
    all_R_2 = stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['R_2']
    all_ap = stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['ap']

    mean_CountDiff = np.mean(all_CountDiff)
    mean_AbsCountDiff = np.mean(all_AbsCountDiff)
    mean_CountAgreement = np.mean(all_CountAgreement)
    mean_MSE = np.mean(all_MSE)
    mean_R_2 = np.mean(all_R_2)

    std_CountDiff = np.std(all_CountDiff)
    std_AbsCountDiff = np.std(all_AbsCountDiff)
    std_CountAgreement = np.std(all_CountAgreement)
    std_MSE = np.std(all_MSE)
    std_R_2 = np.std(all_R_2)

    #ap values
    if args.calc_det_performance:
        mean_ap = round(np.mean(all_ap), 3)
        std_ap = round(np.std(all_ap), 3)
    else:
        mean_ap = None
        std_ap = None

    print()
    print('All CV results:', 'train on:', args.train_on, ',', 'test on:', args.test_on )
    print('The CountDiff values:', [ '%.3f' % elem for elem in all_CountDiff ])
    print('The AbsCountDiff values:', [ '%.3f' % elem for elem in all_AbsCountDiff])
    print('The CountAgreement values:', [ '%.3f' % elem for elem in all_CountAgreement])
    print('The MSE values:', [ '%.3f' % elem for elem in all_MSE])
    print('The R_2 values:', ['%.3f' % elem for elem in all_R_2])

    print('early_stopping_patience:', args.early_stopping_patience)
    print('reduceLR_patience:', args.reduceLR_patience)
    print('reduceLR_factor:', args.reduceLR_factor)
    print('lr:', args.lr)

    if args.calc_det_performance:
        print('The ap values:', ['%.3f' % elem for elem in all_ap])
    else:
        print('The ap values:', 'None')

    print()

    print('Summarized results for: train on:', args.train_on, ',', 'test on:', args.test_on)
    print('mean_CountDiff:', round(mean_CountDiff, 3), 'std_CountDiff:', round(std_CountDiff, 3))
    print('mean_AbsCountDiff:', round(mean_AbsCountDiff,3), 'std_AbsCountDiff:', round(std_AbsCountDiff, 3))
    print('mean_CountAgreement:', round(mean_CountAgreement, 3), 'std_CountAgreement:', round(std_CountAgreement, 3))
    print('mean_MSE:', round(mean_MSE, 3), 'std_MSE:', round(std_MSE, 3))
    print('mean_R_2:', round(mean_R_2, 3), 'std_R_2:', round(std_R_2, 3))

    if args.calc_det_performance:
        print('mean_ap:', round(mean_ap, 3), 'std_ap:', round(std_ap, 3))
        ap_data =[['%.3f' % elem for elem in all_ap]]
    else:
        print('mean_ap:', 'None', 'std_ap:', 'None')
        ap_data = ['None']


    df = pd.read_csv(args.save_res_path)
    new_data = pd.DataFrame({#'early_stopping_patience': [str(args.early_stopping_patience)],
                             #'reduceLR_patience': [str(args.reduceLR_patience)],
                             #'reduceLR_factor': [str(args.reduceLR_factor)],
                             "lr": [str(args.lr)],
                             "Exp":[str(args.exp_num)], "Augmantation": str(args.random_transform), "Train_set":[args.train_on], "Test_set":[args.test_on],
                             "mean_Dic":[str(round(mean_CountDiff, 3))], "std_Dic":[str(round(std_CountDiff, 3))],
                             "mean_AbsDic":[str(round(mean_AbsCountDiff, 3))],  "std_AbsDic":[str(round(std_AbsCountDiff, 3))],
                             "mean_Agreement":[str(round(mean_CountAgreement, 3))],"std_Agreement":[str(round(std_CountAgreement, 3))],
                             "mean_MSE":[str(round(mean_MSE,3))], "std_MSE":[str(round(std_MSE, 3))],
                             "mean_R_2":[str(round(mean_R_2,3))], "std_R_2":[str(round(std_R_2, 3))],
                             "mean_ap": [str(mean_ap)], "std_ap": [str(std_ap)],
                             "all_dic": [['%.3f' % elem for elem in all_CountDiff]], "all_AbsDic": [['%.3f' % elem for elem in all_AbsCountDiff]],
                             "all_CountAgreement": [[ '%.3f' % elem for elem in all_CountAgreement]], "all_MSE": [[ '%.3f' % elem for elem in all_MSE]],
                             "all_R_2": [['%.3f' % elem for elem in all_R_2]], "all_ap": ap_data})

    df = df.append(new_data)
    df.to_csv(args.save_res_path,index=False)

    return mean_CountAgreement




def main(args=None):
    random.seed(50)

    args.pipe =  'keyPfinder' #'reg' or 'keyPfinder'

    args.random_transform = True

    args.exp_num = 8000
    args.exp_name = 'detection_option_20'
    args.epochs = 100
    args.gpu = '0'

    args.do_dropout = False

    tune_hyper_params = False
    args.hyper_max_evals = 100
    # initial values of hyper-params - unless changed via hyper parameters tuning
    args.lr = 1e-5
    args.reduce_lr = True
    args.reduceLR_patience = 5
    args.reduceLR_factor = 0.05

    args.early_stopping_indicator = "AbsCountDiff"
    args.early_stopping_patience = 50

    args.step_multi = 5

    #important?
    args.multi_gpu = False
    args.multi_gpu_force = False

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
        # key point detection options:
        # 10 - best option, as in the paper
        # 20 - changing GT Gaussian maps for the sub-model
        args.option = 20

        # the detection performance is done using the PCK metric - see our paper for mor information
        args.calc_det_performance = False #True

    else:
        print("Choose a relevant pipe - keyPfinder or reg")
        return

    args.save_res_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results", "results_" + args.pipe + '_exp_'+ args.exp_name+ '_'+str(args.exp_num) + ".csv")


    images_num = {}
    images_num["A1"] = 128
    images_num["A2"] = 31
    images_num["A3"] = 27
    images_num["A4"] = 624

    chosen_datasets = ['A1', 'A2', 'A3', 'A4']

    def hyperopt_train_test(params=None):

        agreement_per_ds = {}
        total_num_of_images = 0
        total_mean_agreement = 0

        for ds in chosen_datasets:

            total_num_of_images += images_num[ds]

            args.train_on = ds
            args.test_on = ds

            args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Phenotyping Datasets', 'Plant phenotyping', 'data_2',
                                          'CVPPP2017_LCC_training', 'training', ds)

            stats = {}
            stats['train_on_'+args.train_on+'_test_on_'+args.test_on] ={}

            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountDiff'] = []
            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['AbsCountDiff'] = []
            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountAgreement'] = []
            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['MSE'] = []
            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['R_2'] = []
            stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['ap'] = []

            All_Splitted_Data = data_split(args.data_path, ds)

            num_of_CV = 4
            print('Working on dataset:', ds)
            if tune_hyper_params:
                num_of_folds = 1
            else:
                num_of_folds = 4
            for cv_fold in range(1, 5):
                print()
                print("lr value: ", args.lr)
                print()

                args.snapshot_path = os.path.join(GetEnvVar('ModelsPath'), 'Counting_Models_snapshots', args.pipe,
                                                  'exp_' + str(args.exp_num), 'cv_' + str(cv_fold))
                args.tensorboard_dir = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, 'log_dir',
                                                    'exp_' + str(args.exp_num), 'cv_' + str(cv_fold))

                args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results",
                                              'exp_' + str(args.exp_num), 'cv_' + str(cv_fold))

                args.model = os.path.join(GetEnvVar('ModelsPath'), 'Counting_Models_snapshots', args.pipe,
                                          'exp_' + str(args.exp_num),  'cv_' + str(cv_fold), 'resnet50_csv.h5')  #'resnet50_final.h5'


                Test_fold_num = (cv_fold) % num_of_CV+1
                Val_fold_num = (cv_fold + 1) % num_of_CV+1
                Train_fold_num = [(cv_fold + 2) % num_of_CV+1, (cv_fold + 3) % num_of_CV+1]

                current_data_dict = get_current_data(All_Splitted_Data, Test_fold_num, Val_fold_num, Train_fold_num)

                # print relevant data to files

                train_count_file = os.path.join(args.data_path, ds + '_cv' + str(cv_fold)+ '_exp_'+ str(args.exp_num) +'_Train.csv')
                train_centers_files = os.path.join(args.data_path, ds + '_cv' + str(cv_fold)+ '_exp_'+ str(args.exp_num) +'_Train_leaf_location.csv')

                val_count_file = os.path.join(args.data_path, ds + '_cv' + str(cv_fold)+ '_exp_'+ str(args.exp_num) +'_Val.csv')
                val_centers_files = os.path.join(args.data_path, ds + '_cv' + str(cv_fold)+ '_exp_'+ str(args.exp_num) + '_Val_leaf_location.csv')

                test_count_file = os.path.join(args.data_path, ds + '_cv' + str(cv_fold) + '_exp_'+ str(args.exp_num) +'_Test.csv')
                test_centers_files = os.path.join(args.data_path, ds + '_cv' + str(cv_fold) + '_exp_'+ str(args.exp_num) +'_Test_leaf_location.csv')

                #Remove files from prev runs if mistakenly exist
                if os.path.isfile(train_count_file):
                    os.remove(train_count_file)

                if os.path.isfile(train_centers_files):
                    os.remove(train_centers_files)

                if os.path.isfile(val_count_file):
                    os.remove(val_count_file)

                if os.path.isfile(val_centers_files):
                    os.remove(val_centers_files)

                if os.path.isfile(test_count_file):
                    os.remove(test_count_file)

                if os.path.isfile(test_centers_files):
                    os.remove(test_centers_files)

                create_sub_csv_file(train_count_file, train_centers_files, current_data_dict['Train_leaf_counts'], current_data_dict['Train_leaf_location_coord'])
                create_sub_csv_file(val_count_file, val_centers_files, current_data_dict['Val_leaf_counts'], current_data_dict['Val_leaf_location_coord'])
                create_sub_csv_file(test_count_file, test_centers_files, current_data_dict['Test_leaf_counts'], current_data_dict['Test_leaf_location_coord'])

                args.train_csv_leaf_number_file = train_count_file
                args.train_csv_leaf_location_file = train_centers_files

                args.val_csv_leaf_number_file = val_count_file
                args.val_csv_leaf_location_file = val_centers_files


                #Train the model based on current split
                if args.pipe == 'keyPfinder':
                    train.main(args)
                elif args.pipe == 'reg':
                    train_reg.main(args)

                # Test the model

                #update args for evaluation

                args.val_csv_leaf_number_file = test_count_file
                args.val_csv_leaf_location_file = test_centers_files

                if args.calc_det_performance:
                    CountDiff, AbsCountDiff, CountAgreement, MSE, R_2, ap = evaluate_LCC.main(args)
                else:
                    CountDiff, AbsCountDiff, CountAgreement, MSE, R_2 = evaluate_LCC.main(args)
                    ap = None

                print('Result of cv_',str(cv_fold),'-', 'testing ', ds)
                print('CountDiff:',CountDiff, 'AbsCountDiff:', AbsCountDiff, 'CountAgreement', CountAgreement, 'MSE:', MSE)

                stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountDiff'].append(CountDiff)
                stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['AbsCountDiff'].append(AbsCountDiff)
                stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['CountAgreement'].append(CountAgreement)
                stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['MSE'].append(MSE)
                stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['R_2'].append(R_2)
                stats['train_on_' + args.train_on + '_test_on_' + args.test_on]['ap'].append(ap)


                # Delete current temp csv files
                os.remove(args.train_csv_leaf_number_file)
                os.remove(args.train_csv_leaf_location_file)

                os.remove(val_count_file)
                os.remove(val_centers_files)

                os.remove(test_count_file)
                os.remove(test_centers_files)

            args.exp_num += 1
            # get mean and std errors, and save to results file

            if not os.path.isfile(args.save_res_path) :
                with open(args.save_res_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([#'early_stopping_patience','reduceLR_patience', 'reduceLR_factor',
                                     "lr",
                                     "Exp", "Augmantation", "Train_set", "Test_set", "mean_Dic", "std_Dic",
                                     "mean_AbsDic", "std_AbsDic", "mean_Agreement", "std_Agreement",
                                     "mean_MSE", "std_MSE", 'mean_R_2', "std_R_2", "mean_ap", "std_ap",
                                     "all_dic", "all_AbsDic", "all_CountAgreement", "all_MSE",
                                     "all_R_2", "all_ap"])

            mean_CountAgreement = get_aggregated_results_withHyper(args, stats)
            agreement_per_ds[ds] = mean_CountAgreement

        # get weighted average of count agreement
        for ds in chosen_datasets:
            total_mean_agreement += agreement_per_ds[ds]*(images_num[ds]/total_num_of_images)

        return total_mean_agreement


    # if tune_hyper_params:
    #     fspace = {
    #         'args.lr': hp.loguniform('args.lr',np.log(0.00001), np.log(0.001)) #10**(np.random.uniform(-4,-2)), 10**(np.random.uniform(-2,0))
    #         #'args.early_stopping_patience': hp.choice('args.early_stopping_patience', range(5, 60)),
    #         #'args.reduceLR_patience': hp.choice('args.reduceLR_patience', range(5, 60)),
    #         #'args.reduceLR_factor': hp.uniform('args.reduceLR_factor', 0.0001, 0.01)
    #     }
    #
    #
    #     def f(params):
    #         #args.early_stopping_patience = params['args.early_stopping_patience']
    #         #args.reduceLR_patience = params['args.reduceLR_patience']
    #         #args.reduceLR_factor = params['args.reduceLR_factor']
    #         args.lr = params['args.lr']
    #
    #         acc = hyperopt_train_test(params)
    #         return {'loss': -acc, 'status': STATUS_OK}
    #
    #     trials = Trials()
    #     best = fmin(f, fspace, algo=tpe.suggest, max_evals= args.hyper_max_evals, trials=trials)
    #
    #     print ('best:', best)
    #
    #     # Plot the parameters
    #     #parameters = ['args.early_stopping_patience', 'args.reduceLR_patience', 'args.reduceLR_factor']
    #     parameters = ['args.lr']
    #     cols = len(parameters)
    #     f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(15, 5))
    #     cmap = plt.cm.jet
    #     for i, val in enumerate(parameters):
    #         xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    #         ys = [-t['result']['loss'] for t in trials.trials]
    #         points = zip(xs, ys)
    #         sorted_points = sorted(points)
    #         xs = [point[0] for point in sorted_points]
    #         ys = [point[1] for point in sorted_points]
    #         ys = np.array(ys)
    #         axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i) / len(parameters)))
    #         axes[i].set_title(val)
    #
    #     plot_path = os.path.join(GetEnvVar('ExpResultsPath'), 'LCC_exp_res', args.pipe, "results",
    #                                               'exp_' + str(args.exp_num-1) + '\\Hyper_Params_plot.png')
    #     plt.savefig(plot_path)
    #     plt.close(plot_path)
    #
    # else:
    total_mean_agreement = hyperopt_train_test()
    print('total_mean_agreement:', total_mean_agreement)

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

    args.random_transform = True
    args.evaluation = True
    # TODO - choose min and max image size
    args.image_min_side = 800
    args.image_max_side = 1333

    args.dataset_type = 'csv'


    main(args)
