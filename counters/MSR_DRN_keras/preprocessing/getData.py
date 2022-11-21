import csv
import os
import shutil

import matplotlib.image as mpimg
import numpy as np

import GetEnvVar as env
import create_csv_of_leaf_center

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"



'''
For a given dataset, and chosen split percentages (train_SplitPerc = 0.5, val_SplitPerc = 0.25, test_SplitPerc = 0.25),
this script creats sub folders, for the train, val, test sets. 
Each of this sub folder will have the relevant csv file with the leaf count values, all relevant images, and the 
XXX_leaf_location.csv file that is generated by calling the "create_csv_of_leaf_center" script".

This script can also do the same for a new concatenation of datasets, where the sub_datasets array will containn not just 
one set name, but several.
For example, if sub_datasets=["A1", "A2"], then' it will generate a new directory called "A1A2", that will have all the 
data from "A1" and "A2" , splitted to train, val and test sub directories.
'''

##################################################################################################################

myStoragePath = env.GetEnvVar('storagePath')
myDatasetsPath = env.GetEnvVar('DatasetsPath')
myExpResultsPath = env.GetEnvVar('ExpResultsPath')
myModelsPath = env.GetEnvVar('ModelsPath')



def create_splited_dirs(DATASET_DIR, sub_datasets, dirType, All_Splitted_Data):
    rgb_images = All_Splitted_Data[dirType+"_rgb_images"]
    leaf_counts = All_Splitted_Data[dirType+"_leaf_counts"]
    leaf_location_coord = All_Splitted_Data[dirType+"_leaf_location_coord"]

    current_dataset = ""
    if len(sub_datasets) == 1:
        current_dataset = sub_datasets[0]
    else:
        for set in sub_datasets:
            current_dataset = current_dataset + set
    current_dataset_path = os.path.join(DATASET_DIR, current_dataset)
    if not os.path.exists(current_dataset_path):
        os.makedirs(current_dataset_path)

    # Copy data to the relevant folders
    dst = os.path.join(current_dataset_path, current_dataset + "_"+dirType)
    if not os.path.exists(dst):
        os.makedirs(dst)

    for i in range(len(rgb_images)):
        keys = rgb_images[i][0].keys()
        for key in keys:
            original_dataset = key.split("_")[0]
            image_name = key.split("_")[1]

            # copy rgb images
            src_file = os.path.join(DATASET_DIR, original_dataset, image_name + "_rgb.png")
            dst_file = os.path.join(dst, original_dataset + "_" + image_name + "_rgb.png")
            shutil.copyfile(src_file, dst_file)

            # copy centers images
            src_file = os.path.join(DATASET_DIR, original_dataset, image_name + "_centers.png")
            dst_file = os.path.join(dst, original_dataset + "_" + image_name + "_centers.png")
            shutil.copyfile(src_file, dst_file)

            # copy fg images
            src_file = os.path.join(DATASET_DIR, original_dataset, image_name + "_fg.png")
            dst_file = os.path.join(dst, original_dataset + "_" + image_name + "_fg.png")
            shutil.copyfile(src_file, dst_file)

    # Create a csv file of leaf counts for the relevant set
    new_counts_file_path = os.path.join(dst, current_dataset + "_" +dirType+".csv")
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
    new_centers_file_path = os.path.join(dst, current_dataset + "_" + dirType+ "_leaf_location.csv")
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


def main(DATASET_DIR, train_SplitPerc, val_SplitPerc, sub_datasets):
    leaf_counts = []
    rgb_images = []
    masks = []
    centers_Images = []

    # the center coordinates of the leaves of each image at each dataset
    coord_dict = {}
    for data in sub_datasets:
        data_path = os.path.join(DATASET_DIR, data)

        # read the leaf counts csv file
        csvPath = os.path.join(data_path, data + ".csv")
        with open(csvPath) as csvfile:
            readCSV = csv.reader(csvfile)
            print("Working on dataset: ", data, "\n")
            count = 0
            for row in readCSV:
                print(row)
                rgbImage_name = row[0]
                plant_name = rgbImage_name.split("_")[0]

                current_leaf_count = {}
                current_leaf_count[data+"_"+plant_name] = int(row[1])
                current_leaf_count = [current_leaf_count]
                leaf_counts.append(current_leaf_count)

                rgb_image = mpimg.imread(os.path.join(data_path,rgbImage_name))
                current_rgb = {}
                current_rgb[data + "_" + plant_name] = rgb_image
                current_rgb = [current_rgb]
                rgb_images.append(current_rgb)

                mask = mpimg.imread(os.path.join(data_path,plant_name+"_fg.png"))
                current_mask = {}
                current_mask[data+"_"+plant_name] = mask
                current_mask = [current_mask]
                masks.append(current_mask)

                center = mpimg.imread(os.path.join(data_path,plant_name+"_centers.png"))
                current_center = {}
                current_center[data+"_"+plant_name] = center
                current_center = [current_center]
                centers_Images.append(current_center)

                count+=1
            print()

            #create the centers coordinates csv, if doesn't exist yet
            leaf_location_csvPath = os.path.join(data_path, data + "_leaf_location.csv")
            if os.path.isfile(leaf_location_csvPath) == False:
                create_csv_of_leaf_center.main(data_path, data)

            # read the leaf_location csv file
            with open(leaf_location_csvPath) as csvfile_2:
                readCSV_2 = csv.reader(csvfile_2)
                print("Reading leaf coordinates: ", data, "\n")
                # create a dictionary for the center coordinates of each plant in each dataset
                for row in readCSV_2:
                    plant_name = row[0].split("_")[0]
                    x = int(row[1])
                    y = int(row[2])
                    key = data+"_"+plant_name
                    if len(coord_dict)==0:
                        coord_dict[key] = []
                        coord_dict[key].append([x,y])
                    else:
                        if key in coord_dict.keys():
                            coord_dict[key].append([x, y])
                        else:
                            coord_dict[key] = []
                            coord_dict[key].append([x, y])

        print("Done, ", data, "set - has", count, "images \n")


    # create a list where each item is a pair: plant key, list of center coordinates
    leaf_location_coord = []
    for key,value in coord_dict.items():
        leaf_location_coord.append([key,value])

    print("Done reading the datasets, start random split of the data... \n")

    # Create a random datasets split
    num_of_images = len(rgb_images)
    N_train = round(train_SplitPerc * num_of_images)
    N_val = round(val_SplitPerc*num_of_images)
    N_test = num_of_images-N_train-N_val

    np.random.seed(0)

    Perm = np.random.permutation(num_of_images)   # Randomly permute a sequence
    trInx = Perm[0:N_train]                       # indices for training
    valInx = Perm[N_train:N_train+N_val]          # indices for validation
    tsInx = Perm[N_train+N_val:]                  # indices for testing
    # print(Perm)
    # print(trInx)
    # print(valInx)
    # print(tsInx)

    #Create Train data
    Train_rgb_images = [rgb_images[i] for i in trInx]
    Train_leaf_counts = [leaf_counts[i] for i in trInx]
    Train_centers_Images = [centers_Images[i] for i in trInx]
    Train_masks = [masks[i] for i in trInx]
    Train_leaf_location_coord = [leaf_location_coord[i] for i in trInx]

    #Create Val data
    Val_rgb_images = [rgb_images[i] for i in valInx]
    Val_leaf_counts = [leaf_counts[i] for i in valInx]
    Val_centers_Images = [centers_Images[i] for i in valInx]
    Val_masks = [masks[i] for i in valInx]
    Val_leaf_location_coord = [leaf_location_coord[i] for i in valInx]

    #Create Test data
    Test_rgb_images = [rgb_images[i] for i in tsInx]
    Test_leaf_counts = [leaf_counts[i] for i in tsInx]
    Test_centers_Images = [centers_Images[i] for i in tsInx]
    Test_masks = [masks[i] for i in tsInx]
    Test_leaf_location_coord = [leaf_location_coord[i] for i in tsInx]

    print("Done splitting the data..")
    print ("Total num of images: ", num_of_images)
    print ("Num of Train images: ", len(trInx))
    print ("Num of Val images: ", len(valInx))
    print ("Num of Test images: ", len(tsInx))
    print()


    All_Splitted_Data = {}

    All_Splitted_Data["Train_rgb_images"] = Train_rgb_images
    All_Splitted_Data["Train_leaf_counts"] = Train_leaf_counts
    All_Splitted_Data["Train_centers_Images"] = Train_centers_Images
    All_Splitted_Data["Train_masks"] = Train_masks
    All_Splitted_Data["Train_leaf_location_coord"] = Train_leaf_location_coord

    All_Splitted_Data["Val_rgb_images"] = Val_rgb_images
    All_Splitted_Data["Val_leaf_counts"] = Val_leaf_counts
    All_Splitted_Data["Val_centers_Images"] = Val_centers_Images
    All_Splitted_Data["Val_masks"] = Val_masks
    All_Splitted_Data["Val_leaf_location_coord"] = Val_leaf_location_coord

    All_Splitted_Data["Test_rgb_images"] = Test_rgb_images
    All_Splitted_Data["Test_leaf_counts"] = Test_leaf_counts
    All_Splitted_Data["Test_centers_Images"] = Test_centers_Images
    All_Splitted_Data["Test_masks"] = Test_masks
    All_Splitted_Data["Test_leaf_location_coord"] = Test_leaf_location_coord

    dir_Types = ["Train", "Val", "Test"]
    for dirType in dir_Types:
        create_splited_dirs(DATASET_DIR,sub_datasets, dirType, All_Splitted_Data)


    return(All_Splitted_Data)


if __name__ == "__main__":

    # Define the data split percenteges
    train_SplitPerc = 0.5
    val_SplitPerc = 0.25
    test_SplitPerc = 0.25

    # Enter the names of the datasets you want to use:
    # choose any of A1-A4, or any combination of those
    # or just define sub_datasets = AC
    AC = ["A1", "A2", "A3", "A4"]
    sub_datasets = ["A1","A2", "A3", "A4"]
    DATASET_DIR = os.path.join(myDatasetsPath, "Phenotyping Datasets\\Plant phenotyping\\data_2\\CVPPP2017_LCC_training\\training")
    All_Splitted_Data = main(DATASET_DIR, train_SplitPerc, val_SplitPerc, sub_datasets)

    print("Done :) ")