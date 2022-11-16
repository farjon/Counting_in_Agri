import csv
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import GetEnvVar as env

myDatasetsPath = env.GetEnvVar('DatasetsPath')

DATASET_DIR = os.path.join(myDatasetsPath, "Phenotyping Datasets\\Plant phenotyping", "data_2", "CVPPP2017_LCC_training\\training")
dataset = ["A1", "A2", "A3", "A4"]

percentile = 99

dist_of_AC = []
for i in range(len(dataset)):

    all_euc_dist = []

    data = dataset[i]

    data_path = os.path.join(DATASET_DIR, data)

    csvPath = os.path.join(data_path, data + "_leaf_location.csv")
    points = {}

    with open(csvPath) as csvfile:
        readCSV = csv.reader(csvfile)
        print("Working on dataset: ", data, "\n")
        #get per image info
        for row in readCSV:
            image_name = row[0]
            x = row[1]
            y = row[2]

            if len(points) == 0:
                points[image_name] = []
                points[image_name].append([x, y])
            else:
                if image_name in points.keys():
                    points[image_name].append([x, y])
                else:
                    points[image_name] = []
                    points[image_name].append([x, y])



    #find per image relative distances
    for key in points.keys():
        im_points = points[key]
        numOfPoints = len(im_points)

        for i in range(numOfPoints-1):
            for j in range(i+1,numOfPoints):

                x_dist = abs(int(im_points[i][0])-int(im_points[j][0]))
                y_dist = abs(int(im_points[i][1]) - int(im_points[j][1]))
                euc_dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
                all_euc_dist.append(euc_dist)

                dist_of_AC.append(euc_dist)


    numOfDist = numOfPoints*(numOfPoints-1)/2
    min_euc = np.min(all_euc_dist)
    max_euc = np.max(all_euc_dist)
    avg_euc = np.mean(all_euc_dist)


    print("For dataset", data, ":")
    print("[min_euc, max_euc, avg_euc]: ", "[", round(min_euc, 3),round(max_euc, 3), round(avg_euc, 3), "]")

    perc_value = np.percentile(all_euc_dist, percentile)
    print("perc_value:", perc_value)

    #np.histogram(all_euc_dist,10)
    plt.hist(all_euc_dist)
    plt.title("dataset:" + data)
    plt.xlabel("dist")
    plt.ylabel("Frequency")
    #plt.show()

    plt.savefig(os.path.join(DATASET_DIR, "perc_percentile_"+data+".jpg"))
    plt.close(image_name)
    print()


# All datasets summary:

min_euc_AC = np.min(dist_of_AC)
max_euc_Ac = np.max(dist_of_AC)
avg_euc_AC = np.mean(dist_of_AC)

print("For dataset AC")
print("[min_euc, max_euc, avg_euc]: ", "[", round(min_euc_AC, 3), round(max_euc_Ac, 3), round(avg_euc_AC, 3), "]")

perc_value = np.percentile(dist_of_AC, percentile)
print("perc_value:", perc_value)

# np.histogram(all_euc_dist,10)
plt.hist(dist_of_AC)
plt.title("dataset AC")
plt.xlabel("dist")
plt.ylabel("Frequency")
#plt.show()
plt.savefig(os.path.join(DATASET_DIR, "perc_percentile_AC.jpg"))
plt.close(image_name)

print()