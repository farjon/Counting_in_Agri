import csv
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import GetEnvVar as env

myDatasetsPath = env.GetEnvVar('DatasetsPath')

DATASET_DIR =  'D:/training'#os.path.join(myDatasetsPath, "Phenotyping Datasets\\Plant phenotyping", "CVPPP2017_LCC_training\\training")
dataset = ["A1", "A2", "A3", "A4"]


for i in range(len(dataset)):
    all_euc_dist = []

    # per_image_info_x_rel = {} # [min_x_rel, max_x_rel, avg_x_rel][min_y_rel, max_y_rel, avg_y_rel]
    # per_image_info_y_rel = {} #[min_y_rel, max_y_rel, avg_y_rel]
    #
    # per_image_info_euc_rel = {}

    per_image_info_euc = {}

    data = dataset[i]

    data_path = os.path.join(DATASET_DIR, data)

    csvPath = os.path.join(data_path, data + "_leaf_location.csv")
    points = {}
    image_size = {}

    # all_relative_dist_X = []
    # all_relative_dist_Y = []
    # all_relative_dist_euclidean = []

    all_dist_euclidean = []
    all_dist_x=[]
    all_dist_y = []

    with open(csvPath) as csvfile:
        readCSV = csv.reader(csvfile)
        print("Working on dataset: ", data, "\n")
        #get per image info
        for row in readCSV:
            #print(row)
            image_name = row[0]
            x = row[1]
            y = row[2]

            image = cv2.imread(os.path.join(data_path,image_name))
            if image is None:
                a=1
            else:
                height, width, channels = image.shape

            if len(image_size) == 0:
                image_size[image_name] = [width, height]
            else:
                if not image_name in image_size.keys():
                    image_size[image_name]=[width, height]

            if len(points) == 0:
                points[image_name] = []
                points[image_name].append([x, y])
            else:
                if image_name in points.keys():
                    points[image_name].append([x, y])
                else:
                    points[image_name] = []
                    points[image_name].append([x, y])


    # global_min_x_rel = {data: 1000000}
    # global_max_x = {data: 0}
    #
    # global_min_y_rel = {data: 1000000}
    # global_max_y = {data: 0}
    #
    # global_avgsum_x = {data: 0}
    # global_avgsum_y = {data: 0}
    #
    # global_avg_x = {data: 0}
    # global_avg_y = {data: 0}
    #
    # global_min_euc_rel = {data: 1000000}
    # global_max_euc_rel = {data: 0}
    # global_avgsum_euc_rel = {data: 0}
    # global_avg_euc_rel = {data: 0}

    global_min_euc = {data: 1000000}
    global_max_euc = {data: 0}
    global_avgsum_euc = {data: 0}
    global_avg_euc = {data: 0}

    global_min_x = {data: 1000000}
    global_max_x = {data: 0}
    global_avgsum_x = {data: 0}
    global_avg_x = {data: 0}

    global_min_y = {data: 1000000}
    global_max_y = {data: 0}
    global_avgsum_y = {data: 0}
    global_avg_y = {data: 0}

    flag_first = 1
    #find per image relative distances
    for key in points.keys():
        im_points = points[key]
        im_sizes = image_size[key]

        # im_relative_dist_array_x = []
        # im_relative_dist_array_y = []
        # im_relative_dist_array_euc = []

        w = im_sizes[0]
        h = im_sizes[1]

        # min_x_rel = 1000000
        # min_y_rel = 1000000
        # max_x_rel = 0
        # max_y_rel = 0
        # sum_x_rel = 0
        # sum_y_rel = 0
        #
        # min_euc_rel = 1000000
        # max_euc_rel = 0
        # sum_euc_rel = 0


        min_euc = 1000000
        max_euc = 0
        sum_euc = 0

        min_x = 1000000
        min_y = 1000000
        max_x = 0
        max_y = 0
        sum_x = 0
        sum_y = 0


        numOfPoints = len(im_points)
        for i in range(numOfPoints-1):
            for j in range(i+1,numOfPoints):

                # x_rel_dist = abs(int(im_points[i][0])-int(im_points[j][0]))/w
                # y_rel_dist = abs(int(im_points[i][1]) - int(im_points[j][1])) / h
                # euc_rel_dist = np.sqrt(x_rel_dist ** 2 + y_rel_dist ** 2)

                x_dist = abs(int(im_points[i][0])-int(im_points[j][0]))
                y_dist = abs(int(im_points[i][1]) - int(im_points[j][1]))
                euc_dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
                all_euc_dist.append(euc_dist)

                # im_relative_dist_array_x.append(x_rel_dist)
                # im_relative_dist_array_y.append(y_rel_dist)
                # im_relative_dist_array_euc.append(euc_rel_dist)

                # if x_rel_dist < min_x_rel:
                #     min_x_rel = x_rel_dist
                # if y_rel_dist < min_y_rel:
                #     min_y_rel = y_rel_dist
                # if euc_rel_dist < min_euc_rel:
                #     min_euc_rel = euc_rel_dist

                if euc_dist < min_euc:
                    min_euc = euc_dist


                # if x_rel_dist > max_x_rel:
                #     max_x_rel = x_rel_dist
                # if y_rel_dist > max_y_rel:
                #     max_y_rel = y_rel_dist
                # if euc_rel_dist > max_euc_rel:
                #     max_euc_rel = euc_rel_dist

                if euc_dist > max_euc:
                    max_euc = euc_dist


                # sum_x_rel += x_rel_dist
                # sum_y_rel += y_rel_dist
                # sum_euc_rel += euc_rel_dist

                sum_euc += euc_dist


        numOfDist = numOfPoints*(numOfPoints-1)/2

        # avg_x_rel = sum_x_rel / numOfDist
        # avg_y_rel = sum_y_rel / numOfDist
        # avg_euc_rel = sum_euc_rel / numOfDist

        avg_euc = sum_euc / numOfDist

        if flag_first:
            # per_image_info_x_rel[data] = []
            # per_image_info_y_rel[data] = []
            # per_image_info_euc_rel[data] = []

            per_image_info_euc[data] = []

            # per_image_info_x_rel[data].append([min_x_rel, max_x_rel, avg_x_rel])
            # per_image_info_y_rel[data].append([min_y_rel, max_y_rel, avg_y_rel])
            # per_image_info_euc_rel[data].append([min_euc_rel, max_euc_rel, avg_euc_rel])

            per_image_info_euc[data].append([min_euc, max_euc, avg_euc])
            flag_first = 0



    for i in range(len(per_image_info_euc[data])):

        # if global_min_x_rel[data] > per_image_info_x_rel[data][i][0]:
        #     global_min_x_rel[data] = per_image_info_x_rel[data][i][0]
        #
        # if global_min_y_rel[data] > per_image_info_y_rel[data][i][0]:
        #     global_min_y_rel[data] = per_image_info_y_rel[data][i][0]
        #
        # if global_min_euc_rel[data] > per_image_info_euc_rel[data][i][0]:
        #     global_min_euc_rel[data] = per_image_info_euc_rel[data][i][0]

        if global_min_euc[data] > per_image_info_euc[data][i][0]:
            global_min_euc[data] = per_image_info_euc[data][i][0]



        # if global_max_x[data] < per_image_info_x_rel[data][i][1]:
        #     global_max_x[data] = per_image_info_x_rel[data][i][1]
        #
        # if global_max_y[data] < per_image_info_y_rel[data][i][1]:
        #     global_max_y[data] = per_image_info_y_rel[data][i][1]
        #
        # if global_max_euc_rel[data] < per_image_info_euc_rel[data][i][1]:
        #     global_max_euc_rel[data] = per_image_info_euc_rel[data][i][1]

        if global_max_euc[data] < per_image_info_euc[data][i][1]:
            global_max_euc[data] = per_image_info_euc[data][i][1]



        # global_avgsum_x[data] += per_image_info_x_rel[data][i][2]
        # global_avgsum_y[data] += per_image_info_y_rel[data][i][2]
        # global_avgsum_euc_rel[data] += per_image_info_euc_rel[data][i][2]

        global_avgsum_euc[data] += per_image_info_euc[data][i][2]


    # global_avg_x[data] = global_avgsum_x[data]/len(per_image_info_x_rel[data])
    # global_avg_y[data] = global_avgsum_y[data]/len(per_image_info_y_rel[data])
    # global_avg_euc_rel[data] = global_avgsum_euc_rel[data] / len(per_image_info_euc_rel[data])

    global_avg_euc[data] = global_avgsum_euc[data] / len(per_image_info_euc[data])


    print("For dataset", data, ":")
    # print("[min_relative_x, max_relative_x, avg_relative_x]: ", "[", round(global_min_x_rel[data], 3), round(global_max_x[data], 3), round(global_avg_x[data], 3), "]")
    # print("[min_relative_y, max_relative_y, avg_relative_y]: ", "[", round(global_min_y_rel[data], 3), round(global_max_y[data], 3), round(global_avg_y[data], 3), "]")
    # print("[min_relative_euc, max_relative_euc, avg_relative_euc]: ", "[", round(global_min_euc_rel[data], 3),
    #       round(global_max_euc_rel[data], 3), round(global_avg_euc_rel[data], 3), "]")

    print("[min_euc, max_euc, avg_euc]: ", "[", round(global_min_euc[data], 3),round(global_max_euc[data], 3), round(global_avg_euc[data], 3), "]")

    perc_five = np.percentile(all_euc_dist, 5)
    print("perc_value:", perc_five)

    #np.histogram(all_euc_dist,10)
    plt.hist(all_euc_dist)
    plt.title("dataset:" + data)
    plt.xlabel("dist")
    plt.ylabel("Frequency")
    plt.show()


    print()