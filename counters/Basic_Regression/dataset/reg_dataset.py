import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pathlib import Path

class Reg_Agri_Dataset_csv(Dataset):
    def __init__(self, data_path, set, transform):
        super(Reg_Agri_Dataset_csv, self).__init__()
        self.datapath = data_path
        self.set_name = set
        self.transform = transform
        self.anno_file = self.load_annotations(os.path.join(data_path, 'annotations', set + '.csv'))
    def load_annotations(self, anno_file):
        # load the annotations from the .csv file
        annotations = pd.read_csv(anno_file)
        dir_list = os.listdir(os.path.join(self.datapath, self.set_name))
        if f'{self.set_name}.csv' in dir_list:
            dir_list.remove(f'{self.set_name}.csv')

        # check if annotations file holds images names with extension
        if Path(annotations.iloc[0]['image_name']).stem == annotations.iloc[0]['image_name']:
            self.ext_list = {}
            for image_name in dir_list:
                img_base_name, ext = os.path.splitext(image_name)
                self.ext_list[img_base_name] = ext
            dir_list = [os.path.splitext(x)[0] for x in dir_list]

        for i, row in annotations.iterrows():
            # check if file exists
            if row.iloc[0] not in dir_list:
                annotations.drop(i, inplace=True)
            else:
                # change the value in the dataframe to the value in the list
                if row.iloc[0] != dir_list[dir_list.index(row.iloc[0])]:
                    annotations.iloc[i, 0] = dir_list[dir_list.index(row.iloc[0])]
        return annotations

    def __len__(self):
        # the length of the dataset is the number of the .csv rows
        return self.anno_file.shape[0]

    def __getitem__(self, idx):
        img_details = self.anno_file.iloc[[idx]].iloc[0]
        if Path(img_details.iloc[0]).stem == img_details.iloc[0]:
            image_path = img_details.iloc[0] + self.ext_list[img_details.iloc[0]]
        else:
            image_path = img_details.iloc[0]
        img = Image.open(os.path.join(self.datapath, self.set_name, image_path))
        img = self.transform(img)
        annot = torch.tensor(img_details.iloc[1], dtype=torch.float32)
        if self.set_name == 'test':
            return img, annot, img_details.iloc[0]
        return img, annot