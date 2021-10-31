import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class Reg_Agri_Dataset_csv(Dataset):
    def __init__(self, args, data_path, set, transform):
        super(Reg_Agri_Dataset_csv, self).__init__()

        self.anno_file = pd.read_csv(os.path.join(os.path.join(data_path, 'regression annotations', set + '.csv')))
        self.im_names = self.anno_file['image_name'].values
        self.datapath = data_path
        self.set_name = set
        self.transform = transform

    def __len__(self):
        # the length of the dataset is the number of the .csv rows
        return self.anno_file.shape[0]

    def __getitem__(self, idx):
        img_details = self.anno_file.iloc[[idx]]

        img = Image.open(os.path.join(self.datapath, 'images', self.set_name, img_details.iloc[0]['image_name']))
        img = self.transform(img)
        annot = torch.tensor(img_details.iloc[0]['GT_number'], dtype=torch.float32)

        return img, annot