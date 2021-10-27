import cv2
import torch
from torch.utils.data import Dataset

class Reg_Agri_Dataset(Dataset):
    def __init__(self, args, set):
        super(Reg_Agri_Dataset, self).__init__()
