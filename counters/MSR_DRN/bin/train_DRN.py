import os
import numpy as np
import torch
import argparse

from counters.MSR_DRN.models import DRN
from torch.utils.data import DataLoader
from counters.MSR_DRN.utils.dataloader import CSV_OC



def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='A1', help='choose a dataset')
    parser.add_argument('-lf', '--labels_format', type=str, default='csv', help='labels format can be .csv / coco (.json)')
    # --------------------------- Training Arguments -----------------------
    parser.add_argument('-m', '--model_type', type=str, default='DRN', help='choose a model variant - DRN / MSR')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='only works with batch size = 1')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='number of epochs for training')
    parser.add_argument('-exp', '--exp_number', type=int, default=0, help='number of current experiment')
    parser.add_argument('-c', '--criteria', type=str, default='mae', help='criteria can be mse / mae')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='set learning rate')
    parser.add_argument('-o', '--optim', type=str, default='sgd', help='choose optimizer adam / adamw / sgd')
    parser.add_argument('-ve', '--val_interval', type=int, default=2, help='run model validation every X epochs')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    args = parser.parse_args()
    return args

def main(args):
    # --------------------------- Don't edit --------------------------- #
    # define device (use cuda if available)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # torch and numpy reproducibility setup
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(10)
    np.random.seed(10)

    # --------------------------- Start edit --------------------------- #
    # setting up path to save pretrained models
    torch.hub.set_dir(os.path.join(args.ROOT_DIR, 'Trained_Models', 'pretrained'))

    train_dataset_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}
    # train dataset loader
    base_dir = os.path.join(args.ROOT_DIR, args.data)
    csv_ON_path = os.path.join(base_dir, 'train', args.data + '_Train.csv')
    csv_OC_path = os.path.join(base_dir, 'train', args.data + '_Train_leaf_location.csv')
    train_dataset = DataLoader(
        CSV_OC(csv_ON_path, csv_OC_path, base_dir, 320),
        **train_dataset_params)

    model = DRN.resnet50(args, 1, True)

if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri\\Data\\LCC\\training'
    main(args)
