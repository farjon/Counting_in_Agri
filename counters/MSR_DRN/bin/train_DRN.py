import os
import numpy as np
import torch
import argparse

from counters.MSR_DRN.models import DRN
from torch.utils.data import DataLoader
from counters.MSR_DRN.utils.dataloader import CSV_OC
from counters.MSR_DRN.bin.train_loop import train_MSR_DRN


def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default=f'LCC/training/A1', help='choose a dataset')
    parser.add_argument('-lf', '--labels_format', type=str, default='csv', help='labels format can be .csv / coco (.json)')
    # --------------------------- Training Arguments -----------------------
    parser.add_argument('-m', '--model_type', type=str, default='DRN', help='choose a model variant - DRN / MSR')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='only works with batch size = 1')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('-exp', '--exp_number', type=int, default=0, help='number of current experiment')
    parser.add_argument('-c', '--criteria', type=str, default='mae', help='criteria can be mse / mae')
    parser.add_argument('-lr', '--lr', type=float, default=1e-5, help='set learning rate')
    parser.add_argument('-o', '--optim', type=str, default='adam', help='choose optimizer adam / adamw / sgd')
    parser.add_argument('-mm', '--monitor_metric', type=str, default='absDiC', help='choose what metric to monitor absDiC / agreement / MSE')
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
    args.save_trained_models = os.path.join(args.ROOT_DIR, 'Trained_Models', args.data.split('\\')[-1])
    os.makedirs(args.save_trained_models, exist_ok=True)
    torch.hub.set_dir(os.path.join(args.ROOT_DIR, 'Trained_Models', 'pretrained'))

    train_dataset_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}
    # train dataset loader
    train_base_dir = os.path.join(args.ROOT_DIR, 'Data', args.data, 'train')
    train_csv_ON_path = os.path.join(train_base_dir, args.data + '_Train.csv')
    train_csv_OC_path = os.path.join(train_base_dir, args.data + '_Train_leaf_location.csv')
    train_dataset = DataLoader(
        CSV_OC(train_csv_ON_path, train_csv_OC_path, train_base_dir),
        **train_dataset_params)
    val_base_dir = os.path.join(args.ROOT_DIR, 'Data', args.data, 'val')
    val_csv_ON_path = os.path.join(val_base_dir, args.data + '_Val.csv')
    val_csv_OC_path = os.path.join(val_base_dir, args.data + '_Val_leaf_location.csv')
    val_dataset = DataLoader(
        CSV_OC(val_csv_ON_path, val_csv_OC_path, val_base_dir),
        **train_dataset_params)
    model = DRN.resnet50(args, 1, True)

    # define loss function
    if args.criteria == 'mse':
        count_loss_func = torch.nn.MSELoss(reduction='mean')
    elif args.criteria == 'mae':
        count_loss_func = torch.nn.L1Loss(reduction='mean')

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    #train loop
    train_MSR_DRN(args, train_dataset, val_dataset, model, count_loss_func, optimizer, scheduler)

if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)
