import os
import numpy as np
import torch
import argparse
import wandb
from counters.Basic_Regression.bin.train_loop import train_and_eval
from counters.Basic_Regression.bin.test_loop import test_models
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='Grapes', help='choose a dataset')
    parser.add_argument('-lf', '--labels_format', type=str, default='csv', help='labels format can be .csv / coco (.json)')
    parser.add_argument('-si', '--split_images', type=bool, default=True, help='should we split the images into tiles')
    parser.add_argument('-nt', '--num_of_tiles', type=int, default=10, help='number of tiles')
    parser.add_argument('-p', '--padding', type=int, default=10, help='padding size in case of splitting')
    # --------------------------- Training Arguments -----------------------
    parser.add_argument('-m', '--model_type', type=str, default='resnet50', help='choose a deep model')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='number of epochs for training')
    parser.add_argument('-exp', '--exp_number', type=int, default=0, help='number of current experiment')
    parser.add_argument('-c', '--criteria', type=str, default='mse', help='criteria can be mse / mae')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='set learning rate')
    parser.add_argument('-o', '--optim', type=str, default='sgd', help='choose optimizer adam / adamw / sgd')
    parser.add_argument('-ve', '--val_interval', type=int, default=5, help='run model validation every X epochs')
    parser.add_argument('--es_patience', type=int, default=5,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    args = parser.parse_args()
    return args

def create_dataset(args, train_dataset_params, test_dataset_params):
    if args.split_images:
        from utils.split_raw_images import split_to_tiles
        print('Notice - to split the images, bbox annotations are needed')
        split_to_tiles(args, args.num_of_tiles, args.padding)
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data + '_split')
    else:
        args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data)
    if args.labels_format == 'csv':
        from counters.Basic_Regression.dataset.reg_dataset import Reg_Agri_Dataset_csv as Reg_Agri_Dataset
    elif args.labels_format == 'coco':
        from counters.Basic_Regression.dataset.reg_dataset import Reg_Agri_Dataset_json as Reg_Agri_Dataset
    # train dataset loader
    train_dataset = DataLoader(
        Reg_Agri_Dataset(args, args.data_path, 'train',
                        transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(20),
                            transforms.RandomPerspective(distortion_scale=0.2),
                            transforms.Resize([args.input_size, args.input_size]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
        **train_dataset_params)

    # validation dataset loader
    val_dataset = DataLoader(
        Reg_Agri_Dataset(args, args.data_path, 'val',
                        transform=transforms.Compose([
                            transforms.Resize([args.input_size, args.input_size]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
        **test_dataset_params)

    # test dataset loader
    test_dataset = DataLoader(
        Reg_Agri_Dataset(args, args.data_path, 'test',
                        transform=transforms.Compose([
                            transforms.Resize([args.input_size, args.input_size]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
        **test_dataset_params)
    return train_dataset, val_dataset, test_dataset


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
    wandb.init(project=f'{args.data}_{args.exp_number}', name=f'{args.data}_{args.exp_number}')
    # setting up path to save pretrained models
    args.save_downloaded_weights = os.path.join(args.ROOT_DIR, 'Trained_Models', 'pretrained')
    torch.hub.set_dir(args.save_downloaded_weights)
    args.save_checkpoint_path = os.path.join(args.ROOT_DIR, 'Trained_Models', args.data, 'Basic_Regression', str(args.exp_number))
    os.makedirs(args.save_checkpoint_path, exist_ok=True)
    # Create data loaders
    if args.model_type == 'resnet50':
        args.input_size = 224
    train_dataset_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}
    test_dataset_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0}

    train_dataset, val_dataset, test_dataset = create_dataset(args, train_dataset_params, test_dataset_params)

    # Create model
    if args.model_type == 'resnet50':
        from counters.Basic_Regression.models.ResNet50_reg import ResNet_50_regressor as Reg_Model
    model = Reg_Model()

    # define loss function
    if args.criteria == 'mse':
        loss_func = torch.nn.MSELoss(reduction='mean')
    elif args.criteria == 'mae':
        loss_func = torch.nn.L1Loss(reduction='mean')

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.05, patience=5, verbose=True, threshold=1e-6)

    # Train model
    train_and_eval(args, train_dataset, val_dataset, model, loss_func, optimizer, scheduler)

    best_model = Reg_Model()
    best_model.load_state_dict(torch.load(os.path.join(
        args.save_checkpoint_path, f'best_{args.model_type}_model.pth'
    )))
    final_model = Reg_Model()
    final_model.load_state_dict(torch.load(os.path.join(
        args.save_checkpoint_path, f'final_{args.model_type}_model.pth'
    )))

    # Test Models
    models_scores = test_models(args, test_dataset, loss_func, models = [final_model, best_model])

    # Report results
    # wandb is printing a summary of the results

if __name__ =='__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)
