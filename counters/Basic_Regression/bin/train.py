import os
import numpy as np
import torch
import argparse
from counters.Basic_Regression.bin.train_loop import train_and_eval
from counters.Basic_Regression.bin.test_loop import test_models
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    parser.add_argument('-m', '--model_type', type=str, default='resnet50', help='choose a deep model')
    parser.add_argument('-d', '--data', type=str, help='choose a dataset')
    parser.add_argument('-si', '--split_images', type=str, help='should we split the images into tiles')
    args = parser.parse_args()
    return args

def create_dataset(args, train_dataset_params, test_dataset_params):
    train_dataset = {}
    # train dataset loader
    train_dataset['train'] = DataLoader(
        Reg_Agri_Dataset(args.data_path, 'train', args,
                        transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(20),
                            transforms.RandomPerspective(distortion_scale=0.2),
                            transforms.Resize([args.input_size, args.input_size]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
        **train_dataset_params)

    # validation dataset loader
    train_dataset['validation'] = DataLoader(
        FacesDataReader(args.data_path, 'validation', args,
                        transform=transforms.Compose([
                            transforms.Resize([args.input_size, args.input_size]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
        **test_dataset_params)

    # test dataset loader
    test_dataset = DataLoader(
        FacesDataReader(args.data_path, 'test', args,
                        transform=transforms.Compose([
                            transforms.Resize([args.input_size, args.input_size]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
        **test_dataset_params)
    return train_dataset, test_dataset


def main(args):
    # --------------------------- Don't edit --------------------------- #
    # define device (use cuda if available)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # torch and numpy reproducibility setup
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(10)
    np.random.seed(10)

    # setting up path to save pretrained models
    torch.hub.set_dir(os.path.join(args.ROOT_DIR, 'Trained_Models', 'pretrained'))

    # Create data loaders
    if args.model_type == 'resnet50':
        args.input_size = 224
    train_dataset_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}
    test_dataset_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0}

    train_dataset, test_dataset = create_dataset(args, train_dataset_params, test_dataset_params)



    # Create model
    if args.model_type == 'resnet50':
        from counters.Basic_Regression.models.ResNet50_reg import ResNet_50_regressor as Reg_Model
    model = Reg_Model()

    # Train model
    final_model, best_model = train_and_eval(args, train_dataset, model)

    # Test Model
    final_scores, best_scores = test_models(args, test_dataset, models = [final_model, best_model])

    # Report results


if __name__ =='__main__':
    args = parse_args()
    args.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    main(args)
