import os
import numpy as np
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    parser.add_argument('-m', '--model', type=str, default='resnet50', help='choose a deep model')
    parser.add_argument('-d', '--data', type=str, help='choose a dataset')
    args = parser.parse_args()
    return args

def main(args):
    pass


if __name__ =='__main__':
    args = parse_args()
    main(args)
