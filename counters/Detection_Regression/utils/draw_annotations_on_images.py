import os
import cv2
import argparse
from tqdm import tqdm
from detectron2_windows.detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2_windows.detectron2.utils.visualizer import Visualizer
from detectron2_windows.detectron2.data import MetadataCatalog
from detectron2_windows.detectron2.data.catalog import DatasetCatalog
from detectron2_windows.detectron2.data.datasets import register_coco_instances

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='AppleFlowers_split', help='choose a dataset')
    parser.add_argument('-o', '--output_path', type=str, default='', help='path to save images, defualt is NONE')
    args = parser.parse_args()
    return args

def main(args):
    args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data)
    args.output_path = os.path.join(args.data_path, 'images_with_annotations')
    os.makedirs(args.output_path, exist_ok=True)
    # register_coco_instances : name, metadata, json_file, image_root
    register_coco_instances("train", {}, f'{args.data_path}/coco/annotations/instances_train.json', f'{args.data_path}/coco/train')
    register_coco_instances("val", {}, f'{args.data_path}/coco/annotations/instances_val.json', f'{args.data_path}/coco/val')
    register_coco_instances("test", {}, f'{args.data_path}/coco/annotations/instances_test.json', f'{args.data_path}/coco/test')

    sets_to_vis = ['train', 'val', 'test']

    for current_set in sets_to_vis:
        dataset_metadata = MetadataCatalog.get(current_set)
        dataset_dicts = DatasetCatalog.get(current_set)
        if args.output_path != 'NONE':
            set_output_path = os.path.join(args.output_path, current_set)
            os.makedirs(set_output_path, exist_ok=True)
        for i, sample in enumerate(tqdm(dataset_dicts)):
            image_name = sample['file_name'].split('\\')[-1]
            img = cv2.imread(sample["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(sample)
            if args.output_path == 'NONE':
                cv2.imshow('',img)
                cv2.imshow('',vis.get_image()[:, :, ::-1])
                cv2.waitKey(0)
            else:
                cv2.imwrite(os.path.join(set_output_path, image_name), vis.get_image()[:, :, ::-1])

if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)