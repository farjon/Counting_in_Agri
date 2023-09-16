# Counting in Agriculture

This repository contains an implementation of counting algorithms for the agriculture domain.

## Repository Structure

The repository is structured as follows:


## Datasets


|-- data\
| |-- coco\
| |-- train\
| | |-- animals\
| | |-- vehicles\
| | |-- landscapes\
| |-- test\
| | |-- animals\
| | |-- vehicles\
| | |-- landscapes\
|-- src\
| |-- model.py\
| |-- preprocess.py\
| |-- evaluate.py\
|-- notebooks\
| |-- example.ipynb\
|-- README.md

* The training pipes assume the data is already tiled into train-val-test datasets
* If the images should be tiled into tiles, the first run will split it 


## Detection based counting

### yolov5
To train the yolov5, two .yaml files should be prepard
1. a 'yolo5_i_Dataset.yaml' where i = {s,m,l,x}, and 'Dataset' is your dataset name
2. a 'Dataset.yaml' where 'Dataset' is your dataset name

They should contain the information about your dataset\
for 'yolo5_i_Dataset.yaml' - it should include n=X where X is the number of classes\
for 'Dataset.yaml' - it should contain the classes' names

### detectron2 repository for Faster-RCNN and RetinaNet




