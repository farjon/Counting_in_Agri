# Counting in Agriculture

## Datasets
All data should be in 'coco' style: \
Dataset_dir\
&nbsp;&nbsp;&nbsp;&nbsp; images\
&nbsp;&nbsp;&nbsp;&nbsp; annotations\
&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp; train &nbsp;&nbsp;&nbsp;&nbsp; |
&nbsp;&nbsp;&nbsp;&nbsp; test &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | 
&nbsp;&nbsp;&nbsp;val 

* The training pipes assume the data is already splitted into train-val-test datasets
* If the images should be splitted into tiles, the first run will split it 


## Detectors

### yolov5
To train the yolov5, two .yaml files should be prepard
1. a 'yolo5_i_Dataset.yaml' where i = {s,m,l,x}, and 'Dataset' is your dataset name
2. a 'Dataset.yaml' where 'Dataset' is your dataset name

They should contain the information about your dataset\
for 'yolo5_i_Dataset.yaml' - it should include n=X where X is the number of classes\
for 'Dataset.yaml' - it should contain the classes' names





