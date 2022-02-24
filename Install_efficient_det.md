# How to install efficientDet pytorch implementation
we use conda to run each of the environments 
## Step 1 - create a new conda environment
```
conda create -n efficientDet python=3.7
conda activate efficientDet
```


## Step 2 - Install cuda
download and install cuda and cudnn for your GPU \
must be at least cuda 10.1 (later versions will also work)

## Step 3 - Install PyTorch
This manual was written on the 19-Oct. 2021 \
Currently, you need PyTorch 1.7 at least so that works yolov5 will work

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Step 4 - clone the efficientDet repository
```
git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git
cd yolov5
```

## Step 5 - Finally, install all requirements
first, verify you have cython so that pycocotools will be installed later
```
pip install cython
```
now install the following
```
pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
```

# That's it - you now have efficientDet-pytorch implementation
However, don't forget to get the models' weights from\
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch  

