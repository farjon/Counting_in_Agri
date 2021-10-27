#How to install yolov5
we use conda to run each of the environments 
##Step 1 - create a new conda environment
```
conda create -n yolo python=3.7
conda activate yolo
```


##Step 2 - Install cuda
download and install cuda and cudnn for your GPU \
must be at least cuda 10.1 (later versions will also work)

##Step 3 - Install PyTorch
This manual was written on the 19-Oct. 2021 \
Currently, you need PyTorch 1.7 at least so that works yolov5 will work

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

##Step 4 - clone the yolov5 repository
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
```

##Step 5 - Finally, install all requirements
```
pip install -r requirements.txt
```

#That's it - you now have yolov5 

