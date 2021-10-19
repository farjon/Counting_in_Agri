#How to install detectron2 on windows-10
we use conda to run each of the environments 
##Step 1 - create a new conda environment
```
conda create -n detectron2 python=3.7
conda activate detectron2
```


##Step 2 - Install cuda
download and install cuda and cudnn for your GPU \
must be at least cuda 10.1 (later versions will also work)

##Step 3 - Install PyTorch
This manual was written on the 19-Oct. 2021 \
Currently, PyTorch 1.6 is the latest version that works for detectron2 on windows

```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

##Step 4 - Install Cython and pycocotools
pycocotools has its issues on windows \
but we must have it to run detectron installation
first install cython
```
conda install cython
```
now install pycocotools from the following git
```
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

##Step 5 - Finally, install detectron2
make sure you have git installed on your machine
clone the detectron2-windows version from
```
git clone https://github.com/DGMaxime/detectron2-windows.git
```
This is a fork of the original detectron2 git \
with specific modifications for windows

now install detectron2
```
cd detectron2-windows
pip install -e .
```
don't forget the dot at the end 

#That's it - you now have detectron2 

To test it, make sure you have opencv-python installed and try
```
python tests/test_windows_install.py
```

###if you have a problem with the fvcore library 
if you get the following error
```
cannot import name 'FakeQuantizeBase' from 'torch.quantization'
```
you need to install a previoius version of fvcore library
```
pip install git+https://github.com/facebookresearch/fvcore@4525b814c8bb0f70510e37e68247c958010eb285
```
