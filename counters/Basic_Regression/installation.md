conda create -n regression python=3.8 -y
conda activate regression

conda install pytorch torchvision -c pytorch
conda install tqdm
pip install tensorboard
pip install opencv-python
pip install pandas
pip install pycocotools