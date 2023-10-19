# SRResNet: A Super-Resolution Residual Network for seismic signal denoising based on PyTorch

**Copyright (c) 2023 Zhengjie Zhang (zhangzhengjie@mail.ustc.edu.cn)**  

We would like to thank Dr. H. Wang for her inspiration and help in this work.

- This is the core code of the project **Traffic seismic data denoising based on machine learning**.  
- We do not share the data for other purposes.
- **Note** that the use of `UNet.py` in this package still needs to be optimized, and we are still debugging and modifying it.   


## Network Architecture
![image](https://github.com/zhangzj1209/SRResNet/blob/main/figure/SRResNet_architecture.png)

## Installation

### Via Anaconda (recommended):
```
conda create -n SRResNet python=3.8
conda activate SRResNet
conda install numpy==1.23.4 matplotlib==3.6.3 obspy==1.3.0 torch==1.10.1 scikit-learn==1.2.2 torchsummary==1.5.1 pandas==1.5.3
```

### Clone source codes
set your working directory at `/data/`
```
cd /data/
git clone https://github.com/zhangzj1209/SRResNet.git
unzip SRResNet.zip
cd SRResNet/
```

## Description 
Please create several folders in path `/data/SRResNet/`   
```
mkdir -r data           ! used to store training data and validation data
mkdir -r label          ! used to store training label and validation label
mkdir -r save           ! used to store train model and predict result
mkdir -r predict_data   ! used to store prediction data
mkdir -r predict_label  ! used to store prediction label
```

- If you want to use this network to do your work, please modify the contents of `My_Dataset` in `dataset.py`.
- The number of residual block layers of the network can also be modified in **line 29** of `SRResNet.py`.
