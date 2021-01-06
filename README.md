## Dependencies and Installation

This code is based on [BasicSR](https://github.com/xinntao/BasicSR)


+ Python 3 (Recommend to use Anaconda)
+ PyTorch >= 1.0
+ NVIDIA GPU + CUDA

## Pre-trained models
+ Models for challenge results
--  [DF2K](https://drive.google.com/open?id=1pWGfSw-UxOkrtbh14GeLQgYnMLdLguOF) for corrupted images with processing noise.
--  [DPED](https://drive.google.com/open?id=1zZIuQSepFlupV103AatoP-JSJpwJFS19) for real images taken by cell phone camera.
+ Extended models
--  [DF2K-JPEG](https://drive.google.com/open?id=1w8QbCLM6g-MMVlIhRERtSXrP-Dh7cPhm) for compressed jpeg image.


## Training code
cd ./codes
### DF2K/DPED dataset
1. Modify the configuration file preprocess/paths.yml

+ python preprocess/creat_validation.py
+ python preprocess/collect_noise.py
+ python preprocess/creat_validation.py
2. prepare dataset and edit options/dped/train_kernel_noise.yml
+ CUDA_VISIBLE_DEVICES=1,2,3  python train.py -opt options/dped/train_kernel_noise.yml 



## test code
cd ./codes
### DF2K/DPED dataset
1. Modify the configuration file options/dped/options/test_dped.yml

+   python test.py -opt options/dped/test_dped.yml 
2. The output images is saved in '../results/'
