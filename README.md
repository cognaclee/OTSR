## OTSR
Real-World Super-Resolution under the guidance of Optimal Transmission

(Official PyTorch Implementation)
### Introduction

The essence of super-resolution(SR) is to find a map from low resolution image space to high resolution image space,  which implies that the solution of SR problem is not unique. In the real world, the lack of image pairs makes image super-resolution a trickier blind problem known as Blind-SR. Existing methods are mainly based on prior knowledge to achieve the tradeoff between detail restoration and noise artifact suppression, which makes it impossible to get the optimal mapping in both aspects.  To solve this problem, we propose OTSR: an image super-resolution method based on the optimal transport theory. OTSR aims to find the optimal solution to the ill-posed SR problem, so that the model can not only restore high-frequency detail accurately, but also suppress noise and artifacts well. Extensive experiments show that our method out- performs the state-of-the-art methods in terms of both detail repair and noise artifact suppression.

## Results
<!--![DF2K](https://github.com/cognaclee/OTSR/blob/main/result/DF2K.png)-->
<img src="https://github.com/cognaclee/OTSR/blob/main/result/DF2K.png" width="100%" height="160%">
<img src="https://github.com/cognaclee/OTSR/blob/main/result/DPED.png" width="100%" height="160%">
<!--<img src="https://github.com/cognaclee/OTSR/blob/main/result/DF2K.png" alt="00003" style="zoom:150%;" /> -->

## Dependencies and Installation

This code is based on [Realsr](https://github.com/jixiaozhong/RealSR)


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
+ python preprocess/create_kernel_dataset.py

2. Prepare dataset and edit options/dped/train_kernel_noise.yml or options/df2k/train_bicubic_noise.yml
+ CUDA_VISIBLE_DEVICES=1,2,3  python train.py -opt options/dped/train_kernel_noise.yml or 
+ CUDA_VISIBLE_DEVICES=1,2,3  python train.py -opt options/df2k/train_bicubic_noise.yml 



## Testing code
cd ./codes
### DF2K/DPED dataset
1. Modify the configuration file options/dped/options/test_dped.yml or options/df2k/test_df2k.yml

+ CUDA_VISIBLE_DEVICES=1 python test.py -opt options/dped/test_dped.yml or 
+ CUDA_VISIBLE_DEVICES=1 python test.py -opt options/df2k/test_df2k.yml

2. The output images is saved in '../results/'
