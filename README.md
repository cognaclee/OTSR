# OTSR
Blind Super-Resolution under the guidance of Optimal Transmission

Zezeng Li  , Na Lei  , Hao Xue , Ji Shi

(*Official PyTorch Implementation*)

## Introduction

 Recent state-of-the-art super-resolution methods provide some solutions, but these methods make their model convergent to a specific maps under the constraint of prior knowledge, resulting in severe loss of details when these methods are dealing with low-resolution images in the real world. To solve these problems, we propose OTSR: an image super-resolution method based on the optimal transmission theory.

If you are interested in this work, please cite our [paper](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf)

    @InProceedings{
                   booktitle = {Blind Super-Resolution under the guidance of Optimal Transmission},
                   month = {June},
                   year = {2020}
         }



​    



## Dependencies and Installation

This code is based on [BasicSR](https://github.com/xinntao/BasicSR).

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`

## Pre-trained models

- Models for challenge results
  - [DF2K](https://drive.google.com/open?id=1pWGfSw-UxOkrtbh14GeLQgYnMLdLguOF) for corrupted images with processing noise.
  - [DPED](https://drive.google.com/open?id=1zZIuQSepFlupV103AatoP-JSJpwJFS19) for real images taken by cell phone camera.
- Extended models
  - [DF2K-JPEG](https://drive.google.com/open?id=1w8QbCLM6g-MMVlIhRERtSXrP-Dh7cPhm) for compressed jpeg image. 

## Testing

Download dataset from [NTIRE 2020 RWSR](https://competitions.codalab.org/competitions/22220#participate) and unzip it to your path.

For convenient, we provide [Corrupted-te-x](https://drive.google.com/open?id=1GrLxeE-LruddQoAePV1Z7MFclXdZWHMa) and [DPEDiphone-crop-te-x](https://drive.google.com/open?id=19zlofWRxkhsjf_TuRA2oI9jgozifGvxp).


## Training

