# OTSR
Blind Super-Resolution under the guidance of Optimal Transmission

(*Official PyTorch Implementation*)

## Introduction

The essence of super-resolution(SR) is to find a map from low resolution image space to high resolution image space, which implies that the solution of SR problem is not unique. In the real world, the lack of image pairs makes image super-resolution a trickier blind problem known as Blind-SR. Existing methods are mainly based on prior knowledge to achieve the tradeoff between detail restoration and noise artifact suppression, which makes it impossible for them to get the optimal mapping in both aspects. To solve this problem, we propose OTSR : an image super-resolution method based on the optimal transport theory. OTSR aims to find the optimal solution to the ill-posed SR problem, so that the model can not only restore high-frequency detail accurately, but also suppress noise and artifacts well. Extensive experiments show that our method out-
performs the state-of-the-art methods in terms of both detail repair and noise artifact suppression.



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

