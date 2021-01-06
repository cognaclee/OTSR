import torch
import numpy as np
from torch import nn

def edge_conv2d(im,use_cuda):
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)

    
    if use_cuda:
        conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    else:
        conv_op.weight.data = torch.from_numpy(sobel_kernel)
        
    edge_detect = conv_op(im)
    return edge_detect

def edge_extraction(img, use_cuda=True):
    img = img[np.newaxis, :]
    img = torch.Tensor(img)
    edge_detect = edge_conv2d(img, use_cuda)
    return edge_detect
    
def sobel(img, use_cuda=True):
    image_shape = img.shape
    edge_detect = torch.zeros(image_shape)

    for i in range(image_shape[0]):
        edge_detect[i] = edge_conv2d(img[i].reshape((1, 3, image_shape[2], image_shape[3])),use_cuda)
    
    return edge_detect