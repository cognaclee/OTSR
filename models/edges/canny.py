import torch
from torch.autograd import Variable
from models.edges.net_canny import Net
import cv2
import numpy as np

def canny(img, use_cuda=True):
    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()

    image_shape = img.shape
    edge_detect = torch.zeros(image_shape)

    for i in range(image_shape[0]):
        _, edge_detect[i][0], _,edge_detect[i][1], _ ,edge_detect[i][2] = net(torch.stack([img[i]]).float())

    return edge_detect

def canny_test(img, use_cuda=True):
    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()

    if use_cuda:
        data = Variable(img).cuda()
    else:
        data = Variable(img)

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)

    cv2.imwrite('gradient_magnitude.png',grad_mag.data.cpu().numpy())
    cv2.imwrite('grad_orientation.png', grad_orientation.data.cpu().numpy())
    cv2.imwrite('final.png', (thresholded.data.cpu().numpy() > 0.0).astype(float))
    cv2.imwrite('thresholded.png', early_threshold.data.cpu().numpy())


    return thin_edges, early_threshold,thresholded


