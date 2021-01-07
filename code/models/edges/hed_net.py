#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

class HedNet(torch.nn.Module):
	def __init__(self):
		super(HedNet, self).__init__()

		self.netVggOne = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggTwo = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggThr = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggFou = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggFiv = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

		self.netCombine = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			torch.nn.Sigmoid()
		)

		weights = torch.load('./models/edges/network-bsds500.pytorch')
		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in weights.items()})

	def forward(self, tenInput):
		tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
		tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
		tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434

		tenInput = torch.cat([ tenBlue, tenGreen, tenRed ], 1)

		tenVggOne = self.netVggOne(tenInput)
		tenVggTwo = self.netVggTwo(tenVggOne)
		tenVggThr = self.netVggThr(tenVggTwo)
		tenVggFou = self.netVggFou(tenVggThr)
		tenVggFiv = self.netVggFiv(tenVggFou)

		tenScoreOne = self.netScoreOne(tenVggOne)
		tenScoreTwo = self.netScoreTwo(tenVggTwo)
		tenScoreThr = self.netScoreThr(tenVggThr)
		tenScoreFou = self.netScoreFou(tenVggFou)
		tenScoreFiv = self.netScoreFiv(tenVggFiv)

		tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

		return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))


def estimate(netNetwork, tenInput):

	intWidth = tenInput.shape[2]
	intHeight = tenInput.shape[1]

	assert(intWidth == 480)
	assert(intHeight == 320)

	return netNetwork(tenInput.cuda().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
