import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import torchvision.models as models


class StylizedEnhancer(nn.Module):
	def __init__(self):
		super(StylizedEnhancer, self).__init__()
		resnet18 = models.resnet18(pretrained=True)
		self.down0 = nn.Sequential(resnet18.conv1, resnet18.bn1, resnet18.relu)
		self.down1 = nn.Sequential(resnet18.maxpool, resnet18.layer1)
		self.down2 = resnet18.layer2
		self.down3 = resnet18.layer3
		self.down4 = nn.Sequential(resnet18.layer4, resnet18.avgpool)

		self.up1 = PUB(512, 256, 32)
		self.up2 = PUB(256, 128, 64)
		self.up3 = PUB(128, 64, 128)
		self.up4 = PUB(64, 64, 256, useresblock=False)
		self.up5 = nn.Upsample(scale_factor=2, mode='bilinear')
		self.up6 = conv3x3(64*3, 3)

	def forward(self, x, preference):
		x0 = self.down0(x)
		x1 = self.down1(x0)
		x2 = self.down2(x1)
		x3 = self.down3(x2)
		x4 = self.down4(x3)

		x5 = self.up1(x4, x3, preference)
		x6 = self.up2(x5, x2, preference)
		x7 = self.up3(x6, x1, preference)
		x8 = self.up4(x7, x0, preference)
		delta = self.up6(self.up5(x8))
		return delta


class PUB(nn.Module):
	def __init__(self, inplanes, planes, size, useresblock=True):
		super(PUB, self).__init__()
		if size == 32:
			self.upsample1 = nn.Upsample(scale_factor=32, mode='bilinear')
		else:
			self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
		self.upsample2 = nn.Upsample(scale_factor=size, mode='bilinear')

		self.conv1 = ConvBlock(inplanes, planes)
		self.conv2 = ConvBlock(512, planes, False, kernel_size=1)

		if useresblock:
			self.resblock = ResBlk(planes*3, planes)
		self.useresblock = useresblock

	def forward(self, x, skip, preference):
		x = self.conv1(self.upsample1(x))
		preference = self.upsample2(self.conv2(preference))
		if self.useresblock:
			return self.resblock(torch.cat([x, skip, preference], dim=1))
		else:
			return torch.cat([x, skip, preference], dim=1)


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlk(nn.Module):
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(ResBlk, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

		if inplanes != planes:
			self.conv3 = conv1x1(inplanes, planes)
			self.bn3 = nn.BatchNorm2d(planes)
			self.downsample = True

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample:
		    identity = self.bn3(self.conv3(x))

		out += identity
		out = self.relu(out)

		return out


class ConvBlock(nn.Module):
	def __init__(self, inplanes, planes, usebatchnorm=True, kernel_size=3):
		super(ConvBlock, self).__init__()
		if kernel_size == 3:
			self.conv = conv3x3(inplanes, planes)
		elif kernel_size == 1:
			self.conv = conv1x1(inplanes, planes)
		if usebatchnorm:
			self.batchnorm = nn.BatchNorm2d(planes)
		self.usebatchnorm = usebatchnorm

	def forward(self, x):
		x = self.conv(x)
		if self.usebatchnorm:
			x = self.batchnorm(x)
		return F.relu(x)
