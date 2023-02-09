import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import torchvision.models as models


class ContentEmbedding(nn.Module):
	def __init__(self, size):
		super(ContentEmbedding, self).__init__()
		self.resnet18 = models.resnet18(pretrained=True)
		self.avgpool = torch.nn.AdaptiveAvgPool2d((size, size))
		self.conv1x1 = nn.Conv2d(512, int(512 / size / size), kernel_size=1, bias=False)
		self.size = size

	def forward(self, x_):
		assert x_.shape[-1] == 256, "[Error]: Inputs for the content_embedding must be 256 x 256."
		x = self.resnet18.conv1(x_)
		x = self.resnet18.bn1(x)
		x = self.resnet18.relu(x)
		x = self.resnet18.maxpool(x)

		x = self.resnet18.layer1(x)
		x = self.resnet18.layer2(x)
		x = self.resnet18.layer3(x)
		x = self.resnet18.layer4[0](x)

		identity = x
		x = self.resnet18.layer4[1].conv1(x)
		x = self.resnet18.layer4[1].bn1(x)
		x = self.resnet18.layer4[1].relu(x)
		x = self.resnet18.layer4[1].conv2(x)
		x = self.resnet18.layer4[1].bn2(x)
		x += identity
		x = self.resnet18.relu(x)

		x = self.avgpool(x)
		x = self.conv1x1(x)

		x = torch.flatten(x, 1)

		return x
