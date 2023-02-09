import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import torchvision.models as models


class StyleEmbedding(nn.Module):
	def __init__(self):
		super(StyleEmbedding, self).__init__()
		self.resnet18 = models.resnet18(pretrained=True)

	def forward(self, x, y):
		batch_size = x.shape[0]
		feature = self.forward_each(self.resnet18, torch.cat([x, y], dim=0))
		feature_x = feature[:batch_size]
		feature_y = feature[batch_size:]
		feature_diff = feature_y - feature_x
		return feature_diff

	def forward_each(self, model, x_):
		assert x_.shape[-1] == 256, "[Error]: Inputs for the style_embedding must be 256 x 256."
		x = model.conv1(x_)
		x = model.bn1(x)
		x = model.relu(x)
		x = model.maxpool(x)

		x = model.layer1(x)
		x = model.layer2(x)
		x = model.layer3(x)
		x = model.layer4[0](x)

		identity = x
		x = model.layer4[1].conv1(x)
		x = model.layer4[1].bn1(x)
		x = model.layer4[1].relu(x)
		x = model.layer4[1].conv2(x)
		x = model.layer4[1].bn2(x)
		x += identity

		x = model.avgpool(x)
		x = torch.flatten(x, 1)

		return x
