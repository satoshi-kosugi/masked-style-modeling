import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LossPienet(nn.Module):
	def __init__(self, args):
		super().__init__()

		self.pienet_embedding = PienetEmbedding()
		self.pienet_embedding = nn.DataParallel(self.pienet_embedding).cuda()
		self.pienet_embedding.load_state_dict(torch.load('pretrained_models/pienet_embedding.pth.tar')['state_dict'])

	def forward(self, x, delta, y):
		l1loss = torch.nn.L1Loss()

		color_loss = l1loss(x+delta, y)

		self.pienet_embedding.train()
		output_embed = self.pienet_embedding(nn.functional.interpolate(x+delta, size=(256, 256), mode='bilinear'))
		self.pienet_embedding.eval()
		gt_embed = self.pienet_embedding(nn.functional.interpolate(y, size=(256, 256), mode='bilinear'))
		perceptual_loss = l1loss(output_embed, gt_embed)

		tv_loss = self.calc_tv_loss(x, delta)

		loss = color_loss + perceptual_loss * 0.4 + tv_loss * 0.01

		return loss

	def calc_tv_loss(self, x, delta):
		input_y_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
		input_x_diff = x[:, :, :, 1:] - x[:, :, :, :-1]
		delta_y_diff = delta[:, :, 1:, :] - delta[:, :, :-1, :]
		delta_x_diff = delta[:, :, :, 1:] - delta[:, :, :, :-1]
		wy = torch.exp(-torch.abs(input_y_diff))
		wx = torch.exp(-torch.abs(input_x_diff))

		l1loss = torch.nn.L1Loss()

		return l1loss(delta_y_diff * wy, torch.zeros_like(delta_y_diff)) + \
					l1loss(delta_x_diff * wx, torch.zeros_like(delta_x_diff))


class PienetEmbedding(nn.Module):
	def __init__(self):
		super(PienetEmbedding, self).__init__()
		self.resnet18 = models.resnet18(pretrained=True)

	def forward(self, x_):
		assert x_.shape[-1] == 256, "error"
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

		x = self.resnet18.avgpool(x)
		x = torch.flatten(x, 1)

		x = x / torch.sum(x ** 2, 1, keepdim=True) ** 0.5
		return x
