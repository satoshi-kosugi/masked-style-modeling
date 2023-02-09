import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import torchvision.models as models


class MaskedToken(nn.Module):
	def __init__(self):
		super(MaskedToken, self).__init__()
		self.masked_token = nn.Linear(1, 512, bias=False)

	def forward(self, x):
		assert (x.min() == 1) and (x.max() == 1), "error"
		x = self.masked_token(x)
		x = x / torch.sum(x ** 2, 1, keepdim=True) ** 0.5
		return x
