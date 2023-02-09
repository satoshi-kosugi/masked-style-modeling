import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import torchvision.models as models


class TransformerEncoder(nn.Module):
	def __init__(self):
		super(TransformerEncoder, self).__init__()

		self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True), num_layers=6)
		self.fc = nn.Linear(1024, 512)

	def forward(self, A, src_key_padding_mask=None):
		return self.fc(self.transformer_encoder(A, src_key_padding_mask=src_key_padding_mask)[:, -1])
