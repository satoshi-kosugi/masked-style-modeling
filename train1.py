import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import sys

from model import LossPienet, StylizedEnhancer, StyleEmbedding
from utils import AverageMeter, reverse_normalize
from dataset.loader import Dataset1


parser = argparse.ArgumentParser()
parser.add_argument('--t_batch_size', default=64, type=int, help='mini batch size for training')
parser.add_argument('--v_batch_size', default=10, type=int, help='mini batch size for validation')
parser.add_argument('--num_workers', default=12, type=int, help='number of workers')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=30, type=int, help='number of training epochs')
parser.add_argument('--eval_freq', default=1, type=int, help='frequency of validation')
parser.add_argument('--save_dir', default='./save_model/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/'	, type=str, help='path to dataset')
args = parser.parse_args()


def train(train_loader, style_embedding, stylized_enhancer, optimizer, loss_pienet):
	losses = AverageMeter()

	style_embedding.train()
	stylized_enhancer.train()

	with tqdm(train_loader, file=sys.stdout) as t:
		for (x, y, x_256, y_256) in t:
			x = x.cuda(non_blocking=True)
			y = y.cuda(non_blocking=True)
			y_256 = y_256.cuda(non_blocking=True)
			x_256 = x_256.cuda(non_blocking=True)


			s = style_embedding(x_256, y_256)
			delta = stylized_enhancer(x, s.reshape(list(s.shape)+[1,1]))

			loss1 = loss_pienet(x, delta, y)

			losses.update(loss1.item(), args.t_batch_size)

			optimizer.zero_grad()
			loss1.backward()
			optimizer.step()

			mse = F.mse_loss(reverse_normalize(x+delta), reverse_normalize(y), reduction='none').mean((1, 2, 3))
			psnr = 10 * torch.log10(1 / mse).mean()

			t.postfix = "PSNR: " + str(psnr.item())[:6]
			t.update()

	return losses.avg


def valid(val_loader, style_embedding, stylized_enhancer):
	PSNR = AverageMeter()

	torch.cuda.empty_cache()

	style_embedding.eval()
	stylized_enhancer.eval()

	for (x, y, x_256, y_256) in tqdm(val_loader, file=sys.stdout):
		x = x.cuda(non_blocking=True)
		y = y.cuda(non_blocking=True)
		x_256 = x_256.cuda(non_blocking=True)
		y_256 = y_256.cuda(non_blocking=True)

		with torch.no_grad():
			s = style_embedding(x_256, y_256)
			delta = stylized_enhancer(x, s.reshape(list(s.shape)+[1,1]))

		predicted_y = delta + x
		mse = F.mse_loss(reverse_normalize(predicted_y), reverse_normalize(y), reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse).mean()
		PSNR.update(psnr.item(), args.v_batch_size)

	return PSNR.avg


if __name__ == '__main__':
	style_embedding = StyleEmbedding()
	style_embedding = nn.DataParallel(style_embedding).cuda()

	stylized_enhancer = StylizedEnhancer()
	stylized_enhancer = nn.DataParallel(stylized_enhancer).cuda()

	loss_pienet = LossPienet(args)

	train_dataset = Dataset1('train', True, args)
	train_loader = DataLoader(train_dataset,
                              batch_size=args.t_batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

	valid_dataset = Dataset1('valid', False, args)
	valid_loader = DataLoader(valid_dataset,
                            batch_size=args.v_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)

	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)

	if not os.path.exists(os.path.join(args.save_dir, 'stylized_enhancer.pth.tar')):
		optimizer = torch.optim.Adam([
      		{'params': stylized_enhancer.parameters(), 'lr': args.lr},
      		{'params': style_embedding.parameters(), 'lr': args.lr}
        ])

		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

		best_psnr = 0
		for epoch in range(args.epochs + 1):
			print("Epoch: ", epoch, "/", args.epochs)
			loss = train(train_loader, style_embedding, stylized_enhancer, optimizer, loss_pienet)
			print('Train [{0}]\t'
			      'Loss: {loss:.4f}\t '
				  'Best Val PSNR: {psnr:.2f}'.format(epoch, loss=loss, psnr=best_psnr))

			scheduler.step()

			if epoch % args.eval_freq == 0:
				avg_psnr = valid(valid_loader, style_embedding, stylized_enhancer)
				print('Valid: [{0}]\tPSNR: {psnr:.2f}'.format(epoch, psnr=avg_psnr))

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'state_dict': stylized_enhancer.state_dict()},
							   os.path.join(args.save_dir, 'stylized_enhancer.pth.tar'))
					torch.save({'state_dict': style_embedding.state_dict()},
							   os.path.join(args.save_dir, 'style_embedding.pth.tar'))

	else:
		print('==> Existing trained model')
		exit(1)
