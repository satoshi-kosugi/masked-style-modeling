import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
from sklearn.decomposition import PCA

from model import ContentEmbedding, StyleEmbedding, MaskedToken, TransformerEncoder, StylizedEnhancer
from utils import AverageMeter, reverse_normalize
from dataset.loader import Dataset1, Dataset2Train, Dataset2ValidPref, Dataset2ValidUnseen


parser = argparse.ArgumentParser()
parser.add_argument('--t_batch_size', default=32, type=int, help='mini batch size for training')
parser.add_argument('--v_batch_size', default=50, type=int, help='mini batch size for validation')
parser.add_argument('--num_workers', default=12, type=int, help='number of workers')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=30, type=int, help='number of training epochs')
parser.add_argument('--eval_freq', default=1, type=int, help='frequency of validation')
parser.add_argument('--save_dir', default='./save_model/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/'	, type=str, help='path to dataset')
parser.add_argument('--num_pref', default=10, type=int)
parser.add_argument('--spatial_size', default=2, type=int)
args = parser.parse_args()


def train(train_loader, transformer_encoder, masked_token, content_embedding, style_embedding, criterion, optimizer):
	losses = AverageMeter()

	transformer_encoder.train()
	masked_token.train()
	content_embedding.train()
	style_embedding.eval()

	pca_components = np.load(os.path.join(args.save_dir, "pca_components.npy"))
	pca_mean = np.load(os.path.join(args.save_dir, "pca_mean.npy"))
	pca_components = torch.from_numpy(pca_components.astype('float32')).cuda(non_blocking=True)
	pca_std = (pca_components**2).sum(dim=1)**0.5
	pca_components = pca_components / pca_std[:,None]
	pca_mean = torch.from_numpy(pca_mean.astype('float32')).cuda(non_blocking=True)

	with tqdm(train_loader, file=sys.stdout) as t:
		for (x_256, y_256, preferred_x_256, preferred_y_256, padding_mask) in t:
			x_256 = x_256.cuda(non_blocking=True)
			y_256 = y_256.cuda(non_blocking=True)
			preferred_x_256 = preferred_x_256.cuda(non_blocking=True)
			preferred_y_256 = preferred_y_256.cuda(non_blocking=True)
			padding_mask = padding_mask.cuda(non_blocking=True).to(torch.uint8)

			batch_size = preferred_x_256.shape[0]
			num_pref_images = preferred_x_256.shape[1]

			with torch.no_grad():
				s = style_embedding(x_256, y_256)
				preferred_s = style_embedding(preferred_x_256.reshape([-1, 3, 256, 256]),
												preferred_y_256.reshape([-1, 3, 256, 256]))
			preferred_s = preferred_s.reshape((batch_size, num_pref_images, preferred_s.shape[-1]))
			preferred_c = content_embedding(preferred_x_256.reshape([-1, 3, 256, 256]))
			preferred_c = preferred_c.reshape((batch_size, num_pref_images, preferred_c.shape[-1]))
			c = content_embedding(x_256)

			concat_s = torch.cat([preferred_s, masked_token(torch.ones_like(c[:,0:1]))[:, None]], dim=1)
			concat_c = torch.cat([preferred_c, c[:, None]], dim=1)
			A = torch.cat([torch.zeros_like(concat_s), torch.zeros_like(concat_s)], dim=2)
			A[:,:,::2] = concat_s
			A[:,:,1::2] = concat_c

			predicted_s = transformer_encoder(A, src_key_padding_mask=padding_mask)

			loss2 = criterion(predicted_s * pca_std[None],
				torch.mm(s - pca_mean[None], pca_components.permute(1,0)))
			losses.update(loss2.item(), args.t_batch_size)

			optimizer.zero_grad()
			loss2.backward()
			optimizer.step()

			t.postfix = "loss2: " + str(loss2.item())[:6]
			t.update()

	return losses.avg


def valid(valid_unseen_loader, valid_preferred_loader,
	transformer_encoder, masked_token, content_embedding, style_embedding, stylized_enhancer,
	style_dirs, num_pref_images):

	PSNR = [AverageMeter() for i in range(len(style_dirs))]

	torch.cuda.empty_cache()

	transformer_encoder.eval()
	masked_token.eval()
	content_embedding.eval()
	style_embedding.eval()
	stylized_enhancer.eval()

	pca_components = np.load(os.path.join(args.save_dir, "pca_components.npy"))
	pca_mean = np.load(os.path.join(args.save_dir, "pca_mean.npy"))
	pca_components = torch.from_numpy(pca_components.astype('float32')).cuda(non_blocking=True)
	pca_std = (pca_components**2).sum(dim=1)**0.5
	pca_components = pca_components / pca_std[:,None]
	pca_mean = torch.from_numpy(pca_mean.astype('float32')).cuda(non_blocking=True)

	counts = torch.zeros((len(style_dirs), 1)).cuda(non_blocking=True)

	preferred_y_256_set = torch.zeros((len(style_dirs), num_pref_images, 3, 256, 256)).cuda(non_blocking=True)
	preferred_x_256_set = torch.zeros((len(style_dirs), num_pref_images, 3, 256, 256)).cuda(non_blocking=True)

	with torch.no_grad():
		for (y_256, x_256, style_id, img_name) in tqdm(valid_preferred_loader, file=sys.stdout):
			y_256 = y_256.cuda(non_blocking=True)
			x_256 = x_256.cuda(non_blocking=True)

			for i in range(y_256.shape[0]):
				counts[style_id[i], 0] += 1
				preferred_y_256_set[style_id[i], counts[style_id[i], 0].to(torch.int64)-1] = y_256[i]
				preferred_x_256_set[style_id[i], counts[style_id[i], 0].to(torch.int64)-1] = x_256[i]

		for (x, y, x_256, style_id, original_shape, img_name) in tqdm(valid_unseen_loader, file=sys.stdout):
			x = x.cuda(non_blocking=True)
			y = y.cuda(non_blocking=True)
			style_id = style_id.cuda(non_blocking=True)

			preferred_x_256 = preferred_x_256_set[style_id.argmax(dim=1)]
			preferred_y_256 = preferred_y_256_set[style_id.argmax(dim=1)]
			batch_size = preferred_x_256.shape[0]
			num_pref_images = preferred_x_256.shape[1]

			preferred_s = style_embedding(preferred_x_256.reshape([-1, 3, 256, 256]),
											preferred_y_256.reshape([-1, 3, 256, 256]))
			preferred_s = preferred_s.reshape((batch_size, num_pref_images, preferred_s.shape[-1]))
			preferred_c = content_embedding(preferred_x_256.reshape([-1, 3, 256, 256]))
			preferred_c = preferred_c.reshape((batch_size, num_pref_images, preferred_c.shape[-1]))
			c = content_embedding(x_256)

			concat_s = torch.cat([preferred_s, masked_token(torch.ones_like(c[:,0:1]))[:, None]], dim=1)
			concat_c = torch.cat([preferred_c, c[:, None]], dim=1)
			A = torch.cat([torch.zeros_like(concat_s), torch.zeros_like(concat_s)], dim=2)
			A[:,:,::2] = concat_s
			A[:,:,1::2] = concat_c

			predicted_s = transformer_encoder(A)
			predicted_s = torch.mm(predicted_s * pca_std[None], pca_components) + pca_mean[None]

			delta = stylized_enhancer(x, predicted_s.reshape(list(predicted_s.shape)+[1,1]))
			predicted_y = delta + x

			mse = F.mse_loss(reverse_normalize(predicted_y), reverse_normalize(y), reduction='none').mean((1, 2, 3))
			for i in range(mse.shape[0]):
				psnr = 10 * torch.log10(1 / mse[i])
				PSNR[style_id[i].argmax().item()].update(psnr.item(), 1)

	return np.array(list(map(lambda x: x.avg, PSNR))).mean()

def calcPCA(train_loader_PCA, style_embedding):
	style_embedding.eval()
	s_set = np.zeros((0, 512))
	with torch.no_grad():
		for (x, y, x_256, y_256) in tqdm(train_loader_PCA, file=sys.stdout):
			y_256 = y_256.cuda(non_blocking=True)
			s = style_embedding(x_256, y_256)
			s_set = np.concatenate([s_set, np.array(s.cpu())])

	pca = PCA(n_components=512)
	features_pca = pca.fit_transform(s_set)
	pca_components = pca.components_ * (pca.explained_variance_**0.5)[:,None]

	np.save(os.path.join(args.save_dir, "pca_components.npy"), pca_components)
	np.save(os.path.join(args.save_dir, "pca_mean.npy"), pca.mean_)

if __name__ == '__main__':
	transformer_encoder = TransformerEncoder()
	transformer_encoder = nn.DataParallel(transformer_encoder).cuda()

	masked_token = MaskedToken()
	masked_token = nn.DataParallel(masked_token).cuda()

	style_embedding = StyleEmbedding()
	style_embedding = nn.DataParallel(style_embedding).cuda()
	style_embedding.load_state_dict(torch.load(os.path.join(args.save_dir, 'style_embedding.pth.tar'))['state_dict'])

	content_embedding = ContentEmbedding(args.spatial_size)
	content_embedding = nn.DataParallel(content_embedding).cuda()

	stylized_enhancer = StylizedEnhancer()
	stylized_enhancer = nn.DataParallel(stylized_enhancer).cuda()
	stylized_enhancer.load_state_dict(torch.load(os.path.join(args.save_dir, 'stylized_enhancer.pth.tar'))['state_dict'])

	criterion = torch.nn.L1Loss()

	train_dataset = Dataset2Train('train', args)
	train_loader = DataLoader(train_dataset,
                              batch_size=args.t_batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

	valid_num_pref_images = 20
	valid_pref_dataset = Dataset2ValidPref('valid_ref', args, valid_num_pref_images, 0)
	valid_pref_loader = DataLoader(valid_pref_dataset,
                            batch_size=args.v_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)
	valid_unseen_dataset = Dataset2ValidUnseen('valid', args)
	valid_unseen_loader = DataLoader(valid_unseen_dataset,
                            batch_size=args.v_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)
	valid_style_dirs = list(map(lambda x: os.path.basename(x), valid_unseen_dataset.style_dirs))

	if not os.path.exists(os.path.join(args.save_dir, 'pca_components.npy')):
		train_dataset_PCA = Dataset1('train', True, args)
		train_loader_PCA = DataLoader(train_dataset_PCA,
	                              batch_size=args.t_batch_size,
	                              shuffle=True,
	                              num_workers=args.num_workers,
	                              pin_memory=True,
	                              drop_last=True)
		calcPCA(train_loader_PCA, style_embedding)

	if not os.path.exists(os.path.join(args.save_dir, 'transformer_encoder.pth.tar')):
		optimizer = torch.optim.Adam([
      		{'params': transformer_encoder.parameters(), 'lr': args.lr},
      		{'params': masked_token.parameters(), 'lr': args.lr},
      		{'params': content_embedding.parameters(), 'lr': args.lr},
		])
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

		best_psnr = 0
		for epoch in range(args.epochs + 1):
			print("Epoch: ", epoch, "/", args.epochs)
			loss = train(train_loader, transformer_encoder, masked_token, content_embedding, style_embedding, criterion, optimizer)
			print('Train [{0}]\t'
			      'Loss: {loss:.4f}\t '
				  'Best Val PSNR: {psnr:.2f}'.format(epoch, loss=loss, psnr=best_psnr))

			scheduler.step()

			if epoch % args.eval_freq == 0:
				valid_psnr = valid(valid_unseen_loader, valid_pref_loader,
					transformer_encoder, masked_token, content_embedding, style_embedding, stylized_enhancer,
					valid_style_dirs, valid_num_pref_images)

				if valid_psnr > best_psnr:
					best_psnr = valid_psnr
					torch.save({'state_dict': transformer_encoder.state_dict()},
							os.path.join(args.save_dir, 'transformer_encoder.pth.tar'))
					torch.save({'state_dict': masked_token.state_dict()},
							os.path.join(args.save_dir, 'masked_token.pth.tar'))
					torch.save({'state_dict': content_embedding.state_dict()},
							os.path.join(args.save_dir, 'content_embedding.pth.tar'))

				print('Valid: [{0}]\tPSNR: {psnr:.2f}'.format(epoch, psnr=valid_psnr))

	else:
		print('==> Existing trained model')
		exit(1)
