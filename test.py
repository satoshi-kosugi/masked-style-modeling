import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import cv2
from pytorch_msssim import ssim

from model import ContentEmbedding, StyleEmbedding, MaskedToken, TransformerEncoder, StylizedEnhancer
from utils import AverageMeter, reverse_normalize, rgb2lab
from dataset.loader import Dataset2Train, Dataset2ValidPref, Dataset2ValidUnseen


parser = argparse.ArgumentParser()
parser.add_argument('--v_batch_size', default=10, type=int, help='mini batch size for validation')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./save_model/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/'	, type=str, help='path to dataset')
parser.add_argument('--spatial_size', default=2, type=int)
parser.add_argument('--save_images', action='store_true')
args = parser.parse_args()


def test(test_unseen_loader, test_preferred_loader,
	transformer_encoder, masked_token, style_embedding, content_embedding, stylized_enhancer,
	style_dirs, num_pref_images):

	PSNR = [AverageMeter() for i in range(len(style_dirs))]
	SSIM = [AverageMeter() for i in range(len(style_dirs))]
	DELTAab = [AverageMeter() for i in range(len(style_dirs))]

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
	preferred_s_set = torch.zeros((len(style_dirs), num_pref_images, 512)).cuda(non_blocking=True)
	preferred_c_set = torch.zeros((len(style_dirs), num_pref_images, 512)).cuda(non_blocking=True)

	with torch.no_grad():
		for (y_256, x_256, style_id, img_name) in tqdm(test_preferred_loader, file=sys.stdout):
			y_256 = y_256.cuda(non_blocking=True)
			x_256 = x_256.cuda(non_blocking=True)

			for i in range(y_256.shape[0]):
				counts[style_id[i], 0] += 1
				preferred_y_256_set[style_id[i], counts[style_id[i], 0].to(torch.int64)-1] = y_256[i]
				preferred_x_256_set[style_id[i], counts[style_id[i], 0].to(torch.int64)-1] = x_256[i]
				preferred_s_set[style_id[i], counts[style_id[i], 0].to(torch.int64)-1] = style_embedding(x_256[i:i+1], y_256[i:i+1])[0]
				preferred_c_set[style_id[i], counts[style_id[i], 0].to(torch.int64)-1] = content_embedding(x_256[i:i+1])[0]

				if args.save_images:
					os.symlink("../../../"+img_name[i],
						"test_results/{0}/{1}/preferred/{2}".format(style_dirs[style_id[i].cpu()], num_pref_images, os.path.basename(img_name[i])))

		if args.save_images:
			for i in range(len(style_dirs)):
				preferred_image_matrix = np.zeros((128*(num_pref_images//5), 128*5, 3), dtype=np.uint8)
				for j in range(num_pref_images):
					preferred_y_256_np = np.array(reverse_normalize(preferred_y_256_set[i, j]).cpu())[0]
					preferred_y_256_np = np.clip(preferred_y_256_np.transpose((1,2,0)) * 255, 0, 255).astype(np.uint8)[:,:,::-1]
					preferred_image_matrix[128*(j//5):128*(j//5+1), 128*(j%5):128*(j%5+1)] = cv2.resize(preferred_y_256_np, (128, 128))
				cv2.imwrite("test_results/{0}/{1}/preferred/references.jpg".format(style_dirs[i], num_pref_images), preferred_image_matrix)

		for (x, y, x_256, style_id, original_shape, img_name) in tqdm(test_unseen_loader, file=sys.stdout):
			x = x.cuda(non_blocking=True)
			y = y.cuda(non_blocking=True)
			style_id = style_id.cuda(non_blocking=True)

			preferred_c = preferred_c_set[style_id.argmax(dim=1)]
			preferred_s = preferred_s_set[style_id.argmax(dim=1)]
			batch_size = preferred_c.shape[0]
			num_pref_images = preferred_c.shape[1]

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

			predicted_y_rgb = reverse_normalize(predicted_y)
			y_rgb = reverse_normalize(y)
			predicted_y_lab = rgb2lab(predicted_y_rgb)
			y_lab = rgb2lab(y_rgb)

			mse = F.mse_loss(predicted_y_rgb, y_rgb, reduction='none').mean((1, 2, 3))
			for i in range(batch_size):
				psnr = 10 * torch.log10(1 / mse[i])
				PSNR[style_id[i].argmax().item()].update(psnr.item(), 1)
				DELTAab[style_id[i].argmax().item()].update((((predicted_y_lab[i] - y_lab[i]) ** 2).sum(dim=0)**0.5).mean().item(), 1)

				_, _, H, W = y_rgb.size()
				down_ratio = max(1, round(min(H, W) / 256))
				ssim_score = ssim(F.adaptive_avg_pool2d(predicted_y_rgb[i:i+1], (int(H / down_ratio), int(W / down_ratio))),
								F.adaptive_avg_pool2d(y_rgb[i:i+1], (int(H / down_ratio), int(W / down_ratio))),
								data_range=1, size_average=False).item()
				SSIM[style_id[i].argmax().item()].update(ssim_score, 1)

			if args.save_images:
				predicted_y_np = np.array(predicted_y_rgb.cpu())
				y_np = np.array(reverse_normalize(y).cpu())
				x_np = np.array(reverse_normalize(x).cpu())
				for i in range(batch_size):
					predicted_y_np_i = np.clip(predicted_y_np[i].transpose((1,2,0)) * 255, 0, 255).astype(np.uint8)[:,:,::-1]
					predicted_y_np_i = cv2.resize(predicted_y_np_i, original_shape[i][:2].tolist()[::-1], interpolation=cv2.INTER_LINEAR)
					y_np_i = np.clip(y_np[i].transpose((1,2,0)) * 255, 0, 255).astype(np.uint8)[:,:,::-1]
					y_np_i = cv2.resize(y_np_i, original_shape[i][:2].tolist()[::-1], interpolation=cv2.INTER_LINEAR)
					x_np_i = np.clip(x_np[i].transpose((1,2,0)) * 255, 0, 255).astype(np.uint8)[:,:,::-1]
					x_np_i = cv2.resize(x_np_i, original_shape[i][:2].tolist()[::-1], interpolation=cv2.INTER_LINEAR)

					output_name = "test_results/{0}/{1}/output/{2}".format(style_dirs[style_id[i].argmax().cpu()], num_pref_images, img_name[i]).replace(".tif", ".jpg")
					cv2.imwrite(output_name, np.hstack([x_np_i, predicted_y_np_i, y_np_i]))

	print("PSNR:{0:.2f} SSIM:{1:.3f} DELTAab:{2:.2f}".format(np.array(list(map(lambda x: x.avg, PSNR))).mean(),
												np.array(list(map(lambda x: x.avg, SSIM))).mean(),
												np.array(list(map(lambda x: x.avg, DELTAab))).mean()))
	return list(map(lambda x: x.avg, PSNR)), list(map(lambda x: x.avg, SSIM)), list(map(lambda x: x.avg, DELTAab))


if __name__ == '__main__':
	transformer_encoder = TransformerEncoder()
	transformer_encoder = nn.DataParallel(transformer_encoder).cuda()
	transformer_encoder.load_state_dict(torch.load(os.path.join(args.save_dir, 'transformer_encoder.pth.tar'))['state_dict'])

	masked_token = MaskedToken()
	masked_token = nn.DataParallel(masked_token).cuda()
	masked_token.load_state_dict(torch.load(os.path.join(args.save_dir, 'masked_token.pth.tar'))['state_dict'])

	style_embedding = StyleEmbedding()
	style_embedding = nn.DataParallel(style_embedding).cuda()
	style_embedding.load_state_dict(torch.load(os.path.join(args.save_dir, 'style_embedding.pth.tar'))['state_dict'])

	content_embedding = ContentEmbedding(args.spatial_size)
	content_embedding = nn.DataParallel(content_embedding).cuda()
	content_embedding.load_state_dict(torch.load(os.path.join(args.save_dir, 'content_embedding.pth.tar'))['state_dict'])

	stylized_enhancer = StylizedEnhancer()
	stylized_enhancer = nn.DataParallel(stylized_enhancer).cuda()
	stylized_enhancer.load_state_dict(torch.load(os.path.join(args.save_dir, 'stylized_enhancer.pth.tar'))['state_dict'])


	test_dataset = Dataset2ValidUnseen('test', args)
	style_dirs = list(map(lambda x: os.path.basename(x), test_dataset.style_dirs))

	if not args.save_images:
		num_pref_images_list = [20, 50, 100]
	else:
		num_pref_images_list = [20]

	PSNRresults = np.zeros((10, len(style_dirs), len(num_pref_images_list)))
	SSIMresults = np.zeros((10, len(style_dirs), len(num_pref_images_list)))
	DELTAabresults = np.zeros((10, len(style_dirs), len(num_pref_images_list)))

	for k, num_pref_images in enumerate(num_pref_images_list):
		for i in (range(10) if not args.save_images else [2]):
			print("[Number of preferred images: {0}] {1} / 10".format(num_pref_images, i))
			test_ref_dataset = Dataset2ValidPref('test_ref', args, num_pref_images, start_idx=i)
			test_preferred_loader = DataLoader(test_ref_dataset,
		                            batch_size=args.v_batch_size,
		                            num_workers=5,
		                            pin_memory=True)

			test_dataset = Dataset2ValidUnseen('test', args)
			test_unseen_loader = DataLoader(test_dataset,
		                            batch_size=args.v_batch_size,
		                            num_workers=5,
		                            pin_memory=True)

			if args.save_images:
				for style_dir in style_dirs:
					os.makedirs("test_results/{0}/{1}/preferred/".format(style_dir, num_pref_images))
					os.makedirs("test_results/{0}/{1}/output/".format(style_dir, num_pref_images))

			PSNR, SSIM, DELTAab = test(test_unseen_loader, test_preferred_loader,
				transformer_encoder, masked_token, style_embedding, content_embedding, stylized_enhancer,
				style_dirs, num_pref_images)

			for j, dir in enumerate(style_dirs):
				PSNRresults[i, j, k] = PSNR[j]
				SSIMresults[i, j, k] = SSIM[j]
				DELTAabresults[i, j, k] = DELTAab[j]

	if args.save_images:
		print("Results are saved in test_results/")
	else:
		print("[Results]")
		for k, num_pref_images in enumerate(num_pref_images_list):
			print("[Number of preferred images: {0}]".format(num_pref_images))
			print("PSNR: {0:.2f}@{1:.2f}".format(PSNRresults[:,:,k].mean(axis=1).mean(), PSNRresults[:,:,k].mean(axis=1).std()))
			print("SSIM: {0:.3f}±{1:.3f}".format(SSIMresults[:,:,k].mean(axis=1).mean(), SSIMresults[:,:,k].mean(axis=1).std()))
			print("DELTAab: {0:.2f}±{1:.2f}".format(DELTAabresults[:,:,k].mean(axis=1).mean(), DELTAabresults[:,:,k].mean(axis=1).std()))
