import os
import random
import numpy as np
import cv2
from glob import glob
import torch

from torch.utils.data import Dataset
from utils import hwc_to_chw, reverse_normalize, normalize


def random_crop(imgs=[], size=(0,0), min_ratio=0.75):
	H, W, _ = imgs[0].shape

	if random.random() < 0.1:
		Hc, Wc = H, W
	elif random.randint(0, 1) == 1:
		randr = random.uniform(min_ratio, 1.0)
		Hc = int(H * randr)
		Wc = int(W * randr)
	else:
		Hc = int(H * random.uniform(min_ratio, 1.0))
		Wc = int(W * random.uniform(min_ratio, 1.0))

	Hs = random.randint(0, H-Hc)
	Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = cv2.resize(imgs[i][min(imgs[i].shape[0]-1, Hs):(Hs+Hc), min(imgs[i].shape[1]-1, Ws):(Ws+Wc), :],
							 size, interpolation=cv2.INTER_LINEAR)

	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=0)
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.transpose(imgs[i], (1, 0, 2))
	return imgs

def random_brightness(image):
	return image * random.uniform(0.5, 1.5)

def color_jitter(image):
	augment_params = np.random.uniform(-1, 1, (6,))
	image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) * 1.
	image_hsv[:,:,1] = image_hsv[:,:,1] * (1 + augment_params[0] * 0.2)
	image_hsv = np.clip(image_hsv, 0, 255).astype(np.uint8)
	image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR) * 1.
	image_bgr = image_bgr * (1 + augment_params[2] * 0.5)
	image_bgr = image_bgr * (1 + augment_params[3:6][None, None] * 0.1)
	image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
	return image_bgr


class Dataset1(Dataset):
	def __init__(self, sub_dir, is_train, args):
		self.is_train = is_train

		if sub_dir == "train":
			self.style_dirs = []
			for style_dir in sorted(glob(os.path.join(args.data_dir, "train", '*')) + \
					glob(os.path.join(args.data_dir, "valid", '*')) + \
					glob(os.path.join(args.data_dir, "valid_ref", '*'))):
				if ("_input" not in style_dir) and (os.path.basename(style_dir)[0] != "."):
					self.style_dirs.append(style_dir)
		elif sub_dir == "valid":
			self.style_dirs = []
			for style_dir in sorted(glob(os.path.join(args.data_dir, "train", '*')) + \
					glob(os.path.join(args.data_dir, "valid", '*')) + \
					glob(os.path.join(args.data_dir, "valid_ref", '*'))):
				if ("_input" not in style_dir) and (os.path.basename(style_dir)[0] != ".") and ("Preset" in style_dir):
					self.style_dirs.append(style_dir)

		self.num_style = len(self.style_dirs)

		self.img_names = []
		for i, style_dir in enumerate(self.style_dirs):
			for img_name in sorted(os.listdir(style_dir)):
				if img_name[0] != ".":
					self.img_names.append([i, img_name])
		self.img_num = len(self.img_names)

		if sub_dir == "valid":
			self.random_aug_params = np.random.uniform(-1, 1, (self.img_num, 7))

		self.sub_dir = sub_dir

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		style_id = self.img_names[idx][0]
		img_name = self.img_names[idx][1]

		if self.is_train:
			x = normalize(random_brightness(cv2.imread(os.path.join(self.style_dirs[style_id]+"_input", img_name))))
		else:
			x = normalize(cv2.imread(os.path.join(self.style_dirs[style_id]+"_input", img_name)))
		y = normalize(cv2.imread(os.path.join(self.style_dirs[style_id], img_name)))

		if self.is_train:
			[x, y] = random_crop([x, y], (512, 512))
			x_256 = cv2.resize(x, (256, 256), interpolation=cv2.INTER_LINEAR)
			y_256 = cv2.resize(y, (256, 256), interpolation=cv2.INTER_LINEAR)
		else:
			x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)
			y = cv2.resize(y, (512, 512), interpolation=cv2.INTER_LINEAR)
			x_256 = cv2.resize(x, (256, 256), interpolation=cv2.INTER_LINEAR)
			y_256 = cv2.resize(y, (256, 256), interpolation=cv2.INTER_LINEAR)

		return hwc_to_chw(x), hwc_to_chw(y), hwc_to_chw(x_256), hwc_to_chw(y_256)


class Dataset2ValidPref(Dataset):
	def __init__(self, sub_dir, args, num_pref_images, start_idx):
		self.sub_dir = sub_dir

		self.style_dirs = []
		for style_dir in sorted(glob(os.path.join(args.data_dir, sub_dir, '*'))):
			if ("_input" not in style_dir) and (os.path.basename(style_dir)[0] != "."):
				self.style_dirs.append(style_dir)
		self.num_style = len(self.style_dirs)

		self.img_names = []
		for i, style_dir in enumerate(self.style_dirs):
			img_names_i = sorted(os.listdir(self.style_dirs[i]))
			if num_pref_images*(start_idx+1) <= len(img_names_i):
				self.img_names.append(img_names_i[num_pref_images*start_idx:num_pref_images*(start_idx+1)])
			else:
				self.img_names = random.choices(img_names_i, k=num_pref_images)

		self.num_pref_images = num_pref_images

	def __len__(self):
		l = self.num_pref_images * self.num_style
		return l

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		style_id = idx // self.num_pref_images
		img_id = idx % self.num_pref_images

		image_name = self.img_names[style_id][img_id]

		y = normalize(cv2.imread(os.path.join(self.style_dirs[style_id], image_name)))
		y_256 = cv2.resize(y, (256, 256), interpolation=cv2.INTER_LINEAR)
		x = normalize(cv2.imread(os.path.join(self.style_dirs[style_id]+"_input", image_name)))
		x_256 = cv2.resize(x, (256, 256), interpolation=cv2.INTER_LINEAR)

		return hwc_to_chw(y_256), hwc_to_chw(x_256), style_id, os.path.join(self.style_dirs[style_id], image_name)


class Dataset2ValidUnseen(Dataset):
	def __init__(self, sub_dir, args):

		self.style_dirs = []
		for style_dir in sorted(glob(os.path.join(args.data_dir, sub_dir, '*'))):
			if ("_input" not in style_dir) and (os.path.basename(style_dir)[0] != "."):
				self.style_dirs.append(style_dir)
		self.style_num = len(self.style_dirs)

		self.img_names = []
		self.style_img_names = []
		for i, style_dir in enumerate(self.style_dirs):
			self.style_img_names.append([])
			for img_name in sorted(os.listdir(style_dir)):
				if img_name[0] != ".":
					self.img_names.append([i, img_name])
					self.style_img_names[i].append(img_name)
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		style_id = self.img_names[idx][0]
		img_name = self.img_names[idx][1]

		x = normalize(cv2.imread(os.path.join(self.style_dirs[style_id]+"_input", img_name)))
		y = normalize(cv2.imread(os.path.join(self.style_dirs[style_id], img_name)))

		original_shape = x.shape

		x_256 = cv2.resize(x, (256, 256), interpolation=cv2.INTER_LINEAR)
		y_256 = cv2.resize(y, (256, 256), interpolation=cv2.INTER_LINEAR)
		x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)
		y = cv2.resize(y, (512, 512), interpolation=cv2.INTER_LINEAR)

		return hwc_to_chw(x), hwc_to_chw(y), hwc_to_chw(x_256), np.identity(self.style_num, dtype=np.float32)[style_id], np.array(original_shape), img_name


class Dataset2Train(Dataset):

	def __init__(self, sub_dir, args):

		self.style_dirs = []
		for style_dir in sorted(glob(os.path.join(args.data_dir, sub_dir, '*'))):
			if ("_input" not in style_dir) and (os.path.basename(style_dir)[0] != "."):
				self.style_dirs.append(style_dir)
		self.num_style = len(self.style_dirs)

		self.img_names = []
		self.style_img_names = []
		for i, style_dir in enumerate(self.style_dirs):
			self.style_img_names.append([])
			for img_name in sorted(os.listdir(style_dir)):
				if img_name[0] != ".":
					self.img_names.append([i, img_name])
					self.style_img_names[i].append(img_name)

		self.img_num = len(self.img_names)
		self.num_pref = args.num_pref

		self.rule_based = ["Preset1", "Preset2", "Preset3", "Preset4", "Preset5",
                "Preset6", "Preset7", "Preset8", "Preset9", "Preset10",
                "Preset11", "Preset12", "Preset13", "Preset14", "Preset15",
                "GCEHistMod", "LLF", "JieP", "PIE", "NPEA", "SRIE", "LIME", "LDR", "CDSA", "CLAHE"]

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		style_id = self.img_names[idx][0]
		img_name = self.img_names[idx][1]

		x_raw = cv2.imread(os.path.join(self.style_dirs[style_id]+"_input", img_name))
		x = normalize(random_brightness(color_jitter(x_raw)))
		y = normalize(cv2.imread(os.path.join(self.style_dirs[style_id], img_name)))

		original_shape = x_raw.shape

		pref_img_names = random.sample(list(set(self.style_img_names[style_id]) - set([img_name])), self.num_pref)

		preferred_x = []
		preferred_y = []
		for pref_img_name in pref_img_names:
			if os.path.basename(self.style_dirs[style_id]) in self.rule_based:
				preferred_y.append(y)
				preferred_x.append(normalize(random_brightness(color_jitter(x_raw))))
			else:
				if len(preferred_x)+1 == self.num_pref and random.randint(0, 1) == 0:
					preferred_y.append(y)
					preferred_x.append(normalize(random_brightness(color_jitter(x_raw))))
				else:
					preferred_y.append(normalize(cv2.imread(os.path.join(self.style_dirs[style_id], pref_img_name))))
					preferred_x_i_raw = cv2.imread(os.path.join(self.style_dirs[style_id]+"_input", pref_img_name))
					preferred_x.append(normalize(random_brightness(color_jitter(preferred_x_i_raw))))

		[x_256, y_256] = random_crop([x, y], (256, 256))
		preferred_x_256 = []
		preferred_y_256 = []
		for i in range(len(preferred_y)):
			[cropped_y, cropped_x] = random_crop([preferred_y[i], preferred_x[i]], (256, 256))
			preferred_y_256.append(cropped_y)
			preferred_x_256.append(cropped_x)

		if os.path.basename(self.style_dirs[style_id]) in self.rule_based:
			padding_mask = np.ones(self.num_pref+1)
			padding_mask[-2:] = 0
		else:
			padding_mask = np.zeros(self.num_pref+1)

		return hwc_to_chw(x_256), hwc_to_chw(y_256), np.array(list(map(lambda x: hwc_to_chw(x), preferred_x_256))), np.array(list(map(lambda x: hwc_to_chw(x), preferred_y_256))), padding_mask
