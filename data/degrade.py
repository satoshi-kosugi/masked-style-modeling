import os.path as osp
import os
import logging
import time
import argparse
from collections import OrderedDict
import cv2
import torch

import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr, read_img
from data import create_dataset, create_dataloader
from models import create_model
import torchvision.transforms as TF

#### options
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='options/test/test_for_fiveK.yml', help='Path to options YMAL file.')
parser.add_argument('--path_to_data', type=str)
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)
path_to_data = parser.parse_args().path_to_data
opt["path"]['pretrain_model_G'] = osp.join(path_to_data, "degrading_model.pth")

model = create_model(opt)
dir_names = ["train", "valid", "valid_ref"]

with torch.no_grad():
    for dir_name in dir_names:
        user_ids = os.listdir(os.path.join(path_to_data, dir_name))

        for user_id in user_ids:
            if ("_N" not in user_id) or ("_input" in user_id):
                continue

            high_dir = os.path.join(path_to_data, dir_name, user_id)
            low_dir = os.path.join(path_to_data, dir_name, user_id+"_input")

            if not os.path.exists(low_dir):
                os.mkdir(low_dir)

            for image_name in os.listdir(high_dir):
                print(os.path.join(high_dir, image_name))
                high_im = read_img(os.path.join(high_dir, image_name))
                # high_im = high_im[:, :, [2, 1, 0]]
                # low_im = np.expand_dims(low_im[:, :, [2, 1, 0]],axis=0)
                input = TF.ToTensor()(high_im.copy())
                input = input.unsqueeze(0)
                output = model.netG(input)
                output_img = util.tensor2img(output)
                util.save_img(output_img, os.path.join(low_dir, image_name))
