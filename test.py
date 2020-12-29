import os
import argparse
import copy
import numpy as np
from scipy import misc

import torch
import torch.nn.functional as F
import torchvision

from model.rd3d import RD3D
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to model file')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--test_datasets', type=str, default=['NJU2000-test'], nargs='+', help='test dataset')
parser.add_argument('--data_path', type=str, default='', help='test dataset')
parser.add_argument('--save_path', type=str, help='test dataset')
# model
parser.add_argument('--multi_load', action='store_true', help='whether to load multi-gpu weight')

opt = parser.parse_args()

dataset_path = opt.data_path
test_datasets = opt.test_datasets
if opt.save_path is not None:
    save_root = opt.save_path
else:
    mode_dir_name = os.path.dirname(opt.model_path)
    stime = mode_dir_name.split('\\')[-1]
    save_root = os.path.join(mode_dir_name, f'{stime}_results')

# build model
resnet = torchvision.models.resnet50(pretrained=True)
model = RD3D(32, copy.deepcopy(resnet)).cuda()

if opt.multi_load:
    state_dict_multi = torch.load(opt.model_path)
    state_dict = {k[7:]: v for k, v in state_dict_multi.items()}
else:
    state_dict = torch.load(opt.model_path)
model.load_state_dict(state_dict)
model.cuda()
model.eval()

for dataset in test_datasets:
    save_path = os.path.join(save_root, dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = os.path.join(dataset_path, dataset, 'images')
    gt_root = os.path.join(dataset_path, dataset, 'gts')
    depth_root = os.path.join(dataset_path, dataset, 'depths')
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        images = image.unsqueeze(2)
        depths = depth.unsqueeze(2)
        image = torch.cat([images, depths], 2)
        res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(os.path.join(save_path, name), res)
        print(f"{os.path.join(save_path, name)} saved !")
