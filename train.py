import os
import copy
import time
import json
import argparse

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from model.rd3d import RD3D
from model.rd3d_plus import RD3D_plus
from data import get_loader
from utils.func import AvgMeter, clip_gradient
from utils.lr_scheduler import get_scheduler
from utils.logger import setup_logger


def parse_option():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--hflip', action='store_true', help='hflip data')
    parser.add_argument('--vflip', action='store_true', help='vflip data')
    parser.add_argument('--data_dir', type=str, default='data_dir', help='data director')

    # training
    parser.add_argument('--model', type=str, default='', help='RD3D or RD3D+')
    parser.add_argument('--epochs', type=int, default=50, help='epoch number')
    parser.add_argument('--optim', type=str, default='adamW', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.000125, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup_epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_steps', type=int, default=20,
                        help='for step scheduler. step size to decay lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    # io
    parser.add_argument('--output_dir', type=str, default='./output', help='output director')

    opt, unparsed = parser.parse_known_args()
    opt.output_dir = os.path.join(opt.output_dir, str(int(time.time())))
    return opt


def build_loader(opt):
    num_gpus = torch.cuda.device_count()
    print(f"========>num_gpus:{num_gpus}==========")

    image_root = os.path.join(opt.data_dir, 'RGB-train/')
    gt_root = os.path.join(opt.data_dir, 'GT-train/')
    depth_root = os.path.join(opt.data_dir, 'depth-train/')
    train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize * num_gpus,
                              trainsize=opt.trainsize)
    return train_loader


def build_model(opt):
    # build model
    resnet = torchvision.models.resnet50(pretrained=True)
    if opt.model=="RD3D":
        model = RD3D(32, copy.deepcopy(resnet))
    else:
        model = RD3D_plus(32, copy.deepcopy(resnet))
    print(model)
    model = nn.DataParallel(model).cuda()
    return model


def main(opt):
    # build dataloader
    train_loader = build_loader(opt)
    n_data = len(train_loader.dataset)
    logger.info(f"length of training dataset: {n_data}")
    # build model
    model = build_model(opt)
    CE = torch.nn.BCEWithLogitsLoss().cuda()
    # build optimizer
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'sdg':
        optimizer = torch.optim.SGD(model.parameters(), opt.lr / 10.0 * opt.batchsize, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError
    scheduler = get_scheduler(optimizer, len(train_loader), opt)

    # routine
    for epoch in range(1, opt.epochs + 1):
        tic = time.time()
        train(train_loader, model, optimizer, CE, scheduler, epoch, opt)
        logger.info('epoch {}, total time {:.2f}, learning_rate {}'.format(epoch, (time.time() - tic),
                                                                           optimizer.param_groups[0]['lr']))
        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(opt.output_dir, f"RD3D_{epoch}_ckpt.pth"))
            logger.info("model saved {}!".format(os.path.join(opt.output_dir, f"RD3D_{epoch}_ckpt.pth")))
    torch.save(model.state_dict(), os.path.join(opt.output_dir, f"RD3D_last_ckpt.pth"))
    logger.info("model saved {}!".format(os.path.join(opt.output_dir, f"RD3D_ckpt.pth")))
    return os.path.join(opt.output_dir, f"RD3D_last_ckpt.pth")


# training
def train(train_loader, model, optimizer, criterion, scheduler, epoch, opt):
    # multi-scale training
    size_rates = [0.75, 1, 1.25]

    model.train()
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts, depths = pack
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()

            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                images = images.unsqueeze(2)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                depths = F.upsample(depths, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                depths = depths.unsqueeze(2)
                images = torch.cat([images, depths], 2)

            if rate == 1:
                images = images.unsqueeze(2)
                depths = depths.unsqueeze(2)
                images = torch.cat([images, depths], 2)

            # forward
            pred_s = model(images)
            loss = criterion(pred_s, gts)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            scheduler.step()
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 100 == 0 or i == len(train_loader):
            logger.info('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                        format(epoch, opt.epochs, i, len(train_loader),
                               loss_record.show()))


if __name__ == '__main__':
    opt = parse_option()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)

    logger = setup_logger(output=opt.output_dir, name="rd3d")
    path = os.path.join(opt.output_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
    logger.info("Full config saved to {}".format(path))

    ckpt_path = main(opt)
