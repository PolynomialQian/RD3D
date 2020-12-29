import os
import pickle
from PIL import Image
import random
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize, hflip=False, vflip=False):
        self.trainsize = trainsize
        self.hflip = hflip
        self.vflip = vflip
        self.p = 0.5
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [os.path.join(depth_root, f) for f in os.listdir(depth_root) if f.endswith('.bmp')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)

        filename = os.path.join(image_root, '..', 'processed_data.pkl')
        if not os.path.exists(filename):
            images = []
            gts = []
            depths = []
            for i in range(self.size):
                image = self.rgb_loader(self.images[i])
                gt = self.binary_loader(self.gts[i])
                depth = self.binary_loader(self.depths[i])
                images.append(image)
                gts.append(gt)
                depths.append(depth)
            self.image_data = images
            self.gt_data = gts
            self.depth_data = depths
            with open(filename, 'wb') as f:
                pickle.dump((self.image_data, self.gt_data, self.depth_data), f)
            print(f"data saved in {filename}")
        else:
            print(f"data loaded in {filename}")
            with open(filename, 'rb') as f:
                self.image_data, self.gt_data, self.depth_data = pickle.load(f)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = self.image_data[index]
        gt = self.gt_data[index]
        depth = self.depth_data[index]
        if self.hflip:
            if random.random() < self.p:
                image = F.hflip(image)
                gt = F.hflip(gt)
                depth = F.hflip(depth)
        if self.vflip:
            if random.random() < self.p:
                image = F.vflip(image)
                gt = F.vflip(gt)
                depth = F.vflip(depth)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depth_transform(depth)
        depth = depth.repeat(3, 1, 1)
        return image, gt, depth

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
            if depth.size == gt.size:
                depths.append(depth_path)

        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, hflip=False, vflip=False,
               num_workers=12, pin_memory=True):
    dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize, hflip=hflip, vflip=vflip)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [os.path.join(depth_root, f) for f in os.listdir(depth_root) if f.endswith('.bmp')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self._transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depth_transform(depth).unsqueeze(0)
        depth = depth.repeat(1, 3, 1, 1)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
