import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import apply_gaussian_noise

class DenoiseDataset(Dataset):
    def __init__(self, img_dir, img_size=256, sigma="sigma_10", transform=None, cutblur_prob=0.5):
        self.img_size = img_size
        self.sigma = sigma
        self.transform = transform
        self.cutblur_prob = cutblur_prob
        self.is_real = "real_noise" in sigma
        
        self.gt_dir = os.path.join(img_dir, "GT")
        self.noisy_dir = os.path.join(img_dir, "NOISY")

        if self.is_real:
            if not (os.path.exists(self.gt_dir) and os.path.exists(self.noisy_dir)):
                raise FileNotFoundError(f"真实噪声模式要求存在 GT 和 NOISY 文件夹: {img_dir}")
            gt_list = set(os.listdir(self.gt_dir))
            ny_list = set(os.listdir(self.noisy_dir))
            self.filenames = sorted(list(gt_list & ny_list))
        else:
            self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                             if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.filenames) if self.is_real else len(self.img_paths)

    def _cutblur(self, noisy, clean):
        if random.random() > self.cutblur_prob:
            return noisy
        c, h, w = noisy.shape
        ch = random.randint(h // 3, h // 2)
        cw = random.randint(w // 3, w // 2)
        cy = random.randint(0, h - ch)
        cx = random.randint(0, w - cw)
        noisy[:, cy:cy+ch, cx:cx+cw] = clean[:, cy:cy+ch, cx:cx+cw]
        return noisy

    def __getitem__(self, idx):
        if self.is_real:
            fname = self.filenames[idx]
            clean = Image.open(os.path.join(self.gt_dir, fname)).convert("RGB")
            noisy = Image.open(os.path.join(self.noisy_dir, fname)).convert("RGB")
        else:
            clean = Image.open(self.img_paths[idx]).convert("RGB")
            noisy = clean.copy()

        seed = np.random.randint(2147483647) 
        random.seed(seed)
        torch.manual_seed(seed)
        clean_t = self.transform(clean)
        random.seed(seed)
        torch.manual_seed(seed)
        noisy_t = self.transform(noisy)

        if not self.is_real:
            noisy_t = apply_gaussian_noise(noisy_t, self.sigma)
            if self.cutblur_prob > 0:
                noisy_t = self._cutblur(noisy_t, clean_t)

        return torch.clamp(noisy_t, 0, 1), torch.clamp(clean_t, 0, 1)

def get_dataloader(img_dir, batch_size=4, img_size=256, sigma="sigma_25", shuffle=True, is_train=True):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomAdjustSharpness(1.5)], p=0.3),
            transforms.ToTensor()])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor()])

    if is_train and "real_noise" not in sigma:
        cb_p = 0.2
    else:
        cb_p = 0
    dataset = DenoiseDataset(img_dir=img_dir, img_size=img_size, sigma=sigma, 
                             transform=transform, cutblur_prob=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    #return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True, persistent_workers=True)