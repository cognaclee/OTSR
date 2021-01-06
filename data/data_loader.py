import torch.utils.data as data
import torch
from torchvision import transforms
from PIL import Image
import glob


class noiseDataset(data.Dataset):
    def __init__(self, dataset='x2/', size=32):
        super(noiseDataset, self).__init__()

        base = dataset
        import os
        assert os.path.exists(base)

        self.noise_imgs = sorted(glob.glob(base + '*.png'))
        self.pre_process = transforms.Compose([transforms.RandomCrop(size),
                                               transforms.ToTensor()])

    def __getitem__(self, index):
        noise = self.pre_process(Image.open(self.noise_imgs[index]))
        norm_noise = (noise - torch.mean(noise, dim=[1, 2], keepdim=True))
        return norm_noise

    def __len__(self):
        return len(self.noise_imgs)
