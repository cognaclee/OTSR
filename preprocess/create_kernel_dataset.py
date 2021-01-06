import argparse
import os
import torch.utils.data
import yaml
import glob
#utils不是codes下的utils
import utils
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from KernelGAN.imresize import imresize
from scipy.io import loadmat
import numpy as np

kernel_dir = '/home/public/data/kernelGan/dped_kernel_x_x4_2/'
parser = argparse.ArgumentParser(description='Apply the trained model to create a dataset')
parser.add_argument('--kernel_path', default=kernel_dir, type=str, help='kernel path to use')
parser.add_argument('--artifacts', default='clean', type=str, help='selecting different artifacts type')
parser.add_argument('--name', default='', type=str, help='additional string added to folder path')
parser.add_argument('--dataset', default='dped', type=str, help='selecting different datasets')
parser.add_argument('--track', default='train', type=str, help='selecting train or valid track')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--cleanup_factor', default=1, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

with open('./preprocess/paths.yml', 'r') as stream:
    PATHS = yaml.load(stream)

    path_tdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + opt.track + opt.name + '_tdsr/'
    input_source_dir = PATHS[opt.dataset][opt.artifacts]['hr'][opt.track]
    input_target_dir = None
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]

tdsr_hr_dir = path_tdsr + 'HR'
tdsr_lr_dir = path_tdsr + 'LR'

if not os.path.exists(tdsr_hr_dir):
    os.makedirs(tdsr_hr_dir)
if not os.path.exists(tdsr_lr_dir):
    os.makedirs(tdsr_lr_dir)

kernel_paths = glob.glob(os.path.join(opt.kernel_path, '*/*_kernel_x4.mat'))
kernel_num = len(kernel_paths)
print('kernel_num: ', kernel_num)

with torch.no_grad():
    for file in tqdm(source_files, desc='Generating images from source'):
        # load HR image
        input_img = Image.open(file)
        input_img = TF.to_tensor(input_img)

        resize2_img = utils.imresize(input_img, 1.0 / opt.cleanup_factor, True)
        _, w, h = resize2_img.size()
        w = w - w % opt.upscale_factor
        h = h - h % opt.upscale_factor
        resize2_cut_img = resize2_img[:, :w, :h]


        try:
            file_name = os.path.basename(file)
            file_id = file_name.split('.')[0]
            kernel_path = os.path.join(opt.kernel_path,file_id+'/'+file_id+'_kernel_x4.mat')
            mat = loadmat(kernel_path)

            path = os.path.join(tdsr_hr_dir, os.path.basename(file))
            if not os.path.exists(path):
                print(f'create_HrLr kernel_path:{kernel_path}')
                HR_img = TF.to_pil_image(input_img)
                HR_img.save(path, 'PNG')

                k = np.array([mat['Kernel']]).squeeze()
                resize_img = imresize(np.array(HR_img), scale_factor=1.0 / opt.upscale_factor, kernel=k)

                path = os.path.join(tdsr_lr_dir, os.path.basename(file))
                TF.to_pil_image(resize_img).save(path, 'PNG')
            else:
                print(f'skip kernel_path:{kernel_path}')

        except Exception as e:
            print(e)
