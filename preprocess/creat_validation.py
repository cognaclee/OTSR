import argparse
import os
import torch.utils.data
import yaml
import utils
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Apply the trained model to create a dataset')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to use')
parser.add_argument('--artifacts', default='clean', type=str, help='selecting different artifacts type')
parser.add_argument('--name', default='', type=str, help='additional string added to folder path')
parser.add_argument('--dataset', default='dped', type=str, help='selecting different datasets')
parser.add_argument('--track', default='valid', type=str, help='selecting train or valid track')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

with open('./preprocess/paths.yml', 'r') as stream:
    PATHS = yaml.load(stream)

if opt.dataset == 'df2k':
    path_tdsr = PATHS['datasets']['df2k'] + '/valid/'
    input_valid_dir = PATHS['df2k']['tdsr']['valid']
    valid_files = [os.path.join(input_valid_dir, x) for x in os.listdir(input_valid_dir) if utils.is_image_file(x)]
else:
    path_tdsr = PATHS['datasets'][opt.dataset] + '/valid/'
    input_source_dir = PATHS[opt.dataset][opt.artifacts]['hr'][opt.track]
    input_target_dir = None
    valid_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = []

tdsr_hr_dir = path_tdsr + 'HR'
tdsr_lr_dir = path_tdsr + 'LR'

if not os.path.exists(tdsr_hr_dir):
    os.makedirs(tdsr_hr_dir)
if not os.path.exists(tdsr_lr_dir):
    os.makedirs(tdsr_lr_dir)

with torch.no_grad():
    for file in tqdm(valid_files, desc='Generating images from validation'):
        input_img = Image.open(file)
        input_img = TF.to_tensor(input_img)

        resize2_img = utils.imresize(input_img, 1.0 / opt.cleanup_factor, True)
        _, w, h = resize2_img.size()
        w = w - w % opt.upscale_factor
        h = h - h % opt.upscale_factor
        resize2_cut_img = resize2_img[:, :w, :h]

        path = os.path.join(tdsr_hr_dir, os.path.basename(file))
        TF.to_pil_image(resize2_cut_img).save(path, 'PNG')

        resize3_cut_img = utils.imresize(resize2_cut_img, 1.0 / opt.upscale_factor, True)

        path = os.path.join(tdsr_lr_dir, os.path.basename(file))
        TF.to_pil_image(resize3_cut_img).save(path, 'PNG')


