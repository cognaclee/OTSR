from PIL import Image
import numpy as np
import os.path as osp
import glob
import os
import argparse
import yaml

parser = argparse.ArgumentParser(description='create a dataset')
parser.add_argument('--dataset', default='dped', type=str, help='selecting different datasets')
parser.add_argument('--artifacts', default='clean', type=str, help='selecting different artifacts type')
parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

with open('./preprocess/paths.yml', 'r') as stream:
    PATHS = yaml.load(stream)

def noise_patch(rgb_img, sp, max_var, min_mean):
    img = rgb_img.convert('L')
    rgb_img = np.array(rgb_img)
    img = np.array(img)

    w, h = img.shape
    collect_patchs = []

    for i in range(0, w - sp, sp):
        for j in range(0, h - sp, sp):
            patch = img[i:i + sp, j:j + sp]
            var_global = np.var(patch)
            mean_global = np.mean(patch)
            if var_global < max_var and mean_global > min_mean:
                rgb_patch = rgb_img[i:i + sp, j:j + sp, :]
                collect_patchs.append(rgb_patch)

    return collect_patchs
    
def blind_noise_patch(rgb_img, sp, lsp, mu, sigma):
    img = rgb_img.convert('L')
    rgb_img = np.array(rgb_img)
    img = np.array(img)

    w, h = img.shape
    collect_patchs = []

    for i in range(0, w - sp, sp):
        for j in range(0, h - sp, sp):
            patch = img[i:i + sp, j:j + sp]
            var_global = np.var(patch)
            mean_global = np.mean(patch)
            mean_threshold = mean_global * mu
            var_threshold = var_global* sigma
            moothFlag = True
            for m in range(0, sp - lsp, lsp):
                for n in range(0, sp - lsp, lsp):
                    localpatch = patch[m:m + lsp, n:n + lsp]
                    var_local = np.var(localpatch)
                    mean_local = np.mean(localpatch)    
                    if abs(mean_local-mean_global) > mean_threshold or abs(var_local-var_global) > var_threshold or var_global > 50:
                        moothFlag = False
                        break
                if moothFlag == False:
                    break
            
                        
            if moothFlag == True:            
                rgb_patch = rgb_img[i:i + sp, j:j + sp, :]
                collect_patchs.append(rgb_patch)
    patchLen = len(collect_patchs)

    return collect_patchs


if __name__ == '__main__':

    if opt.dataset == 'df2k':
        img_dir = PATHS[opt.dataset][opt.artifacts]['source']
        noise_dir = PATHS['datasets']['df2k'] + '/Corrupted_noise'
        sp = 256
        max_var = 20
        min_mean = 0
    else:
        img_dir = PATHS[opt.dataset][opt.artifacts]['hr']['train']
        noise_dir = PATHS['datasets']['dped'] + '/DPEDiphone_noise'
        sp = 256
        max_var = 20
        min_mean = 50

    assert not os.path.exists(noise_dir)
    os.mkdir(noise_dir)

    img_paths = sorted(glob.glob(osp.join(img_dir, '*.png')))
    lsp = 64
    mu = 0.1
    gamma = 0.25
    for path in img_paths:
        img_name = osp.splitext(osp.basename(path))[0]
        print('**********', img_name, '**********')
        img = Image.open(path).convert('RGB')
        patchs = blind_noise_patch(img, sp, lsp, mu, gamma)
        for idx, patch in enumerate(patchs):
            save_path = osp.join(noise_dir, '{}_{:03}.png'.format(img_name, idx))
            Image.fromarray(patch).save(save_path)
