import os
import math
import argparse
import random
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import matplotlib.pyplot as plt
import numpy as np


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    ymlPath = './options/df2k/train_bicubic_noise.yml'
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=ymlPath, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
        print('resume_state=', opt['path']['resume_state'])
    else:
        resume_state = None

    if rank <= 0:
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    opt = option.dict_to_nonedict(opt)

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True

    dataset_ratio = 1  # enlarge the size of each epoch
    train_loader = None
    val_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    model = create_model(opt)

    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    loss_len = total_epochs + 1-start_epoch
    trian_generoter_loss = np.zeros(loss_len)
    trian_discriminator_loss = np.zeros(loss_len)

    score_len = int(math.floor(loss_len/opt['train']['val_freq']))
    valid_ssim = []
    valid_psnr = []
    
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
            
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            
            loss_and_index = epoch-start_epoch
            trian_generoter_loss[loss_and_index] = model.g_total_loss
            trian_discriminator_loss[loss_and_index] = model.d_total_loss

            if current_step % opt['train']['val_freq'] == 0 and rank <= 0 and val_loader is not None:
                avg_psnr = val_pix_err_f = val_pix_err_nf = val_mean_color_err = avg_ssim = 0.0
                idx = 0
                for val_data in val_loader:
                    if idx > 2:
                        break
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    print("!!!Now start to save Image!!!!")
                    save_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}.png'.format(img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    avg_ssim += util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)


                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                val_pix_err_f /= idx
                val_pix_err_nf /= idx
                val_mean_color_err /= idx
                
                valid_ssim.append(avg_ssim)
                valid_psnr.append(avg_psnr)

                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    epoch, current_step, avg_psnr))
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('val_pix_err_f', val_pix_err_f, current_step)
                    tb_logger.add_scalar('val_pix_err_nf', val_pix_err_nf, current_step)
                    tb_logger.add_scalar('val_mean_color_err', val_mean_color_err, current_step)

            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
   
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0,loss_len),trian_generoter_loss,label="train_generoter_loss")
    plt.plot(np.arange(0,loss_len),trian_discriminator_loss,label="trian_discriminator_loss")

    print(trian_generoter_loss)
    print(trian_discriminator_loss)
    plt.title("Training loss on OTSR")
    plt.xlabel("epoch#")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.savefig("loss.jpg")
    plt.show()
    
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(valid_ssim)),valid_ssim,label="valid_ssim")
    plt.plot(np.arange(0, len(valid_psnr)),valid_psnr,label="valid_psnr")
    plt.title("validation index score on OTSR")
    plt.xlabel("epoch#")
    plt.ylabel("index score")
    plt.legend(loc="upper right")
    plt.savefig("evaluate.jpg")
    plt.show()


if __name__ == '__main__':
    main()
