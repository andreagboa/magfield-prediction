import os
import random
import time
import shutil
import numpy as np
from pathlib import PurePath, Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from trainer import Trainer
from utils.dataset import MagneticFieldDataset
from utils.tools import get_config, random_bbox, mask_image
from utils.logger import get_logger

import matplotlib.pyplot as plt
import matplotlib.colors as col

import wandb

# To be able to run from command prompt
# Changing parameters from config without actually doing so
parser = ArgumentParser()
parser.add_argument(
    '--config', type=str,
    default=Path(__file__).parent.resolve() / 'configs' / 'config.yaml',
    help="training configuration"
)
parser.add_argument('--seed', type=int, help='manual seed')


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # Configure checkpoint path
    cp_path = Path(__file__).parent.resolve() / 'checkpoints' / config['dataset_name'] / config['exp_name']
    if config['test']:
        cp_path /= 'test'
    if not cp_path.exists():
        cp_path.mkdir(parents=True)
    elif config['resume'] is None and not config['test']:
        print('Experiment has already been run! Terminating...')
        exit()
    shutil.copy(args.config, cp_path / PurePath(args.config).name)

    if config['wandb'] and not config['test']:
        wandb.init(
            project="wgan-gp_scalar", 
            entity="andreathesis",
            config=config
        )
    
    bnd = config['boundary']
    logger = get_logger(cp_path)
    logger.info("Arguments: {}".format(args))
    if args.seed is None: args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda: torch.cuda.manual_seed_all(args.seed)
    logger.info("Configuration: {}".format(config))

    try:  # for unexpected error logging
        datapath = Path(__file__).parent.resolve() / 'data'
        logger.info(f"Training on dataset: {config['dataset_name']}")

        train_dataset = MagneticFieldDataset(
            datapath / config['train_data'],
            config['scale_factor'],
            config['netG']['volume']
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True
        )

        val_dataset = MagneticFieldDataset(
            datapath / config['val_data'],
            config['scale_factor'],
            config['netG']['volume']
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=True
        )

        trainer = Trainer(config)
        logger.info("\n{}".format(trainer.netG))
        logger.info("\n{}".format(trainer.globalD))
        
        if cuda:
            trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
            trainer_module = trainer.module
        else:
            trainer_module = trainer

        if config['resume']:
            start_iteration = trainer_module.resume(cp_path, config['resume'])
        else:
            start_iteration = 1
        iterable_train_loader = iter(train_loader)
        iterable_val_loader = iter(val_loader)
        l1_loss = nn.L1Loss()
        
        rng = np.random.default_rng(0)
        rng_val = np.random.default_rng(1)
        time_count = time.time()
        best_score = 1
        
        for iteration in range(start_iteration, config['niter'] + 1):
            try:
                gt, grad_z = next(iterable_train_loader)
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                gt, grad_z = next(iterable_train_loader)
            
            bboxes = random_bbox(config, rng=rng)

            if config['uncond']:
                mask = None
            else:
                x, mask, _ = mask_image(gt, bboxes, config, bnd=bnd)
                if cuda: mask = mask.cuda()

            (t,l,h,w) = bboxes[0,0]
            gt = gt[:,:,t - bnd:t + h + bnd,l - bnd:l + w + bnd]
            
            if config['uncond']:
                # size_f = (config['batch_size'], gt.shape[1], (h + 2*bnd) // 4, (w + 2*bnd) // 4)
                x = torch.normal(0, 1, size=(gt.shape[0], config['netG']['latent_dim']))
            
            if cuda:
                x = x.cuda()
                gt = gt.cuda()
                grad_z = grad_z.cuda()
            
            ###### Forward pass ######
            compute_g_loss = iteration % config['n_critic'] == 0
            losses, inpainted_result, _ = trainer(x, mask, gt, grad_z, compute_g_loss)
            # Scalars from different devices are gathered into vectors
            for k in losses.keys():
                if not losses[k].dim() == 0: losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            # Update D
            trainer_module.optimizer_d.zero_grad()
            losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
            losses['d'].backward()

            # Update G
            if compute_g_loss:
                trainer_module.optimizer_g.zero_grad()
                if config['uncond']:
                    losses['g'] = losses['wgan_g'] * config['gan_loss_alpha']
                else:
                    losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
                                + losses['ae'] * config['ae_loss_alpha'] \
                                + losses['wgan_g'] * config['gan_loss_alpha']
                if config['div_loss']: losses['g'] += losses['div'] * config['div_loss_alpha']
                if config['curl_loss']: losses['g'] += losses['curl'] * config['curl_loss_alpha']
                if config['netG']['gauge']: losses['g'] += losses['gauge'] * config['gauge_loss_alpha']
                losses['g'].backward()
                trainer_module.optimizer_g.step()

            trainer_module.optimizer_d.step()
            trainer_module.scheduler.step()

            if iteration % config['print_iter'] == 0:
                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
                for k, v in losses.items():
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
                logger.info(message)
                
                if config['wandb'] and not config['test']:
                    wandb.log(data=losses, step=iteration)
            
            if iteration % (config['viz_iter']) == 0:
                if config['netG']['volume']:
                    gt = gt[:,:,:,:,1]
                    inpainted_result = inpainted_result[:,:,:,:,1]
                gt_scaled = gt / config['scale_factor']
                res = inpainted_result / config['scale_factor']
                err = abs(gt_scaled - res)
                mode = 'Out' if config['outpaint'] else 'In'
                plt.close('all')
                fig, axes = plt.subplots(nrows=config['netG']['input_dim'], 
                                         ncols=3, sharex=True, sharey=True)
                viz_list = [
                    ('Truth_X', gt_scaled[0,0]), (mode + 'paint_X', res[0,0]), ('Error_X', err[0,0]),
                    ('Truth_Y', gt_scaled[0,1]), (mode + 'paint_Y', res[0,1]), ('Error_Y', err[0,1])
                ]                
                if config['netG']['input_dim'] == 3:
                    viz_list.extend([
                        ('Truth_Z', gt_scaled[0,2]), (mode + 'paint_Z', res[0,2]), ('Error_Z', err[0,2])
                    ])

                for i, (title, data) in enumerate(viz_list):
                    ax = axes.flat[i]
                    ax.set_title(title)
                    if 'Error' in title:
                        _ = ax.imshow(data.cpu().data.numpy(), cmap='cividis',
                            norm=col.Normalize(vmin=0, vmax=0.005), origin="lower")
                    else:
                        im = ax.imshow(data.cpu().data.numpy(), cmap='bwr',
                            norm=col.Normalize(vmin=-0.04, vmax=0.04), origin="lower")

                fig.colorbar(im, ax=axes.ravel().tolist())
                plt.savefig(f'{cp_path}/{iteration}.png')

            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(cp_path, iteration)
            
            # Validation
            if iteration % config['valid_iter'] == 0:
                with torch.no_grad():
                    val_loss = []
                    for _ in range(25):
                        try:
                            gt, _ = next(iterable_val_loader)
                        except StopIteration:
                            iterable_val_loader = iter(val_loader)
                            gt, _ = next(iterable_val_loader)

                        bboxes = random_bbox(config, rng=rng_val)

                        if config['uncond']:
                            x = torch.normal(0, 1, size=(gt.shape[0], config['netG']['latent_dim']))
                            mask = None
                        else:
                            x, mask, _ = mask_image(gt, bboxes, config, bnd=bnd)
                            if cuda: mask = mask.cuda()
                        
                        (t,l,h,w) = bboxes[0,0]
                        gt = gt[:,:,t - bnd:t + h + bnd,l - bnd:l + w + bnd]

                        if cuda:
                            x = x.cuda()
                            gt = gt.cuda()

                        # Inference
                        _, x2, _ = trainer_module.netG(x, mask)
                        if config['outpaint'] or config['x2_bnd']:
                            x2_eval = x2
                        elif config['uncond']:
                            x2_eval = x2 # [:,:,:,:,1]
                        else:
                            x2_eval = x2 * mask + x * (1. - mask)

                        if config['uncond']:
                            global_real_pred, global_fake_pred = trainer_module.dis_forward(trainer_module.globalD, gt, x2_eval.detach())
                            val_loss.append(torch.mean(global_fake_pred - global_real_pred) * config['global_wgan_loss_alpha'])
                        else:
                            val_loss.append(l1_loss(x2_eval, gt))

                    val_err = sum(val_loss) / len(val_loss)
                    if config['wandb'] and not config['test']:
                        wandb.log({"L1-loss (val)": val_err})

                    # Saving best model
                    if val_err < best_score:
                        logger.info(f'Saving new best model...')
                        best_score = val_err
                        trainer_module.save_model(cp_path, iteration, best=True)

                    logger.info(f'Validation: {val_err:.6f}')

    except Exception as e:  # for unexpected error logging
        logger.error("{}".format(e))
        raise e


if __name__ == '__main__':
    main()
