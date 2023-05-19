import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd

from model.networks import Generator, GlobalDis
from model.networks_orig import Generator_Uncond, GlobalDis_Uncond
from utils.tools import get_model_list, calc_div, calc_curl
from utils.logger import get_logger

logger = get_logger()


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.box_patch = self.config['box_patch']
        self.outpaint = self.config['outpaint']
        self.mode = self.config['mode']
        self.coarse_G = self.config['coarse_G']
        self.x2_bnd = self.config['x2_bnd']
        self.uncond = self.config['uncond']

        if self.uncond:
            gen = Generator_Uncond
            dis = GlobalDis_Uncond
        else:
            gen = Generator
            dis = GlobalDis

        self.netG = gen(
            self.config['netG'],
            self.config['coarse_G'],
            self.config['uncond'],
            self.use_cuda,
            self.device_ids,
        )
        self.globalD = dis(
            self.config['netD'],
            self.config['image_shape'],
            self.config['mask_shape'],
            self.config['boundary'],
            self.use_cuda,
            self.device_ids
        )

        self.optimizer_g = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.config['lr'],
            betas=(self.config['beta1'], self.config['beta2'])
        )
        self.optimizer_d = torch.optim.Adam(
            list(self.globalD.parameters()),
            lr=config['lr'],
            betas=(self.config['beta1'], self.config['beta2'])
        )
        lambda0 = lambda epoch: 0.97 ** (epoch * 0.0001)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda=lambda0)

        if self.use_cuda:
            self.netG.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])

    def forward(self, x, mask, gt, grad_z, compute_loss_g=False):
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}

        x1, x2, x_fixed = self.netG(x, mask)
        if self.outpaint:
            if x1 is not None: x1_eval = x1
            x2_eval = x2
        else:
            if x1 is not None: x1_eval = x1 * mask + x * (1. - mask)
            # Change to Boolean x2_eval = x2 if True
            if self.uncond:
                x2_eval = x2 # [:,:,:,:,1]
            #     x2_top = x2[:,:,:,:,0]
            #     x2_bottom = x2[:,:,:,:,2]
            elif self.x2_bnd: 
                x2_eval = x2 
            else: 
                x2_eval = x2 * mask + x * (1. - mask)
            # x2_eval = x2 if self.x2_bnd else x2_eval = x2 * mask + x * (1. - mask)
            
        # D part
        # wgan d loss
        global_real_pred, global_fake_pred = self.dis_forward(self.globalD, gt, x2_eval.detach())
        losses['wgan_d'] = torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']
        # gradients penalty loss
        global_penalty = self.calc_gradient_penalty(self.globalD, gt, x2_eval.detach(), self.config['netG']['volume'])
        losses['wgan_gp'] = global_penalty

        # G part
        if compute_loss_g:
            if not self.uncond:
                losses['l1'] = l1_loss(x2 * mask, gt * mask)
                losses['ae'] = l1_loss(x2 * (1. - mask), gt * (1. - mask))
                if x1 is not None:
                    losses['l1'] += l1_loss(x1_eval, gt) * self.config['coarse_l1_alpha']
                    losses['ae'] += l1_loss(x1 * (1. - mask), gt * (1. - mask)) * self.config['coarse_l1_alpha']
            
            if self.config['div_loss']: 
                losses['div'] = calc_div(x2_eval, None)
            if self.config['curl_loss']:
                losses['curl'] = calc_curl(x2_eval, None)

            if x_fixed is not None:
                losses['gauge'] = l1_loss(x_fixed, torch.zeros_like(x_fixed))
            
            # wgan g loss
            global_real_pred, global_fake_pred = self.dis_forward(self.globalD, gt, x2_eval)
            losses['wgan_g'] = -torch.mean(global_fake_pred) * self.config['global_wgan_loss_alpha']

        return losses, x2_eval, x2

    def dis_forward(self, netD, gt, x_inpaint):
        assert gt.size() == x_inpaint.size()
        batch_size = gt.size(0)
        batch_data = torch.cat([gt, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data, volume=False):
        batch_size = real_data.size(0)
        if volume:
            alpha = torch.rand(batch_size, 1, 1, 1, 1)
        else:
            alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def inference(self, x, mask):
        self.eval()
        _, x2, _ = self.netG(x, mask)
        x2_eval = x2 if self.outpaint else x2 * mask + x * (1. - mask)

        return x2_eval

    def save_model(self, checkpoint_dir, iteration, best=False):
        # Save generators, discriminators, and optimizers
        if best:
            gen_name = os.path.join(checkpoint_dir, 'gen_best.pt')
        else:
            gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
            dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
            opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
            torch.save({'globalD': self.globalD.state_dict()}, dis_name)
            torch.save({'gen': self.optimizer_g.state_dict(),
                        'dis': self.optimizer_d.state_dict()}, opt_name)

        torch.save(self.netG.state_dict(), gen_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        # Load generators
        try:
            last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
            self.netG.load_state_dict(torch.load(last_model_name))
            iteration = int(last_model_name[-11:-3])

            if not test:
                # Load discriminators
                last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
                state_dict = torch.load(last_model_name)
                self.globalD.load_state_dict(state_dict['globalD'])
                # Load optimizers
                state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
                self.optimizer_d.load_state_dict(state_dict['dis'])
                self.optimizer_g.load_state_dict(state_dict['gen'])
        except:
            iteration = 1

        #("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration
