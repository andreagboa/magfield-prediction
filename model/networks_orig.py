import numpy as np
import torch.nn as nn

class Generator_Uncond(nn.Module):
    def __init__(self, config, coarse_G, uncond, use_cuda, device_ids):
        super(Generator_Uncond, self).__init__()
        self.in_dim = config['input_dim']
        self.cnum = config['ngf']
        
        if config['volume']:
            self.img_size = [4, 4, 1]
            self.model = nn.Sequential(            
                # conv(self.cnum, 256, 3, 1),
                nn.ConvTranspose3d(self.cnum, 256, (3, 3, 2), (1, 1, 1)),
                nn.ELU(inplace=True),
                # state size: (256, 6, 6, 2)
                
                nn.ConvTranspose3d(256, 128, (4, 4, 2), (2, 2, 1), (1, 1, 0)),
                nn.ELU(inplace=True),
                # state size: (128, 12, 12, 3)
                
                nn.ConvTranspose3d(128, 64, (4, 4, 3), (2, 2, 1), (1, 1, 1)),
                nn.ELU(inplace=True),
                # state size: (64, 24, 24, 3)

                # conv(64, 32, 4, 2, 1),
                nn.ConvTranspose3d(64, 32, (4, 4, 3), (2, 2, 1), (1, 1, 1)),
                nn.ELU(inplace=True),
                # state size: (32, 48, 48, 3)
                
                nn.ConvTranspose3d(32, self.in_dim, (4, 4, 3), (2, 2, 1), (1, 1, 1)),
                # state size: (3, 96, 96, 3)
            )
        elif isinstance(config['latent_dim'], list):
            self.img_size = None
            self.model = nn.Sequential(
                nn.ConvTranspose2d(config['latent_dim'][0], self.cnum*4, 3, 1, 1),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(self.cnum*4, self.cnum*4, 3, 1, 1),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(self.cnum*4, self.cnum*2, 3, 1, 1),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(self.cnum*2, self.cnum*2, 3, 1, 1),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(self.cnum*2, self.cnum, 3, 1, 1),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(self.cnum, self.cnum//2, 3, 1, 1),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(self.cnum//2, self.in_dim, 3, 1, 1)
            )
        else:
            self.img_size = [4, 4]
            self.model = nn.Sequential(            
                nn.ConvTranspose2d(self.cnum, 256, 3, 1),
                nn.ELU(inplace=True),
                # state size: (256, 6, 6)
                
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.ELU(inplace=True),
                # state size: (128, 12, 12)
                
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ELU(inplace=True),
                # state size: (64, 24, 24)

                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ELU(inplace=True),
                # state size: (32, 48, 48)
                
                nn.ConvTranspose2d(32, self.in_dim, 4, 2, 1),
                # state size: (3, 96, 96)
            )
        
        if self.img_size is not None:
            self.upsample =  nn.Sequential(
                nn.Linear(config['latent_dim'][0], np.prod(self.img_size)*self.cnum),            
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, z, mask):
        if self.img_size is not None:
            img = self.upsample(z)
            img = img.view(img.shape[0], self.cnum, *self.img_size)
        else:
            img = z
        img = self.model(img)
        
        return None, img, None
    

class GlobalDis_Uncond(nn.Module):
    def __init__(self, config, image_shape, mask_shape, bnd, use_cuda=True, device_ids=None):
        super(GlobalDis_Uncond, self).__init__()
        self.input_dim = image_shape[0]
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        if config['volume']:
            self.dis_module = nn.Sequential(
                nn.Conv3d(self.input_dim, self.cnum, (5, 5, 1), (2, 2, 1), (2, 2, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(self.cnum, self.cnum*2, (5, 5, 1), (2, 2, 1), (2, 2, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(self.cnum*2, self.cnum*4, (5, 5, 1), (2, 2, 1), (2, 2, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(self.cnum*4, self.cnum*4, (5, 5, 3), (2, 2, 1), (2, 2, 0)),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.dis_module = nn.Sequential(
                nn.Conv2d(self.input_dim, self.cnum, 5, 2, 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.cnum, self.cnum*2, 5, 2, 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.cnum*2, self.cnum*4, 5, 2, 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.cnum*4, self.cnum*4, 5, 2, 2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.linear = nn.Linear(self.cnum*4 * (mask_shape[0] + 2*bnd) // 16 * (mask_shape[1] + 2*bnd) // 16, 1)

    def forward(self, x):
        x = self.dis_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x