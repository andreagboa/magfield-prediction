import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config, coarse_G, uncond, use_cuda, device_ids):
        super(Generator, self).__init__()
        self.in_dim = config['input_dim']
        self.cnum = config['ngf']
        
        if config['volume']:
            conv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
        else:
            conv = nn.ConvTranspose3d
            bn = nn.BatchNorm3d
        
        self.upsample =  nn.Sequential(
            nn.Linear(config['latent_dim'], 16*self.cnum),            
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model = nn.Sequential(
            # 4 x 4
            conv(self.cnum, self.cnum//2, 5, 2, dilation=2, output_padding=1),
            bn(self.cnum//2),
            nn.PReLU(),
            # 16 x 16
            conv(self.cnum//2, self.cnum//4, 7, 2, dilation=2),
            bn(self.cnum//4),
            nn.PReLU(),
            # 39 x 39
            conv(self.cnum//4, self.cnum//8, 5, 2, dilation=2, output_padding=1),
            bn(self.cnum//8),
            nn.PReLU(),
            # 90 x 90
            conv(self.cnum//8, self.in_dim, 3, 1),
        )

    def forward(self, z, mask):
        img = self.upsample(z)
        img = img.view(img.shape[0], self.cnum, 4, 4)
        img = self.model(img)
        
        return None, img, None
    

class GlobalDis(nn.Module):
    def __init__(self, config, image_shape, mask_shape, bnd, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.in_dim = config['input_dim']

        # TODO 3D + conv in D
        self.model = nn.Sequential(
            nn.Linear(int(image_shape[0] * (mask_shape[0] + 2*bnd)**2 * config['depth']), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity