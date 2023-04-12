import torch
import torch.nn as nn

class UNET(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super().__init__()
        self.in_channels = config['input_dim']
        self.cnum = config['ngf'] #variable cnum for in_channels
        self.out_channels = config['input_dim']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = self.contract_block(self.in_channels + 2, self.cnum, 7, 3)
        self.conv2 = self.contract_block(self.cnum, self.cnum * 2, 3, 1)
        self.conv3 = self.contract_block(self.cnum * 2, self.cnum * 4, 3, 1)

        self.upconv3 = self.expand_block(self.cnum * 4, self.cnum * 2, 3, 1)
        self.upconv2 = self.expand_block(self.cnum * 2 *2, 32, 3, 1)
        self.upconv1 = self.expand_block(self.cnum *2, self.out_channels, 3, 1)

    def __call__(self, x, mask):

        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()

        xnow = torch.cat([x, ones, mask], dim=1)

        # downsampling part
        conv1 = self.conv1(xnow)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand