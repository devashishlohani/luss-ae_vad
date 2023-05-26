import torch.nn as nn
from functools import reduce
from operator import mul
import torch
from torchsummary import summary
import os
from .utils import *

class projection_head(nn.Module):
    def __init__(self, input_chn=256, conv_chn=32):
        super(projection_head, self).__init__()
        self.st_pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.s_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d = nn.Conv2d(input_chn, conv_chn, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.st_pool(x)
        x = x[:, :, 0, :, :] #from bx256x1x8x8 to bx256x8x8
        x = self.s_pool(self.conv2d(x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        return x #(2,512)

class embedding_1(nn.Module):
    def __init__(self, in_size=512, out_size=128):
        super(embedding_1, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fc(x))
        return x

class embedding_2(nn.Module):
    def __init__(self, in_size=128, out_size=32):
        super(embedding_2, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fc(x))
        return x

class class_predictor_v2(nn.Module):
    def __init__(self, in_size=32, num_classes=2):
        super(class_predictor_v2, self).__init__()
        self.fc = nn.Linear(in_size, num_classes)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x

class Reconstruction3DEncoder(nn.Module):
    def __init__(self, chnum_in):
        super(Reconstruction3DEncoder, self).__init__()

        # Dong Gong's paper code
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder_recon_pred(nn.Module):

    def __init__(self, chnum_in):
        super(Decoder_recon_pred, self).__init__()

        self.chnum_in = chnum_in
        feature_num = 96
        feature_num_x2 = 128
        feature_num_x4 = 256
        #feature_num_x8 = 512

        self.Tan = nn.Tanh()

        def st_deconv_block(intInput, intOutput):
            return torch.nn.Sequential(
                nn.ConvTranspose3d(intInput, intOutput, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),output_padding=(1, 1, 1)),#, bias=False),
                nn.BatchNorm3d(intOutput),
                nn.LeakyReLU(0.2, inplace=True)
            )

        def s_deconv(intInput, intOutput):
            return torch.nn.Sequential(
                nn.ConvTranspose3d(intInput, intOutput, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1))
            )

        def s_pred(intInput, intOutput):
            return torch.nn.Sequential(
                nn.Conv3d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3, 3), stride=(16, 1, 1),
                      padding=(1, 1, 1))
            )

        def s_pred(intInput, intOutput):
            return torch.nn.Sequential(
                nn.Conv3d(in_channels=intInput, out_channels=intOutput, kernel_size=(3,3,3), stride=(16, 1, 1), padding=(1, 1, 1)),
                nn.Tanh()
            )

        self.decoder = nn.Sequential(
            st_deconv_block(feature_num_x4, feature_num_x4),
            st_deconv_block(feature_num_x4, feature_num_x2),
            st_deconv_block(feature_num_x2, feature_num),
            s_deconv(feature_num, self.chnum_in)
        )
        self.prediction = s_pred(self.chnum_in, self.chnum_in)

    def forward(self, x):
        recons = self.Tan(self.decoder(x))
        pred = self.prediction(self.decoder(x))
        return pred, recons

if __name__ == '__main__':
    device = torch.device("cuda" if True else "cpu")
    chnum_in_ = 1
    model = Decoder_recon_pred(chnum_in_)
    model.to(device)
    print(summary(model, (1, 16, 256, 256)))
