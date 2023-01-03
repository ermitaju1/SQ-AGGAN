#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

from TimeDistributedLayer import TimeDistributedConv2d, TimeDistributedMaxPool, TimeDistributedUpsampling
from BiConvLSTM import BiConvLSTM

###Generator
class DeepSequentialNet(nn.Module):
    def __init__(self, num_sequence, feature_dim, device):
        super(DeepSequentialNet, self).__init__()
        self.num_sequence = num_sequence
        self.feature_dim = feature_dim
        self.device = device

        self.encoding_block1 = nn.Sequential(
            TimeDistributedConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),     ### input channel = 1(intensity) + 5(slice number info) = 6 
            nn.ELU(inplace=True),
            TimeDistributedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True)
        )
        self.encoding_block2 = nn.Sequential(
            TimeDistributedConv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False, dropout=True),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, dropout=True),
            nn.ELU(inplace=True)
        )
        self.encoding_block3 = nn.Sequential(
            TimeDistributedConv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False, dropout=True),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, dropout=True),
            nn.ELU(inplace=True)
        )

        self.fc_feature = nn.Sequential(
            nn.Linear(self.feature_dim, self.num_sequence*256*8*8, bias=False),
            nn.BatchNorm1d(self.num_sequence*256*8*8)
        )
        
        self.biCLSTM1 = BiConvLSTM(input_size=(8,8), input_dim=256*2, hidden_dim=512, kernel_size=(3,3), num_layers=3, device=self.device)

        self.decoding_block3 = nn.Sequential(
            TimeDistributedConv2d(512+256, 256, kernel_size=3, stride=1, padding=1, bias=False, dropout=True),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, dropout=True),
            nn.ELU(inplace=True)
        )
        self.decoding_block2 = nn.Sequential(
            TimeDistributedConv2d(256+128, 128, kernel_size=3, stride=1, padding=1, bias=False, dropout=True),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, dropout=True),
            nn.ELU(inplace=True)
        )
        self.decoding_block1 = nn.Sequential(
            TimeDistributedConv2d(128+64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True)
        )

        self.biCLSTM2 = BiConvLSTM(input_size=(64,64), input_dim=64, hidden_dim=64, kernel_size=(3,3), num_layers=3, device=self.device)
        
        self.maxpooling = TimeDistributedMaxPool(2, stride=2)
        self.upsampling = TimeDistributedUpsampling(scale_factor=2, mode='nearest')
        
        self.onebyoneConv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, sequence, feature):
        ### encoding
        encoded_vol1 = self.encoding_block1(sequence)
        maxpooled_encoded_vol1 = self.maxpooling(encoded_vol1)
        encoded_vol2 = self.encoding_block2(maxpooled_encoded_vol1)
        maxpooled_encoded_vol2 = self.maxpooling(encoded_vol2)
        encoded_vol3 = self.encoding_block3(maxpooled_encoded_vol2)
        maxpooled_encoded_vol3 = self.maxpooling(encoded_vol3)

        ### feature embedding
        feature_code = self.fc_feature(feature)
        feature_code = feature_code.view(-1, self.num_sequence, 256, 8, 8)

        ### concatenate feature with encoding code
        sequence_feature_code = torch.cat((maxpooled_encoded_vol3, feature_code), 2)

        ### lSTM
        lstm_vol1 = self.biCLSTM1(sequence_feature_code)

        ### decoding
        up_vol3 = self.upsampling(lstm_vol1)
        concat_vol3 = torch.cat((encoded_vol3, up_vol3), 2)
        decoded_vol3 = self.decoding_block3(concat_vol3)
        up_vol2 = self.upsampling(decoded_vol3)
        concat_vol2 = torch.cat((encoded_vol2, up_vol2), 2)
        decoded_vol2 = self.decoding_block2(concat_vol2)
        up_vol1 = self.upsampling(decoded_vol2)
        concat_vol1 = torch.cat((encoded_vol1, up_vol1), 2)
        decoded_vol1 = self.decoding_block1(concat_vol1)

        ### LSTM
        lstm_vol2 = self.biCLSTM2(decoded_vol1)
        lstm_vol2 = torch.sum(lstm_vol2, 1)

        ### Last PANG!~
        synth_code = self.onebyoneConv(lstm_vol2)

        # print("encoded_vol1 : ", encoded_vol1.shape)
        # print("maxpooled_encoded_vol1 : ", maxpooled_encoded_vol1.shape)
        # print("encoded_vol2 : ", encoded_vol2.shape)
        # print("maxpooled_encoded_vol1 : ", maxpooled_encoded_vol2.shape)
        # print("encoded_vol3 : ", encoded_vol3.shape)
        # print("maxpooled_encoded_vol1 : ", maxpooled_encoded_vol3.shape)
        # print("lstm_vol1 : ", lstm_vol1.shape)
        # print("up_vol3 : ", up_vol3.shape)
        # print("concat_vol3 : ", concat_vol3.shape)
        # print("decoded_vol3 : ", decoded_vol3.shape)
        # print("up_vol2 : ", up_vol2.shape)
        # print("concat_vol2 : ", concat_vol2.shape)
        # print("decoded_vol2 : ", decoded_vol2.shape)
        # print("up_vol1 : ", up_vol1.shape)
        # print("concat_vol1 : ", concat_vol1.shape)
        # print("decoded_vol1 : ", decoded_vol1.shape)
        # print("lstm_vol2 : ", lstm_vol2.shape)
        # print("synth_code : ", synth_code.shape)

        return synth_code


#in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),            ### 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),       ### 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1, bias=False),   ### 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(32 * 4, 32 * 8, 4, 2, 1, bias=False),   ### 8 -> 4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32 * 8, 1, 4, 2, 1, bias=False),   ### 4 -> 4
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.feature_fc = nn.Sequential(
            nn.Linear(1*2*2, self.feature_dim, bias=False)
            
        )
        self.yn_fc = nn.Sequential(
            nn.Linear(1*2*2, 1, bias=False)
        )

    def forward(self, pred_slice):
        output = self.encoder(pred_slice)     ### (b, 1, 2, 2)
        
        output_flatten = output.view(pred_slice.shape[0], -1)
        cl_feature = self.feature_fc(output_flatten)
        cl_yn = self.yn_fc(output_flatten)
        return output, cl_feature, cl_yn.view(-1)


######cl_feature cl_yn의 역할이 뭘까

