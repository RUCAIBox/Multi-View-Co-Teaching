import numpy as np
import torch
import torch.nn as nn
import os

class TextCNN(nn.Module):
    def __init__(self, channels, kernel_size, pool_size, dim, method='max'):
        super(TextCNN, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[0]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[1]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, dim))
        )
        if method is 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, dim))
        elif method is 'mean':
            self.pool = nn.AdaptiveAvgPool2d((1, dim))
        else:
            raise ValueError('method {} not exist'.format(method))

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x).squeeze(2)
        x = self.pool(x).squeeze(1)
        return x

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class PJFNN(nn.Module):
    def __init__(self, args):
        super(PJFNN, self).__init__()
        dim = args['model']['dim']
        self.emb = nn.Embedding(args['model']['word_num'], dim, padding_idx=0)
        # self.emb = nn.Embedding.from_pretrained(
        #    torch.from_numpy(np.load( os.path.join(args['dataset']['path'], 'emb.npy') )),
        #    freeze=False,
        #    padding_idx=0
        #)

        self.geek_layer = TextCNN(
            channels=args['dataset']['max_sent_num']['geek'],
            kernel_size=[(5, 1), (3, 1)],
            pool_size=(2, 1),
            dim=dim,
            method='max'
        )

        self.job_layer = TextCNN(
            channels=args['dataset']['max_sent_num']['job'],
            kernel_size=[(5, 1), (5, 1)],
            pool_size=(2, 1),
            dim=dim,
            method='mean'
        )

        self.mlp = MLP(
            input_size=dim,
            output_size=1
        )

    def forward(self, geek_vec, job_vec):
        geek_vec, job_vec = self.emb(geek_vec), self.emb(job_vec)
        geek_vec, job_vec = self.geek_layer(geek_vec), self.job_layer(job_vec)
        x = geek_vec * job_vec
        x = self.mlp(x).squeeze(1)
        return x
