import numpy as np
import torch
import torch.nn as nn
import os

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class BPJFNN(nn.Module):
    def __init__(self, args):
        super(BPJFNN, self).__init__()
        dim = args['model']['dim']
        self.emb = nn.Embedding(args['model']['word_num'], dim, padding_idx=0)
        # self.emb = nn.Embedding.from_pretrained(
            # torch.from_numpy(np.load( os.path.join(args['dataset']['path'], 'emb.npy') )),
            # freeze=False,
            # padding_idx=0
        # )
        hd_size = args['model']['LSTM_hidden_dim']

        self.job_biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.job_pool = nn.AdaptiveAvgPool2d((1, hd_size * 2))

        self.geek_biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.geek_pool = nn.AdaptiveAvgPool2d((1, hd_size * 2))

        self.mlp = MLP(
            input_size=hd_size * 2 * 3,
            output_size=1
        )

    def forward(self, geek_sent, job_sent):
        geek_vec, job_vec = self.emb(geek_sent), self.emb(job_sent)
        geek_vec, _ = self.geek_biLSTM(geek_vec)
        job_vec, _ = self.job_biLSTM(job_vec)
        geek_vec, job_vec = self.geek_pool(geek_vec).squeeze(1), self.job_pool(job_vec).squeeze(1)
        x = torch.cat([job_vec, geek_vec, job_vec - geek_vec], dim=1)
        x = self.mlp(x).squeeze(1)
        return x

