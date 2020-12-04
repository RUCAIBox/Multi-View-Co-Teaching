import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class SelfAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # (N, L, D)
        a = self.attn(x)        # (N, L, 1)
        x = (x * a).sum(dim=1)  # (N, D)
        return x

class JobLayer(nn.Module):
    def __init__(self, dim, hd_size):
        super().__init__()
        self.attn1 = SelfAttentionEncoder(dim)
        self.biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attn2 = SelfAttentionEncoder(dim)

    def forward(self, x):
        # (N, S, L, D)
        x = x.permute(1, 0, 2, 3)   # (S, N, L, D)
        x = torch.cat([self.attn1(_).unsqueeze(0) for _ in x])   # (S, N, D)
        s = x.permute(1, 0, 2)      # (N, S, D)
        c = self.biLSTM(s)[0]       # (N, S, D)
        g = self.attn2(c)           # (N, D)
        return s, g

class CoAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.U = nn.Linear(dim, dim, bias=False)
        self.attn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
            nn.Softmax(dim=0)
        )

    def forward(self, x, s):
        # (N, L, D), (N, S2, D)
        s = s.permute(1, 0, 2)  # (S2, N, D)
        y = torch.cat([ self.attn( self.W(x.permute(1, 0, 2)) + self.U( _.expand(x.shape[1], _.shape[0], _.shape[1]) ) ).permute(2, 0, 1) for _ in s ]).permute(2, 0, 1)
        # (N, D) -> (L, N, D) -> (L, N, 1) -- softmax as L --> (L, N, 1) -> (1, L, N) -> (S2, L, N) -> (N, S2, L)
        sr = torch.cat([torch.mm(y[i], _).unsqueeze(0) for i, _ in enumerate(x)])   # (N, S2, D)
        sr = torch.mean(sr, dim=1)  # (N, D)
        return sr

class GeekLayer(nn.Module):
    def __init__(self, dim, hd_size):
        super().__init__()
        self.co_attn = CoAttentionEncoder(dim)
        self.biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.self_attn = SelfAttentionEncoder(dim)

    def forward(self, x, s):
        # (N, S1, L, D), (N, S2, D)
        x = x.permute(1, 0, 2, 3)   # (S1, N, L, D)
        sr = torch.cat([self.co_attn(_, s).unsqueeze(0) for _ in x])   # (S1, N, D)
        u = sr.permute(1, 0, 2)     # (N, S1, D)
        c = self.biLSTM(u)[0]       # (N, S1, D)
        g = self.self_attn(c)       # (N, D)
        return g

class APJFNN(nn.Module):
    def __init__(self, args):
        super(APJFNN, self).__init__()
        dim = args['model']['dim']
        self.emb = nn.Embedding(args['model']['word_num'], dim, padding_idx=0)
        hd_size = args['model']['LSTM_hidden_dim']

        self.geek_biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.job_biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.job_layer = JobLayer(dim * 2, hd_size)
        self.geek_layer = GeekLayer(dim * 2, hd_size)

        self.mlp = MLP(
            input_size=hd_size * 2 * 3,
            output_size=1
        )

    def forward(self, geek_sent, job_sent):
        geek_vecs, job_vecs = self.emb(geek_sent), self.emb(job_sent)
        geek_vecs = torch.cat([self.geek_biLSTM(_)[0].unsqueeze(0) for _ in geek_vecs])
        job_vecs = torch.cat([self.job_biLSTM(_)[0].unsqueeze(0) for _ in job_vecs])
        
        sj, gj = self.job_layer(job_vecs)
        gr = self.geek_layer(geek_vecs, sj)

        x = torch.cat([gj, gr, gj - gr], axis=1)
        x = self.mlp(x).squeeze(1)
        return x
