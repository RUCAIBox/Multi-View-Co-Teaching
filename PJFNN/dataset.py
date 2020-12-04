import sys, os, random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def load_sents(token, args):
    filepath = args['dataset']['path']
    max_sent_num = args['dataset']['max_sent_num']
    max_sent_len = args['dataset']['max_sent_len']

    sents = {}
    sent_num = {}
    tensor_size = [max_sent_num[token], max_sent_len[token]]
    filepath = os.path.join(filepath, '{}.sent.id'.format(token))

    print('\nLoading from {}'.format(filepath))
    sys.stdout.flush()
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            idx, sent = line.strip().split('\t')
            if idx not in sents:
                sents[idx] = torch.zeros(tensor_size).long()
                sent_num[idx] = 0
            if sent_num[idx] == tensor_size[0]: continue
            sent = torch.LongTensor([int(x) for x in sent.split(' ')])
            sents[idx][sent_num[idx]] = F.pad(sent, (0, tensor_size[1] - len(sent)))
            sent_num[idx] += 1
    return sents

class PJFDataset(Dataset):
    '''
    geek_sents: dict[str, LongTensor(max_sent_num['geek'], max_sent_len['geek'])]
    job_sents: dict[str, LongTensor(max_sent_num['job'], max_sent_len['job'])]
    '''
    def __init__(self, geek_sents, job_sents, args, token):
        super(PJFDataset, self).__init__()
        filepath = args['dataset']['path']
        self.geek_sents = geek_sents
        self.job_sents = job_sents
        self.pairs, self.labels = self.load_pairs(filepath, token)

    def load_pairs(self, filepath, token):
        pairs = []
        labels = []
        assert token in ['train', 'test', 'valid']
        filepath = os.path.join(filepath, 'data.{}'.format(token))

        print('\nLoading from {}'.format(filepath))
        sys.stdout.flush()
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                geek_id, job_id, label = line.strip().split('\t')
                if geek_id not in self.geek_sents or job_id not in self.job_sents: continue
                pairs.append([geek_id, job_id])
                labels.append(int(label))
        return pairs, torch.FloatTensor(labels)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        geek_sent = self.geek_sents[pair[0]]
        job_sent = self.job_sents[pair[1]]
        return geek_sent, job_sent, self.labels[index]


