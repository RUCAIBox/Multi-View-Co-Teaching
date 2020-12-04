import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from dataset import load_sents, PJFDataset
from model import BPJFNN
from sklearn.metrics import roc_auc_score

args = load_config('config.json')
# id2word = load_word_id(os.path.join(args['dataset']['path'], 'word.id'))

time_print('Starting training [{}].'.format(args['token']))

geek_sents = load_sents('geek', args)
job_sents = load_sents('job', args)

train_dataset = PJFDataset(geek_sents, job_sents, args, token='train')
valid_dataset = PJFDataset(geek_sents, job_sents, args, token='valid')
test_dataset = PJFDataset(geek_sents, job_sents, args, token='test')

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args['train']['batch_size'],
    shuffle=True,
    num_workers=args['train']['num_workers']
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=args['valid']['batch_size'],
    shuffle=False,
    num_workers=args['valid']['num_workers']
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args['test']['batch_size'],
    shuffle=False,
    num_workers=args['test']['num_workers']
)

time_print('Finishing data preparation.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args['gpu'])
# device = 'cpu'

bpjfnn = BPJFNN(args).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(bpjfnn.parameters(), lr=args['train']['lr'])
print(bpjfnn)
print(get_parameter_number(bpjfnn))

time_print('Starting train process.')

best_acc = 0
best_epoch = 0
best_result = (0, 0, 0, 0, 0, 0)

total_step = len(train_loader)
for epoch in range(args['num_epochs']):
    bpjfnn.train()
    for i, (geek_sent, job_sent, labels) in enumerate(train_loader):
        geek_sent, job_sent, labels = geek_sent.to(device), job_sent.to(device), labels.to(device)

        outputs = bpjfnn(geek_sent, job_sent)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%1000 == 0:
            time_print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format( epoch+1, args['num_epochs'], i+1, total_step, loss.item() ))
            sys.stdout.flush()

    pre_all = []
    label_all = []

    bpjfnn.eval()
    with torch.no_grad():
        for geek_sent, job_sent, labels in valid_loader:
            geek_sent, job_sent = geek_sent.to(device), job_sent.to(device)

            outputs = bpjfnn(geek_sent, job_sent)

            outputs = torch.sigmoid(outputs)
            pre = [x.item() for x in outputs.cpu()]
            label = [x.item() for x in labels]
            pre_all += pre
            label_all += label

    roc_auc = roc_auc_score(label_all, pre_all)
    TP, FN, FP, TN = classify(label_all, pre_all)
    tot = TP + FN + FP + TN
    acc = (TP + TN) / tot
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1score = 2 * precision * recall / (precision + recall)
    epoch_info = '[epoch-{}]\n\tROC_AUC:\t{}\n\tACC:\t\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}'.format(
        epoch+1, roc_auc, acc, precision, recall, f1score)
    time_print(epoch_info)

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch + 1
        best_result = (best_epoch, roc_auc, acc, precision, recall, f1score)
        torch.save(bpjfnn.state_dict(), os.path.join(args['saved_path'], 'model-{}.ckpt'.format(int(epoch+1))))

    if epoch+1 >= best_epoch + args['end_step']:
        time_print('Finish after epoch {}'.format(epoch+1))
        sys.stdout.flush()
        keep_only_the_best(args, best_epoch)
        out_epoch, roc_auc, acc, precision, recall, f1score = best_result
        best_epoch_info = 'BEST: [epoch-{}]\n\tROC_AUC:\t{:.4f}\n\tACC:\t\t{:.4f}\n\tPrecision:\t{:.4f}\n\tRecall:\t\t{:.4f}\n\tF1 Score:\t{:.4f}'.format(
            out_epoch, roc_auc, acc, precision, recall, f1score)
        time_print(best_epoch_info)
        break

# test
bpjfnn.load_state_dict( torch.load(os.path.join(args['saved_path'], '{}-best.ckpt'.format(args['token']))) )
bpjfnn.eval()

pre_all = []
label_all = []

with torch.no_grad():
    for geek_sent, job_sent, labels in test_loader:
        geek_sent, job_sent = geek_sent.to(device), job_sent.to(device)

        outputs = bpjfnn(geek_sent, job_sent)
        outputs = torch.sigmoid(outputs)
        pre = [x.item() for x in outputs.cpu()]
        label = [x.item() for x in labels]
        pre_all += pre
        label_all += label

        # bad_cases_file = open('bad.cases', 'w', encoding='utf-8')
        # save_bad_case(id2word, bad_cases_file, pre, label, [_.tolist() for _ in sents])
        # bad_cases_file.close()

roc_auc = roc_auc_score(label_all, pre_all)
TP, FN, FP, TN = classify(label_all, pre_all)
tot = TP + FN + FP + TN
acc = (TP + TN) / tot
recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1score = 2 * precision * recall / (precision + recall)
test_info = '[{}-test]\n\tROC_AUC:\t{:.4f}\n\tACC:\t\t{:.4f}\n\tPrecision:\t{:.4f}\n\tRecall:\t\t{:.4f}\n\tF1 Score:\t{:.4f}'.format(
    args['token'], roc_auc, acc, precision, recall, f1score)
time_print(test_info)
