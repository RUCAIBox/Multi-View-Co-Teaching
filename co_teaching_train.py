import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from dataset import load_sents, PJFDataset
from model import PJFNN
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

pjfnn1 = PJFNN(args).to(device)
pjfnn2 = PJFNN(args).to(device)

criterion_weight = nn.BCEWithLogitsLoss(reduction = 'none')
criterion = nn.BCEWithLogitsLoss()

optimizer1 = torch.optim.Adam(pjfnn1.parameters(), lr=args['train']['lr'])
optimizer2 = torch.optim.Adam(pjfnn2.parameters(), lr=args['train']['lr'])
print(pjfnn1)
print(pjfnn2)
print(get_parameter_number(pjfnn1))
print(get_parameter_number(pjfnn2))

time_print('Starting train process.')

best_acc = 0
best_epoch = 0
best_result = (0, 0, 0, 0, 0, 0)

def metrics(text, label, pre):
    roc_auc = roc_auc_score(label, pre)
    TP, FN, FP, TN = classify(label, pre)
    tot = TP + FN + FP + TN
    acc = (TP + TN) / tot
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    if (precision + recall) != 0:
        f1score = 2 * precision * recall / (precision + recall)
    else:
        f1score = 0
    epoch_info = text+'[epoch-{}]\n\tROC_AUC:\t{}\n\tACC:\t\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}'.format(
        epoch+1, roc_auc, acc, precision, recall, f1score)
    time_print(epoch_info)
    return acc,roc_auc, precision, recall, f1score

total_step = len(train_loader)
for epoch in range(args['num_epochs']):
    pjfnn1.train()
    pjfnn2.train()
    for i, (geek_sent, job_sent, labels) in enumerate(train_loader):
        geek_sent, job_sent, labels = geek_sent.to(device), job_sent.to(device), labels.to(device)

        geek_sent1 , geek_sent2 = torch.chunk(geek_sent, 2, 0)
        job_sent1 , job_sent2 = torch.chunk(job_sent, 2, 0)
        labels1, labels2 = torch.chunk(labels, 2, 0)

        outputs2 = pjfnn1(geek_sent2, job_sent2)
        outputs1 = pjfnn2(geek_sent1, job_sent1)

        loss_weight1 = criterion_weight(outputs1, labels1)
        loss_weight2 = criterion_weight(outputs2, labels2)

        # 计算权重
        weight1 = torch.add(torch.ones(labels1.size()).float().to(device), 0.2,
                       torch.mul(torch.sub(labels1, torch.ones(labels1.size()).float().to(device)),
                                 loss_weight1))
        weight2 = torch.add(torch.ones(labels2.size()).float().to(device), 0.2,
                       torch.mul(torch.sub(labels2, torch.ones(labels2.size()).float().to(device)),
                                 loss_weight2))

        exchange_outputs1 = pjfnn1(geek_sent1, job_sent1)
        exchange_outputs2 = pjfnn2(geek_sent2, job_sent2)

        loss1 = criterion(exchange_outputs1, weight1.detach() * labels1)
        loss2 = criterion(exchange_outputs2, weight2.detach() * labels2)

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        if (i+1)%1000 == 0:
            time_print('Teach1 Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format( epoch+1, args['num_epochs'], i+1, total_step, loss1.item() ))
            time_print('Teach2 Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch + 1, args['num_epochs'], i + 1, total_step, loss2.item()))
            sys.stdout.flush()

    pre_all1 = []
    pre_all2 = []
    pre_all_ave = []
    label_all = []

    pjfnn1.eval()
    pjfnn2.eval()
    with torch.no_grad():
        for geek_sent, job_sent, labels in valid_loader:
            geek_sent, job_sent = geek_sent.to(device), job_sent.to(device)

            outputs1 = pjfnn1(geek_sent, job_sent)
            outputs2 = pjfnn2(geek_sent, job_sent)
            outputs1 = torch.sigmoid(outputs1)
            outputs2 = torch.sigmoid(outputs2)
            outputs_ave = (outputs1+outputs2)/2
            pre1 = [x.item() for x in outputs1.cpu()]
            pre2 = [x.item() for x in outputs2.cpu()]
            pre_ave = [x.item() for x in outputs_ave.cpu()]
            label = [x.item() for x in labels]
            pre_all1 += pre1
            pre_all2 += pre2
            pre_all_ave += pre_ave
            label_all += label

    acc1 , roc_auc1, precision1, recall1, f1score1= metrics("Eval 1 ", label_all, pre_all1)
    acc2 , roc_auc2, precision2, recall2, f1score2= metrics("Eval 2 ", label_all, pre_all2)
    acc_ave , roc_auc_ave, precision_ave, recall_ave, f1score_ave = metrics("Eval ave ", label_all, pre_all_ave)

    if acc2 > best_acc:
        best_acc = acc2
        best_epoch = epoch + 1
        best_result = (best_epoch, roc_auc2, acc2, precision2, recall2, f1score2)
        torch.save(pjfnn1.state_dict(), os.path.join(args['saved_path'], 'model1-{}.ckpt'.format(int(epoch+1))))
        torch.save(pjfnn2.state_dict(), os.path.join(args['saved_path'], 'model2-{}.ckpt'.format(int(epoch+1))))


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
pjfnn1.load_state_dict(torch.load(os.path.join(args['saved_path'], 'model1-{}.ckpt'.format(int(best_epoch)))))
pjfnn2.load_state_dict(torch.load(os.path.join(args['saved_path'], 'model2-{}.ckpt'.format(int(best_epoch)))))


pjfnn1.eval()
pjfnn2.eval()

pre_all_t1 = []
pre_all_t2 = []
label_all = []

with torch.no_grad():
    for geek_sent, job_sent, labels in test_loader:
        geek_sent, job_sent = geek_sent.to(device), job_sent.to(device)

        outputs1 = pjfnn1(geek_sent, job_sent)
        outputs2 = pjfnn2(geek_sent, job_sent)
        outputs1 = torch.sigmoid(outputs1)
        outputs2 = torch.sigmoid(outputs2)
        pre1 = [x.item() for x in outputs1.cpu()]
        pre2 = [x.item() for x in outputs2.cpu()]
        label = [x.item() for x in labels]
        pre_all_t1 += pre1
        pre_all_t2 += pre2
        label_all += label

        # bad_cases_file = open('bad.cases', 'w', encoding='utf-8')
        # save_bad_case(id2word, bad_cases_file, pre, label, [_.tolist() for _ in sents])
        # bad_cases_file.close()

metrics("Eval 1 ", label_all, pre_all_t1)
metrics("Eval 2 ", label_all, pre_all_t2)

