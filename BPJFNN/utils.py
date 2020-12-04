import json, sys, os, datetime

def load_config(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        args = json.load(file)
    return args

def load_word_id(filepath):
    id2word = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            wd = line.strip()
            id2word[i] = wd
    return id2word

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def keep_only_the_best(args, best_epoch):
    best_file_path = os.path.join(args['saved_path'], '{}-best.ckpt'.format(args['token']))
    ori_path = os.path.join(args['saved_path'], 'model-{}.ckpt'.format(best_epoch))
    os.system('mv {} {}'.format(ori_path, best_file_path))
    os.system('rm -rf {}'.format(os.path.join(args['saved_path'], 'model-*')))

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%F %T')
    return cur

def time_print(s):
    print('[{}] -- {}'.format(get_local_time(), s))
    sys.stdout.flush()

def classify(ans, pre, threshold=0.5):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    assert len(ans) == len(pre)
    for i in range(len(ans)):
        if int(ans[i]) == 1:
            if pre[i] >= 0.5: TP += 1
            else: FN += 1
        else:
            if pre[i] >= 0.5: FP += 1
            else: TN += 1
    return TP, FN, FP, TN

def save_bad_case(id2word, file, pre, label, sents):
    batch_size = len(pre)
    geek_sents, job_sents = sents
    for i in range(batch_size):
        if (pre[i] >= 0.5 and label[i] == 1) or (pre[i] < 0.5 and label[i] == 0): continue
        file.write('[pre]: {}; [label]: {}\n'.format(pre[i], label[i]))
        if isinstance(geek_sents[i][0], list):
            file.write('[geek text]:\n')
            for sent in geek_sents[i]:
                if sent[0] == 0: break
                wds = []
                for wd_id in sent:
                    if wd_id == 0: break
                    wds.append(id2word[wd_id])
                file.write('{}\n'.format(' '.join(wds)))
            file.write('[job text]:\n')
            for sent in job_sents[i]:
                if sent[0] == 0: break
                wds = []
                for wd_id in sent:
                    if wd_id == 0: break
                    wds.append(id2word[wd_id])
                file.write('{}\n'.format(' '.join(wds)))
        else:
            file.write('[geek text]:\n')
            wds = []
            for wd_id in geek_sents[i]:
                if wd_id == 0: break
                wds.append(id2word[wd_id])
            file.write('{}\n'.format(' '.join(wds)))
            file.write('[job text]:\n')
            wds.clear()
            for wd_id in job_sents[i]:
                if wd_id == 0: break
                wds.append(id2word[wd_id])
            file.write('{}\n'.format(' '.join(wds)))
        file.write('\n')

