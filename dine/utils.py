import os, shutil
import torch
import numpy as np


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    # if isinstance(h, tuple) or isinstance(h, list):
    #     return tuple(repackage_hidden(v) for v in h)
    # else:
    #     return h.detach()

    # Ben's safe repackage
    if hasattr(h, 'detach'):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    if args.cuda:
        data = data.cuda()
    return data


def batchify_f1(data, bsz, args, uniformly=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data['labels'].size(0) // bsz
    tmp_labels = data['labels']
    # Shuffeling data labels if needed
    if uniformly:
        ## This will take uniformly distributed values:
        min = torch.min(tmp_labels)
        max = torch.max(tmp_labels)
        tmp_labels = tmp_labels.data.new(tmp_labels.size()).uniform_(min, max)
        ## This will take normal distributed values, as the labels made from:
        # tmp_labels = tmp_labels[torch.randperm(n=tmp_labels.size()[0])]
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data_ret = tmp_labels.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data_ret = data_ret.view(bsz, -1).t().contiguous().unsqueeze(2)
    if not uniformly:
        print(data_ret.size())
    if args.cuda:
        data_ret = data_ret.cuda()
    return data_ret


def batchify_f2(data, bsz, args, uniformly=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data['labels'].size(0) // bsz
    tmp_labels = data['labels']
    # Shuffeling data labels if needed
    if uniformly:
        ## This will take uniformly distributed values:
        min = torch.min(tmp_labels)
        max = torch.max(tmp_labels)
        tmp_labels = tmp_labels.data.new(tmp_labels.size()).uniform_(min, max)
        ## This will take normal distributed values, as the labels made from:
        # tmp_labels = tmp_labels[torch.randperm(n=tmp_labels.size()[0])]
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    x = data['features'].narrow(0, 0, nbatch * bsz)
    y = tmp_labels.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    x = x.view(bsz, -1).t().contiguous().unsqueeze(2)
    y = y.view(bsz, -1).t().contiguous().unsqueeze(2)
    xy = torch.cat((x, y), 2)
    if not uniformly:
        print(xy.size())
    if args.cuda:
        xy = xy.cuda()
    return xy


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len].detach() if evaluation else source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]
    return data, target


def get_batch_dine(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len].detach() if evaluation else source[i:i + seq_len]
    return data


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, path, finetune=False, f1=False):
    if finetune:
        if f1:
            torch.save(model, os.path.join(path, 'finetune_model_f1.pt'))
            torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer_f1.pt'))
        else:
            torch.save(model, os.path.join(path, 'finetune_model_f2.pt'))
            torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer_f2.pt'))

    else:
        if f1:
            torch.save(model, os.path.join(path, 'model_f1.pt'))
            torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_f1.pt'))
        else:
            torch.save(model, os.path.join(path, 'model_f2.pt'))
            torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_f2.pt'))
