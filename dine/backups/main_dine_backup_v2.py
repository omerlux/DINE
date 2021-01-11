import argparse
import os, sys
import re
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

import gc

import data
import model

from utils import batchify, batchify_f1, batchify_f2, get_batch, get_batch_dine, repackage_hidden, create_exp_dir, \
    save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank Language Model')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--ndim', type=int, default=1,
                    help='inputs vectors dimension')
parser.add_argument('--ncell', type=int, default=16,
                    help='number of LSTMs')
parser.add_argument('--lr1', type=float, default=0.0001,
                    help='initial learning rate, 1e-4 for adam, 1e-2 for sgd')
parser.add_argument('--lr2', type=float, default=0.00008,
                    help='initial learning rate, 1e-4 for adam, 1e-2 for sgd')
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='type of optimizer (Adam, SGD)')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=14, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='EXP',
                    help='path to save the final model')
parser.add_argument('--start_save', type=int, default=200,
                    help='start save validation data from this epoch')
parser.add_argument('--wdecay', type=float, default=0,  # 1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=40,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
parser.add_argument('--gpu_device', type=str, default="0",
                    help='specific use of gpu')
parser.add_argument('--data', type=str, default="AWGN",
                    help='specific use of gpu')
parser.add_argument('--N', type=float, default=1,
                    help='variance of Noise gaussian')
parser.add_argument('--P', type=float, default=1,
                    help='variance of X gaussian')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='alpha * Ui-1 + Ui + Xi = Yi')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
# --- pytorch 1.2 warning spam ignoring ---
import warnings

warnings.filterwarnings('ignore')

if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

if not args.continue_train:
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=['main_ptb.py', 'model.py'])

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

###############################################################################
# Load data
###############################################################################

eval_batch_size = args.batch_size #10
test_batch_size = args.batch_size #1

train_length = 1000000
valid_length = 1000000

###############################################################################
# Build the model
###############################################################################

if args.continue_train:
    model_f1 = torch.load(os.path.join(args.save, 'model_f1.pt'))
    model_f2 = torch.load(os.path.join(args.save, 'model_f2.pt'))
    file = open(os.path.join(args.save, 'log.txt'))
    old_args = file.readline()
    nums = re.findall("[^a-zA-Z:](\-?\d+[\.]?\d*)", old_args)
    args.N = float(re.findall("N=([-+]?\d*\.*\d+)", old_args)[0])
    args.P = float(re.findall("P=([-+]?\d*\.*\d+)", old_args)[0])
    args.batch_size = int(re.findall("batch_size=([-+]?\d*\.*\d+)", old_args)[0])
    args.bptt = int(re.findall("bptt=([-+]?\d*\.*\d+)", old_args)[0])
    args.ndim = int(re.findall("ndim=([-+]?\d*\.*\d+)", old_args)[0])
else:
    model_f1 = model.DIModel(1 * args.ndim, args.nhid, args.ncell, args.wdrop)
    model_f2 = model.DIModel(2 * args.ndim, args.nhid, args.ncell, args.wdrop)
    with open(os.path.join(args.save, 'validvars'), 'wb') as f:
        # Aggregate exponents and outputs to calculate the validation set from the beginning of evaluation time
        pickle.dump([0, 0, 0, 0, 0], f)

if args.cuda:
    if args.single_gpu:
        parallel_model_f1 = model_f1.cuda()
        parallel_model_f2 = model_f2.cuda()
    else:
        parallel_model_f1 = nn.DataParallel(model_f1, dim=1).cuda()
        parallel_model_f2 = nn.DataParallel(model_f2, dim=1).cuda()
else:
    parallel_model_f1 = model_f1
    parallel_model_f2 = model_f2

total_params_f1 = sum(x.data.nelement() for x in model_f1.parameters())
total_params_f2 = sum(x.data.nelement() for x in model_f2.parameters())
logging.info('Args: {}'.format(args))
logging.info('Model F1 total parameters: {}'.format(total_params_f1))
logging.info('Model F2 total parameters: {}'.format(total_params_f2))


###############################################################################
# Training code
###############################################################################

def evaluate(data_source_f1, data_source_f2, batch_size=10):
    with open(os.path.join(args.save, 'validvars'), 'rb') as f:
        # Aggregate exponents and outputs to calculate the validation set from the beginning of evaluation time
        [total_f1, total_f1_exps, total_f2, total_f2_exps, total_samples] = pickle.load(f)

    # Turn on evaluation mode which disables dropout.
    model_f1.eval()
    model_f2.eval()
    hidden_f1 = model_f1.init_hidden(batch_size, model_f1.ncell)
    hidden_f2 = model_f2.init_hidden(batch_size, model_f2.ncell)

    # Creating validation randata
    _val_tmp_randata = data.new_process(args, valid_length)
    # and setting uniform distribution for y~
    valid_randata_f1 = batchify_f1(_val_tmp_randata, batch_size, args, uniformly=True)
    valid_randata_f2 = batchify_f2(_val_tmp_randata, batch_size, args, uniformly=True)

    for i in range(0, data_source_f1.size(0) - 1, args.bptt):
        data_f1 = get_batch_dine(data_source_f1, i, args, evaluation=True)
        data_f2 = get_batch_dine(data_source_f2, i, args, evaluation=True)
        randata_f1 = get_batch_dine(valid_randata_f1, i, args, evaluation=True)
        randata_f2 = get_batch_dine(valid_randata_f2, i, args, evaluation=True)
        # forward
        with torch.no_grad():  # added this line to overcome OOM error
            out_f1, out_reused_f1, hidden_f1 = parallel_model_f1(data_f1, randata_f1, hidden_f1)
            out_f2, out_reused_f2, hidden_f2 = parallel_model_f2(data_f2, randata_f2, hidden_f2)

        # loss aggregation
        total_f1 += torch.sum(out_f1)
        total_f1_exps += torch.sum(torch.exp(out_reused_f1))
        total_f2 += torch.sum(out_f2)
        total_f2_exps += torch.sum(torch.exp(out_reused_f2))
        total_samples += torch.numel(out_f1)

        # hidden repackage
        hidden_f1 = repackage_hidden(hidden_f1)
        hidden_f2 = repackage_hidden(hidden_f2)


    return [total_f1, total_f1_exps, total_f2, total_f2_exps, total_samples]


def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Creating train data
    _train_tmp_data = data.new_process(args, train_length)
    train_data_f1 = batchify_f1(_train_tmp_data, args.batch_size, args)
    train_data_f2 = batchify_f2(_train_tmp_data, args.batch_size, args)

    # Turn on training mode which enables dropout.
    total_loss_fi = [0, 0]
    start_time = time.time()
    hidden_f1 = [model_f1.init_hidden(args.small_batch_size, args.ncell) for _ in
                 range(args.batch_size // args.small_batch_size)]
    hidden_f2 = [model_f2.init_hidden(args.small_batch_size, args.ncell) for _ in
                 range(args.batch_size // args.small_batch_size)]
    # Shuffeling data for y~ (ONLY y!)
    train_randata_f1 = batchify_f1(_train_tmp_data, args.batch_size, args, uniformly=True)
    train_randata_f2 = batchify_f2(_train_tmp_data, args.batch_size, args, uniformly=True)

    log_interval = np.ceil((len(train_data_f1) // args.bptt) / 5)
    batch, i = 0, 0
    while i < train_data_f1.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        # Adjusting learning rate to variable length sequence
        lr_tmp_f1 = optimizer_f1.param_groups[0]['lr']
        optimizer_f1.param_groups[0]['lr'] = lr_tmp_f1 * seq_len / args.bptt
        lr_tmp_f2 = optimizer_f2.param_groups[0]['lr']
        optimizer_f2.param_groups[0]['lr'] = lr_tmp_f2 * seq_len / args.bptt

        # Activating train mode
        model_f1.train()
        model_f2.train()

        data_f1 = get_batch_dine(train_data_f1, i, args, seq_len=seq_len)
        data_f2 = get_batch_dine(train_data_f2, i, args, seq_len=seq_len)
        randata_f1 = get_batch_dine(train_randata_f1, i, args, seq_len=seq_len)
        randata_f2 = get_batch_dine(train_randata_f2, i, args, seq_len=seq_len)

        optimizer_f1.zero_grad()
        optimizer_f2.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data_f1 = data_f1[:, start: end]
            cur_data_f2 = data_f2[:, start: end]
            cur_randata_f1 = randata_f1[:, start: end]
            cur_randata_f2 = randata_f2[:, start: end]

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden_f1[s_id] = repackage_hidden(hidden_f1[s_id])
            hidden_f2[s_id] = repackage_hidden(hidden_f2[s_id])

            out_f1, out_reused_f1, hidden_f1[s_id] = parallel_model_f1(cur_data_f1, cur_randata_f1, hidden_f1[s_id])
            out_f2, out_reused_f2, hidden_f2[s_id] = parallel_model_f2(cur_data_f2, cur_randata_f2, hidden_f2[s_id])

            raw_loss_f1 = torch.mean(out_f1) - torch.log(torch.mean(torch.exp(out_reused_f1)))
            raw_loss_f2 = torch.mean(out_f2) - torch.log(torch.mean(torch.exp(out_reused_f2)))
            loss = [raw_loss_f1, raw_loss_f2]

            loss[0] *= args.small_batch_size / args.batch_size
            loss[1] *= args.small_batch_size / args.batch_size
            total_loss_fi[0] += raw_loss_f1.data * (args.small_batch_size / args.batch_size)
            total_loss_fi[1] += raw_loss_f2.data * (args.small_batch_size / args.batch_size)
            (-loss[0]).backward()  # for gradient ascent we use -loss
            (-loss[1]).backward()

            # optimizer grad check... note: comment this if 'nan' problem solved
            for j, p in enumerate(optimizer_f1.param_groups[0]['params']):
                ans = p.grad.data[torch.isnan(p.grad.data)]
                if ans.size()[0] > 0:
                    logging.info("some nan exists in optimizer_f1 parameter #{} !".format(j))
                    raise KeyboardInterrupt
            for j, p in enumerate(optimizer_f2.param_groups[0]['params']):
                ans = p.grad.data[torch.isnan(p.grad.data)]
                if ans.size()[0] > 0:
                    logging.info("some nan exists in optimizer_f2 parameter #{} !".format(j))
                    raise KeyboardInterrupt

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model_f1.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(model_f2.parameters(), args.clip)

        optimizer_f1.step()
        optimizer_f2.step()

        #learning rate - back to normal
        optimizer_f1.param_groups[0]['lr'] = lr_tmp_f1
        optimizer_f2.param_groups[0]['lr'] = lr_tmp_f2

        if batch % log_interval == 0 and batch > 0:
            curr_loss_fi = [total_loss_fi[0].item() / log_interval,
                            total_loss_fi[1].item() / log_interval]
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:3d}/{:3d} batches | lr1 {:f} | lr2 {:f} | ms/batch {:5.4f} | '
                         'loss F1 {:6.4f} | loss F2 {:6.4f} | loss {:6.4f}'.format(
                epoch, batch, len(train_data_f1) // args.bptt,
                optimizer_f1.param_groups[0]['lr'], optimizer_f2.param_groups[0]['lr'],
                              elapsed * 1000 / log_interval,
                curr_loss_fi[0], curr_loss_fi[1], curr_loss_fi[1] - curr_loss_fi[0]))
            total_loss_fi = [0, 0]
            start_time = time.time()
        ###
        batch += 1
        i += seq_len


# Loop over epochs.
lr_f1 = args.lr1
lr_f2 = args.lr2
best_val_loss = []
stored_loss_f1 = -100000000
stored_loss_f2 = -100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    if args.continue_train:
        optimizer_state_f1 = torch.load(os.path.join(args.save, 'optimizer_f1.pt'))
        optimizer_state_f2 = torch.load(os.path.join(args.save, 'optimizer_f2.pt'))
        if args.optimizer == 'SGD':
            optimizer_f1 = torch.optim.SGD(model_f1.parameters(), lr=args.lr1,
                                           weight_decay=args.wdecay)
            optimizer_f2 = torch.optim.SGD(model_f2.parameters(), lr=args.lr2,
                                           weight_decay=args.wdecay)
        else:  # assuming it's Adam
            optimizer_f1 = torch.optim.Adam(model_f1.parameters(), lr=args.lr1,
                                            weight_decay=args.wdecay)
            optimizer_f2 = torch.optim.Adam(model_f2.parameters(), lr=args.lr2,
                                            weight_decay=args.wdecay)
        optimizer_f1.load_state_dict(optimizer_state_f1)
        optimizer_f2.load_state_dict(optimizer_state_f2)
    else:
        if args.optimizer == 'SGD':
            optimizer_f1 = torch.optim.SGD(model_f1.parameters(), lr=args.lr1,
                                           weight_decay=args.wdecay)
            optimizer_f2 = torch.optim.SGD(model_f2.parameters(), lr=args.lr2,
                                           weight_decay=args.wdecay)
        else:  # assuming it's Adam
            optimizer_f1 = torch.optim.Adam(model_f1.parameters(), lr=args.lr1,
                                            weight_decay=args.wdecay)
            optimizer_f2 = torch.optim.Adam(model_f2.parameters(), lr=args.lr2,
                                            weight_decay=args.wdecay)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        # Creating validation data
        _val_tmp_data = data.new_process(args, valid_length)
        val_data_f1 = batchify_f1(_val_tmp_data, eval_batch_size, args)
        val_data_f2 = batchify_f2(_val_tmp_data, eval_batch_size, args)
        # Evaluation
        val_loss = evaluate(val_data_f1, val_data_f2, batch_size=eval_batch_size)
        if epoch > args.start_save:
            logging.info('Using saved evaluation data!')
            with open(os.path.join(args.save, 'validvars'), 'wb') as f:
                # saving the current aggregated result:
                pickle.dump(val_loss, f)            # = [total_f1, total_f1_exps, total_f2, total_f2_exps, total_samples]
        f1 = torch.div(val_loss[0], val_loss[-1]) - torch.log(torch.div(val_loss[1], val_loss[-1]))
        f2 = torch.div(val_loss[2], val_loss[-1]) - torch.log(torch.div(val_loss[3], val_loss[-1]))
        cap = f2 - f1
        logging.info('-' * 100)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | '
                     'valid loss F1 {:6.4f} | valid loss F2 {:6.4f} | valid loss {:6.4f}'
                     .format(epoch, (time.time() - epoch_start_time),
                             f1, f2, cap))
        logging.info('-' * 100)
        # saving the model
        if f1 > stored_loss_f1:
            save_checkpoint(model_f1, optimizer_f1, args.save, f1=True)
            logging.info('Saving Normal f1!')
            stored_loss = f1

        if f2 > stored_loss_f2:
            save_checkpoint(model_f2, optimizer_f2, args.save, f1=False)
            logging.info('Saving Normal f2!')
            stored_loss = f2

        best_val_loss.append(cap)

except KeyboardInterrupt:
    logging.info('-' * 100)
    logging.info('Exiting from training early')

# Load the best saved model.
model_f1 = torch.load(os.path.join(args.save, 'model_f1.pt'))
model_f2 = torch.load(os.path.join(args.save, 'model_f2.pt'))
parallel_model_f1 = nn.DataParallel(model_f1, dim=1).cuda()
parallel_model_f2 = nn.DataParallel(model_f2, dim=1).cuda()

# Creating test data
_test_tmp_data = data.new_process(args, valid_length)
test_data_f1 = batchify_f1(_test_tmp_data, test_batch_size, args)
test_data_f2 = batchify_f2(_test_tmp_data, test_batch_size, args)
# Run on test data.
test_loss = evaluate(test_data_f1, test_data_f2, batch_size=test_batch_size)
with open(os.path.join(args.save, 'validvars'), 'wb') as f:
    # saving the current aggregated result:
    pickle.dump(test_loss, f)  # = [total_f1, total_f1_exps, total_f2, total_f2_exps, total_samples]
f1 = torch.div(test_loss[0], test_loss[-1]) - torch.log(torch.div(test_loss[1], test_loss[-1]))
f2 = torch.div(test_loss[2], test_loss[-1]) - torch.log(torch.div(test_loss[3], test_loss[-1]))
cap = f2 - f1
logging.info('=' * 100)
logging.info('| End of training | '
             ' test loss F1 {:6.4f} | test loss F2 {:6.4f} | test loss {:6.4f}'
             .format(f1, f2, cap))
logging.info('=' * 100)