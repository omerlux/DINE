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
import matplotlib.pyplot as plt

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
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
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

inp = input("Insert relative path from ./dine/ folder: ")
# dim = int(input("Insert vector dimension: "))
db = float(input("Insert dB SNR value (-999 for no dB mentioned): "))
x_inp = float(input("Insert a value for X: "))
process = input("Insert process type: ")
args.save = inp

if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

assert not args.continue_train, 'Need to work on a trained model...'

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

file = open(os.path.join(args.save, 'log.txt'))
old_args = file.readline()
nums = re.findall("[^a-zA-Z:](\-?\d+[\.]?\d*)", old_args)
args.N = float(nums[7])
args.P = float(nums[8])
args.batch_size = int(nums[10])
args.bptt = int(nums[11])
args.small_batch_size = int(nums[-3])
args.ncell = int(nums[-11])
args.ndim = int(nums[-10])
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

eval_batch_size = args.batch_size  # 10
test_batch_size = args.batch_size  # 1

valid_length = 10000

###############################################################################
# Build the model
###############################################################################

model_f2 = torch.load(os.path.join(args.save, 'model_f2.pt'))

if args.cuda:
    if args.single_gpu:
        parallel_model_f2 = model_f2.cuda()
    else:
        parallel_model_f2 = nn.DataParallel(model_f2, dim=1).cuda()
else:
    parallel_model_f2 = model_f2

total_params_f2 = sum(x.data.nelement() for x in model_f2.parameters())
logging.info('Args: {}'.format(args))
logging.info('Model F2 total parameters: {}'.format(total_params_f2))


###############################################################################
# Training code
###############################################################################

def evaluate_dist(x, support):
    y_vec = np.arange(x - 7.5, x + 7.5, 0.05)
    y_givenx = np.zeros(len(y_vec))
    for k, y in enumerate(y_vec):
        batch_size = 1
        # Creating test data - for x,y respectively
        data_xy = {'features': (torch.ones(50, batch_size, args.ndim) * x),
                   'labels': (torch.ones(50, batch_size, args.ndim) * y)}
        test_data_f2 = batchify_f2(data_xy, batch_size, args)
        # setting uniform distribution for y~
        test_randata_f2 = batchify_f2(data_xy, batch_size, args, uniformly=True)

        # Turn on evaluation mode which disables dropout.
        model_f2.eval()
        hidden_f2 = model_f2.init_hidden(batch_size, model_f2.ncell)
        dist_vector = torch.FloatTensor()

        for i in range(0, test_data_f2.size(0) - 1, args.bptt):
            data_f2 = get_batch_dine(test_data_f2, i, args, evaluation=True)
            randata_f2 = get_batch_dine(test_randata_f2, i, args, evaluation=True)
            # forward
            out_f2, out_reused_f2, hidden_f2 = parallel_model_f2(data_f2, randata_f2, hidden_f2)
            # distribution calculation
            if i:
                dist = torch.exp(out_f2) * (1 / support)
                dist_vector = torch.cat((dist_vector, dist), 0)
            else:
                dist_vector = torch.exp(out_f2) * (1 / support)

            # hidden repackage
            hidden_f2 = repackage_hidden(hidden_f2)

        y_givenx[k] = torch.mean(dist_vector).detach().cpu().numpy()

    y_givenx = y_givenx / np.sum(y_givenx)
    return y_vec, y_givenx


# Load the best saved model.
model_f2 = torch.load(os.path.join(args.save, 'model_f2.pt'))
parallel_model_f2 = nn.DataParallel(model_f2, dim=1).cuda()

# calculating the support
_val_tmp_randata = data.new_process(args, valid_length)
support = torch.max(_val_tmp_randata['labels']) - torch.min(_val_tmp_randata['labels'])
# Run on test data.
y_vec, y_givenx = evaluate_dist(x_inp, support)

plt.rcParams['axes.facecolor'] = 'floralwhite'
plt.fill_between(y_vec, y_givenx)
plt.grid(True, which='both', axis='both')
if db != -999:
    plt.title('DINE Trained Model - ' + r'$P_{Y|X}$' + ' of {}\n'
              'X={}, P={}, N={}, SNR {}dB dim=1'.format(process, x_inp, args.P, args.N, db))
else:
    plt.title('DINE Trained Model - ' + r'$P_{Y|X}$' + ' of {}\n'
              'X={}, P={}, N={}, dim=1'.format(process, x_inp, args.P, args.N))
plt.savefig("./" + inp + "/Py given x={}.png".format(x_inp), dpi=1200)
plt.xlabel(r'$Y$')
plt.ylabel(r'$P_{Y|X}$' + ", X={}".format(x_inp))
plt.show()
