import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

from embed_regularize import embedded_dropout
from locked_dropout import MyLockedDropout as LockedDropout
from weight_drop import WeightDrop

from typing import Tuple, Union, Optional, Callable, Any, List
from torch.nn import Parameter
from enum import IntEnum


class Dim(IntEnum):
    seq = 0
    batch = 1
    feature = 2


class ModifiedLSTMcell(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, dropout=0, variational=False, recycle_hid=False):
        super().__init__()
        self.dropout = dropout
        self.variational = variational
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.recycle_hid = recycle_hid  # will flag to recycle hidden parameters
        self.lstm_cell = nn.LSTMCell(input_size=input_sz, hidden_size=hidden_sz)
        if dropout:
            self.lstm_cell = WeightDrop(self.lstm_cell, ['weight_hh'], dropout=dropout)

    def forward(self, x: torch.Tensor,
                init_states: List[Tuple[torch.Tensor]],
                all_hidden: Optional[Tuple[torch.Tensor]] = None  # will be fed to the layer as (h_t,c_t)
                ):  # -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""

        seq_sz, batch_sz, _ = x.size()
        outputs = []
        recycled = []

        # normal LSTM
        if all_hidden is None:
            h_t, c_t = init_states
            h_t = h_t.squeeze(Dim.seq)
            c_t = c_t.squeeze(Dim.seq)

        # recycle hid parameters
        if self.recycle_hid:
            recycled.append((h_t, c_t))
        # iterate over the time steps
        for t in range(seq_sz):
            x_t = x[t, :, :]
            # normal LSTM
            if all_hidden is None:
                h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))
                if self.recycle_hid:
                    recycled.append((h_t, c_t))
            # modified LSTM - cannot recycle, uses recycled hidden parameters
            else:
                h_t, c_t = self.lstm_cell(x_t, all_hidden[t])
            outputs.append(h_t.unsqueeze(Dim.seq))
        outputs = torch.cat(outputs, dim=Dim.seq)

        if all_hidden is None:
            # normal LSTM
            if self.recycle_hid:
                # return also ALL hidden parameters
                return outputs, (h_t.unsqueeze(Dim.seq), c_t.unsqueeze(Dim.seq)), recycled
            else:
                return outputs, (h_t.unsqueeze(Dim.seq), c_t.unsqueeze(Dim.seq))
        else:
            # modified LSTM
            return outputs  # hidden doesn't matter here...


class DIModel(nn.Module):
    """Container of DINE F1/2 function"""

    def __init__(self, ninp, nhid, ncell=20, wdrop=0):
        super(DIModel, self).__init__()

        self.LSTMunits = [ModifiedLSTMcell(ninp, nhid, dropout=wdrop, recycle_hid=True)
                          for _cell in range(ncell)]  # creating n modules of LSTM
        self.LSTMunits = torch.nn.ModuleList(self.LSTMunits)

        self.fcn = nn.Linear(nhid * ncell, 1)
        self.ninp = ninp
        self.nhid = nhid
        self.ncell = ncell
        self.wdrop = wdrop

    def init_weights(self):
        pass  # for now...

    def forward(self, input, randinput, hidden): # TODO: add random input
        """input is the normal (seq x batch x neurons) input.
        hidden is for normal LSTM, used LSTM will get recycled all_hidden parameter"""
        batch_size = input.size(1)

        raw_outputs = [None]*self.ncell
        raw_outputs_reused = [None]*self.ncell
        recycled_hids = [None]*self.ncell
        new_h = [None]*self.ncell
        for cell, lstm in enumerate(self.LSTMunits):
            # Normal LSTM - enabling recycling
            lstm.recycle_hid = True
            raw_outputs[cell], new_h[cell], recycled_hids[cell] = lstm(input, hidden[cell])
            # Modified LSTM - disabling recycling
            lstm.recycle_hid = False
            raw_outputs_reused[cell] = lstm(randinput, None, all_hidden=recycled_hids[cell])
        # Normal LSTM to fully connected layer
        out = self.fcn(input=torch.cat(raw_outputs, axis=2))
        # Modified LSTM to fully connected layer
        out_reused = self.fcn(input=torch.cat(raw_outputs_reused, axis=2))   # ncell x seq x batch x nhid

        return out, out_reused, new_h

    def init_hidden(self, bsz, ncell):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.nhid).zero_(),
                 weight.new(1, bsz, self.nhid).zero_()) for _ in range(ncell)]


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nhidlast, nlayers,
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,
                 tie_weights=False, ldropout=0.5, n_experts=10):
        super(RNNModel, self).__init__()
        self.use_dropout = True
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnns = [ModifiedLSTMcell(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast,
                                      dropout=wdrop if self.use_dropout else 0) for l in range(nlayers)]
        # if wdrop:
        #     self.rnns = [WeightDrop(rnn, ['weight_hh'], dropout=wdrop if self.use_dropout else 0) for rnn in
        #                  self.rnns]

        self.rnns = torch.nn.ModuleList(self.rnns)

        self.prior = nn.Linear(nhidlast, n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(nhidlast, n_experts * ninp), nn.Tanh())
        self.decoder = nn.Linear(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            # if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ldropout = ldropout
        self.dropoutl = ldropout
        self.n_experts = n_experts
        self.ntoken = ntoken

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('param size: {}'.format(size))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, return_prob=False):
        batch_size = input.size(1)

        # usedp = False if we are at normal eval
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute, usedp=(self.training and self.use_dropout))
        # emb = self.idrop(emb)

        emb = self.lockdrop(emb, dropout=self.dropouti if self.use_dropout else 0)

        raw_output = emb
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                # self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, dropout=self.dropouth if self.use_dropout else 0)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, dropout=self.dropout if self.use_dropout else 0)
        outputs.append(output)  # this i G

        latent = self.latent(output)  # this is H (tanh(W1 * G)
        latent = self.lockdrop(latent, dropout=self.dropoutl if self.use_dropout else 0)
        logit = self.decoder(latent.view(-1, self.ninp))  # this is the logit = W2 * H

        prior_logit = self.prior(output).contiguous().view(-1, self.n_experts)  # W3 * G
        prior = nn.functional.softmax(prior_logit, -1)  # softmax ( W3 * G )

        prob = nn.functional.softmax(logit.view(-1, self.ntoken), -1).view(-1, self.n_experts, self.ntoken)  # N x M
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(prob.add_(1e-8))
            model_output = log_prob

        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_(),
                 weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_())
                for l in range(self.nlayers)]


if __name__ == '__main__':
    model = RNNModel('LSTM', 10, 12, 12, 12, 2)
    input = torch.LongTensor(13, 9).random_(0, 10)
    hidden = model.init_hidden(9)
    model(input, hidden)
    print(model)

    # input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    # hidden = model.init_hidden(9)
    # print(model.sample(input, hidden, 5, 6, 1, 2, sample_latent=True).size())