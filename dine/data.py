import os
import torch
import numpy as np
from collections import Counter


def new_process(args, num):
    """ Creating NEW data_obj as demanded """
    data = args.data
    P = args.P
    N = args.N
    alpha = args.alpha
    ndim = args.ndim
    if data == 'AWGN':
        data_obj = AWGN.create_process(P, N, num, ndim)
    else:
        data_obj = ARMA.create_process(P, N, alpha, num, ndim)
    return data_obj


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class SentCorpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line:
                    continue
                words = line.split() + ['<eos>']
                sent = torch.LongTensor(len(words))
                for i, word in enumerate(words):
                    sent[i] = self.dictionary.word2idx[word]
                sents.append(sent)

        return sents


class BatchSentLoader(object):
    def __init__(self, sents, batch_size, pad_id=0, cuda=False, volatile=False):
        self.sents = sents
        self.batch_size = batch_size
        self.sort_sents = sorted(sents, key=lambda x: x.size(0))
        self.cuda = cuda
        self.volatile = volatile
        self.pad_id = pad_id

    def __next__(self):
        if self.idx >= len(self.sort_sents):
            raise StopIteration

        batch_size = min(self.batch_size, len(self.sort_sents) - self.idx)
        batch = self.sort_sents[self.idx:self.idx + batch_size]
        max_len = max([s.size(0) for s in batch])
        tensor = torch.LongTensor(max_len, batch_size).fill_(self.pad_id)
        for i in range(len(batch)):
            s = batch[i]
            tensor[:s.size(0), i].copy_(s)
        if self.cuda:
            tensor = tensor.cuda()

        self.idx += batch_size

        return tensor

    next = __next__

    def __iter__(self):
        self.idx = 0
        return self


class AWGN(object):
    def __init__(self, P, N, dim):
        self.P = P
        self.N = N
        self.train = self.create_process(P, N, 100000, dim)
        self.valid = self.create_process(P, N, 10000, dim)
        self.test = self.create_process(P, N, 10000, dim)

    @staticmethod
    def create_process(P, N, num, dim):
        xn = np.random.normal(0, np.sqrt(P), (num, dim))
        zn = np.random.normal(0, np.sqrt(N), (num, dim))
        yn = np.add(xn, zn)
        features = torch.FloatTensor(num, dim)
        lables = torch.FloatTensor(num, dim)
        for i in range(num):
            features[i] = torch.FloatTensor(xn[i])  # list(range(i, i+dim)))
            lables[i] = torch.FloatTensor(yn[i])  # list(range(i, i+dim)))
        return {'features': features, 'labels': lables}


class ARMA(object):
    def __init__(self, P, N, alpha, dim):
        self.P = P
        self.N = N
        self.train = self.create_process(P, N, alpha, 100000, dim)
        self.valid = self.create_process(P, N, alpha, 10000, dim)
        self.test = self.create_process(P, N, alpha, 10000, dim)

    @staticmethod
    def create_process(P, N, alpha, num, dim):
        xn = np.random.normal(0, np.sqrt(P), (num, dim))
        zn = np.random.normal(0, np.sqrt(N), (num + 1, dim))
        zn[0] = [0] * dim
        un = np.array([[0] * dim] * num)
        for i in range(num):
            un[i] = zn[i + 1] + alpha * zn[i]
        yn = np.add(xn, un)
        features = torch.FloatTensor(num.dim)
        lables = torch.FloatTensor(num, dim)
        for i in range(num):
            features[i] = torch.FloatTensor(xn[i])
            lables[i] = torch.FloatTensor(yn[i])
        return {'features': features, 'labels': lables}


if __name__ == '__main__':
    # For AWGN Dine
    awgn = AWGN(5, 1)
    for i in range(30):
        print(awgn.train[i])
    # # For PTB
    # corpus = SentCorpus('./data/penn/')
    # loader = BatchSentLoader(corpus.valid, 10)
    # for i, d in enumerate(loader):
    #     print(i, d)
