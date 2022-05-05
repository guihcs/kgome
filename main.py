import copy
import os
import sys

import pandas as pd
from multiprocess import Process

from om.match import onts, aligns
from om.ont import load_ont, namespace, print_node, singleton, split_entity
from om.util import Cross
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from om.match import Runner, print_result, Step
from om.util import Cross, match_format
import math
import random
import json
from sklearn.model_selection import KFold


class WordMap:

    def __init__(self, vocab=None):
        self.i = 0
        self.wi = dict()
        self.iw = dict()
        self.vocab = set()

        if vocab is not None:
            self.add_all(vocab)

    def add(self, word):
        self.vocab.add(word)
        self.wi[word] = self.i
        self.iw[self.i] = word
        self.i += 1

    def add_all(self, vocab):
        for w in vocab:
            self.add(w)

    def __getitem__(self, i):
        return self.wi[i]

    def get_w(self, i):
        return self.iw[i]

    def __len__(self):
        return self.i

    def __contains__(self, x):
        return x in self.wi

    def save(self, path):
        with open(path, 'w') as f:
            f.write(json.dumps({'wi': self.wi, 'iw': self.iw, 'i': self.i, 'vocab': list(self.vocab)}))

    def load(self, path):
        with open(path, 'r') as f:
            d = json.loads(f.read())
            self.wi = d['wi']
            self.iw = d['iw']
            self.i = d['i']
            self.vocab = set(d['vocab'])


def get_vocab(ont):
    vocab = set()

    for e in ont:
        vocab.add(e)
        for p in ont[e]:

            if p == 'comment':
                continue

            if p in ['in', 'out']:
                for v in ont[e][p]:
                    vocab.update(set(v))
                continue
            vocab.add(p)
            for v in ont[e][p]:
                vocab.add(v)

    return vocab


def is_property(n):
    return any(map(lambda x: 'Property' in x, n['type']))


def build_adj(ont):
    ents = list(get_vocab(ont))
    random.shuffle(ents)
    l = len(ents)
    adj = torch.zeros((l, l))
    pdj = -torch.ones((l, l))
    wm = WordMap(ents)
    for e in ents:
        if e not in ont:
            continue
        for p in ont[e]:
            if p == 'comment':
                continue

            if p == 'in':
                for e1, e2 in ont[e][p]:
                    pdj[wm[e]][wm[e1]] = wm[e2]
                    adj[wm[e]][wm[e1]] = -1

            elif p == 'out':
                for e1, e2 in ont[e][p]:
                    pdj[wm[e]][wm[e2]] = wm[e1]
                    adj[wm[e]][wm[e2]] = 1
            elif p == 'superClassOf':
                for v in ont[e][p]:
                    pdj[wm[e]][wm[v]] = wm[p]
                    adj[wm[e]][wm[v]] = -1
            else:
                for v in ont[e][p]:
                    pdj[wm[e]][wm[v]] = wm[p]
                    adj[wm[e]][wm[v]] = 1

    return adj, pdj.long(), ents


def eval_result(result, rang):
    if len(result) <= 1:
        result[0].drop('name', axis=1).mean().plot.bar(rot=0)
        f = [0]
    else:
        p = []
        r = []
        f = []

        for q in result:
            v = q.drop('name', axis=1).mean()

            p.append(v['precision'])
            r.append(v['recall'])
            f.append(v['f1'])

        plt.figure(figsize=(10, 5))
        plt.plot(rang, p, c='r')
        plt.plot(rang, r, c='g')
        plt.plot(rang, f, c='b')
        plt.show()

    print(rang[np.argmax(f)])
    print_result(result[np.argmax(f)])


def get_word_vocab(ont):
    vocab = set()

    for e in ont:
        vocab.update(set(split_entity(e)))
        for p in ont[e]:
            if p == 'comment':
                continue

            if p in ['in', 'out']:
                for e1, e2 in ont[e][p]:
                    vocab.update(set(split_entity(e1)))
                    vocab.update(set(split_entity(e2)))
                continue
            vocab.update(set(split_entity(p)))
            for v in ont[e][p]:
                vocab.update(set(split_entity(v)))

    return vocab


def pos_encod(c, d):
    r = torch.arange(0, d).repeat(c, 1)
    m = torch.arange(0, c).repeat(d, 1).t()
    sm = (r % 2 == 0).float()
    cm = (r % 2 != 0).float()
    r = r - cm
    r = 10000 ** (r / d)
    m = m / r
    s = torch.sin(m) * sm
    c = torch.cos(m) * cm
    return s + c


class MatchDataset(Dataset):

    def __init__(self, r, o1, o2, tb=1):
        self.transform = None
        self.ont1 = load_ont(o1)
        self.ont2 = load_ont(o2)

        self.als = set(map(lambda x: (x[0].split('#')[-1], x[1].split('#')[-1]), aligns(r)))

        self.vocab1 = get_vocab(self.ont1)
        self.vocab2 = get_vocab(self.ont2)
        self.vocab3 = list(self.vocab1.union(self.vocab2))

        self.data = []

        self.adj1, self.pdj1, self.ents1 = build_adj(self.ont1)
        self.adj2, self.pdj2, self.ents2 = build_adj(self.ont2)
        self.wm1 = WordMap(self.ents1)
        self.wm2 = WordMap(self.ents2)

        self.word_vocab1 = get_word_vocab(self.ont1)
        self.word_vocab2 = get_word_vocab(self.ont2)
        self.word_vocab = list(self.word_vocab1.union(self.word_vocab2))

        tc = 0
        nc = 0

        for e1 in tqdm(self.ont1):
            for e2 in self.ont2:
                sim = torch.Tensor([1 if (e1, e2) in self.als else -1])

                for _ in range(1 if sim != 1 else tb):
                    if sim != 1:
                        nc += 1
                    else:
                        tc += 1
                    self.data.append((e1, e2, sim))

        print(f'{len(self)} instances. {tc} true, {nc} false.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class DPAttention(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.ql = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.kl = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.vl = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.nv = 1 / math.sqrt(self.embedding_size)

    def forward(self, s):
        q = s
        k = s
        v = s

        kq = (k @ s.permute(0, 2, 1)) * self.nv
        kqs = F.softmax(kq, dim=2)
        r = kqs @ v
        return r


class MHAttention(nn.Module):

    def __init__(self, embedding_size, h):
        super().__init__()
        self.embedding_size = embedding_size
        self.h = h
        self.heads = [DPAttention(embedding_size) for _ in range(h)]
        self.ln = nn.Linear(embedding_size * h, embedding_size)

    def forward(self, s):
        r = tuple([head(s) for head in self.heads])
        fn = torch.cat(r, dim=2)
        return self.ln(fn)


class EncoderBlock(nn.Module):

    def __init__(self, embedding_size, h):
        super().__init__()
        self.embedding_size = embedding_size
        self.h = h
        self.mh_attention = MHAttention(embedding_size, h)

    def forward(self, s):
        r = self.mh_attention(s)

        return r


class GNN(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pm = nn.Linear(2 * embedding_dim, embedding_dim)
        self.ft = nn.Linear(embedding_dim, embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, features, adj, pdj, i):
        fpc = torch.cat((pdj, features.repeat(pdj.shape[0], 1, 1)), dim=2)
        fpc = self.pm(fpc)
        fpc = self.tanh(fpc)
        ad = adj + torch.eye(adj.shape[0])

        mf = ad.view(ad.shape[0], 1, ad.shape[1]) @ fpc

        return  mf.view(mf.shape[0], mf.shape[2]) + self.tanh(self.ft(features))


class OntMatcher(nn.Module):

    def __init__(self, ln1, embedding_dim, h=1, eh=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.emb1 = nn.Embedding(ln1, embedding_dim, padding_idx=-1)
        self.gnn = GNN(embedding_dim)
        self.encoder = nn.Sequential(*([EncoderBlock(embedding_dim, h=h)] * eh))
        self.tanh = nn.Tanh()

    def build_ft(self, f, p, m):
        w = self.emb1.weight.clone()

        ft = w[f]

        ft = self.encoder(ft) * m
        ft = ft.mean(dim=1, keepdim=True).view(-1, self.embedding_dim)

        cft = torch.cat((ft, torch.zeros(1, self.embedding_dim)), dim=0)
        pdj = cft[p]

        return ft, pdj

    def forward(self, f, a, p, i, m):
        ft, p = self.build_ft(f, p, m)
        g1 = self.gnn_pass(ft, a, p, i)
        return g1

    def gnn_pass(self, ft, a, p, i):
        return self.gnn(ft, a, p, i)


class TrainOm:

    def __init__(self, datasets):
        jv = set()
        es = set()
        self.datasets = datasets
        for d in datasets:
            jv.update(d.word_vocab)
            es.update(d.vocab3)

        jv = list(jv)
        self.wm = WordMap(jv)

        self.split_map = dict()

        for e in es:
            se = split_entity(e)
            self.split_map[e] = list(map(lambda x: self.wm[x], se))

        self.embedding_size = 200
        self.om = OntMatcher(len(jv), self.embedding_size, 1, 1)
        self.crit = nn.CosineEmbeddingLoss(margin=0.5)
        self.optimizer = optim.Adam(self.om.parameters(), lr=0.003)
        self.epochs = 6

    def train(self):
        self.om.share_memory()
        eh = []
        for e in tqdm(range(self.epochs)):
            el = 0
            dte = []
            for data in self.datasets:
                loader = DataLoader(data, batch_size=256, shuffle=True)
                ft1 = list(map(lambda x: self.split_map[x], data.ents1))
                ft2 = list(map(lambda x: self.split_map[x], data.ents2))

                ft1 = copy.deepcopy(ft1)
                ft2 = copy.deepcopy(ft2)
                ml = max(map(len, ft1))
                for s1 in ft1:
                    if len(s1) < ml:
                        s1 += [-1] * (ml - len(s1))

                mask1 = (torch.Tensor(ft1) > -1).float().view(-1, ml, 1)

                ml = max(map(len, ft2))
                for s2 in ft2:
                    if len(s2) < ml:
                        s2 += [-1] * (ml - len(s2))
                mask2 = (torch.Tensor(ft2) > -1).float().view(-1, ml, 1)
                pe1 = pos_encod(len(data), self.embedding_size)

                for i, (e1, e2, s) in enumerate(loader):
                    er = self.train_batch(e1, e2, s, i, i, ft1, ft2, data, mask1, mask2, pe1, i, loader.batch_size)
                    el += er

                el /= len(data)
                dte.append(el)
            eh.append(sum(dte) / len(dte))
        plt.plot(eh)
        plt.show()
        self.wm.save('wm.d')
        torch.save(self.om.state_dict(), 'model.pt')

    def train_batch(self, e1, e2, s, i1, i2, ft1, ft2, data, mask1, mask2, pe1, i, bs):
        self.optimizer.zero_grad()
        e1 = list(map(lambda x: data.wm1[x], e1))
        e2 = list(map(lambda x: data.wm2[x], e2))

        nemb1 = self.om(ft1, data.adj1, data.pdj1, i1, mask1)
        nemb2 = self.om(ft2, data.adj2, data.pdj2, i2, mask2)

        r1, r2 = nemb1[e1].view(-1, self.embedding_size), nemb2[e2].view(-1, self.embedding_size)
        loss = self.crit(r1, r2, s.flatten())
        loss.backward()
        self.optimizer.step()
        return loss.item() * len(e1)


class EqMatcher(Step):

    def __init__(self):
        self.cross = Cross()

    def forward(self, dataset):
        ents = self.cross(dataset)

        res = []
        for e1, e2 in ents:
            res.append(match_format(dataset, e1, e2, 1 if e1 == e2 else 0))

        return [res]


class Matcher(Step):

    def __init__(self, om, wm, rang):
        self.cross = Cross()
        self.om = om
        self.wm = wm
        self.rang = rang

    def forward(self, dataset, i):
        with torch.no_grad():
            i1 = 1
            i2 = 1
            vc1 = get_word_vocab(dataset.ont1)
            vc2 = get_word_vocab(dataset.ont2)

            vc1c = 1 - len(vc1.difference(self.wm.vocab)) / len(vc1)
            vc2c = 1 - len(vc2.difference(self.wm.vocab)) / len(vc2)

            adj1, pdj1, ents1 = build_adj(dataset.ont1)
            adj2, pdj2, ents2 = build_adj(dataset.ont2)
            wm1 = WordMap(ents1)
            wm2 = WordMap(ents2)

            ft1 = list(map(lambda x: [self.wm[q] if q in self.wm else -1 for q in split_entity(x)], ents1))
            ft2 = list(map(lambda x: [self.wm[q] if q in self.wm else -1 for q in split_entity(x)], ents2))

            ft1 = copy.deepcopy(ft1)
            ft2 = copy.deepcopy(ft2)
            ml = max(map(len, ft1))
            for s1 in ft1:
                if len(s1) < ml:
                    s1 += [-1] * (ml - len(s1))

            mask1 = (torch.Tensor(ft1) > -1).float().view(-1, ml, 1)

            ml = max(map(len, ft2))
            for s2 in ft2:
                if len(s2) < ml:
                    s2 += [-1] * (ml - len(s2))
            mask2 = (torch.Tensor(ft2) > -1).float().view(-1, ml, 1)

            ft1, pdj1 = self.om.build_ft(ft1, pdj1, mask1)
            ft2, pdj2 = self.om.build_ft(ft2, pdj2, mask2)

            ents = self.cross(dataset)
            entsv = list(map(lambda x: (x[0], x[1], wm1[x[0]], wm2[x[1]]), ents))

            res = [[] for r in self.rang]
            # res = [[]]
            meta = []
            loader = DataLoader(entsv, batch_size=256)
            for i, (x1, x2, e1, e2) in enumerate(loader):

                nemb1 = self.om.gnn_pass(ft1, adj1, pdj1, i1)
                nemb2 = self.om.gnn_pass(ft2, adj2, pdj2, i2)

                i1 += 1
                i2 += 1
                r1, r2 = nemb1[e1], nemb2[e2]

                s = F.cosine_similarity(r1, r2)

                for w in range(len(x1)):
                    meta.append((x1[w], x2[w], s[w]))

            mt = max(map(lambda x: x[2], meta))
            print(mt)
            mt -= 0.05

            for q1, q2, s in meta:
                for i in range(len(self.rang)):
                    res[i].append(match_format(dataset, q1, q2, 1 if s > self.rang[i] else 0))
                # res[0].append(match_format(dataset, q1, q2, 1 if s > mt else 0))

        return res, {'vc1': vc1c, 'vc2': vc2c}


def train_kf(ki, tri, tei, base):
    train = list(map(lambda x: refs[x], tri))
    test = list(map(lambda x: refs[x], tei))
    datasets = list(map(lambda ps: MatchDataset(ps[0], ps[1], ps[2], tb=600), train))

    tr = TrainOm(datasets)
    tr.train()
    # tr.om.load_state_dict(torch.load('model.pt'))
    # tr.wm.load('wm.d')

    tr.om.eval()
    rang = np.arange(0.1, 1, 0.01)

    matcher = Matcher(tr.om, tr.wm, rang)

    runner = Runner(base + 'conference', base + 'reference', matcher=matcher)

    for i in range(1):

        result = runner.run(refs=list(map(lambda x: x[0], test)), parallel=False)
        if not os.path.exists(f'rs_{ki}'):
            os.mkdir(f'rs_{ki}')
        for ri in range(len(result)):
            result[ri].to_csv(f'rs_{ki}/result_{rang[ri]:.2f}.csv')
        # eval_result(result, rang)


if __name__ == '__main__':
    base = sys.argv[1]
    if not os.path.exists(base + 'conference'):
        raise Exception(f'base ontologies not found in {base}conference')
    refs = list(onts(base + 'conference', base + 'reference'))

    # kf = KFold(n_splits=3)
    #
    # processes = []
    # for ki, (tri, tei) in enumerate(kf.split(refs)):
    #     p = Process(target=train_kf, args=(ki, tri, tei, base))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()
    #
    # results = []
    #
    # for p, d, files in os.walk('/home/guilherme/Downloads/results/rs_1'):
    #     files.sort()
    #     for file in files:
    #         results.append(pd.read_csv('/home/guilherme/Downloads/results/rs_1/' + file))


    # rang = np.arange(0.1, 1, 0.01)
    # eval_result(results, rang)


    data1 = MatchDataset(base + 'reference/cmt-conference.rdf', base + 'conference/cmt.owl', base + 'conference/Conference.owl', tb=800)
    data2 = MatchDataset(base + 'reference/conference-confOf.rdf', base + 'conference/Conference.owl', base + 'conference/confOf.owl', tb=800)
    data3 = MatchDataset(base + 'reference/confOf-edas.rdf', base + 'conference/confOf.owl', base + 'conference/edas.owl', tb=800)
    data4 = MatchDataset(base + 'reference/edas-ekaw.rdf', base + 'conference/edas.owl', base + 'conference/ekaw.owl', tb=800)
    data5 = MatchDataset(base + 'reference/ekaw-sigkdd.rdf', base + 'conference/ekaw.owl', base + 'conference/sigkdd.owl', tb=800)
    data6 = MatchDataset(base + 'reference/iasted-sigkdd.rdf', base + 'conference/iasted.owl', base + 'conference/sigkdd.owl', tb=800)

    datasets = [data1, data2, data3, data4, data5, data6]

    tr = TrainOm(datasets)
    tr.train()
    # tr.om.load_state_dict(torch.load('model.pt'))
    # tr.wm.load('wm.d')

    tr.om.eval()
    rang = np.arange(0.0, 1, 0.01)

    matcher = Matcher(tr.om, tr.wm, rang)

    runner = Runner(base + 'conference', base + 'reference', matcher=matcher)

    for i in range(1):

        result = runner.run(parallel=False)
        eval_result(result, rang)
