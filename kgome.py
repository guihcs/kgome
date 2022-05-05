from om.match import onts, aligns
from om.ont import split_entity, pt, remove_bn
from om.util import Cross
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from om.match import Runner, print_result, Step
from om.util import Cross, get_vocab, WordMap
import math
import random
import pandas as pd
from om.ont import get_n, noisy_copy, remove_bn
from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, OWL
import os
from pymagnitude import *



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

    rd = dict()

    for n in result[0]['name'].unique():
        l = result[0][result[0]['name'] == n].copy()
        l['th'] = 0
        rd[n] = l

    for i in range(len(rang)):
        for n in result[i]['name'].unique():
            l = result[i][result[i]['name'] == n].copy()
            l['th'] = rang[i]

            if l['f1'].iloc[0] > rd[n]['f1'].iloc[0]:
                rd[n] = l

    print('\n\noracle th ---------------------')
    print_result(pd.concat(rd.values()))



def build_adj(g):
    ents = list(get_vocab(g))

    random.shuffle(ents)
    l = len(ents)
    adj = torch.zeros((l, l))
    pdj = -torch.ones((l, l))

    wm = WordMap(ents)

    for s, p, o in g:

        adj[wm[s]][wm[o]] = 1
        pdj[wm[s]][wm[o]] = wm[p]

        if (s, RDFS.domain, None) in g and (s, RDFS.range, None) in g:
            domain = g.value(s, RDFS.domain)
            rg = g.value(s, RDFS.range)

            adj[wm[domain]][wm[rg]] = 1
            pdj[wm[domain]][wm[rg]] = wm[s]

            adj[wm[rg]][wm[domain]] = -1
            pdj[wm[rg]][wm[domain]] = wm[s]

        if (s, RDFS.subClassOf, None) in g:
            for ob in g.objects(s, RDFS.subClassOf):
                adj[wm[s]][wm[ob]] = 1
                pdj[wm[s]][wm[ob]] = wm[RDFS.subClassOf]

                adj[wm[ob]][wm[s]] = -1
                pdj[wm[ob]][wm[s]] = wm[RDFS.subClassOf]

    return ents, adj, pdj.long()




class MDataset(Dataset):

    def __init__(self, dtsn, als, g1, g2, tb=1, fb=1, cache=False, cacheLen=500000):
        self.transform = None
        self.cache = cache
        self.cacheLen = cacheLen
        self.dtsn = dtsn + '-cache.bin'
        if os.path.exists(self.dtsn):
            os.remove(self.dtsn)

        self.g1 = g1
        self.g2 = g2
        self.als = als

        with torch.no_grad():
            self.ents1, self.adj1, self.pdj1 = build_adj(self.g1)
            self.ents2, self.adj2, self.pdj2 = build_adj(self.g2)

        tc = 0
        nc = 0
        self.data = []
        self.dl = 0
        self.li = -1 if self.cache else 0
        self.mi = -1

        for e1 in tqdm(set(self.g1.subjects())):
            for e2 in set(self.g2.subjects()):
                if random.random() < (1 - fb):
                    continue
                sim = torch.Tensor([1 if (e1, e2) in self.als else -1])

                for _ in range(1 if sim != 1 else tb):
                    if sim != 1:
                        nc += 1
                    else:
                        tc += 1
                    self.dl += 1

                    self.data.append((e1, e2, sim))

                    if self.cache and len(self.data) >= self.cacheLen:
                        random.shuffle(self.data)
                        self._cache()
                        self.data = []

        if self.cache:
            random.shuffle(self.data)
            self._cache()
            self.data = []

        print(f'{len(self)} instances. {tc} true, {nc} false.')

    def _cache(self):
        lines = []
        for e1, e2, s in self.data:
            lines.append(str(e1).encode('utf-8').ljust(500) + str(e2).encode('utf-8').ljust(500) + str(s.item()).encode(
                'utf-8').ljust(50))

        with open(self.dtsn, 'ab') as f:
            f.writelines(lines)

    def _loadblock(self, i, ml):
        self.data = []
        ls = 500 + 500 + 50

        with open(self.dtsn, 'rb') as f:
            f.seek(i * ls)
            rd = f.read(ml * ls)

        for fi in range(ml):
            line = rd[fi * ls:(fi + 1) * ls]
            u1 = line[:500].decode('utf-8').strip()
            u2 = line[500:1000].decode('utf-8').strip()
            s = line[1000:1050].decode('utf-8').strip()

            self.data.append((URIRef(u1), URIRef(u2), torch.Tensor([float(s)])))

        self.li = i
        self.mi = i + ml

    def __getitem__(self, i):
        if self.cache and (i >= self.mi or i < self.li):
            self._loadblock(i, min(self.dl - i, 200000))

        return self.data[i - self.li]

    def __len__(self):
        return self.dl


class MatchDataset(MDataset):

    def __init__(self, r, o1, o2, tb=1):
        g1 = Graph()
        g1.parse(o1)
        g2 = Graph()
        g2.parse(o2)

        als = set(map(lambda x: (URIRef(x[0]), URIRef(x[1])), aligns(r)))

        super().__init__('none', als, g1, g2, tb=tb)



def emb_ents(ents, g):
    embs = []

    for i in range(len(ents)):
        e = ents[i]

        if type(e) is BNode:
            emb = torch.zeros((1, 300))

        elif type(e) is Literal:
            words = split_entity(e)
            emb = sum(map(lambda x: torch.Tensor([list(glove.query(x.lower()))]), words)) / len(words)
        else:

            n = e.n3(g.namespace_manager)
            if n.startswith('<'):
                n = n.split('/')[-1]
            else:
                n = n.split(':')[-1]

            words = split_entity(n)
            if len(words) <= 0:
                emb = torch.zeros((1, 300))
            else:
                emb = sum(map(lambda x: torch.Tensor([list(glove.query(x.lower()))]), words)) / len(words)
        embs.append(emb)
    return torch.cat(embs)


class GNAH(nn.Module):

    def __init__(self, emblen):
        super().__init__()
        self.emblen = emblen
        self.w = nn.Linear(emblen, emblen, bias=False)
        self.a = nn.Linear(emblen * 2, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.pc = nn.Linear(emblen * 2, emblen)

    def forward(self, embs, adj, pde):
        adc = torch.clone(adj)

        wf = self.w(embs)
        rm = wf.view(wf.shape[0], 1, wf.shape[1]).repeat(1, wf.shape[0], 1)
        cm = wf.view(1, wf.shape[0], wf.shape[1]).repeat(wf.shape[0], 1, 1)

        pc = torch.cat([cm, pde], dim=2)
        pc = self.lrelu(self.pc(pc))
        pc = self.w(pc)

        mf = self.lrelu(self.a(torch.cat([rm, pc], dim=2)))
        mf = mf.view(mf.shape[0], mf.shape[1])
        me = torch.nan_to_num((adc - 1) * float('inf')).to(torch.device('cuda:0'))
        mf = torch.softmax(mf + me, dim=1)

        nbc = mf @ embs

        res = torch.cat([embs, nbc], dim=1)

        return res


class MHAG(nn.Module):

    def __init__(self, emblen, heads):
        super().__init__()
        self.emblen = emblen
        self.heads = heads
        self.heads = nn.ModuleList([GNAH(emblen) for _ in range(heads)])
        self.ch = nn.Linear(emblen * heads * 2, emblen)

    def forward(self, embs, adj, pdj):
        pe = torch.cat([embs, torch.zeros((1, self.emblen)).to(torch.device('cuda:0'))])
        pde = pe[pdj]

        res = [h(embs, adj, pde) for h in self.heads]

        rm = self.ch(torch.cat(res, dim=1))

        return rm


class GNN(nn.Module):

    def __init__(self, emblen, heads, nodes):
        super().__init__()
        self.emblen = emblen
        self.heads = heads
        self.nodes = nodes
        self.an = nn.ModuleList([MHAG(emblen, heads) for _ in range(nodes)])

    def forward(self, embs, adj, pdj):
        res = embs

        for n in self.an:
            res = n(res, adj, pdj)

        return res




class KGOME:

    def __init__(self):
        self.glove = Magnitude('/moredata/gsantos/canard/glove.840B.300d.magnitude')
        self.emblen = 300
        pass

    def fit(self, datasets):
        device = torch.device('cuda:0')

        epochs = 3
        lr = 0.0003

        crit = nn.CosineEmbeddingLoss(margin=0.4)

        gnn = GNN(self.emblen, 2, 1)
        gnn.to(device)
        optimizer = optim.Adam(gnn.parameters(), lr=lr)
        lh = []

        for epcq in tqdm(range(epochs)):
            el = 0
            for data in datasets:

                dl = 0
                bs = 256

                embs1 = emb_ents(data.ents1, data.g1)
                embs2 = emb_ents(data.ents2, data.g2)

                wm1 = WordMap(data.ents1)
                wm2 = WordMap(data.ents2)

                for e1, e2, s in tqdm(DataLoader(data, batch_size=bs), leave=False):
                    optimizer.zero_grad()

                    m1 = gnn(embs1.to(device), data.adj1.to(device), data.pdj1.to(device))
                    m2 = gnn(embs2.to(device), data.adj2.to(device), data.pdj2.to(device))

                    r1 = list(map(lambda x: m1[wm1[x]].unsqueeze(0), e1))
                    r2 = list(map(lambda x: m2[wm2[x]].unsqueeze(0), e2))

                    r1 = torch.cat(r1).to(torch.device('cpu'))
                    r2 = torch.cat(r2).to(torch.device('cpu'))

                    loss = crit(r1, r2, s.flatten())
                    loss.backward()

                    optimizer.step()
                    dl += loss.item()

                    # break

                dl /= (len(data) / bs)
                el += dl

                # break

            el /= len(datasets)
            lh.append(el)
            torch.save(gnn.state_dict(), f'gnn-conf-{epcq}.pt')
            # break

        plt.plot(lh)
        plt.show()


    def load(self, path):
        self.gnn = GNN(self.emblen, 2, 1)
        self.gnn.load_state_dict(torch.load(path))




    def embed(self, g):
        ents1, adj1, pdj1 = build_adj(g)

        embs1 = emb_ents(ents1, g)

        wi1 = WordMap(ents1)

        self.gnn.to(torch.device('cuda:0'))
        with torch.no_grad():
            m1 = self.gnn(embs1.to(torch.device('cuda:0')), adj1.to(torch.device('cuda:0')), pdj1.to(torch.device('cuda:0'))).to(torch.device('cpu'))


        return m1



class Matcher(Step):

    def __init__(self, gnn, rang):
        self.cross = Cross()
        self.gnn = gnn
        self.rang = rang

    def forward(self, dataset, i):
        ents = self.cross(dataset)

        ents1, adj1, pdj1 = build_adj(dataset.g1)
        ents2, adj2, pdj2 = build_adj(dataset.g2)
        embs1 = emb_ents(ents1, dataset.g1)
        embs2 = emb_ents(ents2, dataset.g2)

        wi1 = WordMap(ents1)
        wi2 = WordMap(ents2)


        with torch.no_grad():
            m1 = self.gnn(embs1.to(torch.device('cuda:0')), adj1.to(torch.device('cuda:0')), pdj1.to(torch.device('cuda:0'))).to(torch.device('cpu'))
            m2 = self.gnn(embs2.to(torch.device('cuda:0')), adj2.to(torch.device('cuda:0')), pdj2.to(torch.device('cuda:0'))).to(torch.device('cpu'))


        res = [[] for r in self.rang]
        for e1, e2 in ents:

            r1 = m1[wi1[e1]].unsqueeze(0)
            r2 = m2[wi2[e2]].unsqueeze(0)

            s = F.cosine_similarity(r1, r2)
            s = s.item()
            for i in range(len(self.rang)):
                sim = 1 if s > self.rang[i] else 0
                res[i].append((e1, e2, sim))

        return res, {}


class GloveMatcher(Step):

    def __init__(self, rang):
        self.cross = Cross()
        self.rang = rang

    def forward(self, dataset, i):
        g1 = dataset.g1
        g2 = dataset.g2

        remove_bn(g1)
        remove_bn(g2)

        ents1 = set(g1.subjects())
        ents2 = set(g2.subjects())

        m1 = dict()

        for e1 in ents1:
            n1 = e1.n3(g1.namespace_manager)
            n1 = map(lambda x: x.lower(), split_entity(n1))
            n1 = list(map(lambda x: torch.Tensor([list(glove.query(x.lower()))]), n1))
            ln = len(n1)
            sv = sum(n1) / ln

            m1[e1] = sv.squeeze(0)

        m2 = dict()

        for e1 in ents2:
            n1 = e1.n3(g2.namespace_manager)
            n1 = map(lambda x: x.lower(), split_entity(n1))
            n1 = list(map(lambda x: torch.Tensor([list(glove.query(x.lower()))]), n1))
            ln = len(n1)
            sv = sum(n1) / ln

            m2[e1] = sv.squeeze(0)

        ents = self.cross(dataset)

        res = [[] for r in self.rang]
        for e1, e2 in ents:

            r1 = m1[e1]
            r2 = m2[e2]

            s = F.cosine_similarity(r1.unsqueeze(0), r2.unsqueeze(0)).item()
            for i in range(len(self.rang)):
                sim = 1 if s > self.rang[i] else 0
                res[i].append((e1, e2, sim))

        return res, {}
