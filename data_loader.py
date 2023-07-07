import json
from random import shuffle
from numpy.core.fromnumeric import _ndim_dispatcher
import torch
import dgl

DGLlayer = 2


class TrainDataLoader(object):


    def __init__(self, g):
        self.device = torch.device(
            ('cuda:%d' % (1)) if torch.cuda.is_available() else 'cpu')
        self.cpu = torch.device('cpu')
        self.batch_size = 256
        self.ptr = 0
        self.data = []
        self.g = g
        self.g1 = None
        self.g2 = None
        data_file = 'data/ASSIST/train_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            student_n, exercise_n, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        self.student_dim = int(student_n)
        self.exercise_dim = int(exercise_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, labels = [], [], [], []
        nids = {}
        nids['stu'] = []
        nids['exer'] = []
        stu_index = []
        exer_index = []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            labels.append(y)
            if log['user_id'] - 1 not in nids['stu']:
                nids['stu'].append(log['user_id'] - 1)
            if log['exer_id'] - 1 not in nids['exer']:
                nids['exer'].append(log['exer_id'] - 1)
            stu_index.append(nids['stu'].index(log['user_id'] - 1))
            exer_index.append(nids['exer'].index(log['exer_id'] - 1))
        nids['k'] = list(range(0, self.knowledge_dim))
        # print(nids)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(DGLlayer)
        gdataloader = dgl.dataloading.NodeDataLoader(
            self.g, nids, sampler, device=self.device, shuffle=False, batch_size=self.batch_size * 2 + self.knowledge_dim, drop_last=False)
        self.ptr += self.batch_size

        for input_nodes, output_nodes, blocks in gdataloader:
            return torch.LongTensor(input_stu_ids).to(self.device), torch.LongTensor(input_exer_ids).to(self.device), torch.Tensor(input_knowedge_embs).to(self.device), torch.LongTensor(labels).to(self.device), torch.LongTensor(stu_index).to(self.device), torch.LongTensor(exer_index).to(self.device), input_nodes, output_nodes, blocks
    
    def get(self, data, name):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(DGLlayer)
        nids = {}
        nids['stu'] = data['stu'].to(self.cpu)
        nids['exer'] = data['exer'].to(self.cpu)
        if name == 1:
            gdataloader = dgl.dataloading.NodeDataLoader(
                self.g1, nids, sampler, device=self.device, shuffle=False, batch_size=self.batch_size * 2, drop_last=False)
        else:
            gdataloader = dgl.dataloading.NodeDataLoader(
                self.g2, nids, sampler, device=self.device, shuffle=False, batch_size=self.batch_size * 2, drop_last=False)
        for input_nodes, edge_subgraph, blocks in gdataloader:
            return input_nodes, edge_subgraph, blocks

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        shuffle(self.data)
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, g):
        self.device = torch.device(
            ('cuda:%d' % (1)) if torch.cuda.is_available() else 'cpu')
        self.ptr = 0
        self.data = []
        self.g = g
        data_file = 'data/ASSIST/test_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        nids = {}
        nids['stu'] = [user_id - 1]
        nids['exer'] = input_exer_ids
        nids['k'] = list(range(0, self.knowledge_dim))
        sizeg = 1 + len(input_exer_ids) + self.knowledge_dim
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(DGLlayer)
        gdataloader = dgl.dataloading.NodeDataLoader(
            self.g, nids, sampler, device=self.device, shuffle=False, batch_size=sizeg, drop_last=False)
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys), gdataloader

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
