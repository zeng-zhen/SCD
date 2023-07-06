import json
import torch
import dgl

DGLlayer = 2


class TrainDataLoader(object):
    '''
    data loader for training
    '''

    def __init__(self,args, g):
        self.device = torch.device(
            ('cuda:%d' % (0)) if torch.cuda.is_available() else 'cpu')
        self.cpu = torch.device('cpu')
        self.batch_size = 256
        self.ptr = 0
        self.data = []
        self.g = g
        self.g1 = None
        self.g2 = None
        data_file = './data/ASSIST/train_set.json'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        self.knowledge_dim = int(args.knowledge_n)
        self.student_dim = int(args.student_n)
        self.exercise_dim = int(args.exer_n)
        self.edge_data = {}
        for log in self.data:
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            self.edge_data[(log['user_id'] - 1, log['exer_id'] - 1)
                           ] = {'y': log['score'], 'knowledge_emb': knowledge_emb}
        n_edges = self.g.number_of_edges('serelate')
        train_seeds = {'serelate': torch.tensor(range(n_edges))}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(DGLlayer)
        self.gdataloader = dgl.dataloading.EdgeDataLoader(
            self.g, train_seeds, sampler, device=self.device, shuffle=True, batch_size=self.batch_size, drop_last=False)

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


class ValTestDataLoader(object):
    def __init__(self,args, g):
        self.device = torch.device(
            ('cuda:%d' % (0)) if torch.cuda.is_available() else 'cpu')
        self.ptr = 0
        self.data = []
        self.g = g
        data_file = './data/ASSIST/test_set.json'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        self.knowledge_dim = int(args.knowledge_n)

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
        sizeg = 1 + len(input_exer_ids)
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
