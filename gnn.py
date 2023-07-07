import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch.hetero as hetero
import dgl.nn.pytorch.conv.gatconv as gatconv
from dgl import function as fn




class GATLayer(hetero.HeteroGraphConv):
    def my_aggregate(self, input, tensors, dsttype):
        if dsttype == 'stu':
            score0 = self.s_attn0(torch.cat([input, tensors[0]], dim=1))
            score = F.softmax(score0, dim=1)
            emb = input + score[:, 0].unsqueeze(1) * tensors[0]
            return emb
        elif dsttype == 'exer':
            score0 = self.e_attn0(torch.cat([input, tensors[0]], dim=1))
            score1 = self.e_attn1(torch.cat([input, tensors[1]], dim=1))
            score = F.softmax(torch.cat([score0, score1], dim=1), dim=1) 
            emb = input + score[:, 0].unsqueeze(1) * tensors[0] + score[:, 1].unsqueeze(1) * tensors[1]
            return emb
        elif dsttype == 'k':
            score0 = self.k_attn0(torch.cat([input, tensors[0]], dim=1))
            score = F.softmax(score0, dim=1) 
            emb = input + score[:, 0].unsqueeze(1) * tensors[0]
            return emb

    def __init__(self, in_dim, out_dim, knowledge_n):
        mods = {'esrelate': gatconv.GATConv(in_dim, out_dim, num_heads=1), 'serelate': gatconv.GATConv(in_dim, out_dim, num_heads=1), 'ekrelate': gatconv.GATConv(in_dim, out_dim,num_heads=1), 'kerelate': gatconv.GATConv(in_dim, out_dim,num_heads=1)}
        super(GATLayer, self).__init__(mods, aggregate=self.my_aggregate)
        self.e_attn0 = nn.Linear(2 * knowledge_n, 1, bias=True)
        self.e_attn1 = nn.Linear(2 * knowledge_n, 1, bias=True)
        self.s_attn0 = nn.Linear(2 * knowledge_n, 1, bias=True)
        self.k_attn0 = nn.Linear(2 * knowledge_n, 1, bias=True)
        
    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](rel_graph, (src_inputs[stype], dst_inputs[dtype]), *mod_args.get(etype, ()), **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](rel_graph, (inputs[stype], inputs[dtype]), *mod_args.get(etype, ()), **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        self.dst_inputs = dst_inputs

        for k in outputs.keys():
            for i in range(len(outputs[k])):
                outputs[k][i] = outputs[k][i].squeeze(dim=1)

        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.my_aggregate(dst_inputs[nty], alist, nty)
        return rsts


