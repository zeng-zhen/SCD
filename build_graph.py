# -*- coding: utf-8 -*-

import dgl
import torch



def build_graph(stu_n, exer_n, k_n):
    data_dict = {}
    edge_list = []
    with open('./data/ASSIST/graph/u_from_e.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            edge_list.append((int(line[0]), int(line[1])))

    src, dst = tuple(zip(*edge_list))
    data_dict[('exer', 'esrelate', 'stu')] = (
        torch.tensor(src), torch.tensor(dst))
    data_dict[('stu', 'serelate', 'exer')] = (
        torch.tensor(dst), torch.tensor(src))

    edge_list = []
    with open('./data/ASSIST/graph/k_from_e.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            edge_list.append((int(line[0]), int(line[1])))

    src, dst = tuple(zip(*edge_list))
    data_dict[('exer', 'ekrelate', 'k')] = (
        torch.tensor(src), torch.tensor(dst))
    data_dict[('k', 'kerelate', 'exer')] = (
        torch.tensor(dst), torch.tensor(src))

    num_nodes_dict = {}
    num_nodes_dict['exer'] = exer_n
    num_nodes_dict['stu'] = stu_n
    num_nodes_dict['k'] = k_n
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    return g


def build_sub_graph(stu_n, exer_n, k_n, name=None):
    if name is None:
        name = ''
    data_dict = {}
    edge_list = []
    with open('./data/ASSIST/graph/u_from_e' + name + '.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            edge_list.append((int(line[0]), int(line[1])))

    src, dst = tuple(zip(*edge_list))

    data_dict[('exer', 'esrelate', 'stu')] = (
        torch.tensor(src), torch.tensor(dst))
    
    edge_list = []
    with open('./data/ASSIST/graph/e_from_u' + name + '.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            edge_list.append((int(line[0]), int(line[1])))

    src, dst = tuple(zip(*edge_list))
    data_dict[('stu', 'serelate', 'exer')] = (
        torch.tensor(src), torch.tensor(dst))

    edge_list = []
    with open('./data/ASSIST/graph/k_from_e.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            edge_list.append((int(line[0]), int(line[1])))

    src, dst = tuple(zip(*edge_list))
    data_dict[('exer', 'ekrelate', 'k')] = (
        torch.tensor(src), torch.tensor(dst))
    data_dict[('k', 'kerelate', 'exer')] = (
        torch.tensor(dst), torch.tensor(src))

    num_nodes_dict = {}
    num_nodes_dict['exer'] = exer_n
    num_nodes_dict['stu'] = stu_n
    num_nodes_dict['k'] = k_n
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    return g
