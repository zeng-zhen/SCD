#from dgl.subgraph import edge_subgraph

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
from utils import CommonArgParser, construct_local_map, build_ue
import time


def train(args, local_map):
    data_loader = TrainDataLoader(local_map)
    device = torch.device(('cuda:%d' % (args.gpu))
                          if torch.cuda.is_available() else 'cpu')

    net = Net(args)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    print('training model...')
    loss_function = nn.NLLLoss()
    for epoch in range(args.epoch_n):
        data_loader.reset()
        time_start = time.time()
        running_loss = 0.0
        batch_count = 0
        net.train()
        while not data_loader.is_end():
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, stu_index, exer_index, input_nodes, output_nodes, blocks = data_loader.next_batch()
            data_loader.g1 = build_ue(args, name="map1")
            data_loader.g2 = build_ue(args, name="map2")
            g1 = data_loader.get(blocks[1].dstdata['_ID'], 1)
            g2 = data_loader.get(blocks[1].dstdata['_ID'], 2)
            optimizer.zero_grad()
            output_1, c_loss = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, input_nodes, output_nodes, blocks, stu_index, exer_index, g1=g1, g2=g2)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)
            loss = loss_function(torch.log(output + 1e-10), labels)
            allloss = loss + args.alpha * c_loss

            allloss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0
            batch_count += 1

        time_end = time.time()
        print('epoch time ', time_end - time_start, ' second')
        # validate and save current model every epoch
        save_snapshot(net, 'model/' + 'model' + '_epoch' + str(epoch + 1))
        predict(args, local_map, net, epoch)


def predict(args, g, net, epoch):
    device = torch.device(('cuda:%d' % (args.gpu))
                          if torch.cuda.is_available() else 'cpu')
    data_loader = ValTestDataLoader(g)
    print('predicting model...')
    data_loader.reset()
    net.eval()
    time_start = time.time()
    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, gdataloader = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(
            device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
        input_nodes, output_nodes, blocks = next(iter(gdataloader))
        output = net.forward_test(input_stu_ids, input_exer_ids, input_knowledge_embs, input_nodes, output_nodes, blocks)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()
    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all)**2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    time_end = time.time()
    print('predict time ', time_end - time_start, ' second')
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' %
          (epoch + 1, accuracy, rmse, auc))
    with open('result/' + 'scd_model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' %
                (epoch + 1, accuracy, rmse, auc))

    return


def test(args, g, epoch):
    device = torch.device(('cuda:%d' % (args.gpu))
                          if torch.cuda.is_available() else 'cpu')
    data_loader = ValTestDataLoader(g)
    print('predicting model...')
    data_loader.reset()
    net = Net(args)
    load_snapshot(net, 'model/' + 'model' + '_epoch' + str(epoch))
    net.to(device)
    net.eval()
    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, gdataloader = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(
            device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
        input_nodes, output_nodes, blocks = next(iter(gdataloader))
        output = net.forward_test(input_stu_ids, input_exer_ids, input_knowledge_embs, input_nodes, output_nodes, blocks)
        output = output.view(-1)
        # compute accuracy
        stu_correct_count = 0.0
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                stu_correct_count += 1
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all)**2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' %
          (epoch, accuracy, rmse, auc))
    with open('result/'+'scd_model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' %
                (epoch, accuracy, rmse, auc))

    return


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    train(args, construct_local_map(args))
