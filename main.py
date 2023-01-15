#from dgl.subgraph import edge_subgraph
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
from utils import CommonArgParser, construct_local_map, build_ue
import time


def train(args, local_map):
    alpha = args.alpha
    data_loader = TrainDataLoader(args, local_map)
    device = torch.device(('cuda:%d' % (args.gpu))
                          if torch.cuda.is_available() else 'cpu')

    net = Net(args)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    print('training model...')
    loss_function = nn.NLLLoss()
    for epoch in range(args.epoch_n):
        time_start = time.time()
        running_loss = 0.0
        batch_count = 0
        net.train()
        for input_nodes, edge_subgraph, blocks in data_loader.gdataloader:
            data_loader.g1 = build_ue(args, name="map1")
            data_loader.g2 = build_ue(args, name="map2")
            g1 = data_loader.get(blocks[1].dstdata['_ID'], 1)
            g2 = data_loader.get(blocks[1].dstdata['_ID'], 2)
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = [], [], [], []
            for stu, exer in zip(edge_subgraph.edges(etype='serelate')[0], edge_subgraph.edges(etype='serelate')[1]):
                stu_id = edge_subgraph.nodes['stu'].data['_ID'][stu]
                exer_id = edge_subgraph.nodes['exer'].data['_ID'][exer]
                input_stu_ids.append(stu_id)
                input_exer_ids.append(exer_id)
                knowledge_emb = data_loader.edge_data[(
                    int(stu_id), int(exer_id))]['knowledge_emb']
                input_knowledge_embs.append(knowledge_emb)
                y = data_loader.edge_data[(int(stu_id), int(exer_id))]['y']
                labels.append(y)
            optimizer.zero_grad()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = torch.LongTensor(input_stu_ids).to(device), torch.LongTensor(
                input_exer_ids).to(device), torch.Tensor(input_knowledge_embs).to(device), torch.LongTensor(labels).to(device)
            output_1, c_loss = net.forward(input_stu_ids, input_exer_ids,
                                           input_knowledge_embs, input_nodes, edge_subgraph, blocks, g1=g1, g2=g2)

            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output + 1e-10), labels)

            allloss = loss + alpha * c_loss

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

        save_snapshot(net, 'model/' + 'epoch' + str(epoch + 1))
        rmse, auc = predict(args, local_map, net, epoch, alpha)


def predict(args, g, net, epoch, alpha):
    device = torch.device(('cuda:%d' % (args.gpu))
                          if torch.cuda.is_available() else 'cpu')
    data_loader = ValTestDataLoader(args,g)
    print('predicting model...')
    data_loader.reset()
    net.eval()
    time_start = time.time()
    correct_count, exer_count = 0, 0
    batch_count = 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, gdataloader = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(
            device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
        input_nodes, output_nodes, blocks = next(iter(gdataloader))
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs,
                             input_nodes, output_nodes, blocks, predict=True)
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

    time_end = time.time()
    print('predict time ', time_end - time_start, ' second')
    print('epoch= %d, accuracy= %f, rmse= %f' %
          (epoch + 1, accuracy, rmse))
    with open('result/' + str(alpha) + 'ncd_model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f' %
                (epoch + 1, accuracy, rmse))

    return


def test(args, g, epoch):
    device = torch.device(('cuda:%d' % (args.gpu))
                          if torch.cuda.is_available() else 'cpu')
    data_loader = ValTestDataLoader(args,g)
    print('predicting model...')
    data_loader.reset()
    net = Net(args)
    load_snapshot(net, 'model/' + 'epoch' + str(epoch))
    net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    student_set = {}
    student_rmse = {}
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, gdataloader = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(
            device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
        input_nodes, output_nodes, blocks = next(iter(gdataloader))
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs,
                             input_nodes, output_nodes, blocks, predict=True)
        output = output.view(-1)
        # compute accuracy
        stuid = int(input_stu_ids[0].item())
        stu_correct_count, stu_exer_count = 0.0, 0.0
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                stu_correct_count += 1
                correct_count += 1
        stu_exer_count = len(labels)
        exer_count += len(labels)

        student_set[stuid] = stu_correct_count / stu_exer_count
        pred_stu = np.array(output.to(torch.device('cpu')).tolist())
        label_stu = np.array(labels.to(torch.device('cpu')).tolist())
        student_rmse[stuid] = np.sqrt(np.mean((label_stu - pred_stu)**2))

        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all)**2))

    print('epoch= %d, accuracy= %f, rmse= %f' %
          (epoch, accuracy, rmse))
    with open('result/'+'ncd_model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f' %
                (epoch, accuracy, rmse))
    with open('./result/' + 'divede_by_student.json', 'w', encoding='utf8') as output_file:
        json.dump(student_set, output_file, indent=4, ensure_ascii=False)
    with open('./result/' + 'divede_by_student_rmse.json', 'w', encoding='utf8') as output_file:
        json.dump(student_rmse, output_file, indent=4, ensure_ascii=False)
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
    test(args, construct_local_map(args), 2)
