import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn import GATLayer


class Net(nn.Module):
    def __init__(self, args):
        self.device = torch.device(
            ('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256
        super(Net, self).__init__()

        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.knowledge_emb = nn.Embedding(
            self.knowledge_dim, self.knowledge_dim)

        self.gnet1 = GATLayer(
            args.knowledge_n, args.knowledge_n, args.knowledge_n)
        self.gnet2 = GATLayer(
            args.knowledge_n, args.knowledge_n, args.knowledge_n)

        self.stu_full = nn.Linear(
            2 * args.knowledge_n, args.knowledge_n, bias=False)
        self.exer_full = nn.Linear(
            2 * args.knowledge_n, args.knowledge_n, bias=False)
        self.prednet_full1 = nn.Linear(
            1 * args.knowledge_n, args.knowledge_n, bias=False)

        self.prednet_full2 = nn.Linear(
            1 * args.knowledge_n, args.knowledge_n, bias=False)

        self.prednet_full3 = nn.Linear(1 * args.knowledge_n, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_r, input_nodes, out_dict, blocks, predict=False, g1=None, g2=None):
        stu_emb = self.student_emb(input_nodes['stu']).to(self.device)
        exer_emb = self.exercise_emb(input_nodes['exer']).to(self.device)
        k_emb = self.knowledge_emb(input_nodes['k']).to(self.device)

        node_emb = {'stu': stu_emb, 'exer': exer_emb, 'k': k_emb}
        # Fusion layer 1
        node_emb1 = self.gnet1(blocks[0], node_emb)
        # Fusion layer 2
        node_emb2 = self.gnet2(blocks[1], node_emb1)

        exer_emb2 = node_emb2['exer']
        stu_emb2 = node_emb2['stu']
        k_emb = node_emb2['k']

        if predict is False:
            stu_emb_g1 = self.student_emb(g1[0]['stu']).to(self.device)
            exer_emb_g1 = self.exercise_emb(g1[0]['exer']).to(self.device)
            k_emb_g1 = self.knowledge_emb(g1[0]['k']).to(self.device)
            node_emb_g1 = {'stu': stu_emb_g1,
                           'exer': exer_emb_g1, 'k': k_emb_g1}
            node_emb1_g1 = self.gnet1(g1[2][0], node_emb_g1)
            node_emb2_g1 = self.gnet2(g1[2][1], node_emb1_g1)
            stu_emb2_g1 = node_emb2_g1['stu'].squeeze()
            exer_emb2_g1 = node_emb2_g1['exer'].squeeze()

            stu_emb_g2 = self.student_emb(g2[0]['stu']).to(self.device)
            exer_emb_g2 = self.exercise_emb(g2[0]['exer']).to(self.device)
            k_emb_g2 = self.knowledge_emb(g2[0]['k']).to(self.device)
            node_emb_g2 = {'stu': stu_emb_g2,
                           'exer': exer_emb_g2, 'k': k_emb_g2}
            node_emb1_g2 = self.gnet1(g2[2][0], node_emb_g2)
            node_emb2_g2 = self.gnet2(g2[2][1], node_emb1_g2)
            stu_emb2_g2 = node_emb2_g2['stu'].squeeze()
            exer_emb2_g2 = node_emb2_g2['exer'].squeeze()

            c_s_h1_loss = self.contrastive_loss(stu_emb2_g1, stu_emb2_g2)
            c_s_h2_loss = self.contrastive_loss(stu_emb2_g2, stu_emb2_g1)
            c_e_h1_loss = self.contrastive_loss(exer_emb2_g1, exer_emb2_g2)
            c_e_h2_loss = self.contrastive_loss(exer_emb2_g2, exer_emb2_g1)

            contrastive_loss = (0.5 * (c_s_h1_loss + c_s_h2_loss) + 0.5 * (c_e_h1_loss + c_e_h2_loss))

        if predict is False:
            batch_stu_id = out_dict.edges(etype='serelate')[0]
            batch_exer_id = out_dict.edges(etype='serelate')[1]
            batch_stu_emb = stu_emb2[batch_stu_id]
            batch_exer_emb = exer_emb2[batch_exer_id]
        else:
            batch_stu_emb = stu_emb2.repeat(exer_emb2.shape[0], 1)
            batch_exer_emb = exer_emb2

        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(
            batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1])

        # get batch exercise data
        # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(
            batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        # CD
        preference = torch.sigmoid(self.prednet_full1(batch_stu_vector))
        diff = torch.sigmoid(self.prednet_full2(batch_exer_vector))
        o = torch.sigmoid(self.prednet_full3(preference - diff))
        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_r, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept

        if predict is False:
            return output, contrastive_loss
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def contrastive_loss(self, h1, h2):
        t = 0.5
        batch_size = h1.shape[0]
        negatives_mask = (~torch.eye(batch_size, batch_size,
                          dtype=bool)).to(self.device).float()
        z1 = h1#F.normalize(h1, dim=1)
        z2 = h2#F.normalize(h2, dim=1)
        similarity_matrix1 = F.cosine_similarity(
            z1.unsqueeze(1), z2.unsqueeze(0), dim=2)
        positives = torch.exp(torch.diag(similarity_matrix1) / t)
        negatives = negatives_mask * torch.exp(similarity_matrix1 / t)
        loss_partial = -torch.log(positives / (positives + torch.sum(negatives, dim=1)))
        loss = torch.sum(loss_partial) / batch_size
        return loss


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
