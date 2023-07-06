import argparse
from build_graph import build_graph, build_sub_graph
from build_stu import remake


class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--exer_n', type=int, default=17751,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=123,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int,
                          default=4163, help='The number for student.')
        self.add_argument('--gpu', type=int, default=0,
                          help='The id of gpu, e.g. 0.')
        self.add_argument('--epoch_n', type=int, default=5,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.0001,
                          help='Learning rate')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the testing set in the training process.')
        self.add_argument('--alpha', type=float, default=0.01,
                          help='contrastive learning')



def construct_local_map(args):
    remake(args)
    map = build_graph(args.student_n, args.exer_n, args.knowledge_n)

    return map


def build_ue(args, name=None):
    remake(args,name)
    sub_map = build_sub_graph(
        args.student_n, args.exer_n, args.knowledge_n, name=name)
    return sub_map
