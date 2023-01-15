import random
import torch
k = 2
pmin = 0.4


def remake(name=None):  # u→e
    data_file = 'data/ASSIST/graph/e_from_u.txt'
    s_log_dict = torch.zeros(args.student_n)
    s_log_dict = s_log_dict.tolist()
    e_log_dict = torch.zeros(args.exer_n)
    e_log_dict = e_log_dict.tolist()
    data = []
    with open(data_file, encoding='utf8') as i_f:
        for line in i_f:
            stu, exer = line.replace('\n', '').split('\t')
            stu = int(stu)
            exer = int(exer)
            data.append((stu, exer))
            s_log_dict[stu] = s_log_dict[stu] + 1
            e_log_dict[exer] = e_log_dict[exer] + 1

    u_from_e = ''  # e(src) to k(dst)
    e_from_u = ''  # k(src) to k(dst)

    i = 0
    j = 0
    s_log_dict = torch.Tensor(s_log_dict)
    e_log_dict = torch.Tensor(e_log_dict)
    e_log_dict = k / torch.log(e_log_dict + 1 + 10e-5)
    e = torch.full((args.exer_n,), pmin)
    e_log_dict = torch.where(e_log_dict > pmin, e_log_dict, e)
    s_log_dict = k / torch.log(s_log_dict + 1 + 10e-5)
    s = torch.full((args.student_n,), pmin)
    s_log_dict = torch.where(s_log_dict > pmin, s_log_dict, s)
    random.seed()
    e_rand = torch.rand(len(data)).tolist()
    random.seed()
    s_rand = torch.rand(len(data)).tolist()
    t = 0
    for stu, exer in data:
        if e_rand[t] < e_log_dict[exer]:
            e_from_u += str(stu) + '\t' + str(exer) + '\n'
            i += 1
        if s_rand[t] < s_log_dict[stu]:
            u_from_e += str(exer) + '\t' + str(stu) + '\n'
            j += 1
        t += 1
    if name is not None:
        with open('./data/ASSIST/graph/u_from_e' + name + '.txt', 'w') as f:
            f.write(u_from_e)
        with open('./data/ASSIST/graph/e_from_u' + name + '.txt', 'w') as f:
            f.write(e_from_u)
