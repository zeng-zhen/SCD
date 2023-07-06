import torch
stu_num = 4163
exer_num = 17751


def remake(name=None):  # u→e
    data_file = '/home/q21201141/data/ASSIST_log5_64/graph/e_from_u.txt'
    s_log_dict = torch.zeros(stu_num)
    s_log_dict = s_log_dict.tolist()
    e_log_dict = torch.zeros(exer_num)
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

    s_log_dict = torch.Tensor(s_log_dict)
    e_log_dict = torch.Tensor(e_log_dict)
    e_log_dict = torch.pow(e_log_dict.float().clamp(min=1),-0.25).clamp(min=torch.pow(torch.tensor(123), -0.5).to(e_log_dict.device))
    s_log_dict = torch.pow(s_log_dict.float().clamp(min=1),-0.25).clamp(min=torch.pow(torch.tensor(123), -0.5).to(e_log_dict.device))
    e_rand = torch.rand(len(data))
    s_rand = torch.rand(len(data))
    
    exer_number = torch.LongTensor([e for (s,e) in data])
    stu_number = torch.LongTensor([s for (s,e) in data])

    e_mask = e_log_dict[exer_number] - e_rand
    e_index = torch.nonzero(e_mask > 0).squeeze().tolist()
    s_mask = s_log_dict[stu_number] - s_rand
    s_index = torch.nonzero(s_mask > 0).squeeze().tolist()

    e_from_u = [data[i] for i in e_index]
    u_from_e = [data[i] for i in s_index]

    e_from_u = [str(u) + '\t' + str(e) + '\n' for (u,e) in e_from_u]
    u_from_e = [str(e) + '\t' + str(u) + '\n' for (u,e) in u_from_e]

    e_from_u = "".join(e_from_u)
    u_from_e = "".join(u_from_e)
    if name is not None:
        with open('./view/u_from_e' + name + '.txt', 'w') as f:
            f.write(u_from_e)
        with open('./view/e_from_u' + name + '.txt', 'w') as f:
            f.write(e_from_u)
        
            
        
            


