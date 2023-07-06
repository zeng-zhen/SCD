import json
import random

min_log = 5


def divide_data():
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set, val_set and test_set (0.7:0.1:0.2)
    :return:
    '''
    with open('./data/ASSIST/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)

    stu_i = 0
    l_log = 0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < min_log:
            del stus[stu_i]
            stu_i -= 1
        else:
            l_log += stus[stu_i]['log_num']
        stu_i += 1
    print('stu_i')
    print(stu_i)  
    print('log_num')  
    print(l_log)
    # return
    # 2. divide dataset into train_set, val_set and test_set
    train_slice, train_set, val_set, test_set = [], [], [], []
    for stu in stus:
        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_val = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * 0.6)
        val_size = stu['log_num'] - train_size  # int(stu['log_num'] * 0.1)
        test_size = 0  # stu['log_num'] - train_size - val_size
        logs = []
        for log in stu['logs']:
            logs.append(log)
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_val['log_num'] = val_size
        stu_val['logs'] = logs[train_size:train_size + val_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = []  #logs[-test_size:]
        train_slice.append(stu_train)
        val_set.append(stu_val)
        test_set.append(stu_test)
        # shuffle logs in train_slice together, get train_set
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'], 'knowledge_code': log['knowledge_code']})
    random.shuffle(train_set)
    with open('./data/ASSIST/train_slice.json', 'w', encoding='utf8') as output_file:
        json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
    with open('./data/ASSIST/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('./data/ASSIST/val_set.json', 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=4, ensure_ascii=False)
    with open('./data/ASSIST/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    divide_data()
    print('finish')
