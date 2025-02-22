import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue
import queue

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(user):

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]

        idx = maxlen - 1
        ts = set(map(lambda x: x, user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=128, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def cleanAndsort(User):
    User_filted = dict()
    user_set = set()
    item_set = set()
    user_list = []
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        item_set.update(items)
        user_list.append(user)
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set): #编码
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]]=[item_map[x] for x in items]

    return User_res, len(user_set), len(item_set), user_list, user_map


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = {}
    user_train = {}
    user_valid = {}
    user_test = {}

    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')
    # time_set = set()
    user_count=set()
    item_count=set()

    for line in f:
        u=int(line.rstrip().split(' ')[0])
        i = line.rstrip().split(' ')[1:]
        user_count.add(u)
        item_count.update(i)
        User[u]=i
    f.close()

    User, usernum, itemnum, user_list, user_map = cleanAndsort(User)

    for user in User:
        user_train[user] = User[user][:-2]
        user_valid[user] = []
        user_valid[user].append(User[user][-2])
        user_test[user] = []
        user_test[user].append(User[user][-1])
    print('Preparing done...')
    return [user_train, user_valid, user_test, usernum, itemnum, user_list]

def data_partition_fuse(fname):
    usernum = 0
    itemnum = 0
    User = {}
    user_train = {}
    user_valid = {}
    user_test = {}

    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')

    user_count=set()
    item_count=set()

    for line in f:
        u=line.rstrip().split(' ')[0]
        i = line.rstrip().split(' ')[1:]
        user_count.add(u)
        item_count.update(i)
        User[u]=i
    f.close()

    User, usernum, itemnum, user_list, user_map = cleanAndsort(User)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 5:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print('Preparing done...')
    return [user_train, user_valid, user_test, usernum, itemnum, user_list, user_map]

def evaluate_fuse(model, model2, dataset, dataset_user, args, args_user):
    [train, valid, test, usernum, itemnum, user_list] = copy.deepcopy(dataset)
    [train2, valid2, test2, usernum2, itemnum2, user_list2, user_map] = copy.deepcopy(dataset_user)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    MRR= 0.0

    users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(map(lambda x: x, train[u]))
        rated.add(valid[u][0])
        rated.add(test[u][0])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(300):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        #judge the shift of the user in account
        account=str(user_list[u-1])
        real_user=[]
        for users in user_list2:
            ac=users.split('_')[0]
            if account==ac:
                real_user.append(users)
        shift=np.random.randint(0, len(real_user))
        seq2 = np.zeros([args_user.maxlen], dtype=np.int32)
        u = user_map[real_user[shift]]
        idx = args.maxlen - 1
        for i in reversed(train2[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        predictions2 = -model2.predict(*[np.array(l) for l in [[u], [seq2], item_idx]])
        predictions2 = predictions2[0]

        final_pred=args.balancing_factor*predictions+(1-args.balancing_factor)*predictions2

        rank = final_pred.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 1000 == 0:
            print('.', end='')
            sys.stdout.flush()
        MRR += 1 / (rank + 1)

    return NDCG / valid_user, HT / valid_user, MRR/valid_user, valid_user

def evaluate_valid_fuse(model, model2, dataset, dataset_user, args, args_user):
    [train, valid, test, usernum, itemnum, user_list] = copy.deepcopy(dataset)
    [train2, valid2, test2, usernum2, itemnum2, user_list2, user_map] = copy.deepcopy(dataset_user)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    MRR =0.0
    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(map(lambda x: x, train[u]))
        rated.add(valid[u][0])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(300):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        # judge the shift of the user in account
        account = str(user_list[u - 1])
        real_user = []
        for users in user_list2:
            ac = users.split('_')[0]
            if account == ac:
                real_user.append(users)
        shift = np.random.randint(0, len(real_user))
        seq2 = np.zeros([args_user.maxlen], dtype=np.int32)
        u = user_map[real_user[shift]]
        idx = args.maxlen - 1
        for i in reversed(train2[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        predictions2 = -model2.predict(*[np.array(l) for l in [[u], [seq2], item_idx]])
        predictions2 = predictions2[0]

        final_pred = args.balancing_factor * predictions + (1 - args.balancing_factor) * predictions2
        rank = final_pred.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 1000 == 0:
            print('.', end='')
            sys.stdout.flush()
        MRR += 1 / (rank + 1)
    return NDCG / valid_user, HT / valid_user, MRR/valid_user