import numpy as np
from collections import Counter
from gensim.models import Word2Vec


def data_partition(item_data,time_data):
    User = {}
    Item=set()
    user_train = {}
    user_valid = {}
    user_test = {}
    time_train = {}
    sequence_length=[]
    print('Preparing data...')
    usernum=len(item_data)
    for item in item_data:
        content=item.strip('\n').split(' ')
        # print(content)
        id=content[0]
        for i in content[1:]:
            Item.add(i)
        user_train[id]=content[1:-2]
        user_valid[id]=content[-2]
        user_test[id]=content[-1]
        sequence_length.append(len(content[1:]))
    for time in time_data:
        content = time.strip('\n').split(' ')
        id = content[0]
        time_train[id] = content[1:-2]
    itemnum=len(Item)
    print('Max sequence length: ',max(sequence_length))
    print('Average sequence length: ', np.mean(sequence_length))
    print('Min sequence length: ', min(sequence_length))
    print('The number of users: ', usernum)
    print('The number of items: ', itemnum)
    print('Preparing done...')
    return [user_train, user_valid, user_test, time_train, usernum, itemnum]

def time_interval(time_data): # calculate the time interval
    interval={}
    avg_account=[]
    span=[]
    for times in time_data:
        content=times.strip('\n').split(' ')
        id=content[0]
        if id not in interval.keys():
            interval[id] = []
        for i in range(len(content)-2):
            interval[id].append(int(float(content[i+2]))-int(float(content[i+1])))
    # print(interval.values())
        time_span=list(interval[id])
        time_span=scale_interval(time_span)
        span.append(time_span)
    # avg_account=[sum(time)/len(time) for time in time_span]
        avg_account.append(sum(time_span)/len(time_span))
    avg=sum(avg_account)/len(avg_account)
    return span,avg

def read_item_seq():
    with open('data/train_item_data.txt', 'r') as f:
        lines = f.readlines()
    return [line for line in lines[1:]]

def read_time_seq():
    with open('data/train_item_data.txt', 'r') as f:
        lines = f.readlines()
    return [' '.join(line.split(' ')[1:]) for line in lines]

def positional_encoding(sequence_length, embedding_dim):
    position_enc = np.zeros((sequence_length, embedding_dim))
    for pos in range(sequence_length):
        for i in range(embedding_dim):
            if i % 2 == 0:
                position_enc[pos, i] = np.sin(pos / 10000 ** (2 * i / embedding_dim))
            else:
                position_enc[pos, i] = np.cos(pos / 10000 ** (2 * (i - 1) / embedding_dim))
    return position_enc

def convert_to_index_sequence(input_data, vocab):
    index_sequence = []
    for item in input_data:
        if item in vocab:
            index_sequence.append(vocab[item])
        else:
            index_sequence.append(vocab['<UNK>'])
    return index_sequence

def generate_vocab(input_data, max_vocab_size):
    counter = Counter(input_data)
    most_common = counter.most_common(max_vocab_size - 1)
    vocab = {word: index + 1 for index, (word, _) in enumerate(most_common)}
    vocab['<UNK>'] = 0
    return vocab

def word_vector(item_data):
    # 定义输入序列
    input_sequence = [item.strip('\n').split()[1:] for item in item_data]

    # Build the model instances of Word2Vec
    model = Word2Vec(input_sequence, vector_size=64, window=5, min_count=1, workers=4,seed=42)
    vector={}
    # Output the embedding of each item
    for item in list(model.wv.key_to_index.keys()):
        vector[item]=model.wv[item]
    return vector

def scale_interval(interval):
    span=interval
    min_t = min(span)
    max_t = max(span)
    span_new=[]
    for i in span:
        span_new.append((i-min_t+0.1)/(max_t-min_t+0.1))
    return span_new
def scale_fea(feature):
    span=feature
    span_new=[]
    min_t=min(span)
    max_t=max(span)
    for i in span:
        span_new.append((i-min_t)/(max_t-min_t))
    return span_new
def dbscan(X, eps, min_samples,time_span):
    """
    The implementation of DBSCAN algorithm by Python

    Parameters：
    X: array-like，shape(n_samples, n_features)
    eps: float，to determine the neighborhood
    min_samples: int，to determine the minimal samples of the centers

    Returns：
    labels: array-like，shape(n_samples,)
    """

    # 初始化变量
    X=np.array(X)
    n_samples, n_features = X.shape
    visited = np.zeros(n_samples)  # label it whether visited or not
    labels = np.zeros(n_samples)  # store the labels of clusters
    cluster_id = 0  # current index of the cluster

    # 计算距离矩阵
    dist = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = np.linalg.norm(X[i] - X[j])
            if i==j-1 or i==j+1:
                min_idx=min(i,j)
                alpha=time_span[min_idx]
                dist[i, j] = alpha*d
                dist[j, i] = alpha*d
            else:
                dist[i, j] = d
                dist[j, i] = d

    for i in range(n_samples):
        if visited[i] == 1:
            continue

        visited[i] = 1  # label the current sample visited
        neighbors = [j for j in range(n_samples) if dist[i, j] <= eps]  # find its neighborhood

        # centers or not
        if len(neighbors) >= min_samples:
            cluster_id += 1  # a new cluster
            labels[i] = cluster_id  # classify the sample into this cluster
            expand_cluster(X, neighbors, visited, labels, dist, eps, min_samples, cluster_id)

        # label it as noise
        else:
            labels[i] = -1

    return cluster_id,labels

def expand_cluster(X, neighbors, visited, labels, dist, eps, min_samples, cluster_id):
    """
    Expanded cluster

    Parameters：
    X: array-like，shape(n_samples, n_features)
    neighbors: list，the list of the neighborhood
    visited: array-like，shape(n_samples,)，visited or not
    labels: array-like，shape(n_samples,)，labels of clusters
    dist: array-like，shape(n_samples, n_samples)，distance matrix
    eps: float，eps to determine the neighborhood
    min_samples: int，to determine the minimal samples of the centers
    cluster_id: int，the index of the cluster
    """

    for i in neighbors:
        if visited[i] == 1:
            continue

        visited[i] = 1  # label it as visited
        labels[i] = cluster_id  # classify the sample into this cluster
        next_neighbors = [j for j in range(X.shape[0]) if dist[i, j] <= eps]  # determine its neighbors

        # add its neighbors if it is a center
        if len(next_neighbors) >= min_samples:
            neighbors += next_neighbors

        # label it as the boundary point
        else:
            continue

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

