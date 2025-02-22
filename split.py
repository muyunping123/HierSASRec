from tqdm import *

from utils import *

with open('data/account_item.txt','r') as f1: #load the dataset
    items=f1.readlines()
with open('data/account_time.txt','r') as f2: #load the dataset
    times=f2.readlines()

[user_train, user_valid, user_test, time_train, usernum, itemnum]=data_partition(items,times)
id=list(user_train.keys())
items=list(user_train.values())
times=list(time_train.values())
def save_item_train():
    with open('data/train_item_data.txt','w') as f:
        for i in range(len(id)):
            item = ' '.join(items[i])
            f.write(id[i] + ' ' + item + '\n')
    print('Train set of item data is saved!')

def save_time_train():
    with open('data/train_time_data.txt','w') as f:
        for i in range(len(id)):
            timestamp = ' '.join(times[i])
            f.write(id[i] + ' ' + timestamp + '\n')
    print('Train set of time data is saved!')
save_item_train()
save_time_train()
# load the data
with open('data/train_item_data.txt','r') as f1:
    items=f1.readlines()
with open('data/train_time_data.txt','r') as f2:
    timestamps=f2.readlines()

intervals,time_avg=time_interval(timestamps)
def pos_enc(item_data):
    position=[]
    for item in item_data:
        input_data = item.strip('\n').split(' ')[1:]
        account_id=item.strip('\n').split(' ')[0]
        # generate the vocabulary
        vocab = generate_vocab(input_data, max_vocab_size=10000)

        # convert it into indices
        index_sequence = convert_to_index_sequence(input_data, vocab)

        # get the length of sequences and dimension of embeddings
        sequence_length = len(index_sequence)
        if sequence_length <= 100:
            embedding_dim = 8
        elif sequence_length <= 500 and sequence_length > 100:
            embedding_dim = 32
        elif sequence_length > 500:
            embedding_dim = 64
        # generate the positional encoding
        position_enc = positional_encoding(sequence_length, embedding_dim)

        # print('ID',account_id,'\n',position_enc)
        position.append(position_enc)
    return position
position_encoder=pos_enc(items)

feature=word_vector(items)

def encoder(item_data,time_data,position_encoder,feature_encoder):
    representation=[]
    for idx,pos in enumerate(position_encoder):
        repre=[]
        line=item_data[idx].strip('\n').split(' ')[1:]
        for j in range(len(pos)):
            update=list(feature_encoder[line[j]])
            update=scale_fea(update)
            update.extend(list(pos[j])) # add the positional encoding
            repre.append(update)
        representation.append(repre)
    return representation

representation=encoder(items,times,position_encoder,feature)
def save_embed(representation):
    with open('data/embedding.txt','w') as f:
        for idx,r in enumerate(representation):
            id=items[idx].strip('\n').split(' ')[0]
            for j in r:
                con=' '.join(map(str, j))
                f.write(id+' '+con+'\n')
    print('The vectors of embedding have been saved!')
save_embed(representation)
def cluster(representation,time_interval,eps,min_samples):
    span=scale_interval(time_interval)
    cluster_id,labels=dbscan(representation, eps, min_samples, span)

    return cluster_id,labels

with open('data/roles_v2.txt','w') as f:
    for i in tqdm(range(len(representation))):
        id=items[i].strip('\n').split(' ')[0]
        re=representation[i];ti=intervals[i]
        cluster_id,labels=cluster(re,ti,eps=3.5,min_samples=5)
        content=' '.join(map(str,labels))
        f.write(id+' '+str(cluster_id)+' ')
        for j in labels:
            f.write(str(j)+' ')
        f.write('\n')
