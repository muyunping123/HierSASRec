import os
import time
import torch
import pickle
import argparse
from data_utils import *
from model import *
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='account_item')
parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--maxlen', default=128, type=int)
parser.add_argument('--hidden_units', default=60, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--balancing_factor', default=0.3, type=int)
# user level parameters
parser_user = argparse.ArgumentParser()
parser_user.add_argument('--dataset', default='user_item')
parser_user.add_argument('--train_dir', default='default')
parser_user.add_argument('--batch_size', default=384, type=int)
parser_user.add_argument('--lr', default=0.005, type=float)
parser_user.add_argument('--maxlen', default=16, type=int)
parser_user.add_argument('--hidden_units', default=60, type=int)
parser_user.add_argument('--num_blocks', default=2, type=int)
parser_user.add_argument('--num_epochs', default=20, type=int)
parser_user.add_argument('--num_heads', default=1, type=int)
parser_user.add_argument('--dropout_rate', default=0.2, type=float)
parser_user.add_argument('--l2_emb', default=0.00005, type=float)
parser_user.add_argument('--device', default='cuda:0', type=str)
parser_user.add_argument('--inference_only', default=False, type=str2bool)
parser_user.add_argument('--state_dict_path', default=None, type=str)
# parser.add_argument('--time_span', default=256, type=int)
args, unknown = parser.parse_known_args()
args_user, unknown_user = parser_user.parse_known_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
if not os.path.isdir(args_user.dataset + '_' + args_user.train_dir):
    os.makedirs(args_user.dataset + '_' + args_user.train_dir)
with open(os.path.join(args_user.dataset + '_' + args_user.train_dir, 'args_user.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args_user).items(), key=lambda x: x[0])]))
f.close()
def main():
    # account-level model train——user-level
    dataset = data_partition(args.dataset)
    dataset_user = data_partition_fuse(args_user.dataset)
    [user_train, user_valid, user_test, usernum, itemnum, user_list] = dataset
    [user_train2, user_valid2, user_test2, usernum2, itemnum2, user_list2, user_map] = dataset_user
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    num_batch2 = len(user_train2) // args_user.batch_size
    cc2 = 0.0
    for u in user_train2:
        cc2 += len(user_train2[u])
    print('average sequence length of user level: %.2f' % (cc2 / len(user_train2)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f2 = open(os.path.join(args_user.dataset + '_' + args_user.train_dir, 'log_user.txt'), 'w')

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1)
    model = MGSASRec(usernum, itemnum, args).to(args.device)

    sampler2 = WarpSampler(user_train2, usernum2, itemnum2, batch_size=args_user.batch_size, maxlen=args_user.maxlen, n_workers=1)
    model2 = MGSASRec(usernum2, itemnum2, args_user).to(args_user.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass # just ignore those failed init layers

    model.train() # enable model training

    for name, param in model2.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass # just ignore those failed init layers

    model2.train() # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)

    if args_user.state_dict_path is not None:
        try:
            model2.load_state_dict(torch.load(args_user.state_dict_path))
            tail2 = args_user.state_dict_path[args_user.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx2 = int(tail2[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args_user.state_dict_path)


    bce_criterion = torch.nn.BCEWithLogitsLoss()
    bce_criterion2 = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    adam_optimizer2 = torch.optim.Adam(model2.parameters(), lr=args_user.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            if step%30==0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        for step in range(num_batch2): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u2, seq2, pos2, neg2 = sampler2.next_batch() # tuples to ndarray
            u2, seq2, pos2, neg2 = np.array(u2), np.array(seq2), np.array(pos2), np.array(neg2)
            pos_logits2, neg_logits2 = model2(u2, seq2, pos2, neg2)
            pos_labels2, neg_labels2 = torch.ones(pos_logits2.shape, device=args_user.device), torch.zeros(neg_logits2.shape, device=args_user.device)
            adam_optimizer2.zero_grad()
            indices2 = np.where(pos2 != 0)
            loss2 = bce_criterion2(pos_logits2[indices2], pos_labels2[indices2])
            loss2 += bce_criterion2(neg_logits2[indices2], neg_labels2[indices2])
            for param in model2.item_emb.parameters(): loss2 += args_user.l2_emb * torch.norm(param)
            for param in model2.abs_pos_K_emb.parameters(): loss2 += args_user.l2_emb * torch.norm(param)
            for param in model2.abs_pos_V_emb.parameters(): loss2 += args_user.l2_emb * torch.norm(param)
            loss2.backward()
            adam_optimizer2.step()
            if step%30==0:
                print("loss in epoch {} iteration {}: {} --user level".format(epoch, step, loss2.item()))

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            model2.eval()
            print('Evaluating', end='')
            t_test = evaluate_fuse(model, model2, dataset, dataset_user, args, args_user)
            t_valid = evaluate_valid_fuse(model, model2, dataset, dataset_user, args, args_user)
            avg=T/t_test[3]
            print('query response time:%.4f(s)'%(avg))
            print('epoch:%d, time: %.4f(s), valid (NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f), test (NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_valid[2], t_test[0], t_test[1],t_test[2]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
            model2.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'MGSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
            folder2 = args_user.dataset + '_' + args_user.train_dir
            fname2 = 'MGSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname2 = fname2.format(args_user.num_epochs, args_user.lr, args_user.num_blocks, args_user.num_heads,
                                   args_user.hidden_units, args_user.maxlen)
            torch.save(model2.state_dict(), os.path.join(folder2, fname2))

    f.close()
    f2.close()
    sampler.close()
    sampler2.close()
    print("Done")
if __name__ == '__main__':
    main()