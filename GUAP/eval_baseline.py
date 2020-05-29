#!/usr/bin/env python
# coding: utf-8



from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import math
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import os.path as op
from utils import load_data, accuracy, normalize, load_polblogs_data
from models import GCN
os.environ["CUDA_VISIBLE_DEVICES"]="0" 




# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="citeseer",
                    help='The name of the network dataset.')
parser.add_argument('--radius', type=int, default=12,
                    help='The radius of l2 norm projection')
parser.add_argument('--fake_rate', type=float, default=0.02,
                    help='The ratio of patch nodes to the graph size')
parser.add_argument('--step', type=int, default=10,
                    help='The learning step of updating the connection entries')
parser.add_argument('--sample_percent', type=int, default=40,
                    help='The sampling ratio of train set')
parser.add_argument('--evaluate_mode', type=str, default='universal',
                    help='universal, no_connection, random_connection')
args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



if args.dataset == "polblogs":
    tmp_adj, tmp_feat, labels, train_idx, val_idx, test_idx = load_polblogs_data()
else:
    _, _, labels, train_idx, val_idx, test_idx, tmp_adj, tmp_feat  = load_data(args.dataset)

num_classes = labels.max().item() + 1
# tmp_adj = tmp_adj.toarray()
adj = tmp_adj
adj = np.eye(tmp_adj.shape[0]) + adj
adj, _ = normalize(adj)
adj = torch.from_numpy(adj.astype(np.float32))
feat, _ = normalize(tmp_feat)
feat = torch.FloatTensor(np.array(feat.todense()))
tmp_feat = tmp_feat.todense()




# Model and optimizer
model = GCN(nfeat=feat.shape[1],
            nhid=args.hidden,
            nclass=num_classes,
            dropout=args.dropout
           )
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)



if args.cuda:
    model.cuda()
    features = feat.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = train_idx.cuda()
#     if args.dataset != "polblogs":
    idx_val = val_idx.cuda()
    idx_test = test_idx.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    x = Variable(adj, requires_grad=True)
    output = model(features, x)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
#     print ('output', output.size())
#     print ('labels', labels.size())
    loss_train.backward()

    optimizer.step()


#     if args.dataset != "polblogs": 
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
#     else:
#         print('Epoch: {:04d}'.format(epoch+1),
#               'loss_train: {:.4f}'.format(loss_train.item()),
#               'acc_train: {:.4f}'.format(acc_train.item()),
#               'time: {:.4f}s'.format(time.time() - t))



def test(adj_m):
    model.eval()
    output = model(features, adj_m)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output




t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# torch.save(model, './cora_gcn.pth')
# torch.save(model.state_dict(), 'cora_gcn.pkl')

# Testing
ori_output = test(adj)
correct_res = ori_output[idx_train, labels[idx_train]] #the prediction probability of all train nodes
num_fake = int(tmp_adj.shape[0] * args.fake_rate)




def gaussian_dist(innormal_features):
#     while True: #the generated node feature shouldnt be all 0
    print ('innormal', np.sum(innormal_features))
    feat_mean = np.mean(innormal_features, axis = 0)
    feat_std = np.std(innormal_features, axis = 0)
    feat_fake = np.zeros((num_fake, innormal_features.shape[1]))
    print ('feat_mean', feat_mean)
    print ('feat_std', feat_std)
    for i in range(innormal_features.shape[1]):
        feat_fake[:,i] = np.random.normal(feat_mean[0, i], feat_std[0, i], num_fake).reshape(feat_fake[:,i].shape)
#         print (i, feat_fake[:,i])
    feat_fake = np.where(feat_fake > 0.5, 1, 0).astype(np.float32)
    feat_fake, _ = normalize(feat_fake)
#         if np.sum(feat_fake) >= num_fake:
#             break
    return feat_fake




def add_perturb(input_adj, idx, perturb):
    x = np.zeros((input_adj.shape[0], input_adj.shape[1]))
    x[idx] = perturb  
    x[:,idx] = perturb
#     print ('x', x[idx])

    x1 = np.ones((input_adj.shape[0], input_adj.shape[1])) - x
#     print ('x1', x1[idx])
    adj2 = np.ones((input_adj.shape[0], input_adj.shape[1])) - input_adj
#     print ('adj2', adj2[idx])

    for i in range(input_adj.shape[0]):      
        adj2[i][i] = 0

    perturbed_adj = np.multiply(x1, input_adj) + np.multiply(x, adj2)
    return perturbed_adj



def evaluate_attack(new_adj, new_feat):
    res = []
    # perturb = np.where(perturb>0.5, 1, 0)
#     print ('perturb', perturb)
    v1 = np.zeros(tmp_adj.shape[0]).astype(np.float32)
    v2 = np.ones(num_fake).astype(np.float32)
    perturb = np.concatenate((v1, v2))
    new_pred = []
    all_acc = []
    for i in range(num_classes):
        new_pred.append(0)
    for k in idx_test:
#     for k in range(1):
#         print ('test node', k)
        innormal_x_p = add_perturb(new_adj, k, perturb)
        x_p, degree_p = normalize(innormal_x_p + np.eye(new_adj.shape[0]))
        x_p = torch.from_numpy(x_p.astype(np.float32))
        x_p = x_p.cuda()
        output = model(new_feat, x_p)
        new_pred[int(torch.argmax(output[k]))] += 1
        test_acc = accuracy(output[idx_test], labels[idx_test])
        all_acc.append(test_acc)
        if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
            res.append(0)
            print ('node {} attack failed'.format(k))
        else:
            res.append(1)
            print ('node {} attack succeed'.format(k))
    
    fooling_rate = float(sum(res)/len(res))
    overal_acc = float(sum(all_acc)/len(all_acc))
    print ('the current fooling rate is', fooling_rate)
    return fooling_rate, overal_acc, new_pred



def calculate_entropy(pred):
    h = 0
    all_pred = sum(pred)
    for i in range(num_classes):
        Pi = pred[i]/all_pred
        if Pi != 0:
            h -=  Pi* math.log(Pi)
    return h




new_pred = []
for i in range(num_classes):
    new_pred.append(0)
for k in idx_test:
    new_pred[int(torch.argmax(ori_output[k]))] += 1
entropy = calculate_entropy(new_pred)
print ('the entropy is', entropy)





def modify_adj(adj, mode, edge_num):
    num_ori = adj.shape[0]
    num_new = num_ori + num_fake
    if mode == 'no_connection':
        C = np.zeros((num_ori, num_fake))
        CT = np.zeros((num_fake, num_ori))
        B = np.zeros((num_fake, num_fake))
        ##################
        
    #     B = np.ones((num_fake, num_fake)) - np.eye(num_fake)
        ##################  
    elif mode == 'full_connection':
        #################### full edge
        C = np.ones((num_ori, num_fake))
        CT = np.ones((num_fake, num_ori))
        B = np.ones((num_fake, num_fake)) - np.eye(num_fake)
        ####################

    elif mode == 'random_connection':
        #################### random edge
        # C = np.random.randint(2, size = (num_ori, num_fake))
        # CT = C.transpose()
        # B = np.random.randint(2, size = (num_fake, num_fake))
        ####################

        #################### radom with same number of edge
        BC = np.random.binomial(1, edge_num/(num_fake * tmp_adj.shape[0]), (num_fake + num_ori, num_fake))
        C = BC[:-num_fake, ]
        CT = C.transpose()
        B = BC[-num_fake:, ]
        B = (B + B.transpose()) / 2
        B = np.where(B>0, 1, 0)
        np.fill_diagonal(B, np.float32(0))
        ####################
    adj = np.concatenate((adj, C), axis = 1)
    CTB = np.concatenate((CT, B), axis = 1)
    adj = np.concatenate((adj, CTB), axis = 0)
    return adj





#evaluate the universal attack
if args.evaluate_mode == "universal":
    fool_res = []
    new_acc = []
    p_times = []
    all_entropy = []
    
    for i in range(10):
        folder_path = op.join("./", "step{3}_new_adj/{0}/fake{1}_radius{2}".format(args.dataset, num_fake, args.radius, args.step))
#         folder_path = op.join("./", "sample_step{3}_new_adj/{0}/fake{1}_radius{2}_sample{4}".format(args.dataset, num_fake, args.radius, args.step, args.sample_percent))
        adj_path = op.join(folder_path, 'adj{}.npy'.format(i))
        feat_path = op.join(folder_path, 'feat{}.npy'.format(i))
        new_adj = np.load(adj_path)
        new_feat = np.load(feat_path)
        print ('C adj', np.sum(new_adj[-num_fake:, :-num_fake]))
        print ('B adj', np.sum(new_adj[-num_fake:, -num_fake:]))
        print ('total X feat', np.sum(new_feat[-num_fake:]))
        print ('X feat', np.sum(new_feat[-num_fake:], 1))
        new_feat = torch.from_numpy(new_feat).float()
        new_feat = new_feat.cuda()
        
        res, acc, new_pred = evaluate_attack(new_adj, new_feat)
#         print ('the prediction result is', new_pred)
#         entropy = calculate_entropy(new_pred)
        fool_res.append(res)      
        new_acc.append(acc)
#         p_times.append(len(list(pt)))
#         print ('the perturbation times is', p_times)
        print ('the fooling rates are', fool_res)
        print ('the new accuracy is', new_acc)
        print ('the average fooling rates over 10 times of test is', sum(fool_res)/float(len(fool_res)))
        print ('the average new accuracy over 10 times of test is', sum(new_acc)/float(len(new_acc)))
#         print ('the entropy is', entropy)
#         all_entropy.append(entropy)
#     print ('all the entropy values are', all_entropy)
#     print ('the average entropy is', sum(all_entropy)/float(len(all_entropy)))

elif args.evaluate_mode == "rand_feat":
    fool_res = []
    new_acc = []
    p_times = []
    all_entropy = []
    
    for i in range(10):
        folder_path = op.join("./", "step{3}_new_adj/{0}/fake{1}_radius{2}".format(args.dataset, num_fake, args.radius, args.step))
#         folder_path = op.join("./", "sample_step{3}_new_adj/{0}/fake{1}_radius{2}_sample{4}".format(args.dataset, num_fake, args.radius, args.step, args.sample_percent))
        adj_path = op.join(folder_path, 'adj{}.npy'.format(i))
#         feat_path = op.join(folder_path, 'feat{}.npy'.format(i))
        new_adj = np.load(adj_path)
#         new_feat = np.load(feat_path)
        print ('C adj', np.sum(new_adj[-num_fake:, :-num_fake]))
        print ('B adj', np.sum(new_adj[-num_fake:, -num_fake:]))
        
        

#         fake_feat = gaussian_dist(tmp_feat)
        fake_idx = np.random.choice(feat.shape[0], num_fake)
        fake_feat = feat[fake_idx]
#         fake_feat = np.random.randint(2, size = (num_fake, tmp_feat.shape[1])).astype(np.float32)
#         fake_feat, _ = normalize(fake_feat)
        new_feat = np.concatenate((feat, fake_feat), 0)
        print ('X feat', np.sum(new_feat[-num_fake:], 1))
        print ('total X feat', np.sum(new_feat[-num_fake:]))
        new_feat = torch.from_numpy(new_feat)
        
        new_feat = new_feat.cuda()
        
        res, acc, new_pred = evaluate_attack(new_adj, new_feat)
#         print ('the prediction result is', new_pred)
#         entropy = calculate_entropy(new_pred)
        fool_res.append(res)      
        new_acc.append(acc)
#         p_times.append(len(list(pt)))
#         print ('the perturbation times is', p_times)
        print ('the fooling rates are', fool_res)
        print ('the new accuracy is', new_acc)
        print ('the average fooling rates over 10 times of test is', sum(fool_res)/float(len(fool_res)))
        print ('the average new accuracy over 10 times of test is', sum(new_acc)/float(len(new_acc)))
        

elif args.evaluate_mode == "no_connection" or "random_connection" or "full_connection":
    fool_res = []
    new_acc = []
    p_times = []
    all_entropy = []
    
    for i in range(10):
        folder_path = op.join("./", "step{3}_new_adj/{0}/fake{1}_radius{2}".format(args.dataset, num_fake, args.radius, args.step))
#         folder_path = op.join("./", "new_adj_feat/{0}/fake{1}_radius{2}_{3}".format(args.dataset, num_fake, args.radius1, args.radius2))
        feat_path = op.join(folder_path, 'feat{}.npy'.format(i))
        adj_path = op.join(folder_path, 'adj{}.npy'.format(i))
        adja = np.load(adj_path)
        edge_num = np.sum(adja[-num_fake:])
        new_adj = modify_adj(tmp_adj, args.evaluate_mode, edge_num)

        new_feat = np.load(feat_path)
        print ('C adj', np.sum(new_adj[-num_fake:, :-num_fake]))
        print ('B adj', np.sum(new_adj[-num_fake:, -num_fake:]))
        print ('total X feat', np.sum(new_feat[-num_fake:]))
        print ('X feat', np.sum(new_feat[-num_fake:], 1))
        new_feat = torch.from_numpy(new_feat).float()
        new_feat = new_feat.cuda()
        
        res, acc, new_pred = evaluate_attack(new_adj, new_feat)
#         print ('the prediction result is', new_pred)
#         entropy = calculate_entropy(new_pred)
        fool_res.append(res)      
        new_acc.append(acc)
#         p_times.append(len(list(pt)))
#         print ('the perturbation times is', p_times)
        print ('the fooling rates are', fool_res)
        print ('the new accuracy is', new_acc)
        print ('the average fooling rates over 10 times of test is', sum(fool_res)/float(len(fool_res)))
        print ('the average new accuracy over 10 times of test is', sum(new_acc)/float(len(new_acc)))
#         print ('the entropy is', entropy)
#         all_entropy.append(entropy)
#     print ('all the entropy values are', all_entropy)
#     print ('the average entropy is', sum(all_entropy)/float(len(all_entropy)))

    




