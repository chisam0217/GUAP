#!/usr/bin/env python
# coding: utf-8



from __future__ import division
from __future__ import print_function
import os
import time
import argparse
import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, normalize, load_polblogs_data
from models import GCN
from torch.autograd.gradcheck import zero_gradients
import os.path as op

os.environ["CUDA_VISIBLE_DEVICES"]="1" 




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





num_fake = int(tmp_adj.shape[0] * args.fake_rate)
global new_feat
global new_adj
# args.radius = int(np.sum(tmp_adj)/tmp_adj.shape[0]) 





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
    loss_train.backward()

    optimizer.step()


    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))





def test(feat, adj_m):
    model.eval()
    output = model(feat, adj_m)
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
ori_output = test(features, adj)
correct_res = ori_output[idx_train, labels[idx_train]] #the prediction probability of all train nodes




def add_fake_node(adj, innormal_features, features, file_path):
    #modify the adjencecy matrix
    num_ori = adj.shape[0]
    num_new = num_ori + num_fake
    C = np.zeros((num_ori, num_fake))
    CT = np.zeros((num_fake, num_ori))
    B = np.zeros((num_fake, num_fake))
    ##################
#     B = np.ones((num_fake, num_fake)) - np.eye(num_fake)
    ##################
    adj = np.concatenate((adj, C), axis = 1)
    CTB = np.concatenate((CT, B), axis = 1)
    adj = np.concatenate((adj, CTB), axis = 0)
    
    #add the node features
#     sel_idx = torch.randint(0, num_ori, (num_fake,))
#     feat_fake = features[sel_idx]
    feat_fake = gaussian_dist(innormal_features)
    
    features = np.concatenate((features, feat_fake), 0)
    np.save(file_path, features)
    features = torch.from_numpy(features)
    return adj, features




def gaussian_dist(innormal_features):
    # while True: #the generated node feature shouldnt be all 0
    feat_mean = np.mean(innormal_features, axis = 0)
    feat_std = np.std(innormal_features, axis = 0)
    feat_fake = np.zeros((num_fake, innormal_features.shape[1]))
    for i in range(innormal_features.shape[1]):
        feat_fake[:,i] = np.random.normal(feat_mean[0, i], feat_std[0, i], num_fake).reshape(feat_fake[:,i].shape)
#         print (i, feat_fake[:,i])
    feat_fake = np.where(feat_fake > 0.5, 1, 0).astype(np.float32)
    feat_fake, _ = normalize(feat_fake)
        # if np.sum(feat_fake) >= num_fake:
        #     break
    return feat_fake




# for i in range(10):    
#     folder_path = op.join("./", "version4_new_adj/{0}/fake{1}_radius{2}".format(args.dataset, num_fake, args.radius))
#     adj_path = op.join(folder_path, 'adj{}.npy'.format(i))
#     feat_path = op.join(folder_path, 'feat{}.npy'.format(i))
#     new_adj, new_feat = add_fake_node(tmp_adj, tmp_feat, feat, feat_path)
#     print (torch.sum(new_feat[-num_fake:], 1))




def add_perturb(input_adj, idx, perturb):
    # (1-x)A + x(1-A)
#     input_adj = input_adj.toarray()
    x = np.zeros((input_adj.shape[0], input_adj.shape[1]))
    x[idx] = perturb  
    x[:,idx] = perturb
#     print ('x', x[idx])
#     x += np.transpose(x) #change the idx'th row and column
    x1 = np.ones((input_adj.shape[0], input_adj.shape[1])) - x
#     print ('x1', x1[idx])
    adj2 = np.ones((input_adj.shape[0], input_adj.shape[1])) - input_adj
#     print ('adj2', adj2[idx])

    for i in range(input_adj.shape[0]):   
        adj2[i][i] = 0

    perturbed_adj = np.multiply(x1, input_adj) + np.multiply(x, adj2)
    return perturbed_adj




def modify_adj(input_adj, perturb, idx):
    input_adj = np.add(input_adj,perturb, casting ="unsafe")
    input_adj[idx, -num_fake:] = 1 - input_adj[idx, -num_fake:]
    input_adj[-num_fake:, idx] = input_adj[idx, -num_fake:]
    for i in range(-num_fake, 0):
        input_adj[i,i] = 0
        input_adj[:, i] = proj_lp(input_adj[:, i])
        input_adj[i] = input_adj[:, i]
        
    input_adj = np.clip(input_adj, 0, 1)
    return input_adj




def proj_lp(v, xi=args.radius, p=2):
# def proj_lp(v, xi=8, p=2):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
#     print ('the distance of v', np.linalg.norm(v.flatten(1)))
    
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        v = v
        #################
    v = np.clip(v, 0, 1)
        ########################
#     v = np.where(v<0.1, 0, v)
    #to reduce the number of nonzero elements which means 
    #the times of perturbation, also prevents saddle point

#     v = np.where(v>0.5, 1, 0)
    return v




def universal_attack(attack_epoch, max_epoch, file_path):
    model.eval()
    # delta = 0.02
    fooling_rate = 0.0
    overshoot = 0.02
    max_iter_df = 10
    
#     new_feat = torch.from_numpy(new_feat.astype(np.float32))
#     new_feat = new_feat.cuda()
    
    v1 = np.zeros(tmp_adj.shape[0]).astype(np.float32)
    v2 = np.ones(num_fake).astype(np.float32)
    v = np.concatenate((v1, v2))
    # stdv = 1./math.sqrt(tmp_adj.shape[0])
    # v = np.random.uniform(-stdv, stdv, tmp_adj.shape[0])
    cur_foolingrate = 0.0
    epoch = 0
    early_stop = 0
    results = []
    

    tmp_new_adj = np.copy(new_adj)
    print ('the new adj', np.sum(tmp_new_adj[-num_fake:, :-num_fake]))
    while epoch < max_epoch:

        epoch += 1
        train_idx = idx_train.cpu().numpy()
        np.random.shuffle(train_idx)
        
        ###############################################
        attack_time = time.time()
        for k in train_idx:
            #add v to see if the attack succeeds
            innormal_x_p = add_perturb(tmp_new_adj, k, v)
            ##################whether to use filtering
    #         innormal_x_p = np.where(innormal_x_p<0.5, 0, 1)
            x_p, degree_p = normalize(innormal_x_p + np.eye(tmp_new_adj.shape[0])) #A' = A + I
            x_p = torch.from_numpy(x_p.astype(np.float32))
            x_p = x_p.cuda()
            output = model(new_feat, x_p)

            if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
                dr, iter = IGP(innormal_x_p, x_p, k, num_classes, degree_p)
                if iter < max_iter_df-1:
                    tmp_new_adj = modify_adj(tmp_new_adj, dr, k)
                else:
                    print ('cant attack this node')
            else:
                print ('attack succeeds')
            print ('the new adj', np.sum(tmp_new_adj[-num_fake:, :-num_fake]))
        print ('the IGP time cost is', time.time()-attack_time)
        

        res = []
#         v = np.where(v>0.5, 1, 0)
        tmp_new_adj = np.where(tmp_new_adj>0.5, 1, 0)
        print ('C adjacency matrix', np.sum(tmp_new_adj[-num_fake:, :-num_fake]))
        print ('B adjacency matrix', np.sum(tmp_new_adj[-num_fake:, -num_fake:]))
        for k in train_idx:
            print ('test node', k)
            innormal_x_p = add_perturb(tmp_new_adj, k, v)            
#             innormal_x_p = np.where(innormal_x_p<0.5, 0, 1)
            
            x_p, degree_p = normalize(innormal_x_p + np.eye(tmp_new_adj.shape[0]))
            x_p = torch.from_numpy(x_p.astype(np.float32))
            x_p = x_p.cuda()
            output = model(new_feat, x_p)
            if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
                res.append(0)
            else:
                res.append(1)
        fooling_rate = float(sum(res)/len(res))
        print ('the current train fooling rates are', fooling_rate)
#         test_res = []
#         print ('testing')
#         test_idx = idx_test.cpu().numpy()
#         for k in test_idx:
#             print ('test node', k)
#             innormal_x_p = add_perturb(tmp_new_adj, k, v)            
# #             innormal_x_p = np.where(innormal_x_p<0.5, 0, 1)
            
#             x_p, degree_p = normalize(innormal_x_p + np.eye(tmp_new_adj.shape[0]))
#             x_p = torch.from_numpy(x_p.astype(np.float32))
#             x_p = x_p.cuda()
#             output = model(new_feat, x_p)
#             if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
#                 test_res.append(0)
#             else:
#                 test_res.append(1)
#         test_fooling_rate = float(sum(test_res)/len(test_res))
#         print ('the current test fooling rates are', test_fooling_rate)

        if fooling_rate > cur_foolingrate:
            
            cur_foolingrate = fooling_rate
            np.save(file_path, tmp_new_adj)
            
        results.append(fooling_rate)

        
    return cur_foolingrate
        




def calculate_grad_class(pert_adj, idx, classes):
    x = Variable(pert_adj, requires_grad=True)
    output = model(new_feat, x)
    grad = []
#     for i in range(classes):
    for i in classes:
        cls = torch.LongTensor(np.array(i).reshape(1)).cuda()
        loss = F.nll_loss(output[idx:idx+1], cls) 
        loss.backward(retain_graph=True)
        grad.append(x.grad[idx].cpu().numpy())
#     print ('grad', grad)
    return np.array(grad)   





def calculate_grad(pert_adj, idx): ######exclude idx?
    x = Variable(pert_adj, requires_grad=True)
    output = model(new_feat, x)
    ex_idx_train = train_idx.numpy()
    ex_idx_train = np.delete(ex_idx_train, np.where(ex_idx_train == idx))
    ex_idx_train = torch.LongTensor(ex_idx_train).cuda()
    loss_train = F.nll_loss(output[ex_idx_train], labels[ex_idx_train])
    loss_train.backward(retain_graph=True)
    gradient = np.array(x.grad.cpu().numpy())
    gradient[idx] = 0
    gradient[:,idx] = 0
    gradient[:-num_fake,:-num_fake] = 0
    ###############
#     gradient[-num_fake:, -num_fake:] = 0
    ###############

    np.fill_diagonal(gradient, np.float32(0)) #let the diagonal of the adjancecy matrix always be 0
#     print ('type', type(gradient))
    gradient = (gradient + gradient.transpose())/2
    return gradient




def normalize_add_perturb(ori_adj, pert, single_node, idx, rate):
    if single_node:
        a = ori_adj
        a[idx] += pert[idx] * rate
        a[:,idx] += pert[:,idx] * rate
    else:
        pert[idx] = pert[idx] * rate
        pert[:, idx] = pert[:, idx] * rate
        a = ori_adj + pert
    inv_d = 1 + np.sum(pert, 1)
    inv_d = 1.0/inv_d
    ## filter the perturbed matrix so that >= 0 
#     a = np.where(a<0, 0, a)
    ori_adj = np.multiply(a.transpose(), inv_d).transpose()
    
    return ori_adj




def IGP(innormal_adj, ori_adj, idx, num_classes, degree, overshoot=0.02, max_iter=30):
    #innormal_adj: the perturbed adjacency matrix not normalized
    #ori_adj: the normalized perturbed adjacency matrix 
    model.eval()
    
#     new_feat = torch.from_numpy(new_feat.astype(np.float32))
#     new_feat = new_feat.cuda()
    pred = model(new_feat, ori_adj)[idx]
    pred = pred.detach().cpu().numpy()
    
#     step = 1
    I = pred.argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]    
    f_i = np.array(pred).flatten()
    k_i = int(np.argmax(f_i))  
    w = np.zeros(ori_adj.shape[0])
    r_tot = np.zeros((ori_adj.size(0), ori_adj.size(0)))
    
#     pert_adj = ori_adj
    pert_adj = ori_adj.detach().cpu().numpy()
    pert_adj_tensor = ori_adj
#     degree_idx = degree
    loop_i = 0
#     print ('the correct class', label)
    while k_i == label and loop_i < max_iter:
        pert = np.inf
#         gradients = calculate_grad(pert_adj_tensor, idx, num_classes)
        gradients = calculate_grad_class(pert_adj_tensor, idx, I)
#         r_tot_tmp = np.copy(r_tot)
        
        for i in range(1, num_classes):
            # set new w_k and new f_k
            w_k = gradients[i, :] - gradients[0, :]
            f_k = f_i[I[i]] - f_i[I[0]]
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
            if pert_k < pert:
                pert = pert_k
                w = w_k
            r_i =  pert * w / np.linalg.norm(w)
            r_tot[idx, -num_fake:] = r_tot[idx, -num_fake:] + r_i[-num_fake:]
            r_tot[-num_fake:, idx] = r_tot[idx, -num_fake:]
            
        pert_adj_k = normalize_add_perturb(pert_adj, r_tot, True, idx, (1+overshoot))
            
        pert_adj_k = np.clip(pert_adj_k, 0, 1)
        pert_adj_k = torch.from_numpy(pert_adj_k.astype(np.float32))
        pert_adj_k = pert_adj_k.cuda()
        grad = calculate_grad(pert_adj_k, idx)
#         print ('C grad', np.sum(grad[-num_fake:]))
#         print ('B grad', np.sum(grad[-num_fake:, -num_fake:]))
#         r_tot -= np.sign(grad) * step
        print ('grad', np.sum(grad))
        r_tot -= grad*args.step
        pert_adj = normalize_add_perturb(pert_adj, r_tot, False, idx, (1+overshoot))
        pert_adj = np.clip(pert_adj, 0, 1)
        pert_adj_tensor = torch.from_numpy(pert_adj.astype(np.float32))
        pert_adj_tensor = pert_adj_tensor.cuda()
        f_i = np.array(model(new_feat, pert_adj_tensor)[idx].detach().cpu().numpy()).flatten()
        k_i = int(np.argmax(f_i))
        
        loop_i += 1
        if k_i != label:
            print ('attack succeeds')
    r_tot[:, -num_fake:] = np.multiply(r_tot[:, -num_fake:].transpose(), degree).transpose()
    r_tot[-num_fake:, :-num_fake] = np.multiply(r_tot[-num_fake:, :-num_fake], degree[:-num_fake])
#########
#     r_tot = np.multiply(r_tot.transpose(), degree).transpose()
    
    
    r_tot[idx:] = r_tot[idx:] * (1 + overshoot)
    r_tot[:,idx] = r_tot[:,idx] * (1 + overshoot)

    return r_tot, loop_i




def convert_to_v(adj, pert_m, deg, idx):

    a = np.multiply(pert_m, deg)
    inv_m = np.ones(adj.shape[0]) - np.multiply(adj[idx], 2) 
    inv_m = np.power(inv_m, -1)
    res = np.multiply(a, inv_m)  
    return res
    



train_foolrate = []

for i in range(0,10):
    global new_adj
    global new_feat
    
    folder_path = op.join("./", "step{3}_new_adj/{0}/fake{1}_radius{2}".format(args.dataset, num_fake, args.radius, args.step))
    if not op.exists(folder_path):
        os.mkdir(folder_path)
    adj_path = op.join(folder_path, 'adj{}.npy'.format(i))
    feat_path = op.join(folder_path, 'feat{}.npy'.format(i))
    new_adj, new_feat = add_fake_node(tmp_adj, tmp_feat, feat, feat_path)

    new_feat = new_feat.cuda()
    # fool_rate = universal_attack(i, 50, adj_path)
    fool_rate = universal_attack(i, 50, adj_path)
    train_foolrate.append(fool_rate)

print ('the final train fool rate', train_foolrate)





