## contains supporting functions 

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import f1_score
import dgl
from dgl.data.ppi import LegacyPPIDataset as PPIDataset
# from dgl.data import PPIDataset
from gat import GAT, GCN

# generic func to evalate model, return score & loss 
def evaluate(feats, model, subgraph, labels, loss_fcn):
    model.eval() # turn on evaluation mode 
    with torch.no_grad():
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(),
                         predict, average='micro')
    model.train()   # turn on train mode 
    
    return score, loss_data.item()

# evaluate a model on specific test set 
def test_model(test_dataloader, model, device, loss_fcn):
    test_score_list = []
    model.eval()
    with torch.no_grad():
        for batch, test_data in enumerate(test_dataloader):
            subgraph, feats, labels = test_data
            subgraph = subgraph.to(device)
            feats = feats.to(device) # device (maybe gpu) will load inputs
            labels = labels.to(device)
            test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
        mean_score = np.array(test_score_list).mean()
        print(f"F1-Score on testset:        {mean_score:.4f}")
    model.train()
    return mean_score

# get prediction
def generate_label(t_model, subgraph, feats, device):
    '''generate pseudo lables given a teacher model
    '''
    # t_model.to(device)
    t_model.eval()
    with torch.no_grad():
        t_model.g = subgraph
        for layer in t_model.gat_layers:
            layer.g = subgraph
        # soft labels
        logits_t = t_model(feats.float())
        #pseudo_labels = torch.where(t_logits>0.5, 
        #                            torch.ones(t_logits.shape).to(device), 
        #                            torch.zeros(t_logits.shape).to(device))
        #labels = logits_t
    return logits_t.detach()
    
# evaluate model based on train & valid test 
def evaluate_model(valid_dataloader, train_dataloader, device, s_model, loss_fcn):
    score_list = []
    val_loss_list = []
    s_model.eval()
    with torch.no_grad():
        for batch, valid_data in enumerate(valid_dataloader):
            subgraph, feats, labels = valid_data
            subgraph = subgraph.to(device)
            feats = feats.to(device)
            labels = labels.to(device)
            score, val_loss = evaluate(feats.float(), s_model, subgraph, labels.float(), loss_fcn)
            score_list.append(score)
            val_loss_list.append(val_loss)
    mean_score = np.array(score_list).mean()
    mean_val_loss = np.array(val_loss_list).mean()
    print(f"F1-Score on valset  :        {mean_score:.4f} ")
    s_model.train()
    return mean_score

    """
    train_score_list = []
    for batch, train_data in enumerate(train_dataloader):
        subgraph, feats, labels = train_data
        feats = feats.to(device)
        labels = labels.to(device)
        train_score_list.append(evaluate(feats, s_model, subgraph, labels.float(), loss_fcn)[0])
    print(f"F1-Score on trainset:        {np.array(train_score_list).mean():.4f}")
    """

def collate(sample):
    # print('type of sample: ', type(sample))
    # print("len of sample: ", len(sample))
    graphs, feats, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels

def collate_w_gk(sample):
    '''
    collate with graph_khop
    '''
    graphs, feats, labels, graphs_gk =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    graph_gk = dgl.batch(graphs_gk)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels, graph_gk


def get_teacher(args, data_info):
    '''args holds the common arguments
    data_info holds some special arugments
    '''
    heads = ([args.t_num_heads] * args.t_num_layers) + [args.t_num_out_heads]
    model = GAT(data_info['g'],
            args.t_num_layers,
            data_info['num_feats'],
            args.t_num_hidden,
            data_info['n_classes'],
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.alpha,
            args.residual)
    return model
    
def get_student(args, data_info):
    '''args holds the common arguments
    data_info holds some special arugments
    '''
    heads = ([args.s_num_heads] * args.s_num_layers) + [args.s_num_out_heads]
    model = GAT(data_info['g'],
            args.s_num_layers,
            data_info['num_feats'],
            args.s_num_hidden,
            data_info['n_classes'],
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.alpha,
            args.residual)
    return model

def get_feat_info(args):
    feat_info = {}
    feat_info['s_feat'] = [args.s_num_heads*args.s_num_hidden] * args.s_num_layers
    feat_info['t_feat'] = [args.t_num_heads*args.t_num_hidden] * args.t_num_layers
    #assert len(feat_info['s_feat']) == len(feat_info['t_feat']),"number of hidden layer for teacher and student are not equal"
    return feat_info


def get_data_loader(args):
    '''create the dataset
    return 
        three dataloaders and data_info
    '''
    train_dataset = PPIDataset(mode='train')
    # print('len of an element in train_dataset', len(train_dataset[15]))
    valid_dataset = PPIDataset(mode='valid')
    # print('len of an element in valid_dataset', len(valid_dataset[1]))
    test_dataset = PPIDataset(mode='test')
    # print('len of an element in test_dataset', len(test_dataset[1]))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4, shuffle=True)
    fixed_train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)

    n_classes = train_dataset.labels.shape[1]
    num_feats = train_dataset.features.shape[1]
    g = train_dataset.graph
    data_info = {}
    data_info['n_classes'] = n_classes
    data_info['num_feats'] = num_feats
    data_info['g'] = g
    return (train_dataloader, valid_dataloader, test_dataloader, fixed_train_dataloader), data_info


def save_checkpoint(model, path):
    '''Saves model
    '''
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), path)
    print(f"save model to {path}")

def load_checkpoint(model, path, device):
    '''load model
    '''
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Load model from {path}")

