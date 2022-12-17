import os
import copy
import numpy as np
import torch
import torch.nn.functional as F

import dgl
import argparse
from gat import GAT
from utils import evaluate, collate
from utils import get_data_loader, save_checkpoint, load_checkpoint
from utils import evaluate_model, test_model, generate_label
from auxilary_loss import gen_fit_loss, optimizing, gen_mi_loss, loss_fn_kd, gen_att_loss
from auxilary_model import collect_model
from auxilary_optimizer import block_optimizer
from plot_utils import loss_logger, parameters
import time
import matplotlib.pyplot as plt
import collections
import random

torch.set_num_threads(1)

def train_student(args, auxiliary_model, data, device):
    '''
    mode:
        teacher: KD
        full:   training student use full supervision
        mi:     LSP
        fit: FitNet
        att: Attention Transfer
    
    args: 
        auxiliary_model - dict
            {
                "model_name": {'model','optimizer','epoch_num'}
            }
    '''
    best_score = 0
    best_loss = 1000.0

    train_dataloader, valid_dataloader, test_dataloader, fixed_train_dataloader = data
    
    # multi class loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    loss_mse = torch.nn.MSELoss()

    t_model = auxiliary_model['t_model']['model']
    s_model = auxiliary_model['s_model']['model']
    
    losslogger = loss_logger()
    step_n = 0
    has_run = False
    # print('here 1')
    val_scores = []
    test_scores = []
    for epoch in range(args.s_epochs):
        s_model.train()
        loss_list = []
        additional_loss_list = []
        t0 = time.time()
        for batch, batch_data in enumerate( zip(train_dataloader,fixed_train_dataloader) ):
            # prepare data
            step_n += 1
            shuffle_data, fixed_data = batch_data
            subgraph, feats, labels = shuffle_data
            fixed_subgraph, fixed_feats, fixed_labels = fixed_data

            # move data to gpu
            subgraph = subgraph.to(device)
            feats = feats.to(device)
            labels = labels.to(device)
            fixed_feats = fixed_feats.to(device)
            fixed_labels = fixed_labels.to(device)

            # define graph of model 
            s_model.g = subgraph
            for layer in s_model.gat_layers:
                layer.g = subgraph
            
            logits, middle_feats_s = s_model(feats.float(), middle=True)
            
            if epoch >= args.tofull:
                args.mode = 'full'

            if args.mode == 'full':
                '''use the original labels'''
                additional_loss = torch.tensor(0)
            else:
                logits_t = generate_label(t_model, subgraph, feats, device)
            
            if args.mode=='full':
                ce_loss = loss_fcn(logits, labels.float())
            else:
                class_loss = loss_fn_kd(logits, logits_t)
                #ce_loss = torch.mean( class_loss )
                ce_loss = loss_fcn(logits, labels.float())
                class_loss_detach = class_loss.detach()

            if args.mode == 'teacher':
                additional_loss = torch.tensor(0).to(device)
            elif args.mode == 'mi':
                if epoch>args.warmup_epoch:
                    if not has_run:
                        #block_optimizer(args, auxiliary_model, "s_model", [args.lr*0.1,args.lr*0.2,args.lr*0.5,args.lr, args.lr])
                        has_run = True
                    args.loss_weight = 0
                    mi_loss = ( torch.tensor(0).to(device) if args.loss_weight==0 else
                                gen_mi_loss(auxiliary_model, middle_feats_s[args.target_layer], subgraph, feats, 
                                            fixed_subgraph, fixed_feats, device, class_loss_detach) )
                    
                    additional_loss = mi_loss * args.loss_weight
                else:
                    #ce_loss *= 0
                    mi_loss = gen_mi_loss(auxiliary_model, middle_feats_s[args.target_layer], subgraph, feats, 
                                            fixed_subgraph, fixed_feats, device, class_loss_detach)
                    additional_loss = mi_loss * args.loss_weight
            
            elif args.mode == 'att':
                if epoch > args.warmup_epoch:
                    if not has_run:
                        # block_optimizer(args, auxiliary_model, "s_model", [args.lr*0.1,args.lr*0.2,args.lr*0.5,args.lr, args.lr])
                        has_run = True
                    args.loss_weight = 0
                    att_loss = (torch.tensor(0).to(device) if args.loss_weight == 0 else
                               gen_att_loss(auxiliary_model, middle_feats_s[args.target_layer], subgraph, feats,
                                           device))

                    additional_loss = att_loss * 0.1
                else:
                    # ce_loss *= 0
                    att_loss = gen_att_loss(auxiliary_model, middle_feats_s[args.target_layer], subgraph, feats,
                                           device)
                    additional_loss = att_loss * 0.1

            elif args.mode == 'fit':
                if epoch > args.warmup_epoch:
                    if not has_run  :
                        # block_optimizer(args, auxiliary_model, "s_model", [args.lr*0.1,args.lr*0.2,args.lr*0.5,args.lr, args.lr])
                        has_run = True
                    args.loss_weight = 0
                    fit_loss = (torch.tensor(0).to(device) if args.loss_weight == 0 else
                               gen_fit_loss(auxiliary_model, middle_feats_s[args.target_layer], subgraph, feats,
                                           device))

                    additional_loss = fit_loss * args.loss_weight
                    ce_loss = 0 
                else:
                    # ce_loss *= 0
                    fit_loss = (torch.tensor(0).to(device) if args.loss_weight == 0 else
                                gen_fit_loss(auxiliary_model, middle_feats_s[args.target_layer], subgraph, feats,
                                             device))

                    additional_loss = fit_loss * args.loss_weight
                    ce_loss = 0

                    
            loss = ce_loss + additional_loss

            #optimizing(auxiliary_model, loss, ['s_model', 'local_model', 'local_model_s'])
            optimizing(auxiliary_model, loss, ['s_model'])
            loss_list.append(loss.item())
            additional_loss_list.append(additional_loss.item() if additional_loss!=0 else 0)

        loss_data = np.array(loss_list).mean()
        additional_loss_data = np.array(additional_loss_list).mean()
        print(f"Epoch {epoch:05d} | Loss: {loss_data:.4f} | Mi: {additional_loss_data:.4f} | Time: {time.time()-t0:.4f}s")
        if epoch % 10 == 0:
            score = evaluate_model(valid_dataloader, train_dataloader, device, s_model, loss_fcn)
            if score > best_score or loss_data < best_loss: # only evaluate on testset if the model does improve
                best_score = score
                best_loss = loss_data
                test_score = test_model(test_dataloader, s_model, device, loss_fcn)
            val_scores.append(score)
            test_scores.append(test_score)
    


def train_teacher(args, model, data, device):
    train_dataloader, valid_dataloader, test_dataloader, _ = data
    
    best_model = None
    best_val = 0
    # define loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.t_epochs):
        model.train()
        loss_list = []
        for batch, batch_data in enumerate(train_dataloader):
            subgraph, feats, labels = batch_data
            feats = feats.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.gat_layers:
                layer.g = subgraph
            logits = model(feats.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        print(f"Epoch {epoch + 1:05d} | Loss: {loss_data:.4f}")
        if epoch % 10 == 0:
            score_list = []
            val_loss_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, feats, labels = valid_data
                feats = feats.to(device)
                labels = labels.to(device)
                score, val_loss = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn)
                score_list.append(score)
                val_loss_list.append(val_loss)
            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            print(f"F1-Score on valset  :        {mean_score:.4f} ")
            if mean_score > best_val:
                best_model = copy.deepcopy(model)

            train_score_list = []
            for batch, train_data in enumerate(train_dataloader):
                subgraph, feats, labels = train_data
                feats = feats.to(device)
                labels = labels.to(device)
                train_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
            print(f"F1-Score on trainset:        {np.array(train_score_list).mean():.4f}")

    # model = best_model

    test_score_list = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, feats, labels = test_data
        feats = feats.to(device)
        labels = labels.to(device)
        test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
    print(f"F1-Score on testset:        {np.array(test_score_list).mean():.4f}")
    


def main(args):
    device = torch.device("cpu") if args.gpu<0 else torch.device("cuda:" + str(args.gpu - 1))
    data, data_info = get_data_loader(args)
    model_dict = collect_model(args, data_info)

    t_model = model_dict['t_model']['model']

    # load or train the teacher
    if os.path.isfile("./models/t_model.pt"):
        load_checkpoint(t_model, "./models/t_model.pt", device)
    else:
        print("############ train teacher #############")
        train_teacher(args, t_model, data, device)
        save_checkpoint(t_model, "./models/t_model.pt")
    


    # verify the teacher model
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    train_dataloader, _, test_dataloader, _ = data
    print(f"test acc of teacher:")
    test_model(test_dataloader, t_model, device, loss_fcn)
    print(f"train acc of teacher:")
    test_model(train_dataloader, t_model, device, loss_fcn)
    

    print("############ train student with teacher #############")
    train_student(args, model_dict, data, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")


    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")

    parser.add_argument("--t-epochs", type=int, default=60,
                        help="number of training epochs")
    parser.add_argument("--t-num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--t-num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--t-num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--t-num-hidden", type=int, default=256,
                        help="number of hidden units")

    parser.add_argument("--s-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--s-num-heads", type=int, default=2,
                        help="number of hidden attention heads")
    parser.add_argument("--s-num-out-heads", type=int, default=2,
                        help="number of output attention heads")
    parser.add_argument("--s-num-layers", type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument("--s-num-hidden", type=int, default=68,
                        help="number of hidden units")
    parser.add_argument("--target-layer", type=int, default=2,
                        help="the layer of student to learn")
    
    parser.add_argument("--mode", type=str, default='mi')
    parser.add_argument("--train-mode", type=str, default='together',
                        help="training mode: together, warmup")
    parser.add_argument("--warmup-epoch", type=int, default=600,
                        help="steps to warmup")
    
    parser.add_argument('--loss-weight', type=float, default=1.0,
                        help="weight coeff of additional loss")
    parser.add_argument('--seed', type=int, default=100,
                        help="seed")
    parser.add_argument('--tofull', type=int, default=30,
                        help="change mode to full after tofull epochs")

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main(args)
