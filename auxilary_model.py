import torch
from utils import get_teacher, get_student, get_feat_info
from discriminator_model import get_discriminator
from atransfor_model import get_gcn_transformer, get_transformer_model
from local_structure import get_local_model, get_upsampling_model

    
def collect_model(args, data_info):
    device = torch.device("cpu") if args.gpu<0 else torch.device("cuda:" + str(args.gpu))
    
    feat_info = get_feat_info(args)
    
    t_model = get_teacher(args, data_info)
    # t_model = t_model.to(device)
    t_model.to(device)
    s_model = get_student(args, data_info)                         
    # s_model = s_model.to(device)
    s_model.to(device)
    #d_model = get_discriminator(args, feat_info);                   d_model.to(device)
    #gcn_transformer_model = get_gcn_transformer(args, feat_info);   gcn_transformer_model.to(device)
    #transformer_model = get_transformer_model(args, feat_info);     transformer_model.to(device)
    local_model = get_local_model(feat_info);                       local_model.to(device)
    local_model_s = get_local_model(feat_info, upsampling=True);                     local_model_s.to(device)
    upsampling_model = get_upsampling_model(feat_info);             upsampling_model.to(device)

    # construct optimizers
    s_model_optimizer = torch.optim.Adam(s_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t_model_optimizer = torch.optim.Adam(t_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    local_model_optimizer = None #torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    local_model_s_optimizer = None #torch.optim.Adam(local_model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    upsampling_model_optimizer = torch.optim.Adam(upsampling_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    #transformer_model_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #gcn_transformer_model_optimizer = torch.optim.Adam(gcn_transformer_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # construct model dict
    model_dict = {}
    model_dict['s_model'] = {'model':s_model, 'optimizer':s_model_optimizer}
    model_dict['local_model'] = {'model':local_model, 'optimizer':local_model_optimizer}
    model_dict['local_model_s'] = {'model':local_model_s, 'optimizer':local_model_s_optimizer}
    model_dict['t_model'] = {'model':t_model, 'optimizer':t_model_optimizer}
    model_dict['upsampling_model'] = {'model':upsampling_model, 'optimizer': upsampling_model_optimizer}
    return model_dict