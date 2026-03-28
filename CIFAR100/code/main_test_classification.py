#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 21:03:27 2021

@author: hasan
"""
import datetime
import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms 
import modules.utils_torchvision as utils
from modules.NirvanaLoss import center_loss_nirvana, get_l2_pred_nosubcenter, cosine_similarity, euc_cos
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
import modules.resnet2 as resnet
import warnings

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["OMP_NUM_THREADS"]=str(4)
warnings.filterwarnings("ignore")


def eval_dsc(model, centerloss, data_loader, device, args):
    model.eval(), centerloss.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if args.distributed:
        centers = centerloss.module.centers
    else:
        centers = centerloss.centers
    header = 'Test:'
    all_preds_l2,all_preds_l2_norm, all_labels, all_feats, all_dist_euc_cos = [], [], [], [], []
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 500, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            feats, _ = model(image)
            preds_l2 = get_l2_pred_nosubcenter(feats,centers, target)
            preds_l2_norm = cosine_similarity(feats,centers, target)
            dist_euc_cos = euc_cos(feats,centers, target)
            all_dist_euc_cos.append(dist_euc_cos.cpu())
            all_preds_l2.append(preds_l2.cpu())
            all_preds_l2_norm.append(preds_l2_norm.cpu())
            all_labels.append(target.cpu())
            all_feats.append(feats.cpu())

    preds_l2_norm = torch.cat(all_preds_l2_norm, 0)
    preds_l2 = torch.cat(all_preds_l2, 0)
    preds_euc_cos = torch.cat(all_dist_euc_cos, 0)
    labels = torch.cat(all_labels, 0)
    test_acc_l2 = (float((labels == preds_l2).sum())/len(labels))*100.0
    test_acc_l2_norm = (float((labels == preds_l2_norm).sum())/len(labels))*100.0
    test_acc_dist_cos = (float((labels == preds_euc_cos).sum())/len(labels))*100.0
    print('Test Acc@1_L2 %.3f'%test_acc_l2)
    print('Test Acc@1_COSINE %.3f'%test_acc_l2_norm)
    print('Test Acc@EUC_COS %.3f'%test_acc_dist_cos)
    return test_acc_l2
    

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.Seed)  
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010]) 

    data_loader_val = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=100, shuffle=False,
        num_workers=4, pin_memory=True)
        
    args.num_classes = 100
    print("Creating model")
    # model = models.resnet50(pretrained=True)
    model = resnet.ResNet18(num_classes=args.num_classes)
    model.to(device)
    args.feat_dim = model.linear.weight.shape[1]
    print(args)
    centerloss = center_loss_nirvana(args.num_classes, args.feat_dim, True, device, Expand=args.Expand)
    centerloss.to(device)

    # Load Model weights
    checkpoint = torch.load(args.path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    centerloss.load_state_dict(checkpoint['centerloss'])
    val_acc1 = eval_dsc(model,centerloss, data_loader_val, device, args=args)
    print('%.3f Data Val.'%(val_acc1))
    

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--num_classes', default=200, type=int, metavar='N',
                        help='number of classes')
    parser.add_argument('--Expand', default=50, type=int, metavar='N',
                        help='Expand factor of centers')
    parser.add_argument('--Seed', default=0, type=int, metavar='N',
                        help='Seed')
    parser.add_argument('--feat_dim', default=512, type=int, metavar='N',
                        help='feature dimension of the model')
    parser.add_argument('--path', default='/home/mlcv/CevikalpPy/deep_simplex_classifier/models/lr0.100000_nirvana_Expand256_Epoch400_Seed0_Wd0/best_checkpoint.pth', help='additional note to output folder')

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

