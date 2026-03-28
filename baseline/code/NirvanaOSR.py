import os
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torchvision.transforms as tf
import numpy as np

from modules.dchs import  NirvanaOpenset_loss
from Networks.models import classifier32
from Networks.resnet import resnet50, resnet18, resnet34, resnet101, resnet152
from datasets.osr_dataloader import Random300K_Images, BloodMNIST_OSR, OCTMnist_OSR, DermaMNIST_OSR, TissueMNIST_OSR, ASC_OSR
from utils import Logger, save_networks, load_networks
from core import test_ddfm_b9, train_Nirvana_oe, train_Nirvana_oe_reg
from split import splits_2020 as splits

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='bloodmnist', choices=['bloodmnist', 'octmnist', 'dermamnist', 'tissuemnist', 'asc'], help="Dataset selection")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./logs_results', help='Directory to save results')
# Optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--l1-weight', type=float, default=0.0, help='L1 regularization weight')
parser.add_argument('--l2-weight', type=float, default=1e-4, help='L2 regularization weight (weight decay)')
parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'rmsprop'], help='Optimizer type to use (sgd or rmsprop)')
# model
parser.add_argument('--noisy-ratio', type=float, default=0.0, help="noisy ratio for ablation study")
parser.add_argument('--margin', type=float, default=48.0, help="margin for hinge")
parser.add_argument('--Expand', default=500, type=int, metavar='N', help='Expand factor of centers')
parser.add_argument('--uncertainty-weight', type=float, default=5.0, help='Weight for uncertainty regularization loss component')
parser.add_argument('--outlier-weight', type=float, default=1.0, help='Weight for outlier triplet loss component')
parser.add_argument('--model', type=str, default='classifier32', help='resnet50, classifier32, resnet18, resnet34, resnet101, resnet152')
parser.add_argument('--loss', type=str, default='NirvanaOpenset')
parser.add_argument('--pretrained-model', type=str, default=None,help='Path to your fine-tuned model')
# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log_random30k_noisy_rampfalse')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--oe', action=argparse.BooleanOptionalAction, help="Outlier Exposure", default=True)
parser.add_argument('--oe-path', type=str, default=None, help='Path to 300K random images .npy for outlier exposure')
parser.add_argument('--split-idx', type=int, default=None, help='Run only a single split index (default: run all splits)')
parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True, help='Resume from per-split checkpoint if available')

def _auto_device(use_cpu: bool) -> torch.device:
    if use_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def _empty_cache(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.empty_cache()

def main_worker(options):
    is_best_acc_avg = False
    best_acc_avg = 0.
    results_best = dict()
    best_acc_avg_b9 = 0.
    results_b9_best = dict()
    options['ramp_activate'] = False
    torch.manual_seed(options['seed'])
    np.random.seed(options['seed'])
    random.seed(options['seed'])

    device: torch.device = options['device']
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_cuda = device.type == 'cuda'
    options['use_gpu'] = use_cuda

    if device.type == 'cuda':
        print(f"Using device: cuda (CUDA_VISIBLE_DEVICES={options['gpu']})")
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print(f"Using device: {device.type}")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    data_num_workers = 8 if use_cuda else 0
    print(f"Primary dataset loaders will use num_workers={data_num_workers}")
    if options['dataset'] == 'bloodmnist':
        Data = BloodMNIST_OSR(
            known=options['known'],
            dataroot=options['dataroot'],
            use_gpu=use_cuda,
            num_workers=data_num_workers,
            batch_size=options['batch_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader 
        outloader = Data.out_loader

    elif options['dataset'] == 'octmnist':
        split_dict = splits[options['dataset']][options['item']]
        known = split_dict['known']
        unknown = split_dict['unknown']
        Data = OCTMnist_OSR(
            known=known,
            unknown=unknown,
            dataroot=options['dataroot'],
            use_gpu=use_cuda,
            num_workers=data_num_workers,
            batch_size=options['batch_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader
        outloader = Data.out_loader

    elif options['dataset'] == 'dermamnist':
        split_dict = splits[options['dataset']][options['item']]
        known = split_dict['known']
        unknown = split_dict['unknown']
        Data = DermaMNIST_OSR(
            known=known,
            unknown=unknown,
            dataroot=options['dataroot'],
            use_gpu=use_cuda,
            num_workers=data_num_workers,
            batch_size=options['batch_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader
        outloader = Data.out_loader

    elif options['dataset'] == 'tissuemnist':
        split_dict = splits[options['dataset']][options['item']]
        known = split_dict['known']
        unknown = split_dict['unknown']
        Data = TissueMNIST_OSR(
            known=known,
            unknown=unknown,
            dataroot=options['dataroot'],
            use_gpu=use_cuda,
            num_workers=data_num_workers,
            batch_size=options['batch_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader
        outloader = Data.out_loader    

    elif options['dataset'] == 'asc':
        split_dict = splits[options['dataset']][options['item']]
        known = split_dict['known']
        unknown = split_dict['unknown']
        Data = ASC_OSR(
            known=known,
            unknown=unknown,
            dataroot=options['dataroot'],
            use_gpu=use_cuda,
            num_workers=data_num_workers,
            batch_size=options['batch_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader
        outloader = Data.out_loader
        
        options['img_size'] = 224

    else:
        print('No dataset chosen.')
    
    trainloader_oe = None
    if options['oe']:
        print("Outlier exposure mode is on.")
        background_path = options.get('oe_path')
        if not background_path:
            background_path = os.path.join(
                os.path.dirname(options['dataroot']),
                '300K_random_images',
                '300K_random_images.npy',
            )

        if not os.path.exists(background_path):
            raise FileNotFoundError(
                f"Outlier Exposure is enabled but background .npy was not found at: {background_path}\n"
                f"Either download it, or run with --no-oe."
            )

        if options['dataset'] in ['asc']:
            oe_transform = tf.Compose([
                tf.Resize((224,224)),
                tf.RandomCrop(224, padding=20),
                tf.RandomHorizontalFlip(),
                tf.ToTensor()
            ])
        else:
            oe_transform = tf.Compose([
                tf.RandomCrop(32, padding=4),
                tf.RandomHorizontalFlip(),
                tf.ToTensor()
            ])

        oe_data = Random300K_Images(
            file_path=background_path,
            transform=oe_transform,
            extendable=options['noisy_ratio']
        )
        print(f"Loaded background dataset with {len(oe_data)} images")

        oe_num_workers = 8 if use_cuda else 0
        oe_loader_kwargs = dict(
            batch_size=options['batch_size'],
            shuffle=True,
            num_workers=oe_num_workers,
            drop_last=True,
            pin_memory=use_cuda,
        )
        if oe_num_workers > 0:
            oe_loader_kwargs['persistent_workers'] = True
        trainloader_oe = torch.utils.data.DataLoader(oe_data, **oe_loader_kwargs)
        print(f"Background loader created with {len(trainloader_oe)} batches (num_workers={oe_num_workers})")

    if (options['noisy_ratio']):
        oe_data.data = oe_data.data[:30000]
        oe_data.data.extend(list(Data.noisy_data))
        print("#of background images {}".format(len(oe_data)))
        options['ramp_activate'] = True
    
    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format(options['model']))
    if options['model'] == 'classifier32':
        net = classifier32(num_classes=options['num_classes'])
        feat_dim = 128
    elif options['model'] == 'resnet50':
        net = resnet50(pretrained=True,num_classes=options['num_classes'])
        feat_dim = net.feat_dim
    elif options['model'] == 'resnet18':
        net = resnet18(pretrained=True,num_classes=options['num_classes'])
        feat_dim = net.feat_dim
    elif options['model'] == 'resnet34':
        net = resnet34(pretrained=True, num_classes=options['num_classes'])
        feat_dim = net.feat_dim
    elif options['model'] == 'resnet101':
        net = resnet101(pretrained=True,num_classes=options['num_classes'])
        feat_dim = net.feat_dim
    elif options['model'] == 'resnet152':
        net = resnet152(pretrained=True,num_classes=options['num_classes'])
        feat_dim = net.feat_dim
    else:
        raise ValueError('Model not supported in this file.')
        
    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_cuda
        }
    )

    criterion = NirvanaOpenset_loss(
            num_classes=options['num_classes'], 
            feat_dim=options['feat_dim'], 
            precalc_centers=True, 
            margin=options['margin'], 
            Expand=options['Expand'],
            uncertainty_weight=options['uncertainty_weight'],
            outlier_weight=options['outlier_weight']
        )
        
    
    net = net.to(device)
    criterion = criterion.to(device)

    dir_name = '{}_{}_{}_{}'.format(options['model'], options['loss'],options['margin'],options['oe'])
    model_path = os.path.join(options['outf'], 'models', options['dataset'],dir_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    file_name = '{}_{}_{}_{}_{}'.format(options['model'], options['loss'], options['item'], options['margin'],options['noisy_ratio'])
    resume_checkpoint_path = os.path.join(model_path, f'{file_name}_resume.pth')

    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test_ddfm_b9(net, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
        return results

    #optimizer
    l2_weight = options['l2_weight']
    if options['optim'] == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), 
                                lr=options['lr'], 
                                momentum=0.9, 
                                weight_decay=l2_weight)
        print(f"Using SGD optimizer with lr={options['lr']}, momentum=0.9, weight_decay={l2_weight}")
    elif options['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(net.parameters(), 
                                    lr=options['lr'],
                                    alpha=0.95, 
                                    eps=1e-6,
                                    weight_decay=l2_weight, 
                                    momentum=0.9,
                                    centered=False)
        print(f"Using RMSprop optimizer with lr={options['lr']}, alpha=0.95, eps=1e-6, weight_decay={l2_weight}, momentum=0.9")
    else:
        # Fallback to default SGD
        optimizer = torch.optim.SGD(net.parameters(), 
                                lr=options['lr'], 
                                momentum=0.9, 
                                weight_decay=l2_weight)
        print(f"Unknown optimizer '{options['optim']}', falling back to SGD")

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=options['max_epoch']*len(trainloader))
    start_epoch = 0
    if options['resume'] and os.path.exists(resume_checkpoint_path):
        # Resume checkpoints are produced locally by this training script and
        # include optimizer/scheduler state, so they must be loaded as a full
        # pickle under PyTorch 2.6+.
        checkpoint = torch.load(
            resume_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        net.load_state_dict(checkpoint['net'])
        if 'criterion' in checkpoint:
            criterion.load_state_dict(checkpoint['criterion'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        best_acc_avg = float(checkpoint.get('best_acc_avg', best_acc_avg))
        best_acc_avg_b9 = float(checkpoint.get('best_acc_avg_b9', best_acc_avg_b9))
        results_best = checkpoint.get('results_best', results_best)
        results_b9_best = checkpoint.get('results_b9_best', results_b9_best)
        print(f"Resuming from checkpoint: {resume_checkpoint_path} (next epoch={start_epoch + 1})")

    if start_epoch >= options['max_epoch']:
        print(f"Checkpoint already reached max_epoch={options['max_epoch']}; running evaluation only.")
        results, results_b9 = test_ddfm_b9(
            net,
            criterion,
            testloader,
            outloader,
            epoch=max(options['max_epoch'] - 1, 0),
            **options,
        )
        return results_best or results, results_b9_best or results_b9

    start_time = time.time()
    for epoch in range(start_epoch, options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        train_Nirvana_oe_reg(
            net,
            criterion,
            optimizer,
            scheduler,
            trainloader,
            trainloader_oe=trainloader_oe,
            epoch=epoch,
            **options,
        )

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results , results_b9= test_ddfm_b9(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
            print("Normalized - Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results_b9['ACC'], results_b9['AUROC'], results_b9['OSCR']))
            avg_acc = (results['AUROC'] + results['OSCR'])/2.0
            
            if(avg_acc >= best_acc_avg):
                best_acc_avg = avg_acc
                results_best = results
                print("Best Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results_best['ACC'], results_best['AUROC'], results_best['OSCR']))
                save_networks(net, model_path, file_name, ext='best', criterion=criterion)
          
            avg_acc_b9 = (results_b9['AUROC'] + results_b9['OSCR'])/2.0
            if(avg_acc_b9 >= best_acc_avg_b9):
                best_acc_avg_b9 = avg_acc_b9
                results_b9_best = results_b9
                print("Normalized - Best Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results_b9_best['ACC'], results_b9_best['AUROC'], results_b9_best['OSCR']))
                save_networks(net, model_path, file_name, ext='best_b9', criterion=criterion)

            save_networks(net, model_path, file_name, criterion=criterion)

        checkpoint_state = {
            'epoch': epoch,
            'net': net.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc_avg': best_acc_avg,
            'best_acc_avg_b9': best_acc_avg_b9,
            'results_best': results_best,
            'results_b9_best': results_b9_best,
        }
        tmp_resume_path = f"{resume_checkpoint_path}.tmp"
        torch.save(checkpoint_state, tmp_resume_path)
        os.replace(tmp_resume_path, resume_checkpoint_path)
        
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    return results_best, results_b9_best

if __name__ == '__main__':
    args = parser.parse_args()
    device = _auto_device(args.use_cpu)
    print(f'Using device: {device.type}')
    options = vars(args)
    options['device'] = device
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    results = dict()
    results_b9 = dict()    

    if args.split_idx is not None:
        if args.split_idx < 0 or args.split_idx >= len(splits[options['dataset']]):
            raise ValueError(f"--split-idx out of range for dataset={options['dataset']}: {args.split_idx}")
        split_indices = [args.split_idx]
    else:
        split_indices = range(len(splits[options['dataset']]))

    for i in split_indices:
        split_dict = splits[options['dataset']][i]
        known = split_dict['known']
        unknown = split_dict['unknown']
        
        options.update({
            'item': i,
            'known': known,
            'unknown': unknown,
            'img_size': 32
        })

        dir_name = '{}_{}_{}_{}_{}'.format(options['model'], options['loss'],options['margin'],options['oe'],options['noisy_ratio'])
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = options['dataset'] + '.csv'
        file_name_b9 = options['dataset'] + '_b9.csv'

        #run experiment for this split
        res, res_b9 = main_worker(options)
        res['unknown'] = unknown
        res['known'] = known
        res_b9['unknown'] = unknown
        res_b9['known'] = known

        #save results
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))
        results_b9[str(i)] = res_b9
        df_b9 = pd.DataFrame(results_b9)
        df_b9.to_csv(os.path.join(dir_path, file_name_b9))
