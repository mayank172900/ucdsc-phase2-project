"""
Created on Mon Aug  9 21:03:27 2021

@author: hasan
"""
import datetime
import os
import time
import json
import torch
import modules.utils_torchvision as utils
from modules.NirvanaLoss import center_loss_nirvana
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torchvision import transforms as T
from modules.utils_faceevolve import perform_val,get_val_data
from modules.utils import train_one_epoch, evaluate_majority_voting, evaluate_ijba_maj_vot, dataset_preperation
from Networks.model_irse import IR_50
from Networks.iresnet_torch_new import iresnet100
from Networks.DAMNet_new import DAMGeneral

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ["OMP_NUM_THREADS"]=str(4)

def main(args):
    save_dir = os.path.join('logs','NirvanaFacePaper12kE2000Paper','%s_%s_lr%.6f_Nirvana_Expand%d_Epoch%d_Seed%d_div%d'%(args.dataset_name,args.Network,args.lr, args.Expand, args.epochs, args.Seed, args.div))
    utils.mkdir(save_dir)
    with open(os.path.join(save_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    writer = SummaryWriter(log_dir=save_dir)
    utils.init_distributed_mode(args)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device.split('_')[0])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split('_')[1]
        # torch.manual_seed(12345)  
    
    data_loader, data_loader_val = dataset_preperation(args)
    args.num_classes = len(data_loader.dataset.classes)
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_transform = T.Compose([
            T.Resize((112,112)),
            T.ToTensor(),
            normalize,
        ])
    # lfw, lfw_issame = get_val_data('./data')
    # lfw, lfw_issame = get_val_data('./data')
    # lfw, lfw_issame = get_val_data('./data')
    # lfw, lfw_issame = get_val_data('./data')
    lfw, cfp_ff, cfp_fp, agedb_30, calfw, cplfw, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_30_issame, calfw_issame, cplfw_issame = get_val_data('./data')
    print("Creating model")
    #model = IR_50(pretrained=True,input_size=[112,112], num_classes = args.num_classes)
    #model = iresnet100(pretrained=args.pretrained)
    args.feat_dim = 12500
    model = DAMGeneral(input_dim=args.feat_dim, num_layers=1)
    
    checkpointinitial = torch.load("/run/media/mlcv/SSD2/Hasan/NirvanaFace2/logs/NirvanaFacePaper12kE2000/VGGFace2_ResNet50_lr0.000100_Nirvana_Expand2000_Epoch5_Seed0_div10/best_checkpoint.pth")
    model.load_state_dict(checkpointinitial['model'])
    epoch = checkpointinitial['epoch']
    model.to(device)

  #  pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  #  pytorch_total_untrainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad==False)

   # print(f'count_true: {pytorch_total_trainable_params}, count_false: {pytorch_total_untrainable_params}')
    # if args.num_classes > 2048:
    #     args.feat_dim = args.num_classes-1
    
    # model = models.resnet50(pretrained=False)
    # model = get_network(args.feat_dim, args.num_classes)
    # model = resnet.ResNet18(num_classes=args.num_classes)
    # args.feat_dim = args.num_classes-1
    print(args)
    centerloss = center_loss_nirvana(args.num_classes,args.feat_dim,True,device, Expand=args.Expand)
    centerloss.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model_without_ddp = model
    centerloss_without_ddp = centerloss
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        centerloss = torch.nn.parallel.DistributedDataParallel(centerloss, device_ids=[args.gpu])
        centerloss_without_ddp = centerloss.module
    
    best_val_acc1, best_val_acc1_lfw, val_acc1, val_acc1_lfw, best_val_acc1_ijba, val_acc1_ijba, val_acc1_cfp_ff, best_val_acc1_cfp_ff = 0., 0., 0., 0., 0., 0., 0., 0.
    val_acc1_cfp_fp, best_val_acc1_cfp_fp, val_acc1_agedb_30, best_val_acc1_agedb_30, val_acc1_calfw, best_val_acc1_calfw, val_acc1_cplfw, best_val_acc1_cplfw = 0., 0., 0., 0., 0., 0., 0., 0.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        centerloss_without_ddp.load_state_dict(checkpoint['centerloss'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_val_acc1 = checkpoint['best_val_acc1']
        best_val_acc1_lfw = checkpoint['best_val_acc1_lfw']

    if args.test_only:
        val_acc1_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(device, args.feat_dim, 128, model, lfw, lfw_issame)
        val_acc1_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(device, args.feat_dim, 128, model, cfp_ff, cfp_ff_issame)
        val_acc1_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(device, args.feat_dim, 128, model, cfp_fp, cfp_fp_issame)
        val_acc1_agedb_30, best_threshold_agedb_30, roc_curve_agedb_30 = perform_val(device, args.feat_dim, 128, model, agedb_30, agedb_30_issame)
        val_acc1_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(device, args.feat_dim, 128, model, calfw, calfw_issame)
        val_acc1_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(device, args.feat_dim, 128, model, cplfw, cplfw_issame)
        print('%.3f LFW Val., %.3f cfp_ff Val., %.3f cfp_fp, %.3f agedb_30, %.3f calfw, %.3f cplfw'%(val_acc1_lfw,val_acc1_cfp_ff, val_acc1_cfp_fp, val_acc1_agedb_30, val_acc1_calfw, val_acc1_cplfw))

        # return evaluate_majority_voting(model,centerloss, data_loader_val, device, epoch=0,args=args)
        return evaluate_ijba_maj_vot(model, centerloss, train_transform, device, epoch=0,args=args)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()       
        acc1,loss = train_one_epoch(model, centerloss, optimizer, data_loader, device, epoch, args)
        writer.add_scalar('train/acc1', acc1, epoch)
        writer.add_scalar('train/loss', loss, epoch)
        
        # if(epoch>args.only_centers_upto):
        lr_scheduler.step()
        
        if epoch%args.eval_freq == 0:
            model.eval()
            val_acc1_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(device, args.feat_dim, 128, model, lfw, lfw_issame)
            is_best_lfw = val_acc1_lfw > best_val_acc1_lfw
            best_val_acc1_lfw = max(val_acc1_lfw, best_val_acc1_lfw)
            print('%.3f LFW Val., %.3f LFW Best Val.'%(val_acc1_lfw,best_val_acc1_lfw))
            
            val_acc1_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(device, args.feat_dim, 128, model, cfp_ff, cfp_ff_issame)
            is_best_cfp_ff = val_acc1_cfp_ff > best_val_acc1_cfp_ff
            best_val_acc1_cfp_ff = max(val_acc1_cfp_ff, best_val_acc1_cfp_ff)
            print('%.3f cfp_ff Val., %.3f cfp_ff Best Val.'%(val_acc1_cfp_ff,best_val_acc1_cfp_ff))

            val_acc1_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(device, args.feat_dim, 128, model, cfp_fp, cfp_fp_issame)
            is_best_cfp_fp = val_acc1_cfp_fp > best_val_acc1_cfp_fp
            best_val_acc1_cfp_fp = max(val_acc1_cfp_fp, best_val_acc1_cfp_fp)
            print('%.3f cfp_fp Val., %.3f cfp_fp Best Val.'%(val_acc1_cfp_fp,best_val_acc1_cfp_fp))

            val_acc1_agedb_30, best_threshold_agedb_30, roc_curve_agedb_30 = perform_val(device, args.feat_dim, 128, model, agedb_30, agedb_30_issame)
            is_best_agedb_30 = val_acc1_agedb_30 > best_val_acc1_agedb_30
            best_val_acc1_agedb_30 = max(val_acc1_agedb_30, best_val_acc1_agedb_30)
            print('%.3f agedb_30 Val., %.3f agedb_30 Best Val.'%(val_acc1_agedb_30,best_val_acc1_agedb_30))
            
            val_acc1_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(device, args.feat_dim, 128, model, calfw, calfw_issame)
            is_best_agedb_30 = val_acc1_calfw > best_val_acc1_calfw
            best_val_acc1_calfw = max(val_acc1_calfw, best_val_acc1_calfw)
            print('%.3f calfw Val., %.3f calfw Best Val.'%(val_acc1_calfw,best_val_acc1_calfw))

            # val_acc1_ijba = evaluate_ijba_maj_vot(model, centerloss, train_transform, device, epoch=epoch,args=args)
            # is_best_ijba = val_acc1_ijba > best_val_acc1_ijba
            # best_val_acc1_ijba= max(val_acc1_ijba, best_val_acc1_ijba)
            # print('%.3f IJBA Val., %.3f IJBA Best Val.'%(val_acc1_ijba,best_val_acc1_ijba))              
         
            val_acc1_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(device, args.feat_dim, 128, model, cplfw, cplfw_issame)
            is_best_cplfw = val_acc1_cplfw > best_val_acc1_cplfw
            best_val_acc1_cplfw= max(val_acc1_cplfw, best_val_acc1_cplfw)
            print('%.3f cplfw Val., %.3f cplfw Best Val.'%(val_acc1_cplfw,best_val_acc1_cplfw))              
         
            # val_acc1,val_acc1_fc = evaluate_majority_voting(model,centerloss, data_loader_val, device, epoch,args=args)
            # is_best = val_acc1 > best_val_acc1
            # best_val_acc1 = max(val_acc1, best_val_acc1)
            # writer.add_scalar('val/acc1', val_acc1, epoch)
            # writer.add_scalar('val/best_acc1', best_val_acc1, epoch)
            # print('%.3f Data Val., %.3f Data Best Val.'%(val_acc1,best_val_acc1))
            
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'centerloss': centerloss_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'val_acc1': val_acc1,
                # 'val_acc1_fc':val_acc1_fc,
                'best_val_acc1': best_val_acc1,
                'val_acc1_lfw': val_acc1_lfw,
                'best_val_acc1_lfw': best_val_acc1_lfw,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(save_dir, 'model_{}.pth'.format(epoch)))
            if(is_best_lfw):
                utils.save_on_master(
                    checkpoint,
                    os.path.join(save_dir, 'best_checkpoint.pth'))
                is_best=False

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.close()
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--device', default='cuda_0', help='device ex. cuda_7, cuda_6')
    parser.add_argument('-b', '--batch-size', default=24, type=int)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('-d', '--dataset_name', default='VGGFace2', metavar='N',
                        help='CIFAR10, CIFAR100, car196, MNIST, VGGFace2 ')
    parser.add_argument('-n', '--Network', default='ResNet50', metavar='N',
                        help='ResNet18, ResNet50')
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--div', default=10, type=int, metavar='N',
                        help='division number for Nirvana Hinge loss')
    parser.add_argument('--num_classes', default=12000, type=int, metavar='N',
                        help='number of classes')
    parser.add_argument('--Expand', default=2000, type=int, metavar='N',
                        help='Expand factor of centers')
    parser.add_argument('--Seed', default=0, type=int, metavar='N',
                        help='Seed')
    parser.add_argument('--feat_dim', default=512, type=int, metavar='N',
                        help='feature dimension of the model')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=50, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lamda', default=0.5, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=500, type=int, help='print frequency')
    parser.add_argument('--eval-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--test_only', default=False, help='Only Test the model')
    parser.add_argument('--pretrained', default=True, help='True or False')

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
