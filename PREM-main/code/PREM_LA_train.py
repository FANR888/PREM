from asyncore import write
# import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.PREM_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='PREM', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')

parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')

parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')

parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')

parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
parser.add_argument('--k' , type=float, default=0.5, help='weight')
parser.add_argument('--temperature', type=int, default=0.1, help='sharpen_temperature')

args = parser.parse_args()


def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2


def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask1, loss_mask1 = context_mask(img_a, 1)

            volume_batch1 = img_a * img_mask1 + img_b * (1 - img_mask1)
            label_batch1 = lab_a * img_mask1 + lab_b * (1 - img_mask1)
            outputs1, _ = model(volume_batch1)
            loss_ce1 = F.cross_entropy(outputs1, label_batch1)
            loss_dice1 = DICE(outputs1, label_batch1)
            loss_ce = loss_ce1
            loss_dice = loss_dice1
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f'%(iter_num, loss, loss_dice, loss_ce))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    # torch.save(model.state_dict(), save_mode_path)
                    # torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
            param.detach_()   # ema_model set
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    consistency_criterion = losses.mse_loss
    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)
    
    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]
            with torch.no_grad():
                unoutput_a, _ = ema_model(unimg_a)
                unoutput_b, _ = ema_model(unimg_b)
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                img_mask1, loss_mask1 = context_mask(img_a, 1)
                img_mask2, loss_mask2 = context_mask(img_a, 2)
                img_mask3, loss_mask3 = context_mask(img_a, 3)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            mixl_img = img_a * img_mask1 + unimg_a * (1 - img_mask1)
            mixu_img = unimg_b * img_mask1 + img_b * (1 - img_mask1)
            mixl_lab = lab_a * img_mask1 + plab_a * (1 - img_mask1)
            mixu_lab = plab_b * img_mask1 + lab_b * (1 - img_mask1)
            outputs_l, _ = model(mixl_img)
            outputs_u, _ = model(mixu_img)
            out_num = len(outputs_l)
            soft_outputs_l = F.softmax(outputs_l,dim=1)
            soft_outputs_u = F.softmax(outputs_u, dim=1)
            sharp_output_l = sharpening(soft_outputs_l)
            sharp_output_u = sharpening(soft_outputs_u)

            loss_consist1 = 0.0
            for i in range(out_num):
                for j in range(0, 2):
                    loss_consist1 = loss_consist1 + ( consistency_criterion(soft_outputs_l[i][j], sharp_output_l[i][j]) +
                                                      consistency_criterion( soft_outputs_u[i][j], sharp_output_u[i][j]))

            loss_l1 = mix_loss(outputs_l, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_u1 = mix_loss(outputs_u, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            loss1 = loss_l1 + loss_u1
            #Second Layer----------------------------------------------------------------------------------
            mixl_img2 = img_a * img_mask2 + unimg_a * (1 - img_mask2)
            mixu_img2 = unimg_b * img_mask2 + img_b * (1 - img_mask2)
            mixl_lab2 = lab_a * img_mask2 + plab_a * (1 - img_mask2)
            mixu_lab2 = plab_b * img_mask2 + lab_b * (1 - img_mask2)
            outputs_l2, _ = model(mixl_img2)
            outputs_u2, _ = model(mixu_img2)
            soft_outputs_l2 = F.softmax(outputs_l2, dim=1)
            soft_outputs_u2 = F.softmax(outputs_u2, dim=1)
            sharp_output_l2 = sharpening(soft_outputs_l2)
            sharp_output_u2 = sharpening(soft_outputs_u2)

            loss_consist2 = 0.0
            for i in range(out_num):
                for j in range(0, 2):
                    loss_consist2 = loss_consist2 + (consistency_criterion(soft_outputs_l2[i][j], sharp_output_l2[i][j]) +
                                                     consistency_criterion(soft_outputs_u2[i][j], sharp_output_u2[i][j]))

            loss_l2 = mix_loss(outputs_l2, lab_a, plab_a, loss_mask2, u_weight=args.u_weight)
            loss_u2 = mix_loss(outputs_u2, plab_b, lab_b, loss_mask2, u_weight=args.u_weight, unlab=True)
            loss2 = loss_l2 + loss_u2
            # #Third Layer----------------------------------------------------------------------------------
            mixl_img3 = img_a * img_mask3 + unimg_a * (1 - img_mask3)
            mixu_img3 = unimg_b * img_mask3 + img_b * (1 - img_mask3)
            mixl_lab3 = lab_a * img_mask3 + plab_a * (1 - img_mask3)
            mixu_lab3 = plab_b * img_mask3 + lab_b * (1 - img_mask3)
            outputs_l3, _ = model(mixl_img3)
            outputs_u3, _ = model(mixu_img3)
            soft_outputs_l3 = F.softmax(outputs_l3, dim=1)
            soft_outputs_u3 = F.softmax(outputs_u3, dim=1)
            sharp_output_l3 = sharpening(soft_outputs_l3)
            sharp_output_u3 = sharpening(soft_outputs_u3)

            loss_consist3 = 0.0
            for i in range(out_num):
                for j in range(0, 2):
                    loss_consist2 = loss_consist2 + (
                                consistency_criterion(soft_outputs_l3[i][j], sharp_output_l3[i][j]) +
                                consistency_criterion(soft_outputs_u3[i][j], sharp_output_u3[i][j]))

            loss_l3 = mix_loss(outputs_l3, lab_a, plab_a, loss_mask3, u_weight=args.u_weight)
            loss_u3 = mix_loss(outputs_u3, plab_b, lab_b, loss_mask3, u_weight=args.u_weight, unlab=True)
            loss3 = loss_l3 + loss_u3

            loss = (1-args.k) *(loss1 + loss2 + loss3) +args.k*(loss_consist1 + loss_consist2 + loss_consist3)

            loss_l = loss_l1 + loss_l2 + loss_l3
            loss_u = loss_u1 + loss_u2 + loss_u3
            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_MSE', loss_consist1 + loss_consist2 + loss_consist3, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, loss_MSE: %f'%(iter_num, loss, loss_l, loss_u,loss_consist1 + loss_consist2 + loss_consist3))

            update_ema_variables(model, ema_model, 0.99)

             # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(args.model))
                    # save_net_opt(model, optimizer, save_mode_path)
                    # save_net_opt(model, optimizer, save_best_path)
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()
            
            if iter_num % 200 == 1:
                ins_width = 2
                B,C,H,W,D = outputs_l.size()
                snapshot_img = torch.zeros(size = (D, 3, 3*H + 3 * ins_width, W + ins_width), dtype = torch.float32)

                snapshot_img[:,:, H:H+ ins_width,:] = 1
                snapshot_img[:,:, 2*H + ins_width:2*H + 2*ins_width,:] = 1
                snapshot_img[:,:, 3*H + 2*ins_width:3*H + 3*ins_width,:] = 1
                snapshot_img[:,:, :,W:W+ins_width] = 1

                outputs_l_soft = F.softmax(outputs_l, dim=1)
                seg_out = outputs_l_soft[0,1,...].permute(2,0,1) # y
                target =  mixl_lab[0,...].permute(2,0,1)
                train_img = mixl_img[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                
                writer.add_images('Epoch_%d_Iter_%d_labeled'% (epoch, iter_num), snapshot_img)

                outputs_u_soft = F.softmax(outputs_u, dim=1)
                seg_out = outputs_u_soft[0,1,...].permute(2,0,1) # y
                target =  mixu_lab[0,...].permute(2,0,1)
                train_img = mixu_img[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch, iter_num), snapshot_img)

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "../model/PREM/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "../model/PREM/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    print("Starting PREM training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/PREM_LA_train.py', self_snapshot_path)

    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    
