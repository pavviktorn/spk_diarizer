#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,1,3'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from backbones.fasmodel import FASModel
from detection_layers.modules import MultiBoxLoss
from dataset import FASDataset
from lib.util import load_config, update_learning_rate, my_collate
import time
import logging
import math
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
from loss import *
import warnings
from backbones.model import ATR_FAS
from show_with_opencv import show_tensor_as_image

warnings.filterwarnings('ignore')

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/fasme_train.cfg')
    parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.', default='./checkpoints/epoch_27_best_0.005055611729019211.pkl')
    parser.add_argument('--device', type=str, default='0', help='device id, format is like 0,1,2')

    args = parser.parse_args()
    return args


def save_checkpoint(net, opt, save_path, epoch_num, iters=9999999):
    os.makedirs(save_path, exist_ok=True)
    module = net.module
    model_state_dict = OrderedDict()
    for k, v in module.state_dict().items():
        model_state_dict[k] = torch.tensor(v, device="cpu")

    # opt_state_dict = {}
    # opt_state_dict['param_groups'] = opt.state_dict()['param_groups']
    # opt_state_dict['state'] = OrderedDict()
    # for k, v in opt.state_dict()['state'].items():
    #     opt_state_dict['state'][k] = {}
    #     opt_state_dict['state'][k]['step'] = v['step']
    #     if 'exp_avg' in v:
    #         opt_state_dict['state'][k]['exp_avg'] = torch.tensor(v['exp_avg'], device="cpu")
    #     if 'exp_avg_sq' in v:
    #         opt_state_dict['state'][k]['exp_avg_sq'] = torch.tensor(v['exp_avg_sq'], device="cpu")

    checkpoint = {
        'network': model_state_dict,
        # 'opt_state': opt_state_dict,
        'epoch': epoch_num,
    }

    torch.save(checkpoint, f'{save_path}/epoch_{epoch_num}_{iters}.pkl')


def save_best_checkpoint(net, opt, save_path, epoch_num, f_auc):
    os.makedirs(save_path, exist_ok=True)
    module = net.module
    model_state_dict = OrderedDict()
    for k, v in module.state_dict().items():
        model_state_dict[k] = torch.tensor(v, device="cpu")

    # opt_state_dict = {}
    # opt_state_dict['param_groups'] = opt.state_dict()['param_groups']
    # opt_state_dict['state'] = OrderedDict()
    # for k, v in opt.state_dict()['state'].items():
    #     opt_state_dict['state'][k] = {}
    #     opt_state_dict['state'][k]['step'] = v['step']
    #     if 'exp_avg' in v:
    #         opt_state_dict['state'][k]['exp_avg'] = torch.tensor(v['exp_avg'], device="cpu")
    #     if 'exp_avg_sq' in v:
    #         opt_state_dict['state'][k]['exp_avg_sq'] = torch.tensor(v['exp_avg_sq'], device="cpu")

    checkpoint = {
        'network': model_state_dict,
        # 'opt_state': opt_state_dict,
        'epoch': epoch_num,
    }

    torch.save(checkpoint, f'{save_path}/epoch_{epoch_num}_best_{f_auc}.pkl')


def load_checkpoint(ckpt, net, opt, device):
    checkpoint = torch.load(ckpt)

    # gpu_state_dict = OrderedDict()
    # for k, v in checkpoint['network'] .items():
    #     name = "module."+k  # add `module.` prefix
    #     name = k
    #     gpu_state_dict[name] = v.to(device)
    # net.load_state_dict(gpu_state_dict)

    model_state = net.state_dict()
    pretrained_state = checkpoint['network']
    pretrained_state = {k: v for k, v in pretrained_state.items() if
                        k in model_state and v.size() == model_state[k].size()}
    model_state.update(pretrained_state)
    net.load_state_dict(model_state)

    # opt.load_state_dict(checkpoint['opt_state'])
    base_epoch = int(checkpoint['epoch']) + 1

    return net, opt, base_epoch


def loss_depth_func(ground_truth: torch.Tensor, pred: torch.Tensor):
    # groundtruth shape (batch size, 5, 1, 64, 64)
    # predict shape (batch size, 6, 1, 64, 64)
    # mean_gt = ground_truth.mean(dim=1)  # (batch_size, 64, 64) #zzzzzzzz
    # mean_pred = pred.mean(dim=1)  # (batch_size, 64, 64) #zzzzzzzz
    mean_gt = ground_truth  # (batch_size, 64, 64)
    mean_pred = pred  # (batch_size, 64, 64)

    # Flatten spatial dimensions
    mean_gt = mean_gt.view(mean_gt.shape[0], -1)  # (batch_size, 64*64)
    mean_pred = mean_pred.view(mean_pred.shape[0], -1)  # (batch_size, 64*64)

    epsilon = 1e-7
    mean_pred = torch.clamp(mean_pred, min=epsilon, max=1.0)
    loss = -torch.sum(mean_gt * torch.log(mean_pred), dim=1) / (mean_gt.shape[1])

    return loss.mean()


# SSIM loss implementation
class SSIMLoss(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def gaussian(self, window_size, sigma):
        gauss = torch.tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        if img1.size() != img2.size():
            raise ValueError('Input images must have the same dimensions')
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

# Edge loss based on gradient differences
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def gradient_x(self, img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def gradient_y(self, img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def forward(self, pred, target):
        grad_pred_x = self.gradient_x(pred)
        grad_pred_y = self.gradient_y(pred)
        grad_target_x = self.gradient_x(target)
        grad_target_y = self.gradient_y(target)

        loss_x = torch.abs(grad_pred_x - grad_target_x).mean()
        loss_y = torch.abs(grad_pred_y - grad_target_y).mean()

        return loss_x + loss_y

# Combined weighted loss function
class DepthLoss(nn.Module):
    def __init__(self, weight_mae=0.6, weight_edge=0.2, weight_ssim=1.0):
        super(DepthLoss, self).__init__()
        self.weight_mae = weight_mae
        self.weight_edge = weight_edge
        self.weight_ssim = weight_ssim

        self.mae = nn.L1Loss()
        self.edge = GradientLoss()
        self.ssim = SSIMLoss()

    def forward(self, pred, target):
        loss_mae = self.mae(pred, target)
        loss_edge = self.edge(pred, target)
        loss_ssim = self.ssim(pred, target)

        loss = self.weight_mae * loss_mae + \
               self.weight_edge * loss_edge + \
               self.weight_ssim * loss_ssim
        return loss


def train():

    args = args_func()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # load conifigs
    cfg = load_config(args.cfg)

    logging.basicConfig(filename=cfg['model']['save_path']+"/train.log", level=logging.INFO)

    # init model.
    net = ATR_FAS(frame_num=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # loss init
    det_criterion = MultiBoxLoss(
        cfg['det_loss']['num_classes'],
        cfg['det_loss']['overlap_thresh'],
        cfg['det_loss']['prior_for_matching'],
        cfg['det_loss']['bkg_label'],
        cfg['det_loss']['neg_mining'],
        cfg['det_loss']['neg_pos'],
        cfg['det_loss']['neg_overlap'],
        cfg['det_loss']['encode_target'],
        cfg['det_loss']['use_gpu']
    )
    weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    criterion1 = nn.CrossEntropyLoss()
    weights2 = torch.tensor([1.0, 1.0]).to(device)
    criterion2 = nn.CrossEntropyLoss(weight=weights2)

    # optimizer init.
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=4e-3)

    # load checkpoint if given
    base_epoch = 0
    if args.ckpt:
        net, optimzer, base_epoch = load_checkpoint(args.ckpt, net, optimizer, device)
        # base_epoch = 0
        # Freeze all parameters in model.features
        # for param in net.gate.parameters():
        #     param.requires_grad = False

    net = net.to(device)
    net = nn.DataParallel(net)

    # get training data
    print(f"Load Train deepfake dataset from {cfg['dataset']['img_path']}..")
    train_dataset = FASDataset('train', cfg, cfg['dataset']['info_path'])
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=16,#zzzz
                              collate_fn=my_collate
                              )

    # get testing data
    print(f"Load Test deepfake dataset from {cfg['test_dataset']['img_path']}..")
    test_dataset = FASDataset('test', cfg, cfg['test_dataset']['info_path'])
    test_loader = DataLoader(test_dataset,
                              batch_size=cfg['test']['batch_size'],
                              shuffle=False, num_workers=16,#zzzz
                              collate_fn=my_collate
                              )

    contra_fun = ContrastLoss()

    depth_func = DepthLoss(weight_mae=0.6, weight_edge=0.2, weight_ssim=1.0)

    # start training.
    warmup_steps = cfg['train']['warmup_epoch'] * len(train_dataset) // cfg['train']['batch_size']
    g_step = 0
    best_auc = 1000
    pre_time = time.time()
    for epoch in range(base_epoch, cfg['train']['epoch_num']):
        acc_record = AvgrageMeter()
        tloss_record = AvgrageMeter()

        net.train()
        for index, (batch_data, batch_labels) in enumerate(train_loader):
            net.train()

            # lr = update_learning_rate(epoch, g_step, warmup_steps, cfg['train']['warmup_epoch'])
            lr = 0.00001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            labels, location_labels, confidence_labels, depth_labels, domains = batch_labels
            labels = labels.long().to(device)
            location_labels = location_labels.to(device)
            confidence_labels = confidence_labels.long().to(device)
            depth_labels = depth_labels.to(device)
            domains = domains.long().to(device)

            rand_idx = torch.randperm(batch_data.shape[0])

            optimizer.zero_grad()
            locations, confidence, domain_invariant, outputs, feats1, feats2, frame_depth_map, final_cls = net(batch_data, batch_data[rand_idx, :, :, :])
            # frame_depth_map, final_cls = net(batch_data, batch_data[rand_idx, :, :, :])

            if index % 20 == 0:
                sdpth = frame_depth_map.squeeze(1)
                show_tensor_as_image(sdpth, "depth")
                show_tensor_as_image(depth_labels, "truth")

            # loss_depth = loss_depth_func(depth_labels, frame_depth_map)
            loss_depth = depth_func(depth_labels.unsqueeze(1), frame_depth_map)
            final_label = torch.where(labels == 0, 0, 1)
            final_loss_end_cls = criterion2(final_cls, final_label)

            locations = locations.to(device)
            confidence = confidence.to(device)
            outputs = outputs.to(device)

            loss_end_cls = criterion(outputs, labels)
            gate_acc = sum(outputs.max(-1).indices == labels).item() / labels.shape[0]

            contrast_label = labels[:].long() == labels[rand_idx].long()
            contrast_label = torch.where(contrast_label == True, 1, -1)
            constra_loss = contra_fun(feats1, feats2, contrast_label)

            loss_l_0, loss_c_0 = det_criterion((locations[:,0,:,:], confidence[:,0,:,:]), confidence_labels[:,0,:], location_labels[:,0,:])
            loss_l_4, loss_c_4 = det_criterion((locations[:,1,:,:], confidence[:,1,:,:]), confidence_labels[:,1,:], location_labels[:,1,:])
            det_loss = 0.1 * (loss_l_0 + loss_c_0 + loss_l_4 + loss_c_4)

            adv_loss = criterion1(domain_invariant, domains.long())
            loss = loss_end_cls + det_loss + constra_loss + adv_loss + loss_depth + final_loss_end_cls

            # loss = loss_depth + final_loss_end_cls
            acc = sum(final_cls.max(-1).indices == final_label).item() / final_label.shape[0]

            if (math.isinf(loss.item()) and loss.item() > 0) or (math.isinf(loss.item()) and loss.item() < 0):
                print("error: loss is inf : {:.8f}".format(loss.item()))
                logging.error("error: loss is inf : {:.8f}".format(loss.item()))
                continue

            n = labels.shape[0]
            acc_record.update(acc, n)
            tloss_record.update(loss.item(), n)

            loss.backward()

            torch.nn.utils.clip_grad_value_(net.parameters(), 2)
            optimizer.step()
            g_step += 1

            if index % 5 == 0:
                cur_time = time.time()
                outputs = [
                    "{}/{}".format(epoch, index),
                    "acc:{:.2f}".format(acc),
                    "gacc:{:.2f}".format(gate_acc),
                    "tloss:{:.4f} ".format(loss.item()),
                    "f_cls:{:.4f} ".format(final_loss_end_cls.item()),
                    "depth:{:.4f} ".format(loss_depth.item()),
                    "cls:{:.4f} ".format(loss_end_cls.item()),
                    "det:{:.4f} ".format(det_loss.item()),
                    "cont:{:.4f} ".format(constra_loss.item()),
                    "adv:{:.4f} ".format(adv_loss.item()),
                    "lr:{:.5g}".format(lr),
                    "time:{:.1f}s".format(cur_time - pre_time),
                ]
                print(" ".join(outputs))
                logging.info(" ".join(outputs))
                pre_time = cur_time

            if index > 0 and index % 3000 == 0: # and epoch >= cfg['train']['warmup_epoch']:
                # save_checkpoint(net, optimizer,
                #                 cfg['model']['save_path'],
                #                 epoch, index)

                print("Starting testing")
                logging.info("Starting testing")
                BPCER = test(net, test_loader)
                print("Finished testing : BPCER = {:.8f} ".format(BPCER))
                logging.info("Finished testing : BPCER = {:.8f} ".format(BPCER))
                if best_auc >= BPCER:
                    best_auc = BPCER
                    save_best_checkpoint(net, optimizer,
                                    cfg['model']['save_path'],
                                    epoch, BPCER)
                    print("Saved best model : best_BPCER = {:.8f}".format(best_auc))
                    logging.info("Saved best model : best_BPCER = {:.8f}".format(best_auc))

        # save_checkpoint(net, optimizer,
        #                 cfg['model']['save_path'],
        #                 epoch, index)



def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]
    return err, best_th, right_index


def performances_val(frame_label_list, frame_pred_list):
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for idx in range(len(frame_label_list)):
        try:
            count += 1
            score = float(frame_pred_list[idx])
            label = float(frame_label_list[idx])
            val_scores.append(score)
            val_labels.append(label)
            data.append({'map_score': score, 'label': label})
            if label == 0:
                num_real += 1
            else:
                num_fake += 1
        except:
            continue

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    auc_test = auc(fpr, tpr)
    val_err, val_threshold, right_index = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] < val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count

    FRR = 1 - tpr  # FRR = 1 - TPR

    HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate

    print("val_threshold={}".format(val_threshold))

    return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index], auc_test, val_err


def test(net, test_loader):
    frame_pred_list = list()
    frame_label_list = list()
    video_name_list = list()

    pre_time = time.time()
    net.eval()
    with torch.no_grad():
        for index, (batch_data, batch_labels) in enumerate(test_loader):

            labels, video_name, info_meta = batch_labels
            labels = labels.long()

            alive_label = labels[:].long() == 0
            alive_label = torch.where(alive_label == True, 1, 0)

            outputs = net(batch_data, batch_data)
            outputs = outputs[:, 0]
            frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
            frame_label_list.extend(alive_label.detach().cpu().numpy().tolist())
            video_name_list.extend(list(video_name))

            if index % 5 == 0:
                cur_time = time.time()
                info = [
                    "iter:{}".format(index),
                    "label:{}".format(alive_label.detach().cpu().numpy()[0]),
                    "out:{:.4f}".format(outputs.detach().cpu().numpy()[0]),
                    "time:{:.1f}s".format(cur_time - pre_time),
                ]
                print(" ".join(info))
                pre_time = cur_time

            # print(f"{video_name[0]} label:{alive_label.detach().cpu().numpy()[0]} out:{outputs.detach().cpu().numpy()[0]}")

        f_auc = roc_auc_score(frame_label_list, frame_pred_list)
        fpr, tpr, threshold = roc_curve(frame_label_list, frame_pred_list)

        for idx in range(len(tpr)):
            if tpr[idx] >= 0.99:
                break
        if idx == len(tpr):
            BPCER = 1.0
            mythresh = 0.0
        else:
            mythresh = threshold[idx]

            type1 = 0
            type2 = 0
            num_real = 0
            for iii in range(len(frame_label_list)):
                if frame_label_list[iii] == 0:
                    num_real += 1
                if frame_pred_list[iii] < mythresh and frame_label_list[iii] == 1:
                    type1 += 1
                if frame_pred_list[iii] >= mythresh and frame_label_list[iii] == 0:
                    type2 += 1

            BPCER = type2 / num_real  # Bona Fide Presentation Classification Error Rate for 0.01 Attack Presentation Classification Error Rate (APCER)

        print("BPCER={}, val_threshold={}".format(BPCER, mythresh))
        logging.info("BPCER={}, val_threshold={}".format(BPCER, mythresh))

        # val_err, val_threshold, right_index = get_err_threhold(fpr, tpr, threshold)
        # print("val_threshold={}".format(val_threshold))

    return BPCER


if __name__ == "__main__":
    train()

# vim: ts=4 sw=4 sts=4 expandtab
