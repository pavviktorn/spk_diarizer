
import os
import numpy as np
import random
import time
import argparse
from collections import OrderedDict
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,1,0,3'

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


def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)  # Self-explanatory.


in_root = '/datasets/newout/meta'
out_root = '/datasets/newout'
# in_root = 'D:/zLiveness_data/out/meta'
# out_root = 'D:/zLiveness_data/out'


rseed = int(time.time())
np.random.seed(rseed)
random.seed(rseed)

def sel_lines(task, selnum, f_a):
    input_path = os.path.join(in_root, task)
    full_file_paths = get_filepaths(input_path)

    num = 0
    all_num = 0
    for path in full_file_paths:
        if "new_test_20250821" in path:
            continue
        print(path)

        f = open(path, 'r+')
        lines = f.readlines()
        f.close()

        size = len(lines)
        all_num += size
        for idx in range(size):
            rat = selnum / size
            if random.random() < rat:
                f_a.write(lines[idx])
                num += 1

    return num, all_num

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/fasme_train.cfg')
    parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.', default='./checkpoints/epoch_14_best_0.001824505387992474.pkl')
    parser.add_argument('--device', type=str, default='0', help='device id, format is like 0,1,2')

    args = parser.parse_args()
    return args

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


args = args_func()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# load conifigs
cfg = load_config(args.cfg)

# init model.
net = ATR_FAS(frame_num=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# optimizer init.
optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=4e-3)

# load checkpoint if given
base_epoch = 0
if args.ckpt:
    net, optimzer, base_epoch = load_checkpoint(args.ckpt, net, optimizer, device)

net = net.to(device)
net = nn.DataParallel(net)


def main_pts(info_path, isFake):

    # get testing data
    print(f"Load Test deepfake dataset from {cfg['test_dataset']['img_path']}..")
    test_dataset = FASDataset('test', cfg, info_path)
    test_loader = DataLoader(test_dataset,
                              batch_size=cfg['test']['batch_size'],
                              shuffle=False, num_workers=16,#zzzz
                              collate_fn=my_collate)

    scores_dic = {}

    pre_time = time.time()
    net.eval()
    with torch.no_grad():
        for index, (batch_data, batch_labels) in enumerate(test_loader):

            labels, video_name, info_meta = batch_labels
            outputs = net(batch_data, batch_data)
            outputs = outputs[:, 0]
            for ii in range(batch_data.shape[0]):
                score = outputs[ii]
                scores_dic[info_meta[ii]] = score

            if index % 5 == 0:
                cur_time = time.time()
                print("{} : {:.4f}s {}".format(index, cur_time - pre_time, info_meta[0]))
                pre_time = cur_time

    sorted_dic_by_score = sorted(scores_dic.items(), key=lambda x: x[1], reverse=isFake)
    converted_dict = dict(sorted_dic_by_score)
    pending_size = len(converted_dict)
    adding_size = 500*1000
    if pending_size < adding_size:
        adding_size = pending_size

    f_a = open(os.path.join(out_root, 'pts_train_info.txt'), 'a+')
    f_m = open(os.path.join(out_root, 'score.txt'), 'a+')

    idx = 0
    for fname in converted_dict:
        info_line = "{} {}\n".format(converted_dict[fname], fname)
        f_m.write(info_line)
        line = fname
        if idx < adding_size:
            f_a.write(line)
        idx += 1

    f_a.close()
    f_m.close()


f_a = open(os.path.join(out_root, 'pad_info.txt'), 'w')
n_train_pad, pad_total = sel_lines('pad', 200000, f_a)
f_a.close()

f_a = open(os.path.join(out_root, 'df_info.txt'), 'w')
n_train_df, df_total = sel_lines('df', 20000, f_a)
f_a.close()

f_a = open(os.path.join(out_root, 'real_info.txt'), 'w')
n_train_real, real_total = sel_lines('real', 400000, f_a)
f_a.close()

print(f"train real : {n_train_real}")
print(f"train pad : {n_train_pad}")
print(f"train deepfake : {n_train_df}")
print(f"total real : {real_total}")
print(f"total pad : {pad_total}")
print(f"total deepfake : {df_total}")

main_pts(os.path.join(out_root, 'pad_info.txt'), isFake=True)
main_pts(os.path.join(out_root, 'df_info.txt'), isFake=True)
main_pts(os.path.join(out_root, 'real_info.txt'), isFake=False)

f_a = open(os.path.join(out_root, 'pts_train_info.txt'), 'r+')
train_info = f_a.readlines()
f_a.close()

random.shuffle(train_info)

f_a = open(os.path.join(out_root, 'pts_train_info.txt'), 'w')
f_a.writelines(train_info)
f_a.close()
