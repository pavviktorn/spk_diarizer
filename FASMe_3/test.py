#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from backbones.fasmodel import FASModel
from dataset import FASDataset
from lib.util import load_config, get_video_auc
import cv2
from data.util import get_filepaths, get_landmarks
from lib.data_preprocess.cropface import get_cropped
from train import performances_val
import ntpath
import time
import math
import numpy as np
from backbones.model import ATR_FAS
from show_with_opencv import show_tensor_as_image


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/fasme_test.cfg')
    args = parser.parse_args()
    return args


def load_checkpoint(ckpt, net, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'] .items():
        name = "module." + k  # add `module.` prefix
        name = k
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    return net


def test():

    torch.set_flush_denormal(True)

    args = args_func()

    # load conifigs
    cfg = load_config(args.cfg)

    # init model.
    net = ATR_FAS(frame_num=1)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    net = net.to(device)
    # net = nn.DataParallel(net)
    net.eval()
    if cfg['model']['ckpt']:
        net = load_checkpoint(cfg['model']['ckpt'], net, device)

    img_path = r'D:\lhm_work\face_liveness\DATASET\nizar_data\_error\df_testset\2025-01-21\fake\_fake\2025-07-30'
    img_path = r'/datasets/newtest_20250821'

    #############################################################
    dummy_input = torch.randn(1, 3, 256, 512)
    torch.onnx.export(net, dummy_input, "./FASMe_3_onnx/models/out.onnx", keep_initializers_as_inputs=False, verbose=False,
                      opset_version=12)

    import onnx
    onnx_model = onnx.load("./FASMe_3_onnx/models/out.onnx")
    from onnxsim import simplify
    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    import onnxoptimizer
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx.save(onnx_model, "./FASMe_3_onnx/models/out.onnx")

    from onnx import numpy_helper
    total_parameters = 0
    for initializer in onnx_model.graph.initializer:
        total_parameters += numpy_helper.to_array(initializer).size

    full_file_paths = get_filepaths(img_path)

    thresh = 0.00001
    thresh = 0.01

    n_fake = 0
    n_live = 0
    n_fake_err = 0
    n_live_err = 0
    n_det_err = 0

    with torch.no_grad():
        scores_list = []

        for src_path in full_file_paths:
            head, fname = ntpath.split(src_path)
            if os.path.isdir(src_path) is False and (fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.jpeg') or fname.endswith('.jfif')):

                is_live = True
                if "fake" in src_path:
                    is_live = False
                    n_fake += 1
                    label = 1
                elif "real" in src_path:
                    is_live = True
                    n_live += 1
                    label = 0
                else:
                    continue

                pre_time = time.time()

                img = cv2.imread(src_path)
                face_bbox, kps , lmk = get_landmarks(img)
                if len(face_bbox) == 0:
                    print('No faces')
                    continue

                crop_scale = [[7], [1.2]]
                mfs_result = []
                for idx in range(len(crop_scale)):
                    scale = crop_scale[idx][0]
                    cropMfs = get_cropped(img, face_bbox, scale=scale)
                    mfs_result.append(cropMfs)

                img_con = cv2.hconcat(mfs_result)
                img_t = torch.Tensor(img_con.transpose(2, 0, 1)/255)
                img_t = img_t.unsqueeze(0)
                outputs = net(img_t)
                # outputs = outputs[:, 0]
                # score = outputs.detach().cpu().numpy()
                # score = 1 - score[0]
                out = outputs[0]
                score = out.detach().cpu().numpy()
                score = 1 - score[0][0]
                sdpth = outputs[1].squeeze(1)
                show_tensor_as_image(sdpth, "depth")

                proc_time = time.time() - pre_time

                if score == -1:
                    print("{} : {} : {:.2f}s : noface".format(src_path, score, proc_time))
                    n_det_err += 1
                elif score > thresh:
                    print("{} : {} : {:.2f}s : Fake".format(src_path, score, proc_time))
                    if is_live is True:
                        n_live_err += 1
                    if is_live is True and '#testset' not in src_path:
                        head, fname = ntpath.split(src_path)
                        real_path = r'D:\lhm_work\face_liveness\DATASET\nizar_data\_error\check\real_hard'
                        dst_path = os.path.join(real_path, fname)
                        # if src_path != dst_path:
                        #     copyfile(src_path, dst_path)
                else:
                    print("{} : {} : {:.2f}s : Live".format(src_path, score, proc_time))
                    if is_live is False:
                        n_fake_err += 1
                    if is_live is False and '#testset' not in src_path:
                        head, fname = ntpath.split(src_path)
                        fake_path = r'D:\lhm_work\face_liveness\DATASET\nizar_data\_error\check\fake_hard'
                        dst_path = os.path.join(fake_path, fname)
                        # if src_path != dst_path:
                        #     copyfile(src_path, dst_path)

                scores_list.append("{} {}\n".format(score, label))

        n_total = n_live + n_fake
        if n_live > 0:
            type1_err = n_live_err * 100 / n_live
        else:
            type1_err = 0
        if n_fake > 0:
            type2_err = n_fake_err * 100 / n_fake
        else:
            type2_err = 0

        with open('score.txt', 'w') as file:
            file.writelines(scores_list)

        # BPCER (Bona Fide Presentation Classification Error Rate) for APCER = 0.01 (Attack Presentation Classification Error Rate)
        # BPCER, mythresh = performances_val_0('score.txt')
        # print(f'BPCER={BPCER} : Thresh = {mythresh}')
        #
        test_ACC, fpr, FRR, HTER, auc_test, test_err, val_threshold = performances_val('score.txt')
        print("val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}".format(test_ACC, HTER, auc_test, test_err, test_ACC))

        # print("Total {} : Live {} : Fake {} : n_live_err {} : n_fake_err {} : n_det_err {} : BPCER {:.5f} : {}"
        #       .format(n_total, n_live, n_fake, n_live_err, n_fake_err, n_det_err, BPCER, mythresh))




if __name__ == "__main__":
    test()

# vim: ts=4 sw=4 sts=4 expandtab
