import os
import time
import numpy as np
import math
import cv2
import liveness
import ntpath
from shutil import copyfile
from performance import performances_val_0, performances_val

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# make dirs
model_root_path = './models'

# define model
model_path = os.path.join(model_root_path, "out.onnx")
model = liveness.FASMeONNX(model_path)


def get_filepaths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


if __name__ == '__main__':
    img_path = './images'

    full_file_paths = get_filepaths(img_path)

    thresh = 0.01

    n_fake = 0
    n_live = 0
    n_fake_err = 0
    n_live_err = 0
    n_det_err = 0

    scores_list = []

    for src_path in full_file_paths:
        head, fname = ntpath.split(src_path)
        if os.path.isdir(src_path) is False and (fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.jpeg')):

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
            score = model.get(src_path)
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
    BPCER, mythresh = performances_val_0('score.txt')
    # print(f'BPCER={BPCER} : Thresh = {mythresh}')
    #
    test_ACC, fpr, FRR, HTER, auc_test, test_err, val_threshold = performances_val('score.txt')
    print("val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}".format(test_ACC, HTER, auc_test, test_err, test_ACC))

    print("Total {} : Live {} : Fake {} : n_live_err {} : n_fake_err {} : n_det_err {} : BPCER {:.5f} : {}"
          .format(n_total, n_live, n_fake, n_live_err, n_fake_err, n_det_err, BPCER, mythresh))

