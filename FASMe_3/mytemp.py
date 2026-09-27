
import cv2
import numpy as np
import os
import time
import ntpath
import warnings
import random
from data.util import get_filepaths, get_landmarks, get_quality
from decord import VideoReader
from decord import cpu, gpu

warnings.filterwarnings('ignore')

in_root = "/datasets/newout/PAD/dataset/HiFiMask"
out_root = "/datasets/work/vLLM/data/no_delete_mids_train/3D_ATTACKS/HiFiMask"

if not os.path.exists(out_root):
    os.makedirs(out_root, exist_ok=True)

def _get_new_box(src_w, src_h, bbox, scale):
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]

    # scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y

    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2

    if left_top_x < 0:
        # right_bottom_x -= left_top_x
        left_top_x = 0

    if left_top_y < 0:
        # right_bottom_y -= left_top_y
        left_top_y = 0

    if right_bottom_x > src_w - 1:
        # left_top_x -= right_bottom_x-src_w+1
        right_bottom_x = src_w - 1

    if right_bottom_y > src_h - 1:
        # left_top_y -= right_bottom_y-src_h+1
        right_bottom_y = src_h - 1

    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def get_cropped(org_img, face_bbox, scale):

    src_h, src_w, _ = np.shape(org_img)
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, face_bbox, scale)

    img = org_img[left_top_y: right_bottom_y+1, left_top_x: right_bottom_x+1]

    return img, left_top_y, left_top_x


def convert_images():

    idx = 0
    pre_time = time.time()

    full_file_paths = get_filepaths(in_root)
    for src_path in full_file_paths:

        head, fname = ntpath.split(src_path)

        src_path = src_path.replace('\\', '/')
        save_path = head.replace(in_root, out_root).replace('\\', '/')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(src_path) is False and (fname.endswith('.jpg') or fname.endswith('.jpeg') or fname.endswith('.png') or fname.endswith('.bmp') or fname.endswith('.jfif')):

            try:
                if "_depth" in src_path or "_mask" in src_path:
                    continue

                if random.random() > 0.2:
                    continue
                idx += 1
                # if idx % 10 != 0: continue

                stream = open(src_path, "rb")
                bytes = bytearray(stream.read())
                stream.close()
                numpyarray = np.asarray(bytes, dtype=np.uint8)
                img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                bbox, kps, landmarks = get_landmarks(img)
                if len(bbox) == 0:
                    print('No faces in {}'.format(src_path))
                    continue

                cropped, left_top_y, left_top_x = get_cropped(img, bbox, 2)

                file_name_without_ext, ext = os.path.splitext(fname)
                bin_path = os.path.join(save_path, file_name_without_ext+'.jpg')
                if not os.path.exists(bin_path):
                    cv2.imwrite(bin_path, cropped)

                if idx % 100 == 0:
                    cur_time = time.time()
                    print('time:', cur_time - pre_time, ' label:', src_path, ', ', idx)
                    pre_time = cur_time
            except:
                print("error:" + src_path)
                continue

    return


convert_images()
