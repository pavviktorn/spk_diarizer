
import os
import numpy as np
import random
import time
import ntpath
import cv2
from decord import VideoReader
from decord import cpu, gpu


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

    return file_paths  # Self-explanatory.


info_in_root = '/datasets/newout/meta_full'
info_in_path = '/datasets/newout/meta_full/df'
info_out_root = '/datasets/newout/meta_full_crop'
img_read_root = '/datasets/datasets'
img_save_root = '/datasets/newout'
# info_in_root = 'D:/zLiveness_data/out/meta_test'
# info_out_root = 'D:/zLiveness_data/out/meta_modif'
# img_read_root = 'D:/zLiveness_data'
# img_save_root = 'D:/zLiveness_data/out'


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


def crop_face(img, elem):

    class_label = int(elem[0])
    video_or_frame = elem[1]
    quality = elem[2]
    org_path = elem[3]
    src_path = elem[4]

    bbox = np.zeros((4,)).astype(int)
    be = elem[5].split(':')
    bbox[0], bbox[1], bbox[2], bbox[3] = int(be[0]), int(be[1]), int(be[2]), int(be[3]),

    cropped, left_top_y, left_top_x = get_cropped(img, bbox, 7.0)

    bbox[0] = bbox[0] - left_top_x
    bbox[2] = bbox[2] - left_top_x
    bbox[1] = bbox[1] - left_top_y
    bbox[3] = bbox[3] - left_top_y
    bbox_str = f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}"

    landmark = np.zeros((106, 2)).astype(int)
    le = elem[6].split(':')
    ld_str = ''

    # tim = cropped.copy()
    src_h, src_w, _ = np.shape(cropped)
    for i in range(landmark.shape[0]):
        xy = le[i].split('_')
        xy[0] = int(xy[0]) - left_top_x
        xy[1] = int(xy[1]) - left_top_y
        if xy[0] < 0:
            xy[0] = 0
        if xy[0] >= src_w:
            xy[0] = src_w - 1
        if xy[1] < 0:
            xy[1] = 0
        if xy[1] >= src_h:
            xy[1] = src_h - 1
        landmark[i][0], landmark[i][1] = xy[0], xy[1]
        ld_str += f"{landmark[i][0]}_{landmark[i][1]}:"

        # p = tuple(landmark[i])
        # cv2.circle(tim, p, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # cv2.imwrite('new_cropped_ld.jpg', tim)

    ld_str = ld_str[:-1]

    out_info = f"{class_label},{video_or_frame},{quality},{org_path},{src_path},{bbox_str},{ld_str}\n"

    return cropped, out_info


def convert_file(path):
    path = path.replace('\\', '/')

    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    out_path = path.replace(info_in_root, info_out_root)
    head, fname = ntpath.split(out_path)
    if not os.path.exists(head):
        os.makedirs(head, exist_ok=True)

    f_a = open(out_path, 'w')

    info_size = len(lines)
    idx = 0
    prev_vname = ''
    pre_time = time.time()
    while idx < info_size:
        info = lines[idx]

        elem = info.split(',')
        video_or_frame = elem[1]
        org_path = elem[3]

        if video_or_frame == 'f':
            img_path = img_read_root + org_path
            frame = 0
            img = cv2.imread(img_path)
            cropped, out_info = crop_face(img, elem)
            img_path = img_save_root + org_path
            pp, fn = ntpath.split(img_path)
            if not os.path.exists(pp):
                os.makedirs(pp, exist_ok=True)
            fn = fn.split('.')[0] + '.jpg'
            img_path = os.path.join(pp, fn)
            cv2.imwrite(img_path, cropped)
            f_a.write(out_info)
            idx += 1
            if idx % 100 == 0:
                cur_time = time.time()
                print(f'{idx} : {org_path}, {cur_time - pre_time} s')
                pre_time = cur_time
        else:
            pp = elem[3].split(':')
            vname = pp[0]
            if prev_vname != vname:
                prev_vname = vname
            info_list = []
            while prev_vname == vname:
                info_list.append(info)
                idx += 1
                if idx >= info_size:
                    break
                info = lines[idx]
                elem = info.split(',')
                video_or_frame = elem[1]

                if idx % 100 == 0:
                    cur_time = time.time()
                    print(f'{idx} : {elem[3]}, {cur_time - pre_time} s')
                    pre_time = cur_time

                if video_or_frame == 'v':
                    pp = elem[3].split(':')
                    vname = pp[0]
                else:
                    break

            frame_idxs = []
            for info in info_list:
                elem = info.split(',')
                video_or_frame = elem[1]
                if video_or_frame != 'v':
                    print('error info str')
                    continue
                frame = int(elem[3].split(':')[1])
                frame_idxs.append(frame)

            video_path = img_read_root + prev_vname

            rot_angle = video_path.split('/')[-2]

            vr = VideoReader(video_path, ctx=cpu(0))
            frame_count_video = len(vr)
            idx_info = 0
            for iii in range(frame_count_video):
                if iii not in frame_idxs:
                    continue

                try:
                    frame = vr[iii]
                    img = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR)

                    if rot_angle == '90':
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rot_angle == '180':
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    elif rot_angle == '270':
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                    info = info_list[idx_info]
                    idx_info += 1
                    elem = info.split(',')
                    pp = elem[3].split(':')
                    vpath = pp[0]
                    frame_read = int(pp[1])
                    if iii != frame_read:
                        print('error frame info')
                        continue

                    cropped, out_info = crop_face(img, elem)

                    img_path = img_save_root + vpath
                    pp, fn = ntpath.split(img_path)
                    fn = fn.split('.')[0]
                    img_path = os.path.join(pp, fn)
                    if not os.path.exists(img_path):
                        os.makedirs(img_path, exist_ok=True)
                    img_path = img_path + f'/{iii}.jpg'
                    cv2.imwrite(img_path, cropped)

                    f_a.write(out_info)
                except:
                    continue


    f_a.close()

    return

full_file_paths = get_filepaths(info_in_path)

for path in full_file_paths:
    print(path)
    convert_file(path)
