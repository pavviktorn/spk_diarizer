
import cv2
import numpy as np
import os
import time
import ntpath
import warnings
from data.util import get_filepaths, get_landmarks, get_quality
from decord import VideoReader
from decord import cpu, gpu

warnings.filterwarnings('ignore')

FAKE_LABEL = 2
FAKE_NAME = 'df'
TASKS = [
    'web_deeplivecam01',
    'web_deeplivecam02',
]
in_root = "/datasets/datasets"
in_path = "/datasets/datasets/from_nizar1"
out_root = "/datasets/newout"
# in_root = "D:/zLiveness_data"
# out_root = "D:/zLiveness_data/out"

if not os.path.exists(out_root):
    os.makedirs(out_root, exist_ok=True)

NUM_FRAMES = 32

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

    return img, left_top_y, left_top_x, right_bottom_x, right_bottom_y


def convert_images(task):
    input_path = os.path.join(in_path, task, 'fake').replace('\\', '/')

    fake_path = os.path.join(out_root, 'meta_full_crop', FAKE_NAME)
    if not os.path.exists(fake_path):
        os.makedirs(fake_path, exist_ok=True)
    fake_path = os.path.join(fake_path, f"{task}.txt")

    real_path = os.path.join(out_root, 'meta_full_crop', 'real')
    if not os.path.exists(real_path):
        os.makedirs(real_path, exist_ok=True)
    real_path = os.path.join(real_path, f"{task}.txt")

    f_fake = open(fake_path, 'w')
    f_real = open(real_path, 'w')

    idx = 0
    idx_real = 0
    pre_time = time.time()

    full_file_paths = get_filepaths(input_path)
    for src_path in full_file_paths:
        head, fname = ntpath.split(src_path)

        src_path = src_path.replace('\\', '/')
        if '/fake' in src_path:
            label = FAKE_LABEL
        elif '/real' in src_path:
            label = 0
        else:
            continue

        save_path = head.replace(in_root, out_root).replace('\\', '/')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(src_path) is False and (fname.endswith('.jpg') or fname.endswith('.jpeg') or fname.endswith('.png') or fname.endswith('.bmp') or fname.endswith('.jfif')):

            try:
                stream = open(src_path, "rb")
                bytes = bytearray(stream.read())
                stream.close()
                numpyarray = np.asarray(bytes, dtype=np.uint8)
                img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                bbox, kps, landmarks = get_landmarks(img)
                if len(bbox) == 0:
                    print('No faces in {}'.format(src_path))
                    continue

                cropped, left_top_y, left_top_x, right_bottom_x, right_bottom_y = get_cropped(img, bbox, 7.0)

                bbox[0] = bbox[0] - left_top_x
                bbox[2] = bbox[2] - left_top_x
                bbox[1] = bbox[1] - left_top_y
                bbox[3] = bbox[3] - left_top_y
                bbox_str = f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}"

                ld_str = ''
                src_h, src_w, _ = np.shape(cropped)
                for i in range(landmarks.shape[0]):
                    xy = landmarks[i]
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
                    landmarks[i][0], landmarks[i][1] = xy[0], xy[1]
                    ld_str += f"{landmarks[i][0]}_{landmarks[i][1]}:"

                ld_str = ld_str[:-1]

                file_name_without_ext, ext = os.path.splitext(fname)
                bin_path = os.path.join(save_path, file_name_without_ext+'.jpg')
                # np.savez(bin_path, bbox=bbox, kps=kps, landmarks=landmarks)
                if not os.path.exists(bin_path):
                    cv2.imwrite(bin_path, cropped)

                t_path = src_path.replace('/fake/', '/real/')
                stream = open(t_path, "rb")
                bytes = bytearray(stream.read())
                stream.close()
                numpyarray = np.asarray(bytes, dtype=np.uint8)
                t_img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                t_cropped = t_img[left_top_y: right_bottom_y + 1, left_top_x: right_bottom_x + 1]

                t_save_path = save_path.replace('/fake', '/real')
                if not os.path.exists(t_save_path):
                    os.makedirs(t_save_path, exist_ok=True)
                bin_path = os.path.join(t_save_path, file_name_without_ext+'.jpg')
                if not os.path.exists(bin_path):
                    cv2.imwrite(bin_path, t_cropped)

                if label == 0:
                    score = get_quality(img, kps)
                else:
                    score = -1.0

                score = "{:.4f}".format(score)
                src_path = src_path.replace(in_root, '')
                t_path = t_path.replace(in_root, '')

                line = f"{label},f,{score},{src_path},{t_path},{bbox_str},{ld_str}\n"

                if label == 0:
                    f_real.write(line)
                else:
                    f_fake.write(line)

                if idx % 100 == 0:
                    cur_time = time.time()
                    print('time:', cur_time - pre_time, ' label:', src_path, ', ', idx)
                    pre_time = cur_time
                idx += 1
            except:
                print("error:" + src_path)
                continue

        elif os.path.isdir(src_path) is False and (fname.endswith('.mp4') or fname.endswith('.avi') or fname.endswith('.3gp')):

            try:
                file_name_without_ext, ext = os.path.splitext(fname)
                save_path = os.path.join(save_path, file_name_without_ext)
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)

                vr = VideoReader(src_path, ctx=cpu(0))
                frame_count_video = len(vr)
                if frame_count_video > NUM_FRAMES:
                    frame_idxs = np.linspace(0, frame_count_video - 1, NUM_FRAMES, endpoint=False, dtype=int)
                else:
                    frame_idxs = [0] * frame_count_video
                    for i in range(frame_count_video):
                        frame_idxs[i] = i
                if frame_count_video > 1000:
                    skip_temp = 10
                elif frame_count_video > 500:
                    skip_temp = 5
                elif frame_count_video > 200:
                    skip_temp = 2
                else:
                    skip_temp = 1
                for index in range(frame_count_video):
                    # if index not in frame_idxs:
                    #     continue
                    if index % skip_temp > 0:
                        continue

                    frame = vr[index]
                    img = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR)

                    bbox, kps, landmarks = get_landmarks(img)
                    if len(bbox) == 0:
                        print('No faces in {}'.format(src_path))
                        continue

                    cropped, left_top_y, left_top_x = get_cropped(img, bbox, 7.0)

                    bbox[0] = bbox[0] - left_top_x
                    bbox[2] = bbox[2] - left_top_x
                    bbox[1] = bbox[1] - left_top_y
                    bbox[3] = bbox[3] - left_top_y
                    bbox_str = f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}"

                    ld_str = ''
                    src_h, src_w, _ = np.shape(cropped)
                    for i in range(landmarks.shape[0]):
                        xy = landmarks[i]
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
                        landmarks[i][0], landmarks[i][1] = xy[0], xy[1]
                        ld_str += f"{landmarks[i][0]}_{landmarks[i][1]}:"

                    ld_str = ld_str[:-1]

                    bin_path = os.path.join(save_path, f"{index}.jpg")
                    # np.savez(bin_path, bbox=bbox, kps=kps, landmarks=landmarks)
                    if not os.path.exists(bin_path):
                        cv2.imwrite(bin_path, cropped)

                    if label == 0:
                        score = get_quality(img, kps)
                    else:
                        score = -1.0

                    score = "{:.4f}".format(score)
                    src_path = src_path.replace(in_root, '')

                    line = f"{label},v,{score},{src_path}:{index},:,{bbox_str},{ld_str}\n"
                    if label == 0:
                        f_real.write(line)
                    else:
                        f_fake.write(line)

                    if idx % 100 == 0:
                        cur_time = time.time()
                        print('time:', cur_time - pre_time, ' label:', src_path, ', ', idx)
                        pre_time = cur_time
                    idx += 1
            except:
                print("error:" + src_path)
                continue

    print('The last idx:', idx)

    f_fake.close()
    f_real.close()

    return


for task in TASKS:
    convert_images(task)

# for dataset in os.listdir(in_path):
#     convert_images(dataset)