from __future__ import division
import numpy as np
import cv2
import onnxruntime
from insightface.app import FaceAnalysis
import math


app = FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))


def get_face(frame):
    if len(frame.shape) < 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    new_height = 640
    new_width = 640
    exp_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    exp_scale = 1.0
    if frame.shape[0] >= frame.shape[1] and 480 < frame.shape[0]:
        half_h = 640
        half_w = int(frame.shape[1] * half_h / frame.shape[0])
        exp_scale = half_h / frame.shape[0]
        half_img = cv2.resize(frame, (half_w, half_h), interpolation=cv2.INTER_AREA)
        x_pos = int(new_height / 2 - half_img.shape[0] / 2)
        y_pos = int(new_width / 2 - half_img.shape[1] / 2)
        exp_img[x_pos: x_pos + half_img.shape[0], y_pos: y_pos + half_img.shape[1], :] = half_img
        faces = app.get(exp_img)
        if len(faces) == 0:
            half_h = 320
            half_w = int(frame.shape[1] * half_h / frame.shape[0])
            exp_scale = half_h / frame.shape[0]
            half_img = cv2.resize(frame, (half_w, half_h), interpolation=cv2.INTER_AREA)
            x_pos = int(new_height / 2 - half_img.shape[0] / 2)
            y_pos = int(new_width / 2 - half_img.shape[1] / 2)
            exp_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            exp_img[x_pos: x_pos + half_img.shape[0], y_pos: y_pos + half_img.shape[1], :] = half_img
            faces = app.get(exp_img)
            if len(faces) == 0:
                return [], []

    elif frame.shape[0] <= frame.shape[1] and 480 < frame.shape[1]:

        half_w = 640
        half_h = int(frame.shape[0] * half_w / frame.shape[1])
        exp_scale = half_w / frame.shape[1]
        half_img = cv2.resize(frame, (half_w, half_h), interpolation=cv2.INTER_AREA)
        x_pos = int(new_height / 2 - half_img.shape[0] / 2)
        y_pos = int(new_width / 2 - half_img.shape[1] / 2)
        exp_img[x_pos: x_pos + half_img.shape[0], y_pos: y_pos + half_img.shape[1], :] = half_img

        faces = app.get(exp_img)
        if len(faces) == 0:
            half_w = 320
            half_h = int(frame.shape[0] * half_w / frame.shape[1])
            exp_scale = half_w / frame.shape[1]
            half_img = cv2.resize(frame, (half_w, half_h), interpolation=cv2.INTER_AREA)
            x_pos = int(new_height / 2 - half_img.shape[0] / 2)
            y_pos = int(new_width / 2 - half_img.shape[1] / 2)
            exp_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            exp_img[x_pos: x_pos + half_img.shape[0], y_pos: y_pos + half_img.shape[1], :] = half_img
            faces = app.get(exp_img)
            if len(faces) == 0:
                return [], []

    else:
        exp_scale = 1.0
        x_pos = int(new_height / 2 - frame.shape[0] / 2)
        y_pos = int(new_width / 2 - frame.shape[1] / 2)
        exp_img[x_pos: x_pos + frame.shape[0], y_pos: y_pos + frame.shape[1], :] = frame
        faces = app.get(exp_img)
        if len(faces) == 0:
            return [], []

    # get biggest face
    max_idx = 0
    max_area = 0
    for face_idx in range(0, len(faces)):
        face = faces[face_idx]
        area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        if area > max_area:
            max_idx = face_idx
            max_area = area

    face = faces[max_idx]

    bbox = face.bbox
    bbox[0] = int((bbox[0] - y_pos) / exp_scale)
    bbox[1] = int((bbox[1] - x_pos) / exp_scale)
    bbox[2] = int((bbox[2] - y_pos) / exp_scale)
    bbox[3] = int((bbox[3] - x_pos) / exp_scale)

    kps = face.kps
    for i in range(5):
        kps[i][0] = int((kps[i][0] - y_pos) / exp_scale)
        kps[i][1] = int((kps[i][1] - x_pos) / exp_scale)

    return bbox, kps


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


def get_cropped(org_img, face_bbox, scale, out_w=256, out_h=256, crop=True):

    if not crop:
        dst_img = cv2.resize(org_img, (out_w, out_h))
    else:
        src_h, src_w, _ = np.shape(org_img)
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, face_bbox, scale)

        img = org_img[left_top_y: right_bottom_y+1, left_top_x: right_bottom_x+1]
        dst_img = cv2.resize(img, (out_w, out_h))

    return dst_img


def fit_img(img, out_w=256, out_h=256):
    src_h = img.shape[0]
    src_w = img.shape[1]
    exp_img = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
    if src_h > src_w:
        new_h = out_h
        new_w = int(src_w * out_h / src_h)
        exp_scale = new_h / src_h
        cropped = cv2.resize(img, (new_w, new_h))
        x_pos = int(out_h / 2 - cropped.shape[0] / 2)
        y_pos = int(out_w / 2 - cropped.shape[1] / 2)
        exp_img[x_pos: x_pos + cropped.shape[0], y_pos: y_pos + cropped.shape[1], :] = cropped
    else:
        new_w = out_w
        new_h = int(src_h * new_w / src_w)
        exp_scale = new_w / src_w
        cropped = cv2.resize(img, (new_w, new_h))
        x_pos = int(out_h / 2 - cropped.shape[0] / 2)
        y_pos = int(out_w / 2 - cropped.shape[1] / 2)
        exp_img[x_pos: x_pos + cropped.shape[0], y_pos: y_pos + cropped.shape[1], :] = cropped

    return exp_img


def crop(img, bbox, top, bottom, left, right):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if abs(bottom - top) < 0.2 * h:
        bottom = int(bottom + 0.2 * h)
        top = int(top - 0.2 * h)

    if abs(right - left) < 0.2 * w:
        right = int(right + 0.2 * w)
        left = int(left - 0.2 * w)

    if left < 0: left = 0
    if right >= img.shape[1]: right = img.shape[1] - 1
    if bottom >= img.shape[0]: bottom = img.shape[0] - 1
    if top < 0: top = 0

    return img[int(top):int(bottom), int(left):int(right)]


def get_parts(org_img, bbox, kps, out_w=256, out_h=256):

    result = []

    # kps = np.zeros((5, 2)).astype(int)
    # kps[0] = lmk[38]
    # kps[1] = lmk[88]
    # kps[2] = lmk[86]
    # kps[3] = lmk[52]
    # kps[4] = lmk[61]

    dist_eyes = abs(kps[1][0] - kps[0][0])

    # left eye
    left = kps[0][0] - dist_eyes//2
    if left < 0: left = 0
    right = kps[0][0] + dist_eyes//2
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    bottom = kps[0][1] + dist_eyes//3
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    top = bottom - (right - left)
    if top < 0: top = 0
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    result.append(cropped)
    # cv2.imwrite('0_left_eye.jpg', cropped)

    # right eye
    left = kps[1][0] - dist_eyes//2
    if left < 0: left = 0
    right = kps[1][0] + dist_eyes//2
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    bottom = kps[1][1] + dist_eyes//3
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    top = bottom - (right - left)
    if top < 0: top = 0
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    cropped = cv2.flip(cropped, 1)
    result.append(cropped)
    # cv2.imwrite('1_right_eye.jpg', cropped)

    # forehead
    left = bbox[0]
    if left < 0: left = 0
    right = bbox[2]
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    top = bbox[1] - (bbox[3] - bbox[1]) // 8
    if top < 0: top = 0
    bottom = bbox[1] + (bbox[3] - bbox[1]) // 4
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    result.append(cropped)
    # cv2.imwrite('2_forehead.jpg', cropped)

    # left ear
    left = bbox[0] - (bbox[2] - bbox[0]) // 6
    if left < 0: left = 0
    right = bbox[0] + (bbox[2] - bbox[0]) // 6
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    top = bbox[1]
    if top < 0: top = 0
    bottom = bbox[3]
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    result.append(cropped)
    # cv2.imwrite('3_left_ear.jpg', cropped)

    # right ear
    left = bbox[2] - (bbox[2] - bbox[0]) // 6
    if left < 0: left = 0
    right = bbox[2] + (bbox[2] - bbox[0]) // 6
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    top = bbox[1]
    if top < 0: top = 0
    bottom = bbox[3]
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    cropped = cv2.flip(cropped, 1)
    result.append(cropped)
    # cv2.imwrite('4_right_ear.jpg', cropped)

    # chin
    left = bbox[0]
    if left < 0: left = 0
    right = bbox[2]
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    top = int(bbox[3] - (bbox[3] - bbox[1]) / 2.5)
    if top < 0: top = 0
    bottom = bbox[3] + (bbox[3] - bbox[1]) // 10
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    result.append(cropped)
    # cv2.imwrite('5_chin.jpg', cropped)

    return result


class FASMeONNX:
    def __init__(self, model_file=None):
        assert model_file is not None

        self.session = onnxruntime.InferenceSession(model_file, None)
        self.session.set_providers(['CPUExecutionProvider'])

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        # assert len(self.output_names)==1
        self.output_shape = outputs[0].shape


    def get(self, image_path):
        img = cv2.imread(image_path)
        face_bbox, kps = get_face(img)
        if len(face_bbox) == 0:
            # print('No faces')
            return -1

        crop_scale = [[7], [1.2]]
        mfs_result = []
        for idx in range(len(crop_scale)):
            scale = crop_scale[idx][0]
            cropMfs = get_cropped(img, face_bbox, scale=scale)
            mfs_result.append(cropMfs)

        img = cv2.hconcat(mfs_result)

        input_size = self.input_size
        blob = cv2.dnn.blobFromImage(img, 1.0/255, input_size, (0, 0, 0), swapRB=False)
        # x = self.session.run(self.output_names, {self.input_name: blob})[0]
        x = self.session.run(self.output_names, {self.input_name: blob})

        return x #1 - x[0][0][0], x[1].squeeze()


