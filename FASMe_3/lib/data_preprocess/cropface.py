#!/usr/bin/env python3
import argparse

import os
import cv2
import torch
import random
import numpy as np

from functools import lru_cache
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from .augmentor import rand_range


MEAN_FACE = np.array([
    [-0.17607, -0.172844],  # left eye pupil
    [0.1736, -0.17356],  # right eye pupil
    [-0.00182, 0.0357164],  # nose tip
    [-0.14617, 0.20185],  # left mouth corner
    [0.14496, 0.19943],  # right mouth corner
])


def get_mean_face(mf, face_width, canvas_size):
    ratio = face_width / (canvas_size * 0.34967)
    left_eye_pupil_y = mf[0][1]
    ratioy = (left_eye_pupil_y * ratio + 0.5) * (1 + 1.42)
    mf[:, 0] = (mf[:, 0] * ratio + 0.5) * canvas_size
    mf[:, 1] = (mf[:, 1] * ratio + 0.5) * canvas_size / ratioy

    return mf


def get_align_transform(lm, mf):
    mx = mf[:, 0].mean()
    my = mf[:, 1].mean()
    dmx = lm[:, 0].mean()
    dmy = lm[:, 1].mean()

    ux = mf[:, 0] - mx
    uy = mf[:, 1] - my
    dux = lm[:, 0] - dmx
    duy = lm[:, 1] - dmy
    c1 = (ux * dux + uy * duy).sum()
    c2 = (ux * duy - uy * dux).sum()
    c3 = (dux**2 + duy**2).sum()
    a = c1 / c3
    b = c2 / c3

    kx, ky = 1, 1

    transform = np.zeros((2, 3))
    transform[0][0] = kx * a
    transform[0][1] = kx * b
    transform[0][2] = mx - kx * a * dmx - kx * b * dmy
    transform[1][0] = -ky * b
    transform[1][1] = ky * a
    transform[1][2] = my - ky * a * dmy + ky * b * dmx
    return transform


def align_5p(
        images, ld, face_width, canvas_size,
        translation=[0, 0], rotation=0,
        scale=1, sa=1, sb=1
):
    '''crop face with landmark.

    images: input images. -> ndarray.
    ld: face landmark of input images. -> ndarray; shape -> (5, 2)
    face_width: face width ratio of the cropped images. -> float
    canvas_size: shape of the cropped face.
    return list of cropped images. -> list(ndarray)
    '''
    # nose_tip = ld[30]
    # left_eye = np.mean(ld[36:42], axis=0).astype('int')
    # right_eye = np.mean(ld[42:48], axis=0).astype('int')
    # left_mouth, right_mouth = ld[48], ld[54]
    nose_tip = ld[86]
    left_eye = np.mean(ld[33:43], axis=0).astype('int')
    right_eye = np.mean(ld[87:97], axis=0).astype('int')
    left_mouth, right_mouth = ld[52], ld[61]

    lm = np.array([left_eye, right_eye, nose_tip, left_mouth, right_mouth])

    mf = MEAN_FACE * scale
    mf = get_mean_face(mf, face_width, canvas_size)

    M1 = np.eye(3)
    M1[:2] = get_align_transform(lm, mf)

    M2 = np.eye(3)
    M2[:2] = cv2.getRotationMatrix2D((canvas_size/2, canvas_size/2), rotation, 1)

    def stretch(va, vb, s):
        m = (va+vb)*0.5
        d = (va-vb)*0.5
        va[:] = m+d*s
        vb[:] = m-d*s

    mf = mf[[0, 1, 3, 4]].astype(np.float32)
    mf2 = mf.copy()
    stretch(mf2[0], mf2[1], sa)
    stretch(mf2[2], mf2[3], 1.0/sa)
    stretch(mf2[0], mf2[2], sb)
    stretch(mf2[1], mf2[3], 1.0/sb)

    mf2 += np.array(translation)

    M3 = cv2.getPerspectiveTransform(mf, mf2)

    M = M3.dot(M2).dot(M1)

    dshape = (canvas_size, canvas_size)
    images = [cv2.warpPerspective(img, M, dshape) for img in images]

    # warp landmark.
    ld = np.array(ld)
    ld = ld.dot(M[:, :2].T) + M[:, 2].T

    return images, ld[:, :2]


def get_align5p(images, ld, rng, config, training=False):

    config = config['crop_face']

    images, landmark = align_5p(
        images, ld=ld,
        face_width=config['face_width'], canvas_size=config['output_size'],
        scale=(rng.randn()*0.1+0.9 if training else config['scale']),
        translation=([
            rand_range(rng, -25, 25), rand_range(rng, -25, 25)
        ] if training else [0, 0]),
        rotation=(30*rand_range(rng, -1, 1)**3 if training else 0),
        sa=(rand_range(rng, .97, 1.03) if training and rng.rand() > 0.8 else 1),
        sb=(rand_range(rng, .97, 1.03) if training and rng.rand() > 0.8 else 1),
    )

    return images, landmark


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
        if len(org_img.shape) == 3:
            src_h, src_w, _ = np.shape(org_img)
        else:
            src_h, src_w = np.shape(org_img)
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, face_bbox, scale)

        img = org_img[left_top_y: right_bottom_y+1, left_top_x: right_bottom_x+1]
        dst_img = cv2.resize(img, (out_w, out_h))

    return dst_img
