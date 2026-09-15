#!/usr/bin/env python3
import argparse

import os
import cv2
import random
import numpy as np

from .augmentor import elastic_transform
from .cropface import get_align5p, get_cropped
from .partial_swap_mask import cal_dssim, _sliding_bbox, generate_partial_swap_mask

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def alpha_blending_func(srcRgb, targetRgb, mask):
    '''Alpha blending function

    Argument:

    srcRgb: background image. type -> ndarray.
    targetRgb: target image. type -> ndarray.
    mask: blending mask. type -> ndarray.
    '''
    return (mask * targetRgb + (1 - mask) * srcRgb).astype(np.uint8)


def poisson_blending_func(srcRgb, targetRgb, mask, points):
    '''Poisson blending function

    Argument:

    srcRgb: background image. type -> ndarray.
    targetRgb: target image. type -> ndarray.
    mask: blending mask. type -> ndarray.
    points: blending position. type -> ndarray.
    '''

    points = np.array(points)
    points = points.reshape((-1, 2))
    center_x = (max(points[:, 0]) + min(points[:, 0])) / 2
    center_y = (max(points[:, 1]) + min(points[:, 1])) / 2
    # center = (int(center_y), int(center_x)) # This was a bug zzzzzz
    center = (int(center_x), int(center_y))

    return cv2.seamlessClone(targetRgb, srcRgb, mask, center, cv2.NORMAL_CLONE)


def global_facial_swap(srcRgb, targetRgb, landmark, training=False):
    '''Global Facial Swap.

    Argument:

    srcRgb: background image. type -> ndarray.
    targetRgb: target image. type -> ndarray.
    landmark: position of face landmarks.
    '''

    mask = np.zeros(srcRgb.shape, dtype=np.uint8)

    points = cv2.convexHull(
        np.array(landmark).astype('float32')
    )
    corners = np.expand_dims(points, axis=0).astype(np.int32)
    cv2.fillPoly(mask, corners, (255,)*3)

    if training:
        rng = np.random

        if rng.rand() > 0.5:
            # cv2.imwrite('elastic_transform_before.jpg', mask)
            mask = elastic_transform(
                mask, random.randint(300, 500), mask.shape[0] * 0.08, 0
            )
            # cv2.imwrite('elastic_transform_after.jpg', mask)

    # gaussianblur.
    blured = cv2.GaussianBlur(mask, (5, 5), 3).astype('float32')

    return mask, blured, points


def partial_facial_swap(srcRgb, targetRgb, landmark, sliding_win):
    '''Partial Facial Swap.

    Argument:

    srcRgb: background image. type -> ndarray.
    targetRgb: target image. type -> ndarray.
    sliding_win: size of sliding window.
    '''

    global_swap_mask, blured, _ = global_facial_swap(srcRgb, targetRgb, landmark)

    partial_swap_mask, points = generate_partial_swap_mask(
        targetRgb, srcRgb, global_swap_mask, landmark, sliding_win
    )
    blured = cv2.GaussianBlur(partial_swap_mask, (5, 5), 3).astype('float32')

    return partial_swap_mask, blured, points


def get_partial_bbox_gt(cutMfs, cutSrc, sliding_win):
    dssim = cal_dssim(cutMfs, cutSrc)
    if len(dssim[dssim > 1e-3]) > 40**2*3:
        return _sliding_bbox(dssim, sliding_win[0], sliding_win[1])
    return None


def get_mfs_results(mfs_fake, srcRgb, face_bbox, config, sliding_win=[256,256]):
    cropMfs_list = []
    bbox_list = []

    swap_level = 'global' if sliding_win == srcRgb.shape[:2] else 'partial'

    config = config['crop_face']
    crop_scale = config['crop_scale']

    for idx in range(len(crop_scale)):
        bbox = [[0, 0, 256, 256]]
        scale = crop_scale[idx][0]
        cropMfs = get_cropped(mfs_fake, face_bbox, scale=scale)
        cropSrc = get_cropped(srcRgb, face_bbox, scale=scale)

        if swap_level == 'partial':
            partial_bbox = get_partial_bbox_gt(cropMfs, cropSrc, sliding_win)
            if partial_bbox is not None and partial_bbox != [0,0,0,0] and partial_bbox != [0,0,256,256]: #zzzzz
                bbox.append(partial_bbox)

        cropMfs_list.append(cropMfs)
        bbox_list.append(np.array(bbox))

    return cropMfs_list, bbox_list


def multi_scale_facial_swap(
        srcRgb, targetRgb, landmark, config,
        sliding_win, face_bbox, blending_type='poisson', training=False):

    '''Multi-scale Facial Swap function.

    Argument.

    srcRgb: source image. type -> ndarray
    targetRgb: target image. type -> ndarray
    landmark: position of face landmarks.
    blending_type: image blending function. (poisson or alpha).
    '''

    assert min(sliding_win) > 0
    assert blending_type == 'poisson' or blending_type == 'alpha'

    bbox = [[0, 0, 256, 256]]

    if srcRgb.shape != targetRgb.shape:
        h, w = targetRgb.shape[:2]
        srcRgb = cv2.resize(srcRgb, (w, h))

    assert srcRgb.shape == targetRgb.shape

    swap_level = 'global' if sliding_win == targetRgb.shape[:2] else 'partial'

    # generate blending mask.

    # global facial swap. (the size of sliding window is equal to image shape)
    if swap_level == 'global':

        mask, blured, points = global_facial_swap(
            srcRgb, targetRgb, landmark, training)

    # partial facial swap.
    else:
        mask, blured, points = partial_facial_swap(
            srcRgb, targetRgb, landmark, sliding_win)

    # blending image with blending mask.

    # cv2.imwrite('blending_before_mask.jpg', mask)
    # cv2.imwrite('blending_before_blured.jpg', blured)

    if blending_type == 'poisson':
        try:
            mfs_fake = poisson_blending_func(srcRgb, targetRgb, mask, points)
        except Exception:
            mfs_fake = alpha_blending_func(srcRgb, targetRgb, blured/255.)
    else:
        mfs_fake = alpha_blending_func(srcRgb, targetRgb, blured/255.)

    # cv2.imwrite('blending_after.jpg', mfs_fake)

    cropMfs_list, bbox_list = get_mfs_results(mfs_fake, srcRgb, face_bbox, config, sliding_win)

    # images, landmark = get_align5p(
    #     [mfs_fake, srcRgb], landmark, np.random, config, training)
    #
    # cropMfs, cropSrc = images
    #
    # if swap_level == 'partial':
    #     partial_bbox = get_partial_bbox_gt(cropMfs, cropSrc, sliding_win)
    #     # if partial_bbox is None:
    #     #     return None, 'partial swap err.'
    #     # bbox.append(partial_bbox)
    #     if partial_bbox is not None and partial_bbox != [0,0,0,0]: #zzzzz
    #         bbox.append(partial_bbox)
    #
    # return cropMfs, np.array(bbox)

    return cropMfs_list, bbox_list


def draw_bounding_box(img, bboxs):
    '''Draw bounding box on the image.

    Argument:
    img: input image.
    bbox: input bounding box.
    '''

    for bbox in bboxs:
        boxed_img = cv2.rectangle(
            img, (bbox[1], bbox[0]), (bbox[3], bbox[2]),
            (255, 255, 255), 2
        )

    return boxed_img


# vim: ts=4 sw=4 sts=4 expandtab
