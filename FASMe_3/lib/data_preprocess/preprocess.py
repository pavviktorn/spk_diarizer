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

from .mfs import multi_scale_facial_swap
from .augmentor import add_noise, resize_aug, image_h_mirror
from .cropface import get_align5p, align_5p, get_cropped

from detection_layers.box_utils import match
from detection_layers import PriorBox
import math
from .digital import digital_augment
from .moire import Moire
import ntpath
from torchvision import transforms
from PIL import Image


moire = Moire()

Prior = None


def get_prior(config):
    global Prior
    if Prior is None:
        Prior = PriorBox(config['adm_det'])


def label_assign(bboxs, config, genuine=False):

    global Prior
    get_prior(config)

    labels = torch.zeros(bboxs.shape[0],)
    defaults = Prior.forward().data  # information of priors

    if genuine:
        return np.zeros(defaults.shape), np.zeros(defaults.shape[0], )

    # anchor matching by iou.

    loc_t = torch.zeros(1, defaults.shape[0], 4)
    conf_t = torch.zeros(1, defaults.shape[0])

    match(
        0.5, torch.Tensor(bboxs), defaults,
        [0.1, 0.2], labels, loc_t, conf_t, 0)

    loc_t, conf_t = np.array(loc_t)[0, ...], np.array(conf_t)[0, ...]

    if loc_t.max() > 10**5:
        return None, 'prior bbox match err. bias is inf!'

    if math.isinf(loc_t.min()):
        return loc_t, conf_t

    return loc_t, conf_t


def prepare_train_input(targetRgb, sourceRgb, depth_img, ld, fb, label, config, img_path, training=True):
    '''Prepare model input images.

    Arguments:
    targetRgb: original images or fake images.
    sourceRgb: source images.
    landmark: face landmark.
    label: deepfake labels. genuine: 0, fake: 1.
    config: deepfake config dict.
    training: return processed image with aug or not.
    '''

    rng = np.random

    images = [targetRgb, sourceRgb, depth_img]

    crop_scale = config['crop_face']['crop_scale']

    resized = False

    if training and rng.rand() >= 0.7:
        images, landmark, face_bbox = resize_aug(images, ld, fb)
        resized = True
    else:
        landmark, face_bbox = ld, fb

    # multi-scale facial swap.

    targetRgb, sourceRgb, depth_img = images

    # cv2.imwrite('targetRgb.jpg', targetRgb)
    # if isinstance(sourceRgb, np.ndarray):
    #     cv2.imwrite('srcRgb.jpg', sourceRgb)

    # if input image is fake image. generate new fake image with mfs.
    if label and isinstance(sourceRgb, np.ndarray):
        blending_type = 'poisson' if rng.rand() >= 0.5 else 'alpha'

        sel = rng.rand()
        if sel >= 0.25:
            # global facial swap.
            sliding_win = targetRgb.shape[:2]

            if sel < 0.5:
                # fake to source global facial swap.
                mfs_result, bbox = multi_scale_facial_swap(
                    targetRgb, sourceRgb, landmark, config,
                    sliding_win, face_bbox, blending_type, training
                )
            elif sel > 0.75:
                # source to fake global facial swap.
                mfs_result, bbox = multi_scale_facial_swap(
                    sourceRgb, targetRgb, landmark, config,
                    sliding_win, face_bbox, blending_type, training
                )
            else:
                # mfs_result, bbox = targetRgb, np.array([[0, 0, 256, 256]])
                # cropMfs, landmark = get_align5p(
                #     [mfs_result], landmark, rng, config, training
                # )
                # mfs_result = cropMfs[0]
                mfs_result = []
                bbox = []
                for idx in range(len(crop_scale)):
                    bb = [[0, 0, 256, 256]]
                    scale = crop_scale[idx][0]
                    cropMfs = get_cropped(targetRgb, face_bbox, scale=scale)
                    mfs_result.append(cropMfs)
                    bbox.append(np.array(bb))

        else:
            # parial facial swap.
            prior_bbox = config['sliding_win']['prior_bbox']
            sliding_win = prior_bbox[np.random.choice(len(prior_bbox))]
            mfs_result, bbox = multi_scale_facial_swap(
                sourceRgb, targetRgb, landmark, config,
                sliding_win, face_bbox, blending_type, training
            )
    else:
        # # crop face with landmark.
        # cropMfs, landmark = get_align5p(
        #     [mfs_result], landmark, rng, config, training
        # )
        # mfs_result = cropMfs[0]
        if label == 0:
            sel = rng.rand()
            if sel <= 0.1:
                head, fname = ntpath.split(img_path)
                pp = fname.split('.')
                mask_path = os.path.join(head, pp[0]+'.pkl')
                if os.path.exists(mask_path):
                    targetRgb = digital_augment(targetRgb, mask_path, resized)
                    label = 2
                else:
                    file_name_without_ext, ext = os.path.splitext(fname)
                    mask_path = os.path.join(head, file_name_without_ext+'.pkl')
                    if os.path.exists(mask_path):
                        targetRgb = digital_augment(targetRgb, mask_path, resized)
                        label = 2
            elif sel > 0.1 and sel <= 0.2:
                targetRgb = moire(targetRgb)
                label = 1
            elif sel > 0.2 and sel <= 0.3:
                label = 1
                color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
                image_rgb = cv2.cvtColor(targetRgb, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                transformed_image_pil = color_jitter(image_pil)
                transformed_image_np = np.array(transformed_image_pil)
                targetRgb = cv2.cvtColor(transformed_image_np, cv2.COLOR_RGB2BGR)

        mfs_result = []
        bbox = []
        for idx in range(len(crop_scale)):
            bb = [[0, 0, 256, 256]]
            scale = crop_scale[idx][0]
            cropMfs = get_cropped(targetRgb, face_bbox, scale=scale)
            mfs_result.append(cropMfs)
            bbox.append(np.array(bb))

    if mfs_result is None:
        return None, 'multi scale facial swap err.'

    if training and label > 0 and rng.rand() >= 0.5: # in case of only fake
        # mfs_result, bbox = image_h_mirror(mfs_result, bbox)
        # mfs_result = add_noise(rng, mfs_result)
        for idx in range(len(crop_scale)):
            mfs_result[idx] = add_noise(rng, mfs_result[idx])

    genuine = True if not label else False

    location_label = []
    confidence_label = []
    for idx in range(len(crop_scale)):
        loc, conf = label_assign(
            bbox[idx].astype('float32') / config['crop_face']['output_size'],
            config, genuine
        )
        location_label.append(loc)
        confidence_label.append(conf)

    depth_label = get_cropped(depth_img, face_bbox, scale=crop_scale[1][0])
    depth_label = cv2.resize(depth_label, (64, 64), interpolation=cv2.INTER_AREA)

    return mfs_result, {'label': label, 'location_label': location_label,
                        'confidence_label': confidence_label, 'depth_label': depth_label}


def prepare_test_input(img, ld, fb, label, config):
    # config = config['crop_face']
    #
    # img, ld = align_5p(
    #     img, ld=ld,
    #     face_width=config['face_width'], canvas_size=config['output_size'],
    #     scale=config['scale']
    # )
    crop_scale = config['crop_face']['crop_scale']
    mfs_result = []
    for idx in range(len(crop_scale)):
        scale = crop_scale[idx][0]
        cropMfs = get_cropped(img, fb, scale=scale)
        mfs_result.append(cropMfs)

    return mfs_result, {'label': label}

# vim: ts=4 sw=4 sts=4 expandtab
