#!/usr/bin/env mdl
import cv2
import numpy as np

# from . import _jpegpy
#
#
# def jpeg_encode(img: np.array, quality=80):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return _jpegpy.encode(img, quality)
#
#
# def jpeg_decode(code: bytes):
#     img = _jpegpy.decode(code)
#     return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# vim: ts=4 sw=4 sts=4 expandtab

def jpeg_encode(img: np.array, quality=80):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    return encimg


def jpeg_decode(code: bytes):
    decimg = cv2.imdecode(code, 1)
    return decimg
