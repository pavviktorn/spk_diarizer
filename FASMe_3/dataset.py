#!/usr/bin/env python3
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from lib.data_preprocess.preprocess import prepare_train_input, prepare_test_input
from data.util import get_landmarks, get_quality, get_frame
import ntpath


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

class FASDataset(Dataset):

    def __init__(self, mode: str, config: dict, info_path: str):
        super().__init__()

        self.config = config
        self.mode = mode
        self.img_root = self.config['dataset']['img_path']
        self.save_root = self.config['dataset']['save_path']
        self.info_path = info_path
        self.rng = np.random
        assert mode in ['train', 'test']
        self.do_train = True if mode == 'train' else False
        self.info_meta_dict = self.load_meta_info(self.info_path)
        self.samples = self.collect_samples()

    def load_meta_info(self, info_path) -> List:
        with open(info_path, 'r') as f:
            info_dict = f.readlines()
        return info_dict

    def collect_samples(self) -> List:
        samples = []
        for idx in range(len(self.info_meta_dict)):
            info = self.info_meta_dict[idx]
            elem = info.split(',')
            class_label = int(elem[0])
            video_or_frame = elem[1]
            quality = elem[2]
            org_path = elem[3]
            src_path = elem[4]
            if src_path != ':':
                src_path = (self.img_root + src_path).replace('\\', '/')
            else:
                src_path = ''

            bbox = np.zeros((4,)).astype(int)
            landmark = np.zeros((106, 2)).astype(int)

            if video_or_frame == 'f':
                path = (self.img_root + org_path).replace('\\', '/')
                frame = 0
            else:
                pp = org_path.split(':')
                path = (self.img_root + pp[0]).replace('\\', '/')
                frame = int(pp[1])

            be = elem[5].split(':')
            bbox[0], bbox[1], bbox[2], bbox[3] = int(be[0]),int(be[1]),int(be[2]),int(be[3]),
            le = elem[6].split(':')
            for i in range(landmark.shape[0]):
                xy = le[i].split('_')
                landmark[i][0], landmark[i][1] = xy[0], xy[1]

            samples.append(
                (path, {'labels': class_label, 'vf': video_or_frame, 'frame': frame, 'landmark': landmark, 'bbox': bbox,
                        'source_path': src_path,
                        'quality': quality})
            )

        return samples


    def get_domain(self, src_path):
        domain = 0
        if '/1m_faces_91__99/' in src_path:
            domain = 0
        elif '/add_free/' in src_path:
            domain = 1
        elif '/AgedSyntheticImages/' in src_path:
            domain = 2
        elif '/CelebDF_v2/' in src_path:
            domain = 3
        elif '/DeeperForensics/' in src_path:
            domain = 4
        elif '/DEEPFAKE_CHALLENGE/' in src_path:
            domain = 5
        elif '/DeepFakeFace/' in src_path:
            domain = 6
        elif '/DeepfakeTIMIT/' in src_path:
            domain = 7
        elif '/DFDC/' in src_path:
            domain = 8
        elif '/DFFD/' in src_path:
            domain = 9
        elif '/DFGC-2021/' in src_path:
            domain = 10
        elif '/DFGC-2022/' in src_path:
            domain = 11
        elif '/DFMNIST+/' in src_path:
            domain = 12
        elif '/disco_gan/' in src_path:
            domain = 13
        elif '/FF++/' in src_path:
            domain = 14
        elif '/FF++_HifiFace/' in src_path:
            domain = 15
        elif '/from_nizar' in src_path:
            domain = 16
        elif '/how_fmc/' in src_path:
            domain = 17
        elif '/iFakeFaceDB/' in src_path:
            domain = 18
        elif '/MegaFS/' in src_path:
            domain = 19
        elif '/new_df/' in src_path:
            domain = 20
        elif '/3DMAD/' in src_path:
            domain = 21
        elif '/CeFA/' in src_path:
            domain = 22
        elif '/CelebA-Spoof/' in src_path:
            domain = 23
        elif '/CSMAD/' in src_path:
            domain = 24
        elif '/CVPR2023-ASF/' in src_path:
            domain = 25
        elif '/datatang_sample/' in src_path:
            domain = 26
        elif '/ERPA/' in src_path:
            domain = 27
        elif '/HiFiMask/' in src_path:
            domain = 28
        elif '/mydata/' in src_path:
            domain = 29
        elif '/mywebcam/' in src_path:
            domain = 30
        elif '/Oulu_NPU/' in src_path:
            domain = 31
        elif '/Replay-Attack/' in src_path:
            domain = 32
        elif '/Replay-Mobile/' in src_path:
            domain = 33
        elif '/Rose/' in src_path:
            domain = 34
        elif '/SiW/' in src_path:
            domain = 35
        elif '/SiW-Mv2/' in src_path:
            domain = 36
        elif '/SiW-Mv2/' in src_path:
            domain = 37
        elif '/SWAX/' in src_path:
            domain = 38
        else:
            domain = 49

        return domain


    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        ld = label_meta['landmark']
        bbox = label_meta['bbox']
        quality = label_meta['quality']
        label = label_meta['labels']
        source_path = label_meta['source_path']
        video_or_frame = label_meta['vf']
        if video_or_frame == 'f':
            path = path.replace(self.img_root, self.save_root)
            head, fname = ntpath.split(path)
            pp = fname.split('.')
            path = head + f'/{pp[0]}.jpg'
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
            else:
                img = None
            if img is None:
                file_name_without_ext, ext = os.path.splitext(fname)
                path = head + f'/{file_name_without_ext}.jpg'
                if os.path.exists(path):
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
            if source_path != '':
                source_path = source_path.replace(self.img_root, self.save_root)
                head, fname = ntpath.split(source_path)
                pp = fname.split('.')
                source_path = head + f'/{pp[0]}.jpg'
                if os.path.exists(source_path):
                    source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
                else:
                    source_img = None
                if source_img is None:
                    file_name_without_ext, ext = os.path.splitext(fname)
                    source_path = head + f'/{file_name_without_ext}.jpg'
                    if os.path.exists(source_path):
                        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
            else:
                source_img = None
        else:
            frame = label_meta['frame']
            path = path.replace(self.img_root, self.save_root)
            img, path = get_frame(path, frame, isFirst=True)
            if source_path != '':
                source_path = source_path.replace(self.img_root, self.save_root)
                source_img, _ = get_frame(source_path, frame, isFirst=False)
            else:
                source_img = None

        if not isinstance(img, np.ndarray) and (img is None or np.shape(img) == ()):
            print('img read error: ' + path)

        if isinstance(img, np.ndarray) and isinstance(source_img, np.ndarray) and img.shape != source_img.shape:
            source_img = None

        domain = self.get_domain(path)

        # checking bbox & lmk
        if bbox[0] < 0: bbox[0] = 0
        if bbox[0] >= img.shape[1]: bbox[0] = img.shape[1] - 1
        if bbox[2] < 0: bbox[2] = 0
        if bbox[2] >= img.shape[1]: bbox[2] = img.shape[1] - 1
        if bbox[1] < 0: bbox[1] = 0
        if bbox[1] >= img.shape[0]: bbox[1] = img.shape[0] - 1
        if bbox[3] < 0: bbox[3] = 0
        if bbox[3] >= img.shape[0]: bbox[3] = img.shape[0] - 1

        for iii in range(ld.shape[0]):
            if ld[iii][0] < 0: ld[iii][0] = 0
            if ld[iii][0] >= img.shape[1]: ld[iii][0] = img.shape[1] - 1
            if ld[iii][1] < 0: ld[iii][1] = 0
            if ld[iii][1] >= img.shape[0]: ld[iii][1] = img.shape[0] - 1

        if self.mode == "train":
            if label == 0:
                head, fname = ntpath.split(path)
                pp = fname.split('.')
                path = head + f'/{pp[0]}_depth.jpg'
                depth_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                # if not isinstance(depth_img, np.ndarray) and (depth_img is None or np.shape(depth_img) == ()):
                #     depth_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                #     points = cv2.convexHull(
                #         np.array(ld).astype('float32')
                #     )
                #     corners = np.expand_dims(points, axis=0).astype(np.int32)
                #     cv2.fillPoly(depth_img, corners, 255)
            else:
                # depth_img = np.full((img.shape[0], img.shape[1]), 0.5, dtype=np.float32)
                # depth_img = (depth_img*255.).astype(np.uint8)
                depth_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                points = cv2.convexHull(
                    np.array(ld).astype('float32')
                )
                corners = np.expand_dims(points, axis=0).astype(np.int32)
                cv2.fillPoly(depth_img, corners, 50)
                # cv2.imwrite('t_org.jpg', img)
                # cv2.imwrite('t_depth.jpg', depth_img)

            img, label_dict = prepare_train_input(
                img, source_img, depth_img, ld, bbox, label, self.config, path, self.do_train
            )
            if isinstance(label_dict, str):
                return None, label_dict

            location_label = torch.Tensor(label_dict['location_label'])
            confidence_label = torch.Tensor(label_dict['confidence_label'])
            img = cv2.hconcat([img[0], img[1]])
            # cv2.imwrite(f'merge_{index}.jpg', img)
            # cv2.imwrite(f'depth_{index}.jpg', label_dict['depth_label'])
            img = torch.Tensor(img.transpose(2, 0, 1)/255)
            depth_label = torch.Tensor(label_dict['depth_label']/255)
            return img, (label, location_label, confidence_label, depth_label, domain)

        elif self.mode == 'test':
            img, label_dict = prepare_test_input(
                img, ld, bbox, label, self.config
            )
            img = cv2.hconcat([img[0], img[1]])
            img = torch.Tensor(img.transpose(2, 0, 1)/255)
            video_name = path
            return img, (label, video_name, self.info_meta_dict[index])

        else:
            raise ValueError("Unsupported mode of dataset!")

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    from lib.util import load_config
    cfg = load_config('./configs/caddm_train.cfg')
    d = FASDataset(mode="train", config=cfg, info_path=cfg['dataset']['info_path'])
    for index in range(len(d)):
        res = d[index]
# vim: ts=4 sw=4 sts=4 expandtab
