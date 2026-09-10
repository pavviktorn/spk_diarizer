
import os
import numpy as np
import random
import time
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


in_root = '/datasets/newout/meta'
out_root = '/datasets/newout/meta_full'
save_root = '/datasets/newout'
# in_root = 'D:/zLiveness_data/out/meta_full'
# out_root = 'D:/zLiveness_data/out/meta'
# save_root = 'D:/zLiveness_data/out'


def convert_file(path):
    path = path.replace('\\', '/')

    head, fname = ntpath.split(path)
    category = head.split('/')[-1]
    out_path = os.path.join(out_root, category)
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(out_path, fname)

    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    f_a = open(out_path, 'w')

    size = len(lines)
    for idx in range(size):
        info = lines[idx]

        elem = info.split(',')
        class_label = int(elem[0])
        video_or_frame = elem[1]
        quality = elem[2]
        org_path = elem[3]

        if video_or_frame == 'f':
            pp, fn = ntpath.split(org_path)
            ld_path = pp + f"/{fn.split('.')[0]}.npz"
            ld_path = save_root + ld_path
            frame = 0
        else:
            pp = org_path.split(':')
            frame = int(pp[1])
            pp, fn = ntpath.split(pp[0])
            ld_path = pp + f"/{fn.split('.')[0]}"
            ld_path = (save_root + ld_path) + f"/{frame}.npz"

        try:
            loaded_data = np.load(ld_path)
            bbox = loaded_data['bbox']
            kps = loaded_data['kps']
            landmark = loaded_data['landmarks']
        except:
            print('error {}'.format(ld_path))
            continue

        bbox_str = f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}"
        ld_str = ''
        for iii in range(landmark.shape[0]):
            ld_str += f"{landmark[iii][0]}_{landmark[iii][1]}:"
        ld_str = ld_str[:-1]

        src_path = elem[4].replace('\n', '')
        if src_path == '':
            src_path = ':'

        out_info = f"{class_label},{video_or_frame},{quality},{org_path},{src_path},{bbox_str},{ld_str}\n"

        # bbox__ = np.zeros((4,)).astype(int)
        # ld___ = np.zeros((106,2)).astype(int)
        # eee = bbox_str.split(':')
        # bbox__[0], bbox__[1], bbox__[2], bbox__[3] = int(eee[0]),int(eee[1]),int(eee[2]),int(eee[3]),
        # eee = ld_str.split(':')
        # for idx in range(ld___.shape[0]):
        #     zzz = eee[idx].split('_')
        #     ld___[idx][0], ld___[idx][1] = zzz[0], zzz[1]

        f_a.write(out_info)

    f_a.close()

    return

full_file_paths = get_filepaths(in_root)

for path in full_file_paths:
    print(path)
    convert_file(path)



