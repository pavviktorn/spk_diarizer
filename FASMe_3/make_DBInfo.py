
import os
import numpy as np
import random
import time

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

    return sorted(file_paths)  # Self-explanatory.


in_root = '/datasets/newout/meta'
out_root = '/datasets/newout'
# in_root = 'D:/zLiveness_data/out/meta'
# out_root = 'D:/zLiveness_data/out'


rseed = int(time.time())
np.random.seed(rseed)
random.seed(rseed)


f_a = open(os.path.join(out_root, 'train_info.txt'), 'w')

def sel_lines(task, selnum):
    input_path = os.path.join(in_root, task)
    full_file_paths = get_filepaths(input_path)

    num = 0
    all_num = 0
    for path in full_file_paths:
        print(path)
        if "new_test_20250821" in path:
            continue

        f = open(path, 'r+')
        lines = f.readlines()
        f.close()

        size = len(lines)
        all_num += size
        for idx in range(size):
            rat = selnum / size
            if random.random() < rat:
                f_a.write(lines[idx])
                num += 1

    return num, all_num

n_train_real, real_total = sel_lines('real', 250000)

n_train_pad_main, _ = sel_lines('pad/main', 20000)
n_train_pad, pad_total = sel_lines('pad', 81000)

n_train_df_main, _ = sel_lines('df/main', 50000)
n_train_df, df_total = sel_lines('df', 7000)

print(f"train pad main : {n_train_pad_main}")
print(f"train pad : {n_train_pad}")
print(f"train deepfake main : {n_train_df_main}")
print(f"train deepfake : {n_train_df}")

print(f"\ntrain real : {n_train_real}")
print(f"train pad : {n_train_pad + n_train_pad_main}")
print(f"train deepfake : {n_train_df + n_train_df_main}")

print(f"\ntotal real : {real_total}")
print(f"total pad : {pad_total}")
print(f"total deepfake : {df_total}")

f_a.close()

f_a = open(os.path.join(out_root, 'train_info.txt'), 'r+')
train_info = f_a.readlines()
f_a.close()

random.shuffle(train_info)

f_a = open(os.path.join(out_root, 'train_info.txt'), 'w')
f_a.writelines(train_info)
f_a.close()
