
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


f_a = open(os.path.join(out_root, 'test_info.txt'), 'w')

def sel_lines(task, selnum):
    input_path = os.path.join(in_root, task)
    full_file_paths = get_filepaths(input_path)

    num = 0
    for path in full_file_paths:
        print(path)

        f = open(path, 'r+')
        lines = f.readlines()
        f.close()

        size = len(lines)
        for idx in range(size):
            rat = selnum / size
            if random.random() < rat:
                f_a.write(lines[idx])
                num += 1

    return num

n_train_real = sel_lines('real', 3000)

n_train_pad = sel_lines('pad/main', 2000)

n_train_df = sel_lines('df/main', 1500)

print(f"train real : {n_train_real}")
print(f"train pad : {n_train_pad}")
print(f"train deepfake : {n_train_df}")

f_a.close()

