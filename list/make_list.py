import os
import glob

root_path = 'i3d-features/RGBTest'    ## the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))
violents = []
normal = []
with open('rgb_test.list', 'w+') as f:  ## the name of feature list
    for file in files:
        if '_label_A' in file:
            normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)

print('finish.')