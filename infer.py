from torch.utils.data import DataLoader
import torch
import numpy as np
from model_target import Model
from dataset import Dataset
from test import test
import option
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    print('perform testing...')
    args = option.parser.parse_args()
    device = torch.device("cuda")

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.to(device)
    pretrained_dict = torch.load('ckpt/xd_a2v__1.pkl',map_location="cpu")
    model_dict = model.state_dict()
    if list(pretrained_dict.keys())[0].startswith("module."):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    gt = np.load(args.gt)
    st = time.time()

    pr_auc = test(test_loader, model, gt)
    time_elapsed = time.time() - st
    print('test AP: {:.4f}\n'.format(pr_auc))
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
