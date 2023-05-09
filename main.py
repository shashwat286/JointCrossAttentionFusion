from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
from torch.utils.data import random_split

from model import Model
from dataset import Dataset
from train import train
from test import test
import option
import copy

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    args = option.parser.parse_args()
    setup_seed(args.seed)
    train_data = Dataset(args, test_mode=False)
    
    train_loader = DataLoader(train_data,
          batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
   
    test_data = Dataset(args, test_mode=True)
    
    test_loader = DataLoader(test_data,
                              batch_size=5, shuffle =False,
                              num_workers=args.workers, pin_memory=True)
    
    model = Model(args)
    if torch.cuda.is_available():
        model.cuda()
    
    pretrained_dict = torch.load('ckpt/xd_a2v__1_2.pkl',map_location="cpu")
    model_dict = model.state_dict()
    if list(pretrained_dict.keys())[0].startswith("module."):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
   
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    is_topk = True
    gt = np.load(args.gt)
    random_ap = test(test_loader, model, gt)
    print('Random initialized AP: {:.4f}\n'.format(random_ap))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_ap = 0.0

    st = time.time()
    for epoch in range(args.max_epoch):
        cls_loss = train(train_loader, model, optimizer, criterion)
        scheduler.step()
        
        ap = test(test_loader, model, gt)
        if ap > best_ap :
            best_ap = ap
            best_model_wts = copy.deepcopy(model.state_dict())

        print('[Epoch {}/{}]: cls loss: {} | epoch AP: {:.4f}'.format(epoch + 1, args.max_epoch, cls_loss, ap))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), './ckpt/' + args.model_name  + '.pkl')

    time_elapsed = time.time() - st
    print('Training completes in {:.0f}m {:.0f}s | '
              'best test AP: {:.4f}\n'.format(time_elapsed // 60, time_elapsed % 60, best_ap))

