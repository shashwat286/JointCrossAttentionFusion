import torch

from layers import *
import torch.nn.functional as F
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)


class CMA_VA(nn.Module):
    def __init__(self, l_v, l_a, l_t, hid_dim=32, d_ff=32, dropout_rate=0.1):
        super(CMA_VA, self).__init__()

        self.joint_cross_attention = JointCrossAttention(hid_dim)
        self.ffn = nn.Sequential(
            nn.Linear(l_t,32),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(l_t)

    def forward(self, f_v, f_a):
        f = torch.cat((f_v,f_a), dim=2)
        #print(f.shape)
        new_f = self.joint_cross_attention(f, f_v, f_a)
        new_f = self.norm(new_f)
        new_f = self.ffn(new_f)

        return new_f


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        n_features = args.feature_size
        n_class = args.num_classes

        self.joint_cross_attention = CMA_VA(l_v= 1024, l_a =128, l_t = 1152, hid_dim=32, d_ff=32)
        self.classifier = nn.Linear(32,1)
        self.apply(weight_init)

    def forward(self, x):
        f_r = x[:, :, :1024]
        f_f = x[:, :, 1024:2048]
        f_a = x[:, :, 2048: ]
        f_v = (f_r + f_f)/2
        new_v = self.joint_cross_attention(f_v, f_a)
        logits = self.classifier(new_v)
        logits = logits.squeeze(dim=1)
        logits = torch.sigmoid(logits)

        return logits
