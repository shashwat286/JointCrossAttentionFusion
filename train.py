import torch
from loss import CLAS
from tqdm import tqdm


def train(dataloader, model, optimizer, criterion):
    t_loss = 0.0

    with torch.set_grad_enabled(True):
        model.train()
        for i, (inputs, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)
            inputs = inputs[:, :torch.max(seq_len), :]
            inputs = inputs.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)
            logits = model(inputs)
            loss = CLAS(logits, label, seq_len, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss+=(loss.item()*len(dataloader))

    return t_loss/len(dataloader.sampler)

