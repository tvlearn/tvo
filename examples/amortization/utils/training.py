import torch
import numpy as np
from tqdm import tqdm

def train(model, dataloader, optimizer, on_finish=None):
    model.train()
    model_device = next(model.parameters()).device
    losses = []
    for batch_idx, (X, Kset, logPs, idx) in enumerate(tqdm(dataloader)):
        if X.device != model_device:
            X = X.to(model_device, non_blocking=True)
            Kset = Kset.to(model_device, non_blocking=True)
            logPs = logPs.to(model_device, non_blocking=True)
            torch.cuda.synchronize()

        optimizer.zero_grad()
        res = model(X, Kset, logPs, indexes=idx)
        loss = res["objective"]
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        #print("Batch {:4d} | loss: {:9.4f}".format(batch_idx, loss))
    
    if on_finish is not None:
        on_finish(X, Kset, logPs, np.array(losses).mean(), res)

    return losses
