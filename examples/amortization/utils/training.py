import numpy as np


def train(model, dataloader, optimizer, on_finish=None):
    model.train()
    losses = []
    for batch_idx, (X, Kset, logPs, idx) in enumerate(dataloader):
        optimizer.zero_grad()
        res = model(X, Kset, logPs, indexes=idx)
        loss = res["objective"]
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        print("Batch {} | loss: {}".format(batch_idx, loss))
    
    if on_finish is not None:
        on_finish(X, Kset, logPs, np.array(losses).mean(), res)

    return losses
