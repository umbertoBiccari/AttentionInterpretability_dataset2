import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from metrics import eval_split_metrics_3out_direct, eval_single_metrics

def run_epoch_single(model, loader, device, loss_fn, optimizer=None):    
    
    train = optimizer is not None
    model.train(train)

    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        out = model(xb)  
        
        loss = loss_fn(out, yb)        
        
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = xb.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)

def train_single(model, loss_fn, optimizer, train_loader, val_loader, device, epochs=220, metrics_every=10):
       
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    # scheduler = None
    
    best_val = float("inf")
    best_state = None
    hist = []

    for ep in range(1, epochs + 1):
        tr_loss = run_epoch_single(model, train_loader, device, loss_fn, optimizer=optimizer)
        va_loss = run_epoch_single(model, val_loader, device, loss_fn, optimizer=None)
        if scheduler is not None:
            scheduler.step(va_loss)

        if ep % metrics_every == 0 or ep == 1:
            X_train = train_loader.dataset.tensors[0].numpy()
            y_train = train_loader.dataset.tensors[1].numpy()

            m_tr = eval_single_metrics(model, X_train, y_train, device)

            X_val = val_loader.dataset.tensors[0].numpy()
            y_val = val_loader.dataset.tensors[1].numpy()
            
            m_va = eval_single_metrics(model, X_val, y_val, device)

            print(f"ep={ep:03d} tr_loss={tr_loss:.4f} | va_loss={va_loss:.4f} | tr R2={m_tr['R2']:.3f} | va R2={m_va['R2']:.3f} | tr RMSE={m_tr['RMSE']:.3f} | va RMSE={m_va['RMSE']:.3f} | tr MAE={m_tr['MAE']:.3f} | va MAE={m_va['MAE']:.3f} | lr={optimizer.param_groups[0]["lr"]}")
            
        hist.append((ep, tr_loss, va_loss, optimizer.param_groups[0]["lr"]))

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    hist_df = pd.DataFrame(hist, columns=["epoch", "train_loss", "val_loss", "lr"])
    
    print(f"best val loss: {best_val:.6f}")
    return hist_df