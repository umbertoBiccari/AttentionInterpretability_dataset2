import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from metrics import eval_split_metrics_3out_direct, eval_single_metrics

def run_epoch_3out(model, loader, device, optimizer=None, lambda_age=1.0, lambda_mets=0.5, lambda_sex=0.1):
    reg_loss = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss()

    train = optimizer is not None
    model.train(train)

    total, n = 0.0, 0
    tot_age, tot_mets, tot_sex = 0.0, 0.0, 0.0

    for xb, agec, mets, sex in loader:
        xb = xb.to(device)        
        mets = mets.to(device)
        sex = sex.to(device)

        out3 = model(xb)  # (n,3) => [age_s, mets_s, sex_logit]
        pred_age = out3[:, 0:1]
        pred_mets = out3[:, 1:2]
        pred_sex = out3[:, 2:3]

        loss_age = reg_loss(pred_age, agec)
        loss_mets = reg_loss(pred_mets, mets)
        loss_sex = bce(pred_sex, sex)

        loss = lambda_age * loss_age + lambda_mets * loss_mets + lambda_sex * loss_sex

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = xb.size(0)
        total += loss.item() * bs
        tot_age += loss_age.item() * bs
        tot_mets += loss_mets.item() * bs
        tot_sex += loss_sex.item() * bs
        n += bs

    return total / max(n, 1), tot_age / max(n, 1), tot_mets / max(n, 1), tot_sex / max(n, 1)

def train_3out(model, train_loader, val_loader,
               tag, device, savedir,
               lr=1e-3, weight_decay=1e-6, epochs=200,
               lambda_age=1.0, lambda_mets=0.5, lambda_sex=0.1,
               metrics_every=10, sex_threshold=0.5):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_state = None
    hist = []

    for ep in range(1, epochs + 1):
        tr, tr_age, tr_mets, tr_sex = run_epoch_3out(
            model, train_loader, device, optimizer=optimizer,
            lambda_age=lambda_age, lambda_mets=lambda_mets, lambda_sex=lambda_sex
        )
        va, va_age, va_mets, va_sex = run_epoch_3out(
            model, val_loader, device, optimizer=None,
            lambda_age=lambda_age, lambda_mets=lambda_mets, lambda_sex=lambda_sex
        )
        scheduler.step(va)

        if ep % metrics_every == 0 or ep == 1:
            X_train_s = train_loader.dataset.tensors[0].numpy()
            age_train = train_loader.dataset.tensors[1].numpy()
            mets_train = train_loader.dataset.tensors[2].numpy()
            sex_train = train_loader.dataset.tensors[3].numpy()
                        
            m_tr = eval_split_metrics_3out_direct(model, 
                                                  X_train_s, age_train, mets_train, sex_train,
                                                  device,
                                                  sex_threshold=sex_threshold)
            
            X_val_s = val_loader.dataset.tensors[0].numpy()
            age_val = val_loader.dataset.tensors[1].numpy()
            mets_val = val_loader.dataset.tensors[2].numpy()
            sex_val = val_loader.dataset.tensors[3].numpy()
            
            m_va = eval_split_metrics_3out_direct(model, X_val_s, age_val, mets_val, sex_val,
                                                  device,
                                                  sex_threshold=sex_threshold)

            print(
                f"[{tag}] ep={ep:03d} best_val={best_val:.6f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"| loss va={va:.4f} (age={va_age:.4f}, mets={va_mets:.4f}, sex={va_sex:.4f})\n"
                f"   TRAIN: age(R2={m_tr['age_R2']:.3f}, RMSE={m_tr['age_RMSE']:.3f}, MAE={m_tr['age_MAE']:.3f}) | "
                f"MetSCORE(R2={m_tr['MetSCORE_R2']:.3f}, RMSE={m_tr['MetSCORE_RMSE']:.3f}, MAE={m_tr['MetSCORE_MAE']:.3f}) | "
                f"sex(ACC={m_tr['sex_ACC']:.3f}, AUC={m_tr['sex_AUC']:.3f}, F1={m_tr['sex_F1']:.3f})\n"
                f"   VAL  : age(R2={m_va['age_R2']:.3f}, RMSE={m_va['age_RMSE']:.3f}, MAE={m_va['age_MAE']:.3f}) | "
                f"MetSCORE(R2={m_va['MetSCORE_R2']:.3f}, RMSE={m_va['MetSCORE_RMSE']:.3f}, MAE={m_va['MetSCORE_MAE']:.3f}) | "
                f"sex(ACC={m_va['sex_ACC']:.3f}, AUC={m_va['sex_AUC']:.3f}, F1={m_va['sex_F1']:.3f})"
            )
        else:
            m_tr = {k: np.nan for k in ["age_R2", "age_RMSE", "age_MAE", "MetSCORE_R2", "MetSCORE_RMSE", "MetSCORE_MAE",
                                       "sex_ACC", "sex_AUC", "sex_F1"]}
            m_va = {k: np.nan for k in ["age_R2", "age_RMSE", "age_MAE", "MetSCORE_R2", "MetSCORE_RMSE", "MetSCORE_MAE",
                                       "sex_ACC", "sex_AUC", "sex_F1"]}

        hist.append((
            ep, tr, va, tr_age, tr_mets, tr_sex, va_age, va_mets, va_sex, optimizer.param_groups[0]["lr"],
            m_tr["age_R2"], m_tr["age_RMSE"], m_tr["age_MAE"],
            m_tr["MetSCORE_R2"], m_tr["MetSCORE_RMSE"], m_tr["MetSCORE_MAE"],
            m_tr["sex_ACC"], m_tr["sex_AUC"], m_tr["sex_F1"],
            m_va["age_R2"], m_va["age_RMSE"], m_va["age_MAE"],
            m_va["MetSCORE_R2"], m_va["MetSCORE_RMSE"], m_va["MetSCORE_MAE"],
            m_va["sex_ACC"], m_va["sex_AUC"], m_va["sex_F1"],
        ))

        if va < best_val - 1e-6:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    hist_df = pd.DataFrame(
        hist,
        columns=[
            "epoch", "train_loss", "val_loss",
            "tr_age_loss", "tr_mets_loss", "tr_sex_loss",
            "va_age_loss", "va_mets_loss", "va_sex_loss",
            "lr",
            "train_age_R2", "train_age_RMSE", "train_age_MAE",
            "train_MetSCORE_R2", "train_MetSCORE_RMSE", "train_MetSCORE_MAE",
            "train_sex_ACC", "train_sex_AUC", "train_sex_F1",
            "val_age_R2", "val_age_RMSE", "val_age_MAE",
            "val_MetSCORE_R2", "val_MetSCORE_RMSE", "val_MetSCORE_MAE",
            "val_sex_ACC", "val_sex_AUC", "val_sex_F1",
        ]
    )
    hist_df.to_csv(savedir / f"train_history_{tag}.csv", index=False)
    print(f"[{tag}] best val loss: {best_val:.6f} | epochs: {len(hist_df)}")
    return hist_df

def run_epoch_single(model, loader, device, kind, optimizer=None):
    reg_loss = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss()

    train = optimizer is not None
    model.train(train)

    total, n = 0.0, 0
    for xb, agec, mets, sex in loader:
        xb = xb.to(device)
        agec = agec.to(device)
        mets = mets.to(device)
        sex = sex.to(device)

        out = model(xb)  # (n,1)

        if kind == "age":
            loss = reg_loss(out, agec)
        elif kind == "mets":
            loss = reg_loss(out, mets)
        elif kind == "sex":
            loss = bce(out, sex)
        else:
            raise ValueError(kind)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = xb.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)

def train_single(model, kind, train_loader, val_loader, tag, device, savedir,
                 lr=1e-3, weight_decay=1e-6, epochs=220, metrics_every=10, sex_threshold=0.5):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_state = None
    hist = []

    for ep in range(1, epochs + 1):
        tr_loss = run_epoch_single(model, train_loader, device, kind, optimizer=optimizer)
        va_loss = run_epoch_single(model, val_loader, device, kind, optimizer=None)
        scheduler.step(va_loss)

        if ep % metrics_every == 0 or ep == 1:
            X_train_s = train_loader.dataset.tensors[0].numpy()
            age_train = train_loader.dataset.tensors[1].numpy()
            mets_train = train_loader.dataset.tensors[2].numpy()
            sex_train = train_loader.dataset.tensors[3].numpy()
            m_tr = eval_single_metrics(model, kind, X_train_s, age_train, mets_train, sex_train,
                                       device, sex_threshold=sex_threshold)

            X_val_s = val_loader.dataset.tensors[0].numpy()
            age_val = val_loader.dataset.tensors[1].numpy()
            mets_val = val_loader.dataset.tensors[2].numpy()
            sex_val = val_loader.dataset.tensors[3].numpy()
            m_va = eval_single_metrics(model, kind, X_val_s, age_val, mets_val, sex_val,
                                       device, sex_threshold=sex_threshold)

            if kind in ("age", "mets"):
                print(f"[{tag}] ep={ep:03d} tr_loss={tr_loss:.4f} | va_loss={va_loss:.4f} | TRAIN R2={m_tr['R2']:.3f} | VAL R2={m_va['R2']:.3f}")
            else:
                print(f"[{tag}] ep={ep:03d} va_loss={va_loss:.4f} | TRAIN AUC={m_tr['AUC']:.3f} | VAL AUC={m_va['AUC']:.3f}")

        hist.append((ep, tr_loss, va_loss, optimizer.param_groups[0]["lr"]))

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    hist_df = pd.DataFrame(hist, columns=["epoch", "train_loss", "val_loss", "lr"])
    hist_df.to_csv(savedir / f"train_history_single_{tag}.csv", index=False)
    print(f"[{tag}] best val loss: {best_val:.6f}")
    return hist_df