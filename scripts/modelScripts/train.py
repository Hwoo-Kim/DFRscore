import copy
import pickle
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DFRscoreCollator, TrainDataset
from .model import DFRscore

MAX_STEP = None


def HingeMSELoss(y_pred, y_true):
    global MAX_STEP
    ZERO = torch.tensor(0).float().to(y_pred.device)
    NEG_MIN = torch.tensor(MAX_STEP + 1).float().to(y_pred.device)
    y_true = y_true.float()
    return torch.mean(
        torch.where(
            y_true == NEG_MIN,
            (torch.where(y_pred > NEG_MIN, ZERO, y_pred - NEG_MIN)) ** 2,
            (y_pred - y_true) ** 2,
        )
    )


def train(model, loss_fn, optimizer, train_data_loader):
    train_loss_list = []
    for i_batch, batch in enumerate(train_data_loader):
        x = batch["feature"].float().cuda()
        A = batch["adj"].float().cuda()
        y = batch["label"].cuda()  # label is in int type
        y_pred = model(x, A)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.data.cpu().numpy())
    return train_loss_list


def validate(model, loss_fn, val_data_loader):
    val_loss_list = []
    for i_batch, batch in enumerate(val_data_loader):
        x = batch["feature"].float().cuda()
        A = batch["adj"].float().cuda()
        y = batch["label"].cuda()
        y_pred = model(x, A)
        loss = loss_fn(y_pred, y)
        val_loss_list.append(loss.data.cpu().numpy())
    return val_loss_list


def test(model, loss_fn, test_data_loader):
    test_loss_list = []
    for i_batch, batch in enumerate(test_data_loader):
        x = batch["feature"].float().cuda()
        A = batch["adj"].float().cuda()
        y = batch["label"].cuda()
        y_pred = model(x, A)
        loss = loss_fn(y_pred, y)
        test_loss_list.append(loss.data.cpu().numpy())
    return test_loss_list


def train_DFRscore(args) -> Tuple[int, float]:
    # 0. initial setting
    data_dir = args.data_dir
    save_dir = args.save_dir
    log = args.logger
    log("\n2. Model Training Phase")
    if args.use_scratch:
        new_data_dir = f"/scratch/hwkim/{args.data_preprocessing}"
        log(f"scratch data is used: {new_data_dir}")
    else:
        new_data_dir = data_dir
    now = datetime.now()
    since = time.time()
    # 1. Set training parameters
    torch.set_num_threads(int(args.num_threads))

    loss_fn = HingeMSELoss
    global MAX_STEP
    MAX_STEP = args.max_step

    predictor = DFRscore(
        conv_dim=args.conv_dim,
        fc_dim=args.fc_dim,
        n_GAT_layer=args.n_conv_layer,
        n_fc_layer=args.n_fc_layer,
        num_heads=args.num_heads,
        feature_size=args.feature_size,
        max_step=args.max_step,
        dropout=args.dropout,
    )
    predictor.cuda()

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.factor,
        patience=args.patience,
        threshold=args.threshold,
        min_lr=args.min_lr,
    )
    lr = args.lr

    log("  ----- Train Config Information -----")
    log(f"  save_dir: {args.save_dir}")
    log(f"  data dir: {args.data_dir}")
    log.log_arguments(args)
    log()
    log("  ----- Training Log -----")

    best_loss = 100000
    # 2. Training with validation
    train_data_loader = DataLoader(
        TrainDataset(
            data_dir=f"{new_data_dir}/generated_data",
            key_dir=f"{new_data_dir}/data_keys",
            mode="train",
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DFRscoreCollator(mode="train"),
        num_workers=int(args.num_threads),
    )
    val_data_loader = DataLoader(
        TrainDataset(
            data_dir=f"{new_data_dir}/generated_data",
            key_dir=f"{new_data_dir}/data_keys",
            mode="val",
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DFRscoreCollator(mode="train"),
        num_workers=int(args.num_threads),
    )
    # test_data_loader = DataLoader(
    #                TrainDataset(data_dir=f'{new_data_dir}/generated_data', key_dir=f'{new_data_dir}/data_keys',mode='test'),
    #                batch_size=args.batch_size,
    #                shuffle = False,
    #                collate_fn=gat_collate_fn
    #                )

    train_loss_history = []
    val_loss_history = []
    for i in range(args.num_epoch):
        epoch_start = time.time()
        # 2-1. Train phase
        predictor.train()
        train_epoch_loss_list = train(predictor, loss_fn, optimizer, train_data_loader)
        train_epoch_loss = np.mean(train_epoch_loss_list)
        train_loss_history.append(train_epoch_loss)
        if (i + 1) % 5 == 0:
            torch.save(predictor.state_dict(), f"{save_dir}/DFR_model_{str(i+1)}.pt")

        # 2-2. Validation phase
        predictor.eval()
        val_epoch_loss_list = validate(predictor, loss_fn, val_data_loader)
        val_epoch_loss = np.mean(val_epoch_loss_list)
        val_loss_history.append(val_epoch_loss)

        if best_loss > val_epoch_loss:
            best_loss = val_epoch_loss
            best_epoch = i + 1
            best_model = copy.deepcopy(predictor.state_dict())
        epoch_end = time.time()

        # 2-3. Logging
        log(
            f"  {i+1}th epoch,",
            f"   training loss: {train_epoch_loss}",
            f"   val loss: {val_epoch_loss}",
            f"   epoch time: {epoch_end-epoch_start:.2f}",
        )
        scheduler.step(val_epoch_loss)
        if optimizer.param_groups[0]["lr"] < lr:
            lr = float(optimizer.param_groups[0]["lr"])
            log(f"   scheduler has reduced lr, current is: {lr}")

    # 3. Finish and save the result
    torch.save(best_model, f"{save_dir}/Best_model_{str(best_epoch)}.pt")
    now = datetime.now()
    finished_at = now.strftime("%Y. %m. %d (%a) %H:%M:%S")
    time_elapsed = int(time.time() - since)
    log()
    log(
        "  ----- Training Finised -----",
        f"  finished at : {finished_at}",
        "  time passed: [%dh:%dm:%ds]"
        % (time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60),
        f"  Best epoch: {best_epoch}",
        f"  Best loss: {best_loss}",
        f'  Decayed_lr: {optimizer.param_groups[0]["lr"]}',
    )
    with open(f"{save_dir}/loss_history.pkl", "wb") as fw:
        pickle.dump({"train": train_loss_history, "val": val_loss_history}, fw)

    return best_epoch, best_loss.item()
