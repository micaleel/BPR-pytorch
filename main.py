import argparse
import os
import time

import click
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter

import config
import data_utils
import evaluate
from model import BPR

click.command()
click.option(
    "--lr", type=click.FLOAT, required=False, default=0.01, help="learning rate"
)
click.option("--reg", type=click.FLOAT, default=0.001, help="model regularization rate")
click.option(
    "--batch_size", type=click.INT, default=4096, help="batch size for training"
)
click.option("--epochs", type=click.INT, default=50, help="training epoches")
click.option("--top_k", type=click.INT, default=10, help="compute metrics@top_k")
click.option(
    "--factor_num",
    type=click.INT,
    default=32,
    help="predictive factors numbers in the model",
)
click.option(
    "--num_ng", type=click.INT, default=4, help="sample negative items for training"
)
click.option(
    "--test_num_ng",
    type=click.INT,
    default=99,
    help="sample part of negative items for testing",
)
click.option("--out", default=True, help="save model or not")
click.option("--gpu", type=click.STRING, default="0", help="gpu card ID")


def run(
    batch_size=4096,
    epochs=50,
    factor_num=32,
    gpu="0",
    lr=0.01,
    num_ng=4,
    out=True,
    reg=0.001,
    test_num_ng=99,
    top_k=10,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cudnn.benchmark = True

    ############################## PREPARE DATASET ##########################
    train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

    # construct the train and test datasets
    train_dataset = data_utils.BPRData(train_data, item_num, train_mat, num_ng, True)
    test_dataset = data_utils.BPRData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=test_num_ng + 1, shuffle=False, num_workers=0
    )

    ########################### CREATE MODEL #################################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = BPR(user_num, item_num, factor_num)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=reg)
    # writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    count, best_hr = 0, 0
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item_i, item_j in train_loader:
            user = user.to(device)
            item_i = item_i.to(device)
            item_j = item_j.to(device)

            model.zero_grad()
            prediction_i, prediction_j = model(user, item_i, item_j)
            loss = -(prediction_i - prediction_j).sigmoid().log().sum()
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, top_k, device)

        elapsed_time = time.time() - start_time
        print(
            "The time elapse of epoch {:03d}".format(epoch)
            + " is: "
            + time.strftime("%H: %M: %S", time.gmtime(elapsed_time))
        )
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if out:
                if not os.path.exists(config.model_path):
                    os.mkdir(config.model_path)
                torch.save(model, "{}BPR.pt".format(config.model_path))

    print(
        "End. Best epoch {:03d}: HR = {:.3f}, \
		NDCG = {:.3f}".format(
            best_epoch, best_hr, best_ndcg
        )
    )


if __name__ == "__main__":
    run()
