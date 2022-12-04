import os
import torch
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from yaml import safe_load

from dataloaders.datasets import DatasetValid, DatasetTrain
from models.Dillated3DUnet import UNet3D
from losses.loss_func import LossL2ColorEdges

import warnings

warnings.filterwarnings("ignore")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer:
    def __init__(self, config_file=None):
        # get arguments
        self.args = self.parse_config(config_file)
        # init dataset and load data
        self.train_dataset = DatasetTrain(self.args["DATABASE_TRAIN"])
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args["TRAIN_PARAMS"]["batch_size"],
            num_workers=self.args["TRAIN_PARAMS"]["num_workers"],
            shuffle=True,
            pin_memory=False,
        )

        self.valid_dataset = DatasetValid(self.args["DATABASE_TRAIN"])
        self.valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=False,
        )
        # get device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # init model
        self.model = UNet3D(init_filters=self.args["MODEL"]["filters"]).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=[0]).cuda()
        # init optimizer
        self.optimizer = AdamW(
            params=self.model.parameters(),
            weight_decay=float(self.args["TRAIN_PARAMS"]["weight_decay"]),
            lr=float(self.args["TRAIN_PARAMS"]["lr"]),
        )
        self.scaler = GradScaler()
        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer, T_max=1, eta_min=1e-5
        )
        # init losses
        self.losses = LossL2ColorEdges()
        # create save location
        if not os.path.isdir(self.args["TRAIN_PARAMS"]["weights_dir"]):
            os.makedirs(self.args["TRAIN_PARAMS"]["weights_dir"])
        # create logging directory
        if not os.path.isdir(self.args["TRAIN_PARAMS"]["log_dir"]):
            os.makedirs(self.args["TRAIN_PARAMS"]["log_dir"])

    @staticmethod
    def parse_config(config_file):
        config = None
        if os.path.isfile(config_file) and config_file.endswith("yml"):
            with open(config_file, "r") as f_config:
                config = safe_load(f_config)
        else:
            print("Invalid config: {}".format(config_file))
        return config

    def train(self):
        """
        Train model. Validate at each saved epoch
        """
        writer_dict = {
            'writer': SummaryWriter(log_dir=self.args['TRAIN_PARAMS']['log_dir']),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

        losses_train = AverageMeter()
        losses_valid = AverageMeter()

        for epoch in tqdm(range(self.args["TRAIN_PARAMS"]["epochs"])):
            print("Training epoch: {}".format(epoch))
            total_loss = 0
            self.model.train()
            train_step = 0
            for sample in tqdm(self.train_loader):
                # get input color images
                in_0 = sample[:, 0, :, :, :][:, None, :, :, :]
                in_1 = sample[:, 2, :, :, :][:, None, :, :, :]
                in_2 = sample[:, 4, :, :, :][:, None, :, :, :]
                in_3 = sample[:, 6, :, :, :][:, None, :, :, :]

                gt_0 = sample[:, 1, :, :, :][:, None, :, :, :]
                gt_1 = sample[:, 3, :, :, :][:, None, :, :, :]
                gt_2 = sample[:, 5, :, :, :][:, None, :, :, :]

                net_in = torch.cat((in_0, in_1, in_2, in_3), dim=1)
                gt = torch.cat((gt_0, gt_1, gt_2), dim=1)
                gt = gt.cuda(non_blocking=True)

                net_in = torch.swapaxes(net_in, 1, 2)
                gt = torch.swapaxes(gt, 1, 2)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast(dtype=torch.float16):
                    # forward pass
                    out = self.model(net_in)
                    # compute loss
                    loss = self.losses(out, gt)
                    total_loss += loss.item()
                # optimize
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_step += 1

                losses_train.update(loss.item(), sample.size(0))

                # log
                if train_step % 10 == 0:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    writer.add_scalar('train_loss', losses_train.val, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

            self.scheduler.step()
            print("Total loss after epoch: {} is {}".format(epoch, total_loss))

            if 0 == ((epoch + 1) % self.args["TRAIN_PARAMS"]["save_freq"]):
                # validate model
                print("Validating")
                total_valid_loss = 0
                self.model.eval()
                for sample in tqdm(self.valid_loader):
                    if sample.shape[1] >= 7:
                        in_0 = sample[:, 0, :, :, :][:, None, :, :, :]
                        in_1 = sample[:, 2, :, :, :][:, None, :, :, :]
                        in_2 = sample[:, 4, :, :, :][:, None, :, :, :]
                        in_3 = sample[:, 6, :, :, :][:, None, :, :, :]

                        gt_0 = sample[:, 1, :, :, :][:, None, :, :, :]
                        gt_1 = sample[:, 3, :, :, :][:, None, :, :, :]
                        gt_2 = sample[:, 5, :, :, :][:, None, :, :, :]

                        net_in = torch.cat((in_0, in_1, in_2, in_3), dim=1).to(
                            self.device
                        )
                        gt = torch.cat((gt_0, gt_1, gt_2), dim=1).to(self.device)

                        net_in = torch.swapaxes(net_in, 1, 2)
                        gt = torch.swapaxes(gt, 1, 2)

                        pred = self.model.forward(net_in)
                        pred = torch.clamp(pred, 0.0, 1.0)

                        valid_loss = self.losses(pred, gt)
                        total_valid_loss += valid_loss.item()
                        losses_valid.update(valid_loss.item(), sample.size(0))

                    writer = writer_dict['writer']
                    global_steps = writer_dict['valid_global_steps']
                    writer.add_scalar('valid_loss', losses_valid.avg, global_steps)
                    writer_dict['valid_global_steps'] = global_steps + 1

                print(
                    "Validation loss after epoch: {} is {}".format(
                        epoch, total_valid_loss
                    )
                )

                # save model
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.args["TRAIN_PARAMS"]["weights_dir"], f"epoch-{epoch}.pth"
                    ),
                )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(config_file=r"configs/trainer_config.yml")
    trainer.train()
