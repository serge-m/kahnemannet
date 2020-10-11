from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.core import LightningModule


def _aggregate_and_log(log: Callable, stage: str, outputs):
    for metric_name in [f"{stage}_loss", f"{stage}_acc1", f"{stage}_acc5"]:
        metric = torch.stack([output[metric_name] for output in outputs]).mean().cpu().item()
        log(metric_name, metric, on_epoch=True, prog_bar=True)


class ModelBase(LightningModule):
    def __init__(
            self,
            lr: float,
            momentum: float,
            weight_decay: int,
            data_path: str,
            batch_size: int,
            workers: int,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = F.cross_entropy(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        output = OrderedDict({
            'loss': loss,
            'acc1': acc1,
            'acc5': acc5,
        })
        return output

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = F.cross_entropy(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        output = OrderedDict({
            'val_loss': loss,
            'val_acc1': acc1,
            'val_acc5': acc5,
        })
        return output

    def validation_epoch_end(self, outputs):
        _aggregate_and_log(self.log, "val", outputs)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: 0.1 ** (epoch // 30)
        )
        return [optimizer], [scheduler]

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, outputs):
        _aggregate_and_log(self.log, "test", outputs)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        return parser


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main(args: Namespace, model_cls) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.distributed_backend == 'ddp':
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    model = model_cls(**vars(args))
    lr_logger = LearningRateLogger(logging_interval='step')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[lr_logger])

    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


def get_main_model(lst_objects, label='_MAIN_MODEL'):
    model_classes = [v for k, v in lst_objects.items() if getattr(v, label, None)]
    assert len(model_classes) == 1, f"there must be exactly one model defining '{label}' to work with this script"
    return model_classes[0]
