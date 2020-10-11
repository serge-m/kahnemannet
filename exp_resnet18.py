import os
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import LearningRateLogger

from model_base import ModelBase, get_main_model


def _train_dataset(path):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_dir = os.path.join(path, 'train')
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset


def _val_dataset(path):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    val_dir = os.path.join(path, 'val')
    dataset = datasets.ImageFolder(val_dir, transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize, ]))
    return dataset


class ExpResNet18(ModelBase):
    # pull out resnet names from torchvision models
    MODEL_NAMES = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )

    _MAIN_MODEL = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = models.__dict__['resnet18'](pretrained=kwargs['pretrained'])

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=_train_dataset(self.data_path),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            _val_dataset(self.data_path),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()


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


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--data-path', metavar='DIR', type=str,
                               help='path to dataset', required=True)
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('--seed', type=int, default=42,
                               help='seed for initializing training.')

    model_cls = get_main_model(globals())
    print("Using model class {}".format(model_cls))
    parser = model_cls.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler=True,
        deterministic=True,
        max_epochs=90,
    )
    args = parser.parse_args()
    main(args, model_cls)


if __name__ == '__main__':
    run_cli()
