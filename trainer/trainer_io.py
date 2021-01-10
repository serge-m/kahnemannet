import datetime
from pathlib import Path
from typing import Callable, Union

from fastprogress.fastprogress import master_bar, progress_bar
from pytorch_nn_tools.train.checkpoint import CheckpointSaver
from pytorch_nn_tools.train.metrics.processor import MetricType
from torch.utils.tensorboard import SummaryWriter


class TrainerIO:
    def __init__(self, log_dir: Union[Path, str], experiment_name: str,
                 checkpoint_condition: Callable[[MetricType], bool]):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.path_experiment = self.log_dir.joinpath(experiment_name)
        self.path_checkpoints = self.path_experiment.joinpath("checkpoints")

        self.checkpoint_saver = CheckpointSaver(self.path_checkpoints, logger=DummyLogger())
        self.checkpoint_condition = checkpoint_condition

        path_logs = self.path_experiment.joinpath(f"{self.experiment_name}_{now_as_str()}")
        path_logs.mkdir(exist_ok=True, parents=True)

        self.tb_summary_writer = SummaryWriter(path_logs)

        self.pbars = PBars()

    def load_last(self, start_epoch: int, end_epoch: int, model, optimizer, scheduler) -> int:
        last = self.checkpoint_saver.find_last(start_epoch, end_epoch)
        if last is not None:
            print(f"found pretrained results for epoch {last}. Loading...")
            self.checkpoint_saver.load(model, optimizer, scheduler, last)
            return last + 1
        else:
            return start_epoch

    def save_checkpoint(self, metrics: MetricType, model, optimizer, scheduler, epoch):
        if self.checkpoint_condition(metrics):
            self.checkpoint_saver.save(model, optimizer, scheduler, epoch)

    def main_progress_bar(self, iterable):
        return self.pbars.main(iterable)

    def secondary_progress_bar(self, iterable):
        return self.pbars.secondary(iterable)

    def set_main_status_msg(self, value):
        self.pbars.main_comment(value)


class PBars:
    def __init__(self):
        self._main = None
        self._second = None

    def main(self, it, **kwargs):
        self._main = master_bar(it, **kwargs)
        return self._main

    def secondary(self, it, **kwargs):
        if self._main is None:
            raise RuntimeError("Cannot instantiate secondary progress bar. The main progress bar is not set.")
        self._second = progress_bar(it, parent=self._main, **kwargs)
        return self._second

    def main_comment(self, comment):
        self._main.main_bar.comment = comment


def now_as_str():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%s_%f")


class DummyLogger:
    def debug(self, *args):
        print(*args)
