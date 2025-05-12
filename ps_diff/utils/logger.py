from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only


class SemlLogger(Logger):
    def __init__(self, experiment, logdir):
        super().__init__()
        self.experiment = experiment
        self.logdir = logdir

    @property
    def name(self):
        return "SemlLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for key, val in metrics.items():
            if key not in self.experiment.current_run.info:
                self.experiment.current_run.info[key] = []
            self.experiment.current_run.info[key].append(val)
