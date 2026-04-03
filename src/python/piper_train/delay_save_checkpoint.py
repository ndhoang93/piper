from pytorch_lightning.callbacks import ModelCheckpoint


class DelayedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that only starts saving after a minimum number of training steps."""

    def __init__(self, start_after_step=5000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_after_step = start_after_step

    def _should_skip(self, trainer):
        return trainer.global_step < self.start_after_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._should_skip(trainer):
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_end(self, trainer, pl_module):
        if not self._should_skip(trainer):
            super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._should_skip(trainer):
            super().on_train_epoch_end(trainer, pl_module)
