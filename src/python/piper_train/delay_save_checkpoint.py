from pytorch_lightning.callbacks import ModelCheckpoint

class DelayedStepCheckpoint(ModelCheckpoint):
    def __init__(self, start_after_step=5000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_after_step = start_after_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Chỉ chạy logic lưu nếu đã vượt qua số step tối thiểu
        if trainer.global_step >= self.start_after_step:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)